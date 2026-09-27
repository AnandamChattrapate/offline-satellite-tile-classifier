# Design note — offline tile classification service

## What I assumed

The brief leaves these open, so I picked the reading that makes the system useful, and listed the questions I would actually ask at the end.

- One isolated machine. No internet, no GPU, no second node, no container orchestrator.
- Tiles are pushed to the service by a local process (a watcher on a drop directory, or an operator script), one tile per call. They are not pulled from a bucket.
- An analyst is a human with a browser or a spreadsheet on the same machine. They are not writing SQL against my schema.
- Tiles are 64×64 RGB PNGs, as supplied. Production imagery will be bigger and multi-band, which is a real risk I address at the end.
- No ground truth arrives at run time. The service cannot know whether it was right.

## The shape of the system

```
          ┌──────────────────────────────────────────────────┐
 tile ──▶ │ POST /tiles                                      │
 (PNG)    │  1 read bytes, cap size                          │
          │  2 sha256 → already stored? ──▶ return that row  │
          │  3 decode, RGB, resize 128, ImageNet normalise   │
          │  4 ONNX Runtime forward pass → 7 logits          │
          │  5 softmax → probabilities                       │
          │  6 threshold → accepted | needs_review           │
          │  7 INSERT row (label, confidence, all 7 probs,   │
          │      status, model_id, utc timestamp)            │
          └────────────────────┬─────────────────────────────┘
                               │
                        SQLite (WAL)
                               │
   ┌───────────────────────────┴──────────────────────────┐
   │ GET /tiles      filter: label, status, min_confidence,│
   │                 time range; paged, newest first       │
   │ GET /tiles/{id} one row                               │
   │ GET /tiles.csv  same filters, spreadsheet export      │
   │ GET /stats      class counts, review backlog, model   │
   │ GET /healthz    model_id, labels, threshold, row count│
   └───────────────────────────────────────────────────────┘
```

Five files do the work: `preprocess.py` (bytes → array), `classifier.py` (array → distribution), `storage.py` (all SQL), `main.py` (HTTP and policy), `config.py` (every tunable value).

## The decisions that actually mattered

### 1. Train with PyTorch, serve with ONNX Runtime

Fine-tuning a pretrained ResNet-18 is the cheapest way to a genuinely useful model on 1050 tiles: the ImageNet backbone already encodes edges and textures, so only the last layer is new. Training from scratch on this little data lands no better than hand-built colour features, for much more work.

But torch is a ~250 MB dependency, and the deployed box is isolated hardware. So training and serving are split:

| | tool | size | runs where |
|---|---|---|---|
| train | torch + torchvision | ~250 MB | developer machine, once |
| serve | onnxruntime | ~15 MB | the isolated box |

The ImageNet weights download once, at training time. The exported `model.onnx` is a single 44.8 MB file with a SHA-256. The service opens no socket, ever.

`scripts/export_onnx.py` refuses to ship an export whose outputs differ from torch's by more than 1e-4 — measured at 3.6e-06. I forced weights inline rather than into a sidecar `.onnx.data`, because otherwise the file hash would cover the graph but not the weights, and the provenance recorded on every row would be worthless.

**What would change this:** a model too large for CPU latency targets, or a need for GPU batching. Then the classifier moves behind a local inference server and the API talks to it over a socket. The `Classifier` protocol is the seam where that swap happens — nothing in the API or storage layer knows what the model is.

### 2. Store the whole distribution, not the winner

Every row keeps all 7 probabilities, plus the `model_id` that produced them.

The alternative — store the label and confidence only — is smaller and simpler, and I rejected it. The threshold is a guess that will be wrong at least once. Keeping the full vector means re-cutting it is an `UPDATE`, not a re-ingest of imagery that may no longer be on the machine. It also makes the analyst's real question answerable: *what did it almost say instead?* On this eval set, 3 of the 8 errors are River predicted as Highway, and in most of them River is the runner-up. That is visible in the stored data, and invisible if only the winner is kept.

Cost: ~200 extra bytes per tile. At a million tiles that is 200 MB, which is acceptable for this scale, and the point at which I would move probabilities to a separate table or drop to top-3.

### 3. Uncertainty is a status, not a deletion

Tiles below the threshold are stored with their predicted label intact and `status="needs_review"`. Nothing is dropped, nothing is relabelled `unknown`.

Rejecting the tile (HTTP 422) or erasing the guess both destroy information on a machine no one is watching. A flagged row with a visible best guess is something an analyst can triage; a missing row is a silent hole.

The threshold, 0.80, comes from the measurement, not from taste:

| threshold 0.80 | tiles | accuracy |
|---|---|---|
| accepted | 193 | 0.995 |
| needs_review | 17 | 0.588 |

Reviewing 8% of the tiles catches 87.5% of all mistakes, and what remains in `accepted` is 99.5% correct. That is the trade-off I would put in front of an analyst to choose: how much review effort buys how much cleanliness.

### 4. SQLite, and the exact point I would leave it

One file, no daemon to administer, real SQL, ships with Python, and WAL mode survives an unclean shutdown. On an isolated single box, Postgres buys nothing and costs an operator.

I would move to Postgres when there is a second writer, or when tiles need geo-referencing — then it is PostGIS and `ST_Contains`, which SQLite cannot do honestly. All SQL is confined to `storage.py`, so that migration touches one file.

### 5. Idempotency on the content hash

The same bytes are never scored twice: the hash is looked up first, and a repeat returns the existing row with HTTP 200 and `duplicate: true`. An unattended feeder that dies mid-batch will resend on restart, and re-scoring would inflate every count the analyst reads. This is safe precisely because inference is deterministic — there is a test asserting that.

The cost: a tile genuinely re-scored under a new model needs a deliberate re-ingest. I would handle that by making the uniqueness `(content_sha256, model_id)` when model rollover becomes real.

### 6. Synchronous ingest

The caller learns the outcome of its own tile. 115 tiles/second end to end, measured, which is far beyond anything a single-node feeder will produce. A queue would add durability across restarts and buy throughput I do not need yet, at the cost of a second moving part to explain and operate. If ingest rate ever exceeds inference rate, the fix is a queue in front of this handler, not a rewrite.

## What "querying the results" means here

Four shapes of question, and the endpoint for each:

- *Show me all the residential tiles.* `GET /tiles?label=Residential`
- *What does the model want help with?* `GET /tiles?status=needs_review`
- *What is in this batch overall?* `GET /stats` — class counts, review backlog, mean confidence, and which model produced the rows.
- *Let me work on this in a spreadsheet.* `GET /tiles.csv` — same filters, one row per tile, probabilities as flat columns.

The CSV matters more than it looks. An analyst on an air-gapped box has Excel, not a BI tool.

## What I deliberately did not build

Named here rather than hidden: authentication, batch and async ingest, storage of the raw tile bytes, geo-referencing, a review UI for the flagged tiles, a retraining pipeline, and any kind of model rollover procedure. The review queue is queryable but there is no endpoint to record an analyst's verdict — that is the first thing I would add, because it is also the only source of ground truth this system will ever get.

## Questions I would ask you

1. **Volume and rate.** Tiles per day, and bursty or steady? This decides queue or no queue, and SQLite or Postgres.
2. **Tile size and bands.** 64×64 RGB is the exercise. Real Sentinel-2 has 13 bands at several resolutions. Multi-band input changes the model's first layer and invalidates the ImageNet weights I am relying on — this is the assumption most likely to break the design.
3. **Who consumes the output, and how?** A human triaging, or a downstream system consuming labels programmatically? The second needs a stable contract and probably a callback, not a query API.
4. **Does ground truth ever arrive?** If an analyst ever corrects a label, that is the feedback loop that makes drift detection and retraining possible. Without it, the service can only ever watch its own confidence.
5. **How do model updates reach an isolated machine?** Physical media, presumably. That makes model versioning an operational procedure, and it is why `model_id` is on every row from day one.
6. **What is the cost of a wrong label?** Is a missed water body worse than a misfiled crop field? Right now every class is treated as equally important, which is almost certainly wrong, and a per-class threshold would follow directly from the answer.

## Weakest part

The 96% number. It is measured on 210 tiles from the same curated EuroSAT distribution as the training tiles — same sensor, same processing, same seasons, cloud-free, single land-use per tile. Real incoming tiles will be mixed-use, cloudy, differently illuminated, and possibly a different resolution. Nothing in the system would tell me accuracy had collapsed, because nothing here sees a true label. The confidence distribution is the only tripwire, and a model can be confidently wrong.
