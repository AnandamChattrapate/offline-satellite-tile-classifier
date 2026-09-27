# Offline satellite tile classification service

A thin slice of the system described in [DESIGN.md](DESIGN.md): one endpoint takes a satellite image tile, classifies its land-use type with a locally-run model, and stores the result; an analyst queries the stored results. No internet at run time, CPU only.

- **Part 1** — design note: [DESIGN.md](DESIGN.md)
- **Part 2** — working code: this repository, run instructions below
- **Part 3** — answers: [ANSWERS.md](ANSWERS.md)

## Results, measured

| | |
|---|---|
| Model | ResNet-18, ImageNet pretrained, fine-tuned on `candidate_tiles` |
| Serving | ONNX Runtime, CPU, single 44.8 MB `model.onnx` |
| Validation accuracy | 96.67% (210 held-out tiles from `candidate_tiles`) |
| **Eval-set accuracy** | **96.19%** (202/210 tiles, never seen during training) |
| Threshold 0.80 | 193 accepted at 99.5% accuracy, 17 flagged, 87.5% of all errors flagged |
| End-to-end throughput | 115 tiles/s, upload → inference → SQLite write |
| Tests | 27 passing |

## Run it

Python 3.11 or 3.12. Python 3.14 does not yet have wheels for all of these.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
```

Unzip the dataset so that `data/be-mlsys-assignment-dataset/` contains `candidate_tiles/`, `eval_set/` and `eval_labels.csv`.

### 1. Train (once, needs network for the ImageNet weights)

```bash
.venv/bin/python -m scripts.train
```

Takes about 100 seconds on a laptop CPU. Real output:

```
classes: ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
train tiles: 840  validation tiles: 210
[head]     epoch 1/2 loss=1.5131 val_acc=0.6952 (7.1s)  <- best so far
[head]     epoch 2/2 loss=0.8956 val_acc=0.8048 (7.1s)  <- best so far
[finetune] epoch 1/6 loss=0.3554 val_acc=0.9286 (13.8s)  <- best so far
[finetune] epoch 2/6 loss=0.1550 val_acc=0.9667 (13.2s)  <- best so far
[finetune] epoch 3/6 loss=0.0828 val_acc=0.9667 (13.2s)
[finetune] epoch 4/6 loss=0.0771 val_acc=0.9571 (13.0s)
[finetune] epoch 5/6 loss=0.1120 val_acc=0.9476 (13.1s)
[finetune] epoch 6/6 loss=0.0499 val_acc=0.9476 (13.1s)

best validation accuracy: 0.9667 (from finetune/epoch2)
```

The best-scoring weights are kept, not the last ones.

### 2. Export to ONNX, with a parity check

```bash
.venv/bin/python -m scripts.export_onnx
```

```
exported: artifacts/model.onnx
max |torch - onnx| = 3.576e-06 (tolerance 1e-04)
size:    44.8 MB
sha256:  c9b3a21bf799f70145407e996e5c95fbd26c33a1dd585df0a0ff98c3cd493aa1
parity check passed
```

The script exits non-zero if the ONNX output disagrees with torch, so a bad export can never reach the service. From here on, nothing needs torch.

### 3. Measure it on the eval set

```bash
.venv/bin/python -m scripts.evaluate
```

```
model:    resnet18-eurosat@c9b3a21bf799
accuracy: 0.9619  (202/210)

per-class recall
  AnnualCrop   0.967   Forest   0.967   Highway  1.000   Industrial  0.967
  Residential  1.000   River    0.900   SeaLake  0.933

confusion matrix (rows = truth, columns = prediction)
             Annual  Forest  Highwa  Indust  Reside   River  SeaLak
  AnnualCrop     29       0       0       0       0       1       0
  Forest          0      29       0       0       0       0       1
  Highway         0       0      30       0       0       0       0
  Industrial      0       0       1      29       0       0       0
  Residential     0       0       0       0      30       0       0
  River           0       0       3       0       0      27       0
  SeaLake         1       1       0       0       0       0      28

confidence vs correctness
  [0.00, 0.50)       3     0.000
  [0.50, 0.70)       9     0.556
  [0.70, 0.80)       5     1.000
  [0.80, 0.90)       6     1.000
  [0.90, 0.99)      57     0.982
  [0.99, 1.01)     130     1.000

threshold sweep
  thresh  flagged  errors_accepted  accepted_accuracy
    0.50        3                5             0.9758
    0.60        5                3             0.9854
    0.70       12                1             0.9949
    0.80       17                1             0.9948
    0.90       23                1             0.9947
    0.95       41                1             0.9941
    0.99       80                0             1.0000

threshold 0.80 (configured)
  accepted:      193 tiles, accuracy 0.995
  needs_review:   17 tiles, accuracy 0.588
  share of all mistakes that land in needs_review: 0.875
```

Those last two tables are where the threshold in `app/config.py` comes from. Note that 0.70 dominates 0.80 on this data — same single error accepted, 5 fewer tiles to review; the reasoning for keeping 0.80 anyway is in [artifacts/model_card.md](artifacts/model_card.md).

### 4. Serve it

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Interactive API docs at `http://127.0.0.1:8000/docs`.

### 5. Ingest the eval set through the live service

```bash
.venv/bin/python -m scripts.ingest_eval_set
```

Add `--in-process` to drive the ASGI app directly, for environments where binding a port is not allowed. Real output:

```
service model_id:  resnet18-eurosat@c9b3a21bf799
threshold:         0.8

tiles sent:      210
created:         210
duplicates:      0
failures:        0
stored correct:  202/210 (0.9619)
needs_review:    17
wall clock:      1.8s (114.8 tiles/s end to end)
```

Run it a second time and every tile is recognised as a duplicate — `created: 0, duplicates: 210` — with no inference re-run.

### 6. Tests

```bash
.venv/bin/python -m pytest -q
```

```
27 passed in 1.25s
```

No network, no sleeps, no shared state between tests. The API tests use a temporary database and a fake classifier; the classifier tests use the real ONNX artifact and skip cleanly if it has not been built yet.

## The API

| Method | Path | Purpose |
|---|---|---|
| POST | `/tiles` | Classify and store one tile. 201 when created, 200 with `duplicate: true` for bytes already stored. |
| GET | `/tiles` | Filter by `label`, `status`, `min_confidence`, `created_after`, `created_before`; paged with `limit`/`offset`. |
| GET | `/tiles/{tile_id}` | One stored prediction. |
| GET | `/tiles.csv` | Same filters, CSV export with one column per class probability. |
| GET | `/stats` | Class counts, review backlog, mean confidence, `model_id` breakdown. |
| GET | `/healthz` | Loaded `model_id`, labels, threshold, stored row count. |

Ingest one tile:

```bash
curl -F file=@data/be-mlsys-assignment-dataset/eval_set/tile_001.png \
     -F tile_id=tile_001 http://127.0.0.1:8000/tiles
```

Real stored row for that tile:

```json
{
  "id": 1,
  "tile_id": "tile_001",
  "content_sha256": "c7bede55c6fe976ab1b2939d44543962dd782d891913ed8c1bb5b2efd719ec7b",
  "source_filename": "tile_001.png",
  "predicted_label": "Forest",
  "confidence": 0.98338,
  "probabilities": {
    "AnnualCrop": 0.000944, "Forest": 0.98338,   "Highway": 0.001895,
    "Industrial": 0.001274, "Residential": 0.003696,
    "River": 0.003907,      "SeaLake": 0.004905
  },
  "status": "accepted",
  "model_id": "resnet18-eurosat@c9b3a21bf799",
  "created_at": "2026-09-26T09:11:52+00:00",
  "duplicate": false
}
```

What the model was unsure about:

```bash
curl "http://127.0.0.1:8000/tiles?status=needs_review&limit=3"
```

## Layout

```
app/
  config.py       every tunable value, overridable by environment variable
  preprocess.py   bytes -> model input; imported by BOTH training and serving
  classifier.py   Classifier protocol + ONNX Runtime implementation
  storage.py      all SQL; nothing else in the codebase writes SQL
  schemas.py      request/response contract
  main.py         routes and the confidence policy
scripts/
  train.py            fine-tune ResNet-18, keep the best checkpoint
  export_onnx.py      export + torch-vs-ONNX parity check
  evaluate.py         accuracy, per-class recall, confusion matrix, threshold evidence
  ingest_eval_set.py  drive the real service over HTTP
tests/              27 tests
artifacts/
  model.onnx      the only model file the service needs
  labels.json     class order, written by the training run
  eval_report.json
  model_card.md   training config, metrics, artifact hash
```

Two dependency sets on purpose: `requirements-train.txt` pulls torch (~250 MB) and is used once on a developer machine; `requirements.txt` is what the isolated box needs (~15 MB of onnxruntime). The service never imports torch.

## Stubbed or skipped, deliberately

- **No authentication.** Single trusted machine assumed.
- **No batch or async ingest.** One tile per call, synchronous. 115 tiles/s measured is well past what a single-node feeder produces.
- **Raw tile bytes are not stored,** only their hash. Re-ingest needs the original file.
- **No geo-referencing.** Tiles have no coordinates in this dataset; spatial queries would mean PostGIS, not SQLite.
- **No review endpoint.** Flagged tiles are queryable, but an analyst cannot record a verdict. This is the most important missing piece — it is the only source of ground truth the system would ever get — and it is argued for in [ANSWERS.md](ANSWERS.md).
- **No model rollover procedure.** `model_id` is recorded per row so rollover is *possible* to reason about, but nothing automates it.

## Notes

- The classifier sits behind a `Classifier` protocol. A different architecture, a torch model, or a fake for tests all drop in without touching the API or storage.
- Weights are embedded in `model.onnx` rather than a sidecar `.onnx.data`, so the file's SHA-256 covers the weights too. Otherwise two different models could share one `model_id`.
- The ONNX export is byte-reproducible: the same checkpoint always yields the same hash.
- Timestamps are UTC ISO-8601, because an isolated box's local clock cannot be trusted.
- Dataset: EuroSAT subset, Helber et al. (2018/2019), Sentinel-2 imagery (Copernicus), MIT licensed.
