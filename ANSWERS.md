# Part 3 — problem solving

## 1. The classifier is wrong about 30% of the time. What do you do, and how do you decide whether it is good enough at all?

First I stop treating "30% wrong" as one number, because it hides everything that matters.

**Find out how it is wrong.** Per-class recall and a confusion matrix, which `scripts/evaluate.py` already prints. 30% spread evenly across 7 classes is a weak model. 30% concentrated in two classes that look alike is a labelling or class-definition problem, and merging or splitting those classes may fix more than any retraining. On my actual model the errors are not uniform: 3 of 8 are River read as Highway. Both are long thin dark ribbons in a 64×64 tile. That tells me something specific and actionable.

**Then check whether the model knows it is wrong.** This is the question that decides whether a 70%-accurate model is useful. I bin predictions by confidence and measure accuracy in each bin:

| confidence | tiles | accuracy |
|---|---|---|
| < 0.50 | 3 | 0.000 |
| 0.50–0.70 | 9 | 0.556 |
| 0.70–0.90 | 11 | 1.000 |
| 0.90–0.99 | 57 | 0.982 |
| ≥ 0.99 | 130 | 1.000 |

A model whose mistakes cluster at low confidence is useful even at 70% accuracy: route the confident majority through automatically, send the rest to a human. A model that is confidently wrong is dangerous at any accuracy, because nothing downstream can tell the difference.

**Then set the operating point from that table, not from a preference.** Sweeping the threshold on my model:

| threshold | flagged | errors accepted | accepted accuracy |
|---|---|---|---|
| 0.60 | 5 | 3 | 0.9854 |
| 0.70 | 12 | 1 | 0.9949 |
| 0.80 | 17 | 1 | 0.9948 |
| 0.90 | 23 | 1 | 0.9947 |
| 0.99 | 80 | 0 | 1.0000 |

Read honestly, 0.70 dominates my configured 0.80 here: the same single error gets through, with 5 fewer tiles to review. I kept 0.80 for margin, because 5 tiles out of 210 is noise at this sample size — but the table is the artefact that matters, and it is the analyst's decision, not mine. Catching all 8 errors needs 0.99, which means reviewing 38% of every batch.

The one error that passes at any threshold below 0.99 is tile_166: true SeaLake, predicted Forest, confidence 0.979. Confidence filtering cannot catch that.

**"Good enough" is not a property of the model.** It is a question about the task, and it has two parts I would put back to you:

- What does a wrong label cost, and is it symmetric? Missing a water body is probably not the same as misfiling a crop field. If it is asymmetric, the answer is per-class thresholds, not a better model.
- What is the alternative? If an analyst currently labels tiles by hand at 50 an hour, a model that clears 90% of them at 99% accuracy is valuable at 30% overall error. If it feeds an automated decision with no human in the loop, 30% is unusable at any confidence.

**What I would actually do, in order:** fix the cheap things first — verify preprocessing matches training (a resize or normalisation mismatch alone can cost 20%), check for label noise in the training set, check class balance. Then augment and train longer. Only then reach for a bigger model. In my case 2 epochs of head-only training gave 80%, and unfreezing the backbone gave 96.7% — the largest single win came from a training-procedure change, not a new architecture.

## 2. It runs offline, with no one watching. A month after deployment, how do you know it is still working correctly?

I cannot measure accuracy, because no true labels arrive. So I monitor the things that *precede* accuracy loss, and I make the service leave enough evidence behind to reconstruct what happened.

**What the service already records, per row:** `model_id` (name plus the artifact's SHA-256), the full probability vector, the status, and a UTC timestamp. That means a month of operation is itself the audit trail.

**Signals I would watch, in order of how much they tell me:**

1. **Confidence distribution over time.** The mean confidence on my eval run is 0.954, and 62% of tiles land above 0.99. If next month's tiles average 0.72, the input has changed even though nothing has errored. This is the closest thing to a drift alarm available without labels.
2. **Share flagged `needs_review`.** 8% today. A jump to 40% is the same signal, expressed as workload.
3. **Class distribution.** A month of tiles over the same area should look roughly like last month's. All 7 classes present and stable is normal; a sudden 60% of one class means either the area changed or the model collapsed onto a majority class.
4. **`model_id` uniformity.** `GET /stats` groups by it. More than one value in a period that should have had one means someone swapped the artifact. This is why the hash covers the weights, not just the graph.
5. **Liveness and throughput.** Rows per day, and `GET /healthz`. Zero rows can mean a healthy service and a dead feeder — those are different faults and must be distinguishable, so the feeder needs its own heartbeat.

**Two things I would add for a real month-long deployment, and have not built here:**

- **A canary set.** Ship 30 labelled tiles with the model and score them on every startup, or nightly. If accuracy on a fixed, known set moves, the model or its environment changed — this separates "the world changed" from "my system broke", which no distribution metric can do.
- **A rolling daily summary written to disk** (counts, mean confidence, flag rate, per-class breakdown) so the evidence survives even if the database is later truncated, and can be collected without running a query.

**And the honest limit:** all of this detects change, not incorrectness. A model that is confidently wrong on genuinely new imagery will look perfectly healthy on every metric above. The only real fix is the feedback loop — an endpoint where an analyst records the correct label for a reviewed tile. That is the first thing I would build next, and it is why I flag rather than drop uncertain tiles.

## 3. Tiles arrive fine, but the stored results look wrong. How do you find the cause?

Working from the outside in, cheapest check first, and not skipping steps just because one feels likelier.

1. **Pin down "wrong".** Wrong labels, wrong rows, or wrong counts? Those are three different bugs. I would ask for two or three specific `tile_id`s the analyst believes are wrong, because "looks wrong" is often the analyst's model of the classes disagreeing with the dataset's — a real finding, but not a code defect.

2. **Reproduce one tile end to end.** Take one bad `tile_id`, `GET /tiles/{id}`, and re-upload the same file. If the duplicate response matches the stored row, storage and inference are consistent and the problem is upstream of storage. If it differs, the model or its artifact changed since that row was written — compare the row's `model_id` against `GET /healthz`. This one comparison separates "bad prediction" from "bad plumbing", and it is why the model id is stored per row.

3. **Look at the stored probability vector, not just the label.** Near-uniform probabilities across all 7 classes mean the model is receiving garbage — that pattern points at preprocessing. Confident and wrong points at the model or at genuinely out-of-distribution imagery.

4. **Check preprocessing, because it is the most likely culprit and the easiest to get wrong.** A channel swap (BGR for RGB), a missed division by 255, the wrong normalisation constants, or a different resize filter each degrade accuracy while erroring nowhere. My defence is structural: training and serving import the identical `image_to_tensor`, and a test asserts a grey tile maps to the exact value the formula predicts. If preprocessing is suspect, I would decode one tile and compare the array against the training pipeline's array element-wise.

5. **Check the artifact itself.** Hash `model.onnx` and compare it against the `model_id` on recent rows. Re-run the torch-versus-ONNX parity check. An export that silently diverged, or a file replaced by a different one, is exactly the failure this check exists to catch.

6. **Check the label mapping.** This is the highest-consequence, lowest-visibility bug in the whole system: if `labels.json` order disagrees with the order the model was trained on, every prediction is a valid-looking label attached to the wrong class. Index 0 must mean what it meant during training. My guard is that both sides derive the order from `sorted(directory names)`, and the list is written by the training run itself.

7. **Only then suspect storage.** Verify the write path with a fresh tile against an empty database, check for a mismatched column order in the insert, and confirm timestamps are UTC — a timezone bug makes rows look reordered and misleads everything above.

8. **Finally, check the input pipeline.** Is the feeder sending the tile it claims to? A filename-to-content mismatch in the feeder produces perfectly correct predictions filed under the wrong ids, which looks exactly like a broken model from the analyst's seat. The stored `content_sha256` and `source_filename` let me prove which bytes were actually classified.

## 4. What is the weakest part of your design, and what breaks it first?

**The weakest part is that the model's accuracy is unverifiable in production, and everything downstream treats its output as fact.**

The 96.19% is measured on 210 tiles drawn from the same curated EuroSAT distribution as the training tiles: one sensor, one processing chain, cloud-free, single land-use per 64×64 tile. Every number in this submission inherits that assumption.

**What breaks it first, in likelihood order:**

1. **Real tiles that are not EuroSAT-like.** Mixed land use in one tile, cloud cover, haze, a different season, a different sun angle. The model must return one of 7 labels and will do so confidently, because nothing taught it to say "none of these" or "several of these". Softmax always sums to 1; it has no way to express "this is not like anything I was trained on".
2. **A different band configuration.** Real Sentinel-2 imagery is 13 bands at several resolutions. My pipeline is hard-wired to 3-channel RGB, and the ImageNet transfer learning that makes 1050 tiles sufficient only works for 3 channels. Multi-band input does not degrade this design; it invalidates it and requires a different first layer and a different pretrained starting point.
3. **Resolution or tile size change.** The ONNX graph is exported at 128×128. A 1024×1024 tile will be squashed to fit and lose the texture the model reads. That is a silent accuracy loss, not an error.
4. **Single-node storage.** SQLite is correct for one box; it will not survive a second writer or a genuine need for spatial queries.

**The specific failure mode that worries me most** is the confidently wrong prediction: tile_166 in my own run, true SeaLake, predicted Forest at 0.979. Confidence filtering — the mechanism this whole design leans on for safety — cannot catch that class of error, and on out-of-distribution imagery that class grows. The mitigations I would reach for are a feedback endpoint so corrections become data, a canary set so drift is detectable, and per-class thresholds once I know what a wrong label costs. None of them are in this thin slice, and I would not claim the system is production-ready without at least the first.
