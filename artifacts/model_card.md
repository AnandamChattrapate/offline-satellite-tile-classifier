# Model card — resnet18-eurosat@c9b3a21bf799

## Identity

| | |
|---|---|
| Artifact | `artifacts/model.onnx`, 44.8 MB, weights embedded (no sidecar file) |
| SHA-256 | `c9b3a21bf799f70145407e996e5c95fbd26c33a1dd585df0a0ff98c3cd493aa1` |
| `model_id` | `resnet18-eurosat@c9b3a21bf799` — recorded on every stored prediction |
| Graph | ONNX opset 18, IR version 10 |
| Input | `tile`, float32, `(batch, 3, 128, 128)` |
| Output | `logits`, float32, `(batch, 7)` — raw scores, softmax applied in application code |
| Runtime | ONNX Runtime 1.30.0, CPU execution provider, 1 intra-op thread |

The export is byte-reproducible: the same checkpoint always produces this hash.

## Task

Single-label land-use classification of satellite image tiles into 7 classes, in fixed index order:

`AnnualCrop, Forest, Highway, Industrial, Residential, River, SeaLake`

Order comes from `sorted()` over the class directory names and is written to `artifacts/labels.json` by the training run. Index 0 must mean `AnnualCrop` at serving time or every label is wrong.

## Training data

EuroSAT subset supplied with the assignment: 1050 tiles, 150 per class, 64×64 RGB PNG, Sentinel-2 (Copernicus), MIT licensed, Helber et al. (2018/2019).

Split 80/20 per class with seed 42 — 840 train, 210 validation. The separate 210-tile `eval_set` was not used for training, validation, or any tuning decision.

## Procedure

| | |
|---|---|
| Base | torchvision `resnet18`, `IMAGENET1K_V1` weights |
| Head | final layer replaced with `Linear(512, 7)` |
| Preprocessing | resize 64→128 bilinear, /255, ImageNet mean/std normalisation |
| Augmentation | random horizontal flip, vertical flip, 90° rotations |
| Stage 1 | 2 epochs, backbone frozen, Adam lr 1e-3 |
| Stage 2 | 6 epochs, all layers trainable, Adam lr 1e-4 |
| Loss | cross-entropy |
| Batch size | 32 |
| Seed | 42 |
| Selection | highest validation accuracy, not the last epoch |
| Cost | ~100 s on a laptop CPU |

Selected checkpoint: finetune epoch 2, validation accuracy 0.9667.

Augmentation is restricted to flips and rotations because satellite tiles have no canonical orientation, so those transforms produce images the sensor could genuinely have produced. Colour jitter was excluded: it would distort the brightness and hue cues that separate water from forest.

## Measured performance

On the 210-tile `eval_set`, scored through the exported ONNX file and the serving preprocessing path.

**Accuracy: 0.9619 (202/210)**

Per-class recall:

| class | recall | |
|---|---|---|
| Highway | 1.000 | 30/30 |
| Residential | 1.000 | 30/30 |
| AnnualCrop | 0.967 | 29/30 |
| Forest | 0.967 | 29/30 |
| Industrial | 0.967 | 29/30 |
| SeaLake | 0.933 | 28/30 |
| River | 0.900 | 27/30 |

All 8 errors:

| tile | true | predicted | confidence | status |
|---|---|---|---|---|
| tile_166 | SeaLake | Forest | 0.979 | **accepted** |
| tile_053 | River | Highway | 0.696 | needs_review |
| tile_111 | Industrial | Highway | 0.630 | needs_review |
| tile_009 | Forest | SeaLake | 0.598 | needs_review |
| tile_050 | River | Highway | 0.551 | needs_review |
| tile_130 | AnnualCrop | River | 0.433 | needs_review |
| tile_174 | SeaLake | AnnualCrop | 0.381 | needs_review |
| tile_046 | River | Highway | 0.355 | needs_review |

The dominant error mode is River read as Highway (3 of 8). Both are narrow dark ribbons crossing a 64×64 tile.

## Calibration and the operating point

| confidence | tiles | accuracy |
|---|---|---|
| < 0.50 | 3 | 0.000 |
| 0.50–0.70 | 9 | 0.556 |
| 0.70–0.80 | 5 | 1.000 |
| 0.80–0.90 | 6 | 1.000 |
| 0.90–0.99 | 57 | 0.982 |
| ≥ 0.99 | 130 | 1.000 |

Mistakes cluster at low confidence, which is what makes a threshold useful.

**Threshold 0.80** (`TILES_CONFIDENCE_THRESHOLD`):

- accepted: 193 tiles, accuracy 0.995
- needs_review: 17 tiles (8.1%), accuracy 0.588
- 87.5% of all errors land in needs_review
- exactly 1 error passes as accepted

Full sweep, since the operating point is a cost decision and not a model property:

| threshold | flagged | errors accepted | accepted accuracy |
|---|---|---|---|
| 0.50 | 3 | 5 | 0.9758 |
| 0.60 | 5 | 3 | 0.9854 |
| 0.70 | 12 | 1 | 0.9949 |
| **0.80** | **17** | **1** | **0.9948** |
| 0.90 | 23 | 1 | 0.9947 |
| 0.95 | 41 | 1 | 0.9941 |
| 0.99 | 80 | 0 | 1.0000 |

Read honestly, 0.70 dominates 0.80 on this data: same single error accepted, 5 fewer tiles to review. The default is 0.80 anyway, for one reason: 5 tiles out of 210 is noise at this sample size, and the wider margin costs little while leaving headroom if incoming imagery drifts and confidence sags. Catching all 8 errors requires 0.99, which means reviewing 38% of every batch — a defensible choice only if a wrong label is expensive.

This is the table I would put in front of an analyst to set the threshold, rather than deciding it myself.

## Limits

- Accuracy is measured on curated EuroSAT imagery: one sensor, one processing chain, cloud-free, one land-use per tile. It does not predict performance on mixed-use, cloudy, or differently illuminated tiles.
- 3-channel RGB only. Multi-band Sentinel-2 input would require a different first layer and invalidates the ImageNet transfer this model depends on.
- Fixed 128×128 input. Larger tiles are downscaled and lose texture, silently.
- The model cannot express "none of these" or "several of these" — softmax always sums to 1 across the 7 classes.
- No confidence calibration was performed (no temperature scaling). The probabilities happen to be usefully ordered, which is not the same as being calibrated.
