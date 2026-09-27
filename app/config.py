"""Single place for every tunable value.

Anything an operator might want to change without editing code is read from an
environment variable, with a default that works out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- data -------------------------------------------------------------------
DATASET_DIR = Path(
    os.environ.get("TILES_DATASET_DIR", PROJECT_ROOT / "data" / "be-mlsys-assignment-dataset")
)
CANDIDATE_TILES_DIR = DATASET_DIR / "candidate_tiles"
EVAL_SET_DIR = DATASET_DIR / "eval_set"
EVAL_LABELS_CSV = DATASET_DIR / "eval_labels.csv"

# --- model artifacts --------------------------------------------------------
ARTIFACTS_DIR = Path(os.environ.get("TILES_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
ONNX_MODEL_PATH = Path(os.environ.get("TILES_MODEL_PATH", ARTIFACTS_DIR / "model.onnx"))
LABELS_PATH = Path(os.environ.get("TILES_LABELS_PATH", ARTIFACTS_DIR / "labels.json"))
CHECKPOINT_PATH = ARTIFACTS_DIR / "resnet18_eurosat.pt"

# --- storage ----------------------------------------------------------------
DATABASE_PATH = Path(os.environ.get("TILES_DB_PATH", PROJECT_ROOT / "tiles.db"))

# --- preprocessing ----------------------------------------------------------
# Tiles arrive at 64x64. ResNet-18 was pretrained at 224x224; 128 is a
# compromise that keeps the pretrained filters useful without paying for a
# 224x224 forward pass on every tile.
INPUT_SIZE = 128

# ImageNet statistics. The pretrained weights expect inputs normalised with
# these exact numbers, so they are part of the model contract, not a free choice.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tiles larger than this are rejected before decoding, to bound memory use.
MAX_UPLOAD_BYTES = int(os.environ.get("TILES_MAX_UPLOAD_BYTES", 8 * 1024 * 1024))

# --- decision policy --------------------------------------------------------
# Predictions at or above this confidence are stored as "accepted"; the rest are
# stored as "needs_review". Measured on the eval set, not guessed: see
# scripts/evaluate.py and artifacts/model_card.md.
CONFIDENCE_THRESHOLD = float(os.environ.get("TILES_CONFIDENCE_THRESHOLD", 0.80))

STATUS_ACCEPTED = "accepted"
STATUS_NEEDS_REVIEW = "needs_review"

# --- training ---------------------------------------------------------------
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
FROZEN_EPOCHS = 2  # train only the new head
FINETUNE_EPOCHS = 6  # then unfreeze the whole backbone
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-4
