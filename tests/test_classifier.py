"""Tests for the real ONNX classifier.

Skipped rather than failed when the artifact is absent, so a fresh clone can run
the rest of the suite before training has been done.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.classifier import OnnxTileClassifier
from app.config import ONNX_MODEL_PATH
from app.preprocess import bytes_to_tensor

pytestmark = pytest.mark.skipif(
    not ONNX_MODEL_PATH.exists(), reason="model.onnx not built; run scripts.train + scripts.export_onnx"
)


@pytest.fixture(scope="module")
def classifier() -> OnnxTileClassifier:
    return OnnxTileClassifier()


def test_labels_and_model_id(classifier: OnnxTileClassifier) -> None:
    assert len(classifier.labels) == 7
    assert classifier.labels == sorted(classifier.labels)  # order is the graph's contract
    assert "@" in classifier.model_id  # name@artifact-hash


def test_prediction_is_a_valid_distribution(classifier: OnnxTileClassifier, tile_bytes: bytes) -> None:
    prediction = classifier.predict(bytes_to_tensor(tile_bytes))

    assert prediction.label in classifier.labels
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == set(classifier.labels)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    # The reported confidence must be the winning class's probability.
    assert prediction.probabilities[prediction.label] == pytest.approx(prediction.confidence)


def test_inference_is_deterministic(classifier: OnnxTileClassifier, tile_bytes: bytes) -> None:
    """Same tile, same answer, every time.

    This is what makes the idempotency shortcut safe: skipping inference for
    bytes already seen cannot change the stored result.
    """
    first = classifier.predict(bytes_to_tensor(tile_bytes))
    second = classifier.predict(bytes_to_tensor(tile_bytes))
    assert first == second


def test_batch_and_single_paths_agree(classifier: OnnxTileClassifier, tile_bytes: bytes, other_tile_bytes: bytes) -> None:
    """The evaluation script batches; the API does not. They must not diverge."""
    single = [classifier.predict(bytes_to_tensor(raw)) for raw in (tile_bytes, other_tile_bytes)]
    batched = classifier.predict_batch(
        np.concatenate([bytes_to_tensor(tile_bytes), bytes_to_tensor(other_tile_bytes)])
    )

    for prediction, row in zip(single, batched):
        assert prediction.confidence == pytest.approx(float(row.max()), abs=1e-5)


def test_wrong_input_shape_is_rejected(classifier: OnnxTileClassifier) -> None:
    with pytest.raises(ValueError):
        classifier.predict(np.zeros((3, 128, 128), dtype=np.float32))


def test_missing_artifact_raises_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="scripts.train"):
        OnnxTileClassifier(model_path=tmp_path / "absent.onnx")
