"""The classifier the service uses, behind an interface it can be swapped out of.

Only onnxruntime, numpy and Pillow are needed here. torch is a training-time
dependency and is deliberately not imported.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from app.config import LABELS_PATH, ONNX_MODEL_PATH
from app.preprocess import softmax


@dataclass(frozen=True)
class Prediction:
    """What one tile produced. Carries the whole distribution, not just the winner.

    Storing every probability is what lets the confidence threshold be changed
    later without re-reading a single tile.
    """

    label: str
    confidence: float
    probabilities: dict[str, float]
    model_id: str


class Classifier(Protocol):
    """The only thing the API knows about the model.

    Any implementation that satisfies this can be dropped in - a different
    architecture, a torch model, or a fake used by tests - without touching the
    API or the storage layer.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def labels(self) -> list[str]: ...

    def predict(self, tensor: np.ndarray) -> Prediction: ...


class OnnxTileClassifier:
    """Runs the exported ResNet-18 graph with ONNX Runtime."""

    def __init__(
        self,
        model_path: Path = ONNX_MODEL_PATH,
        labels_path: Path = LABELS_PATH,
        model_name: str = "resnet18-eurosat",
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"model artifact missing: {model_path}. "
                "Run 'python -m scripts.train' then 'python -m scripts.export_onnx'."
            )

        self._labels: list[str] = json.loads(labels_path.read_text())

        # One session, created at startup and reused for every request. Creating
        # it per request would cost tens of milliseconds and megabytes each time.
        # Threads are pinned to 1 because uvicorn already serves requests
        # concurrently; letting both layers expand threads oversubscribes the CPU.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # The artifact hash goes on every stored row. A month after deployment
        # this is how you prove which exact file produced a given prediction.
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self._model_id = f"{model_name}@{digest[:12]}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def predict(self, tensor: np.ndarray) -> Prediction:
        """Score one preprocessed tile of shape (1, 3, H, W)."""
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"expected shape (1, 3, H, W), got {tensor.shape}")

        logits = self._session.run([self._output_name], {self._input_name: tensor})[0]
        probabilities = softmax(logits)[0]
        best = int(np.argmax(probabilities))

        return Prediction(
            label=self._labels[best],
            confidence=float(probabilities[best]),
            probabilities={
                label: float(value) for label, value in zip(self._labels, probabilities)
            },
            model_id=self._model_id,
        )

    def predict_batch(self, tensors: np.ndarray) -> np.ndarray:
        """Score a stacked batch, used by scripts/evaluate.py. Returns probabilities."""
        logits = self._session.run([self._output_name], {self._input_name: tensors})[0]
        return softmax(logits)
