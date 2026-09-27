"""Shared test fixtures.

Every test runs against a temporary SQLite file and, where the model itself is
not under test, a fake classifier. No test touches the network, sleeps, or
depends on the order the tests run in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.classifier import Prediction
from app.config import CANDIDATE_TILES_DIR
from app.storage import PredictionStore

LABELS = ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"]


class FakeClassifier:
    """Returns a fixed answer, so API tests measure the API and not the model.

    It satisfies the same Classifier protocol the real one does, which is the
    whole reason the protocol exists.
    """

    def __init__(self, label: str = "Forest", confidence: float = 0.93) -> None:
        self._label = label
        self._confidence = confidence
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "fake-classifier@000000000000"

    @property
    def labels(self) -> list[str]:
        return list(LABELS)

    def predict(self, tensor: np.ndarray) -> Prediction:
        self.calls += 1
        remainder = (1.0 - self._confidence) / (len(LABELS) - 1)
        probabilities = {name: remainder for name in LABELS}
        probabilities[self._label] = self._confidence
        return Prediction(
            label=self._label,
            confidence=self._confidence,
            probabilities=probabilities,
            model_id=self.model_id,
        )


@pytest.fixture
def store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(tmp_path / "test.db")


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def client(store: PredictionStore, fake_classifier: FakeClassifier) -> TestClient:
    """A test client whose store and classifier are replaced via FastAPI's DI."""
    from app import main

    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_classifier] = lambda: fake_classifier
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def tile_bytes() -> bytes:
    """A real tile from the dataset, so decoding is exercised for real."""
    path = sorted((CANDIDATE_TILES_DIR / "Forest").glob("*.png"))[0]
    return path.read_bytes()


@pytest.fixture
def other_tile_bytes() -> bytes:
    path = sorted((CANDIDATE_TILES_DIR / "Residential").glob("*.png"))[0]
    return path.read_bytes()
