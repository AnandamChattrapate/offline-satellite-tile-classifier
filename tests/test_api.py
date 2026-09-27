"""End-to-end tests of the HTTP contract, against a temporary database."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.config import STATUS_ACCEPTED, STATUS_NEEDS_REVIEW
from tests.conftest import FakeClassifier


def post_tile(client: TestClient, data: bytes, filename: str = "tile.png", **form: str):
    return client.post("/tiles", files={"file": (filename, data, "image/png")}, data=form)


def test_ingest_stores_and_returns_the_prediction(client: TestClient, tile_bytes: bytes) -> None:
    response = post_tile(client, tile_bytes, tile_id="tile_001")
    assert response.status_code == 201

    body = response.json()
    assert body["tile_id"] == "tile_001"
    assert body["predicted_label"] == "Forest"
    assert body["status"] == STATUS_ACCEPTED
    assert body["duplicate"] is False
    assert body["source_filename"] == "tile.png"
    assert len(body["content_sha256"]) == 64
    assert sum(body["probabilities"].values()) > 0.99


def test_tile_id_defaults_to_content_hash(client: TestClient, tile_bytes: bytes) -> None:
    body = post_tile(client, tile_bytes).json()
    assert body["tile_id"] == body["content_sha256"][:16]


def test_same_bytes_are_not_scored_twice(
    client: TestClient, tile_bytes: bytes, fake_classifier: FakeClassifier
) -> None:
    """A crashed feeder that resends tiles must not inflate the stored counts."""
    first = post_tile(client, tile_bytes, tile_id="tile_001")
    second = post_tile(client, tile_bytes, tile_id="tile_001_again")

    assert first.status_code == 201
    assert second.status_code == 200  # nothing created
    assert second.json()["duplicate"] is True
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["tile_id"] == "tile_001"  # the original id is kept
    assert fake_classifier.calls == 1  # inference ran once, not twice
    assert client.get("/stats").json()["total_tiles"] == 1


def test_low_confidence_is_flagged_but_still_stored(client: TestClient, tile_bytes: bytes) -> None:
    from app import main

    main.app.dependency_overrides[main.get_classifier] = lambda: FakeClassifier(
        label="River", confidence=0.41
    )
    body = post_tile(client, tile_bytes).json()

    assert body["status"] == STATUS_NEEDS_REVIEW
    assert body["predicted_label"] == "River"  # the guess is kept, not erased
    assert body["confidence"] == 0.41


def test_non_image_upload_is_rejected_with_422(client: TestClient) -> None:
    response = post_tile(client, b"not an image at all")
    assert response.status_code == 422


def test_empty_upload_is_rejected_with_400(client: TestClient) -> None:
    response = post_tile(client, b"")
    assert response.status_code == 400


def test_oversized_upload_is_rejected_with_413(client: TestClient) -> None:
    from app import main

    large = Image.new("RGB", (2000, 2000))
    buffer = io.BytesIO()
    large.save(buffer, format="PNG")

    original = main.MAX_UPLOAD_BYTES
    main.MAX_UPLOAD_BYTES = 1024
    try:
        assert post_tile(client, buffer.getvalue()).status_code == 413
    finally:
        main.MAX_UPLOAD_BYTES = original


def test_get_by_tile_id(client: TestClient, tile_bytes: bytes) -> None:
    post_tile(client, tile_bytes, tile_id="tile_042")
    assert client.get("/tiles/tile_042").json()["tile_id"] == "tile_042"
    assert client.get("/tiles/nope").status_code == 404


def test_filters_and_paging(client: TestClient, tile_bytes: bytes, other_tile_bytes: bytes) -> None:
    post_tile(client, tile_bytes, tile_id="a")
    post_tile(client, other_tile_bytes, tile_id="b")

    page = client.get("/tiles", params={"limit": 1}).json()
    assert page["total"] == 2
    assert len(page["items"]) == 1  # total is the match count, not the page length

    assert client.get("/tiles", params={"label": "Forest"}).json()["total"] == 2
    assert client.get("/tiles", params={"label": "SeaLake"}).json()["total"] == 0
    assert client.get("/tiles", params={"status": STATUS_ACCEPTED}).json()["total"] == 2
    assert client.get("/tiles", params={"min_confidence": 0.99}).json()["total"] == 0


def test_filter_value_cannot_inject_sql(client: TestClient, tile_bytes: bytes) -> None:
    post_tile(client, tile_bytes)
    response = client.get("/tiles", params={"label": "Forest'; DROP TABLE predictions;--"})
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert client.get("/stats").json()["total_tiles"] == 1  # table still there


def test_stats_aggregate(client: TestClient, tile_bytes: bytes, other_tile_bytes: bytes) -> None:
    post_tile(client, tile_bytes)
    post_tile(client, other_tile_bytes)

    stats = client.get("/stats").json()
    assert stats["total_tiles"] == 2
    assert stats["by_label"]["Forest"] == 2
    assert stats["by_status"][STATUS_ACCEPTED] == 2
    assert stats["by_model_id"]["fake-classifier@000000000000"] == 2
    assert stats["mean_confidence"] == 0.93


def test_csv_export_has_one_row_per_tile(client: TestClient, tile_bytes: bytes) -> None:
    post_tile(client, tile_bytes, tile_id="tile_001")
    response = client.get("/tiles.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert len(lines) == 2  # header plus one tile
    assert "p_Forest" in lines[0]
    assert "tile_001" in lines[1]


def test_healthz_reports_the_loaded_model(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_id"] == "fake-classifier@000000000000"
    assert len(body["labels"]) == 7
    assert 0.0 < body["confidence_threshold"] <= 1.0
