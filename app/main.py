"""The service: ingest one tile, classify it, store it, let an analyst query it.

Run it:

    .venv/bin/uvicorn app.main:app --port 8000

The ingest path is the only write path, and it is deliberately synchronous: the
caller learns the outcome of its own tile. If throughput ever matters more than
that, the change is a queue in front of this handler, not a rewrite.
"""

from __future__ import annotations

import csv
import io
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile

from app.classifier import Classifier, OnnxTileClassifier
from app.config import (
    CONFIDENCE_THRESHOLD,
    DATABASE_PATH,
    MAX_UPLOAD_BYTES,
    STATUS_ACCEPTED,
    STATUS_NEEDS_REVIEW,
)
from app.preprocess import InvalidTileError, bytes_to_tensor, sha256_hex
from app.schemas import HealthOut, PredictionOut, PredictionPage, StatsOut
from app.storage import PredictionStore

app = FastAPI(
    title="Satellite Tile Classification Service",
    description="Offline land-use classification of satellite image tiles.",
    version="0.1.0",
)

# Both are created once at import time and shared by every request. The ONNX
# session is thread-safe for inference, and SQLite gets a fresh connection per
# operation inside the store.
_classifier: Classifier = OnnxTileClassifier()
_store = PredictionStore(DATABASE_PATH)


def get_classifier() -> Classifier:
    """Dependency so a test can substitute a fake classifier."""
    return _classifier


def get_store() -> PredictionStore:
    """Dependency so a test can substitute a temporary database."""
    return _store


def decide_status(confidence: float) -> str:
    """The whole uncertainty policy, in one place.

    Low-confidence tiles are stored, not dropped and not relabelled. The
    prediction stays visible, the row is flagged for review, and the full
    probability vector is kept so the threshold can be re-cut later without
    re-reading a single tile.
    """
    return STATUS_ACCEPTED if confidence >= CONFIDENCE_THRESHOLD else STATUS_NEEDS_REVIEW


@app.post("/tiles", response_model=PredictionOut, status_code=201)
async def ingest_tile(
    response: Response,
    file: Annotated[UploadFile, File(description="A single image tile (PNG).")],
    tile_id: Annotated[str | None, Form(description="Caller's id; defaults to the content hash.")] = None,
    classifier: Classifier = Depends(get_classifier),
    store: PredictionStore = Depends(get_store),
) -> PredictionOut:
    """Classify one tile and store the result.

    Returns 201 for a newly stored tile, 200 for a tile whose bytes were already
    stored. Both return the same body shape, with `duplicate` distinguishing them.
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"tile larger than {MAX_UPLOAD_BYTES} bytes"
        )

    content_hash = sha256_hex(data)

    # Idempotency. An unattended feeder that crashes mid-batch will resend
    # tiles on restart; re-scoring them would inflate the analyst's counts for
    # no benefit, since the same bytes and the same model give the same answer.
    existing = store.get_by_content_hash(content_hash)
    if existing is not None:
        response.status_code = 200  # nothing was created, so 201 would be a lie
        return PredictionOut.from_stored(existing, duplicate=True)

    try:
        tensor = bytes_to_tensor(data)
    except InvalidTileError as exc:
        # 422, not 500: the request was understood and is the caller's problem.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = classifier.predict(tensor)

    stored = store.insert(
        tile_id=tile_id or content_hash[:16],
        content_sha256=content_hash,
        source_filename=file.filename,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        status=decide_status(prediction.confidence),
        model_id=prediction.model_id,
    )
    return PredictionOut.from_stored(stored)


@app.get("/tiles", response_model=PredictionPage)
def list_tiles(
    label: str | None = Query(default=None, description="Exact predicted label."),
    status: str | None = Query(default=None, description="accepted or needs_review."),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    created_after: str | None = Query(default=None, description="ISO-8601 UTC, inclusive."),
    created_before: str | None = Query(default=None, description="ISO-8601 UTC, inclusive."),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    store: PredictionStore = Depends(get_store),
) -> PredictionPage:
    """Filtered, paged list of stored predictions, newest first."""
    rows, total = store.query(
        label=label,
        status=status,
        min_confidence=min_confidence,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )
    return PredictionPage(
        total=total,
        limit=limit,
        offset=offset,
        items=[PredictionOut.from_stored(row) for row in rows],
    )


@app.get("/tiles.csv")
def export_csv(
    label: str | None = Query(default=None),
    status: str | None = Query(default=None),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=10_000, ge=1, le=100_000),
    store: PredictionStore = Depends(get_store),
    classifier: Classifier = Depends(get_classifier),
) -> Response:
    """Same filters, delivered as CSV.

    An analyst on an isolated machine has a spreadsheet, not a BI tool. The
    probability columns are flattened so the export is directly sortable.
    """
    rows, _ = store.query(label=label, status=status, min_confidence=min_confidence, limit=limit)
    class_names = classifier.labels

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "tile_id",
            "source_filename",
            "predicted_label",
            "confidence",
            "status",
            "model_id",
            "created_at",
            *(f"p_{name}" for name in class_names),
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.id,
                row.tile_id,
                row.source_filename or "",
                row.predicted_label,
                f"{row.confidence:.6f}",
                row.status,
                row.model_id,
                row.created_at,
                *(f"{row.probabilities.get(name, 0.0):.6f}" for name in class_names),
            ]
        )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="tiles.csv"'},
    )


@app.get("/stats", response_model=StatsOut)
def get_stats(store: PredictionStore = Depends(get_store)) -> StatsOut:
    """Class distribution, review backlog, and which model produced the rows."""
    return StatsOut(**store.stats())


@app.get("/tiles/{tile_id}", response_model=PredictionOut)
def get_tile(tile_id: str, store: PredictionStore = Depends(get_store)) -> PredictionOut:
    """One stored prediction by its tile id."""
    stored = store.get_by_tile_id(tile_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"no stored tile with id {tile_id!r}")
    return PredictionOut.from_stored(stored)


@app.get("/healthz", response_model=HealthOut)
def healthz(
    classifier: Classifier = Depends(get_classifier),
    store: PredictionStore = Depends(get_store),
) -> HealthOut:
    """Liveness plus the facts needed to tell which model is actually loaded.

    Returning the model id and threshold here means a check on an unattended box
    can detect a silently swapped artifact, not just that the port is open.
    """
    return HealthOut(
        status="ok",
        model_id=classifier.model_id,
        labels=classifier.labels,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        database_path=str(store.database_path),
        stored_tiles=int(store.stats()["total_tiles"]),
    )
