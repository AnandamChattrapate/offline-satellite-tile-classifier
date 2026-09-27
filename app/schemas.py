"""Request and response shapes.

These are the service's contract. Keeping them in one file makes it obvious what
a caller can rely on, and pydantic enforces the types at the boundary so no
hand-written validation is scattered through the route handlers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.storage import StoredPrediction


class PredictionOut(BaseModel):
    """One stored prediction, as returned by every endpoint that returns tiles."""

    id: int
    tile_id: str
    content_sha256: str
    source_filename: str | None
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    status: str
    model_id: str
    created_at: str
    duplicate: bool = Field(
        default=False,
        description="True when these bytes were already stored and no new inference ran.",
    )

    @classmethod
    def from_stored(cls, stored: StoredPrediction, duplicate: bool = False) -> "PredictionOut":
        return cls(**vars(stored), duplicate=duplicate)


class PredictionPage(BaseModel):
    """A page of results. `total` is the full match count, not the page length."""

    total: int
    limit: int
    offset: int
    items: list[PredictionOut]


class StatsOut(BaseModel):
    """Aggregates an analyst reads first."""

    total_tiles: int
    by_label: dict[str, int]
    by_status: dict[str, int]
    by_model_id: dict[str, int]
    mean_confidence: float | None
    first_seen: str | None
    last_seen: str | None


class HealthOut(BaseModel):
    """Enough to tell, from one call, that the service and model are sane."""

    status: str
    model_id: str
    labels: list[str]
    confidence_threshold: float
    database_path: str
    stored_tiles: int
