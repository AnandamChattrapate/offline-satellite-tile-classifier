"""SQLite persistence for tile predictions.

SQLite, not Postgres, because the whole point of the scenario is one isolated
machine with no services to administer. It is a file, it ships with Python, it
does real SQL, and it survives a power cut with WAL enabled. If this ever needs
concurrent writers on several nodes, that is the moment to move to Postgres -
and the query layer here is deliberately plain SQL so that move stays cheap.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id          TEXT    NOT NULL UNIQUE,
    content_sha256   TEXT    NOT NULL UNIQUE,
    source_filename  TEXT,
    predicted_label  TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    probabilities    TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    model_id         TEXT    NOT NULL,
    created_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_label   ON predictions(predicted_label);
CREATE INDEX IF NOT EXISTS idx_predictions_status  ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model   ON predictions(model_id);
"""


@dataclass(frozen=True)
class StoredPrediction:
    """One row, with probabilities already decoded from JSON."""

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

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredPrediction":
        return cls(
            id=row["id"],
            tile_id=row["tile_id"],
            content_sha256=row["content_sha256"],
            source_filename=row["source_filename"],
            predicted_label=row["predicted_label"],
            confidence=row["confidence"],
            probabilities=json.loads(row["probabilities"]),
            status=row["status"],
            model_id=row["model_id"],
            created_at=row["created_at"],
        )


def utc_now_iso() -> str:
    """Timestamps are always UTC and always ISO-8601.

    An offline box may have a wrong or drifting local clock; storing UTC at
    least removes timezone ambiguity when the results are finally collected.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PredictionStore:
    """Thin data-access layer. All SQL lives here, none of it in the API."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # WAL lets readers continue while a write is in progress, and
            # survives an unclean shutdown better than the default journal.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        return connection

    # --- writes -------------------------------------------------------------
    def insert(
        self,
        *,
        tile_id: str,
        content_sha256: str,
        source_filename: str | None,
        predicted_label: str,
        confidence: float,
        probabilities: dict[str, float],
        status: str,
        model_id: str,
    ) -> StoredPrediction:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    tile_id, content_sha256, source_filename, predicted_label,
                    confidence, probabilities, status, model_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_id,
                    content_sha256,
                    source_filename,
                    predicted_label,
                    confidence,
                    json.dumps(probabilities),
                    status,
                    model_id,
                    utc_now_iso(),
                ),
            )
            row_id = int(cursor.lastrowid)
        stored = self.get_by_id(row_id)
        assert stored is not None  # just inserted
        return stored

    # --- reads --------------------------------------------------------------
    def get_by_id(self, row_id: int) -> StoredPrediction | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (row_id,)
            ).fetchone()
        return StoredPrediction.from_row(row) if row else None

    def get_by_tile_id(self, tile_id: str) -> StoredPrediction | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE tile_id = ?", (tile_id,)
            ).fetchone()
        return StoredPrediction.from_row(row) if row else None

    def get_by_content_hash(self, content_sha256: str) -> StoredPrediction | None:
        """Used for idempotency: the same bytes are never scored twice."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE content_sha256 = ?", (content_sha256,)
            ).fetchone()
        return StoredPrediction.from_row(row) if row else None

    def query(
        self,
        *,
        label: str | None = None,
        status: str | None = None,
        min_confidence: float | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StoredPrediction], int]:
        """Filtered, paged read. Returns (rows, total matching rows).

        The filters are built as parameterised fragments, never string
        interpolation, so a crafted label value cannot become SQL.
        """
        clauses: list[str] = []
        parameters: list[object] = []

        if label is not None:
            clauses.append("predicted_label = ?")
            parameters.append(label)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            parameters.append(min_confidence)
        if created_after is not None:
            clauses.append("created_at >= ?")
            parameters.append(created_after)
        if created_before is not None:
            clauses.append("created_at <= ?")
            parameters.append(created_before)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM predictions {where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM predictions {where}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (*parameters, limit, offset),
            ).fetchall()

        return [StoredPrediction.from_row(row) for row in rows], total

    def stats(self) -> dict[str, object]:
        """What an analyst wants first: how many of each class, and what needs review."""
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            by_label = {
                row["predicted_label"]: row["n"]
                for row in connection.execute(
                    "SELECT predicted_label, COUNT(*) AS n FROM predictions "
                    "GROUP BY predicted_label ORDER BY n DESC"
                )
            }
            by_status = {
                row["status"]: row["n"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS n FROM predictions GROUP BY status"
                )
            }
            by_model = {
                row["model_id"]: row["n"]
                for row in connection.execute(
                    "SELECT model_id, COUNT(*) AS n FROM predictions GROUP BY model_id"
                )
            }
            average_confidence = connection.execute(
                "SELECT AVG(confidence) FROM predictions"
            ).fetchone()[0]
            first_seen, last_seen = connection.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM predictions"
            ).fetchone()

        return {
            "total_tiles": total,
            "by_label": by_label,
            "by_status": by_status,
            "by_model_id": by_model,
            "mean_confidence": round(average_confidence, 4) if average_confidence else None,
            "first_seen": first_seen,
            "last_seen": last_seen,
        }
