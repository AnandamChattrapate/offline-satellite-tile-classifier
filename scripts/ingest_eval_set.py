"""Push the 210 eval tiles through the running service over HTTP.

    .venv/bin/uvicorn app.main:app --port 8000     # in one shell
    .venv/bin/python -m scripts.ingest_eval_set    # in another

This is not a second evaluation path. scripts/evaluate.py measures the model;
this script exercises the real service - upload, decode, inference, threshold,
SQLite write - and leaves a populated database behind as run evidence.

It also compares the stored labels against eval_labels.csv, which answers a
question the service itself cannot: is what got *stored* actually right?
"""

from __future__ import annotations

import argparse
import csv
import time

import httpx

from app.config import EVAL_LABELS_CSV, EVAL_SET_DIR


def build_client(base_url: str, in_process: bool):
    """Talk to a running server, or to the ASGI app directly.

    The in-process mode exists for environments where binding a TCP port is not
    allowed. It exercises the same routes, dependencies, model and database code;
    the only thing it skips is the socket. httpx's own ASGI transport is
    async-only, so the synchronous Starlette client is used instead - it ships
    with FastAPI, so this adds no dependency.
    """
    if not in_process:
        return httpx.Client(base_url=base_url, timeout=30.0)

    from starlette.testclient import TestClient

    from app.main import app

    return TestClient(app)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0, help="0 means all tiles")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="drive the ASGI app directly instead of over a socket",
    )
    arguments = parser.parse_args()

    with EVAL_LABELS_CSV.open(newline="") as handle:
        truth = {row["filename"]: row["true_label"] for row in csv.DictReader(handle)}

    filenames = sorted(truth)
    if arguments.limit:
        filenames = filenames[: arguments.limit]

    with build_client(arguments.base_url, arguments.in_process) as client:
        health = client.get("/healthz").raise_for_status().json()
        print(f"service model_id:  {health['model_id']}")
        print(f"threshold:         {health['confidence_threshold']}")
        print(f"tiles already in db: {health['stored_tiles']}\n")

        created = duplicates = failures = 0
        correct = flagged = 0
        started = time.perf_counter()

        for name in filenames:
            path = EVAL_SET_DIR / name
            response = client.post(
                "/tiles",
                files={"file": (name, path.read_bytes(), "image/png")},
                data={"tile_id": name.removesuffix(".png")},
            )
            if response.status_code not in (200, 201):
                failures += 1
                print(f"  {name}: HTTP {response.status_code} {response.text[:120]}")
                continue

            body = response.json()
            created += int(response.status_code == 201)
            duplicates += int(body["duplicate"])
            correct += int(body["predicted_label"] == truth[name])
            flagged += int(body["status"] == "needs_review")

        elapsed = time.perf_counter() - started

    processed = len(filenames)
    print(f"tiles sent:      {processed}")
    print(f"created:         {created}")
    print(f"duplicates:      {duplicates}")
    print(f"failures:        {failures}")
    print(f"stored correct:  {correct}/{processed} ({correct / processed:.4f})")
    print(f"needs_review:    {flagged}")
    print(f"wall clock:      {elapsed:.1f}s ({processed / elapsed:.1f} tiles/s end to end)")


if __name__ == "__main__":
    main()
