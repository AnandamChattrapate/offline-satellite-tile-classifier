"""Measure the exported ONNX model on the untouched eval set.

    .venv/bin/python -m scripts.evaluate

This runs the same ONNX file and the same preprocessing the API uses, so the
number it prints is a statement about the deployed system, not about a
research notebook.

Three outputs, each with a job:
  1. accuracy and per-class recall - where the model is weak.
  2. confusion matrix - which classes it confuses with which.
  3. confidence-vs-correctness table - the evidence for choosing a threshold.
     Without this, any threshold is a guess.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.classifier import OnnxTileClassifier
from app.config import ARTIFACTS_DIR, CONFIDENCE_THRESHOLD, EVAL_LABELS_CSV, EVAL_SET_DIR
from app.preprocess import bytes_to_tensor

BATCH_SIZE = 32
CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.01)]


def load_ground_truth(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {row["filename"]: row["true_label"] for row in csv.DictReader(handle)}


def main() -> None:
    truth = load_ground_truth(EVAL_LABELS_CSV)
    classifier = OnnxTileClassifier()
    labels = classifier.labels

    filenames = sorted(truth)
    records: list[dict[str, object]] = []

    # Batched inference: one ONNX call per 32 tiles instead of per tile. The
    # dynamic batch axis set during export is what makes this possible.
    for start in range(0, len(filenames), BATCH_SIZE):
        chunk = filenames[start : start + BATCH_SIZE]
        tensors = np.concatenate(
            [bytes_to_tensor((EVAL_SET_DIR / name).read_bytes()) for name in chunk]
        )
        probabilities = classifier.predict_batch(tensors)

        for name, row in zip(chunk, probabilities):
            best = int(np.argmax(row))
            records.append(
                {
                    "filename": name,
                    "true_label": truth[name],
                    "predicted_label": labels[best],
                    "confidence": float(row[best]),
                    "correct": labels[best] == truth[name],
                }
            )

    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    accuracy = correct / total

    print(f"model:    {classifier.model_id}")
    print(f"eval set: {total} tiles from {EVAL_SET_DIR.name}")
    print(f"accuracy: {accuracy:.4f}  ({correct}/{total})\n")

    # --- per-class recall ---------------------------------------------------
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    for record in records:
        per_class_total[record["true_label"]] += 1
        per_class_correct[record["true_label"]] += int(record["correct"])

    print("per-class recall (of the tiles that truly are this class, how many were found)")
    for label in labels:
        count = per_class_total[label]
        if count:
            print(f"  {label:<12} {per_class_correct[label]/count:.3f}  ({per_class_correct[label]}/{count})")

    # --- confusion matrix --------------------------------------------------
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for record in records:
        matrix[index[record["true_label"]], index[record["predicted_label"]]] += 1

    print("\nconfusion matrix (rows = truth, columns = prediction)")
    print("            " + " ".join(f"{label[:6]:>7}" for label in labels))
    for label in labels:
        row = matrix[index[label]]
        print(f"  {label:<10}" + " ".join(f"{value:>7}" for value in row))

    # --- confidence vs correctness -----------------------------------------
    print("\nconfidence vs correctness")
    print("  bin            tiles  accuracy")
    for low, high in CONFIDENCE_BINS:
        bucket = [r for r in records if low <= r["confidence"] < high]
        if bucket:
            bucket_accuracy = sum(1 for r in bucket if r["correct"]) / len(bucket)
            print(f"  [{low:.2f}, {high:.2f})   {len(bucket):>5}     {bucket_accuracy:.3f}")

    # --- what the current threshold would do -------------------------------
    accepted = [r for r in records if r["confidence"] >= CONFIDENCE_THRESHOLD]
    flagged = [r for r in records if r["confidence"] < CONFIDENCE_THRESHOLD]
    accepted_accuracy = (
        sum(1 for r in accepted if r["correct"]) / len(accepted) if accepted else 0.0
    )
    flagged_accuracy = sum(1 for r in flagged if r["correct"]) / len(flagged) if flagged else 0.0
    errors_caught = (
        sum(1 for r in flagged if not r["correct"]) / (total - correct) if total != correct else 0.0
    )

    # --- threshold sweep ----------------------------------------------------
    # The operating point is a business decision: how much review effort buys
    # how much cleanliness. This table is what that decision should be made on.
    print("\nthreshold sweep")
    print("  thresh  flagged  errors_accepted  accepted_accuracy")
    for candidate in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        kept = [r for r in records if r["confidence"] >= candidate]
        if not kept:
            continue
        let_through = sum(1 for r in kept if not r["correct"])
        kept_accuracy = sum(1 for r in kept if r["correct"]) / len(kept)
        print(
            f"  {candidate:>6.2f}  {total - len(kept):>7}  {let_through:>15}  {kept_accuracy:>17.4f}"
        )

    print(f"\nthreshold {CONFIDENCE_THRESHOLD:.2f} (configured)")
    print(f"  accepted:     {len(accepted):>4} tiles, accuracy {accepted_accuracy:.3f}")
    print(f"  needs_review: {len(flagged):>4} tiles, accuracy {flagged_accuracy:.3f}")
    print(f"  share of all mistakes that land in needs_review: {errors_caught:.3f}")

    report = {
        "model_id": classifier.model_id,
        "eval_tiles": total,
        "accuracy": round(accuracy, 4),
        "per_class_recall": {
            label: round(per_class_correct[label] / per_class_total[label], 4)
            for label in labels
            if per_class_total[label]
        },
        "confusion_matrix": {"labels": labels, "rows_are_truth": matrix.tolist()},
        "threshold": CONFIDENCE_THRESHOLD,
        "accepted_count": len(accepted),
        "accepted_accuracy": round(accepted_accuracy, 4),
        "needs_review_count": len(flagged),
        "needs_review_accuracy": round(flagged_accuracy, 4),
        "share_of_errors_flagged": round(errors_caught, 4),
    }
    output_path = ARTIFACTS_DIR / "eval_report.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
