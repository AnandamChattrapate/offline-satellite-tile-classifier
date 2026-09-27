"""Export the trained checkpoint to ONNX, then prove the export is faithful.

    .venv/bin/python -m scripts.export_onnx

Why export at all: the deployed box then needs onnxruntime (~15 MB) instead of
torch (~250 MB), and the model becomes a single self-contained file with a hash
we can record on every prediction.

The parity check at the end is the point of this script. An export that silently
produces different numbers would be worse than no export, so nothing downstream
is allowed to trust the file until torch and onnxruntime agree on it.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import onnxruntime as ort
import torch

from app.config import CHECKPOINT_PATH, INPUT_SIZE, LABELS_PATH, ONNX_MODEL_PATH
from scripts.train import build_model

# Largest acceptable difference between torch and onnxruntime outputs.
# Float arithmetic reorders between backends, so exact equality is not a
# realistic requirement; 1e-4 on logits is far below anything that could change
# a predicted label.
PARITY_TOLERANCE = 1e-4


def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]

    model = build_model(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # turns off dropout and freezes batch-norm statistics

    # A dummy input only defines the graph's shapes; its values are irrelevant.
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        (dummy,),
        str(ONNX_MODEL_PATH),
        input_names=["tile"],
        output_names=["logits"],
        # Allow a variable batch size so the same file can score one tile from
        # the API or a whole batch from the evaluation script.
        dynamic_axes={"tile": {0: "batch"}, "logits": {0: "batch"}},
        # Opset 18 is what this torch version's exporter produces natively.
        # Asking for 17 makes it try a version downgrade that fails on this
        # graph, prints a traceback, and falls back to 18 anyway.
        opset_version=18,
        # Keep the weights inside the .onnx file. By default the exporter writes
        # them to a sidecar .onnx.data, which would make the file hash cover the
        # graph but not the weights - two different models could then share one
        # model_id, and the provenance recorded on every row would be worthless.
        external_data=False,
    )
    print(f"exported: {ONNX_MODEL_PATH}")

    # --- parity check -------------------------------------------------------
    sample = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        torch_logits = model(sample).numpy()

    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"tile": sample.numpy()})[0]

    largest_difference = float(np.abs(torch_logits - onnx_logits).max())
    print(f"max |torch - onnx| = {largest_difference:.3e} (tolerance {PARITY_TOLERANCE:.0e})")
    if largest_difference > PARITY_TOLERANCE:
        raise SystemExit("ONNX export does not match torch output; refusing to ship it")

    if torch_logits.argmax(axis=1).tolist() != onnx_logits.argmax(axis=1).tolist():
        raise SystemExit("ONNX export changes the predicted class; refusing to ship it")

    LABELS_PATH.write_text(json.dumps(labels, indent=2) + "\n")
    digest = hashlib.sha256(ONNX_MODEL_PATH.read_bytes()).hexdigest()
    size_mb = ONNX_MODEL_PATH.stat().st_size / 1e6

    print(f"labels:  {labels}")
    print(f"size:    {size_mb:.1f} MB")
    print(f"sha256:  {digest}")
    print("parity check passed")


if __name__ == "__main__":
    main()
