"""Fine-tune ResNet-18 on candidate_tiles and save a checkpoint.

Run once, on a machine with network access:

    .venv/bin/python -m scripts.train

The ImageNet weights download here, at training time. The deployed service
loads only the exported ONNX file and never opens a socket.

The eval_set is never touched by this script. Validation uses a held-out slice
of candidate_tiles, so the eval set stays an honest final measurement.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from app.config import (
    INPUT_SIZE,
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CANDIDATE_TILES_DIR,
    CHECKPOINT_PATH,
    FINETUNE_EPOCHS,
    FINETUNE_LEARNING_RATE,
    FROZEN_EPOCHS,
    HEAD_LEARNING_RATE,
    LABELS_PATH,
    RANDOM_SEED,
    VALIDATION_FRACTION,
)
from app.preprocess import image_to_tensor


def set_seeds(seed: int) -> None:
    """Make the run reproducible, so reported numbers can be re-derived."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_labels(root: Path) -> list[str]:
    """Class names come from the directory names, sorted for a stable order.

    The order matters: it fixes which output index means which class, and the
    same list is written to labels.json for the serving side to read.
    """
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def stratified_split(
    root: Path, labels: list[str], validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Split per class, so both sides keep all 7 classes in equal proportion.

    A plain random split over the whole file list could leave a class with very
    few validation examples, which makes the validation number noisy.
    """
    rng = random.Random(seed)
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []

    for index, label in enumerate(labels):
        files = sorted((root / label).glob("*.png"))
        rng.shuffle(files)
        cut = int(len(files) * validation_fraction)
        val_items.extend((path, index) for path in files[:cut])
        train_items.extend((path, index) for path in files[cut:])

    rng.shuffle(train_items)
    return train_items, val_items


class TileDataset(Dataset):
    """Reads PNG tiles and returns (tensor, label_index).

    Augmentation is applied to the PIL image before the shared preprocessing
    step, so training and serving still run the identical resize and
    normalisation code from app.preprocess.
    """

    def __init__(self, items: list[tuple[Path, int]], augment: bool, seed: int = 0) -> None:
        self.items = items
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, image: Image.Image) -> Image.Image:
        """Flips and 90-degree rotations only.

        Satellite tiles have no canonical up direction, so these transforms
        produce images that could genuinely have come from the sensor. Colour
        jitter would not: it would change the very brightness and hue cues the
        model uses to separate water from forest.
        """
        if self.rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        turns = self.rng.randint(0, 3)
        if turns:
            image = image.rotate(90 * turns)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label_index = self.items[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        if self.augment:
            image = self._augment(image)
        tensor = image_to_tensor(image)[0]  # drop the batch dimension
        return torch.from_numpy(tensor), label_index


def build_model(num_classes: int) -> nn.Module:
    """Pretrained ResNet-18 with a new final layer sized to our classes.

    The backbone already knows generic visual structure - edges, textures,
    colour blobs - from ImageNet. Only the last layer is specific to the
    original 1000 categories, so that layer is replaced. This is transfer
    learning, and it is why 1050 tiles are enough.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze everything except the final layer."""
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


class BestCheckpoint:
    """Remembers the highest-scoring weights seen during training."""

    def __init__(self) -> None:
        self.accuracy = -1.0
        self.stage = ""
        self.state_dict: dict[str, torch.Tensor] = {}

    def offer(self, model: nn.Module, accuracy: float, stage: str) -> bool:
        """Keep these weights if they beat the best so far. Returns True if kept."""
        if accuracy <= self.accuracy:
            return False
        self.accuracy = accuracy
        self.stage = stage
        # Copy to CPU tensors now; the live weights keep changing afterwards.
        self.state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
        return True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Accuracy on a loader, with gradients and dropout/batch-norm updates off."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += targets.numel()
    return correct / total if total else 0.0


def run_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    stage: str,
    best: BestCheckpoint,
) -> None:
    """One training stage. Records the best-scoring weights in `best`.

    Validation accuracy does not climb monotonically: once the model starts
    fitting noise in 840 tiles, it drifts back down. Keeping the best-scoring
    weights instead of the last ones is the cheapest possible guard against
    shipping an over-trained model.
    """
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        started = time.perf_counter()

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # gradients accumulate by default; clear them
            outputs = model(inputs)  # forward pass: raw scores per class
            loss = criterion(outputs, targets)  # how wrong those scores are
            loss.backward()  # backward pass: gradient for every weight
            optimizer.step()  # nudge the weights down the gradient
            # .detach() drops the autograd graph before reading the number out.
            running_loss += loss.detach().item() * targets.numel()

        accuracy = evaluate(model, val_loader)
        improved = best.offer(model, accuracy, f"{stage}/epoch{epoch}")
        print(
            f"[{stage}] epoch {epoch}/{epochs} "
            f"loss={running_loss / len(train_loader.dataset):.4f} "
            f"val_acc={accuracy:.4f} "
            f"({time.perf_counter() - started:.1f}s)"
            f"{'  <- best so far' if improved else ''}"
        )


def main() -> None:
    set_seeds(RANDOM_SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = discover_labels(CANDIDATE_TILES_DIR)
    train_items, val_items = stratified_split(
        CANDIDATE_TILES_DIR, labels, VALIDATION_FRACTION, RANDOM_SEED
    )
    print(f"classes: {labels}")
    print(f"train tiles: {len(train_items)}  validation tiles: {len(val_items)}")

    train_loader = DataLoader(
        TileDataset(train_items, augment=True, seed=RANDOM_SEED),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TileDataset(val_items, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_model(len(labels))
    criterion = nn.CrossEntropyLoss()
    best = BestCheckpoint()

    # Stage 1: only the new head learns. The backbone stays frozen so its
    # pretrained features are not destroyed by the large, random gradients a
    # freshly initialised head produces in the first few batches.
    set_backbone_trainable(model, False)
    head_optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LEARNING_RATE)
    run_epochs(
        model, head_optimizer, criterion, train_loader, val_loader, FROZEN_EPOCHS, "head", best
    )

    # Stage 2: everything learns, at a 10x smaller learning rate, so the
    # pretrained filters adapt to satellite imagery without being overwritten.
    set_backbone_trainable(model, True)
    full_optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LEARNING_RATE)
    run_epochs(
        model, full_optimizer, criterion, train_loader, val_loader, FINETUNE_EPOCHS, "finetune", best
    )

    torch.save(
        {
            "state_dict": best.state_dict,
            "labels": labels,
            "validation_accuracy": best.accuracy,
            "selected_at": best.stage,
            "input_size": INPUT_SIZE,
            "seed": RANDOM_SEED,
        },
        CHECKPOINT_PATH,
    )
    LABELS_PATH.write_text(json.dumps(labels, indent=2) + "\n")

    print(f"\nbest validation accuracy: {best.accuracy:.4f} (from {best.stage})")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"labels:     {LABELS_PATH}")


if __name__ == "__main__":
    main()
