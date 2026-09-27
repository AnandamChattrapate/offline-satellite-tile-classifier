"""Turn raw PNG bytes into the exact array the model was trained on.

This module is imported by both the training script and the API. That is
deliberate: if preprocessing differs between training and serving by even a
resize filter, accuracy silently drops and nothing in the system reports it.
One implementation, one place to change it.

Deliberately depends only on Pillow and numpy, so the serving box does not need
torch or torchvision.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


class InvalidTileError(ValueError):
    """Raised when the uploaded bytes are not a usable image tile."""


def sha256_hex(data: bytes) -> str:
    """Content hash of the raw bytes, used as the tile's identity."""
    return hashlib.sha256(data).hexdigest()


def decode_tile(data: bytes) -> Image.Image:
    """Decode bytes to an RGB image, or raise InvalidTileError.

    Coercing to RGB matters: EuroSAT PNGs are RGB here, but a greyscale or
    RGBA tile would otherwise produce the wrong channel count and a confusing
    shape error deep inside the model.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidTileError(f"not a decodable image: {exc}") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_tensor(image: Image.Image) -> np.ndarray:
    """Resize, scale to 0-1, normalise, and return shape (1, 3, H, W) float32.

    Steps, in order, and why each one exists:
      1. resize to INPUT_SIZE with bilinear interpolation - the model has a
         fixed input size baked into its graph.
      2. divide by 255 - the pretrained weights expect 0-1, not 0-255.
      3. transpose HWC -> CHW - torch and ONNX both use channels-first.
      4. subtract the ImageNet mean, divide by the ImageNet std - this is the
         normalisation the pretrained backbone was trained under.
      5. add a leading batch dimension of 1 - the graph always expects a batch.
    """
    if image.size != (INPUT_SIZE, INPUT_SIZE):
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, 3)
    array = np.transpose(array, (2, 0, 1))  # (3, H, W)
    array = (array - _MEAN) / _STD
    return np.expand_dims(array, axis=0)  # (1, 3, H, W)


def bytes_to_tensor(data: bytes) -> np.ndarray:
    """Convenience path used by the API: raw upload bytes to model input."""
    return image_to_tensor(decode_tile(data))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert raw model scores into probabilities that sum to 1.

    The max is subtracted first for numerical stability: exp() of a large
    positive number overflows float32, and subtracting a constant from every
    logit leaves the result unchanged.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)
