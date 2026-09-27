"""Preprocessing is the one place where training and serving must agree exactly."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE
from app.preprocess import (
    InvalidTileError,
    bytes_to_tensor,
    decode_tile,
    sha256_hex,
    softmax,
)


def test_output_shape_and_dtype(tile_bytes: bytes) -> None:
    tensor = bytes_to_tensor(tile_bytes)
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_normalisation_matches_imagenet_statistics() -> None:
    """A mid-grey tile must map to the exact value the formula predicts.

    This is the test that would fail if someone changed the mean, the std, or
    the order of the resize and the divide-by-255.
    """
    grey = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (128, 128, 128))
    buffer = io.BytesIO()
    grey.save(buffer, format="PNG")

    tensor = bytes_to_tensor(buffer.getvalue())
    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert tensor[0, channel].mean() == pytest.approx(expected, abs=1e-5)


def test_greyscale_and_rgba_are_coerced_to_three_channels() -> None:
    for mode, colour in (("L", 200), ("RGBA", (10, 20, 30, 255))):
        buffer = io.BytesIO()
        Image.new(mode, (64, 64), colour).save(buffer, format="PNG")
        assert bytes_to_tensor(buffer.getvalue()).shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_odd_sized_tile_is_resized() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (37, 91), (1, 2, 3)).save(buffer, format="PNG")
    assert bytes_to_tensor(buffer.getvalue()).shape == (1, 3, INPUT_SIZE, INPUT_SIZE)


def test_non_image_bytes_raise_invalid_tile() -> None:
    with pytest.raises(InvalidTileError):
        decode_tile(b"this is not a png")


def test_preprocessing_is_deterministic(tile_bytes: bytes) -> None:
    """Same bytes, same array. Required for the content-hash idempotency to mean anything."""
    assert np.array_equal(bytes_to_tensor(tile_bytes), bytes_to_tensor(tile_bytes))


def test_sha256_is_content_addressed(tile_bytes: bytes, other_tile_bytes: bytes) -> None:
    assert sha256_hex(tile_bytes) == sha256_hex(tile_bytes)
    assert sha256_hex(tile_bytes) != sha256_hex(other_tile_bytes)


def test_softmax_sums_to_one_and_survives_large_logits() -> None:
    probabilities = softmax(np.array([[1000.0, 999.0, 0.0]], dtype=np.float32))
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.isfinite(probabilities).all()  # would be NaN without the max-subtraction
    assert int(np.argmax(probabilities)) == 0
