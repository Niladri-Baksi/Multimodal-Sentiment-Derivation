"""Video transformation contract tests (run when torchvision is installed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("torchvision")

from sentiment_derivation.data.transforms import VideoClipTransform  # noqa: E402


def test_video_transform_returns_float32_with_fixed_batch_shape() -> None:
    transform = VideoClipTransform(image_size=(16, 12))
    frames = torch.full((2, 3, 3, 20, 24), 255, dtype=torch.uint8)

    output = transform(frames)

    assert output.shape == (2, 3, 3, 16, 12)
    assert output.dtype == torch.float32


def test_video_transform_rejects_empty_frame_list() -> None:
    transform = VideoClipTransform()

    with pytest.raises(ValueError, match="must not be empty"):
        transform([])


def test_video_transform_rejects_non_rgb_array() -> None:
    transform = VideoClipTransform()

    with pytest.raises(ValueError, match="3 channels"):
        transform([np.zeros((10, 10), dtype=np.uint8)])
