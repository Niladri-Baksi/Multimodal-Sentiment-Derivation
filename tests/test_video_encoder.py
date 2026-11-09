"""Video encoder unit tests; the ResNet backbone is mocked to avoid weights/downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from torch import nn
    from sentiment_derivation.encoders import video_encoder as video_module
except Exception as exc:  # Explicitly report unavailable/broken vision runtime.
    pytest.skip(f"Video encoder runtime unavailable: {exc}", allow_module_level=True)


class DummyResNet(nn.Module):
    """Mock backbone returning 512 features; does not validate pretrained ResNet."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 1000)
        self.projection = nn.Conv2d(3, 512, kernel_size=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(frames).mean(dim=(-2, -1))


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(video_module.models, "resnet18", lambda **kwargs: DummyResNet())
    return video_module.VideoEncoder(pretrained=False, freeze=False)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_video_encoder_batch_shapes_dtype_finite_cpu(encoder, batch_size: int) -> None:
    frames = torch.randn(batch_size, 4, 3, 32, 32, dtype=torch.float32)
    result = encoder(frames)
    assert result["features"].shape == (batch_size, 4, 512)
    assert result["features"].dtype == torch.float32
    assert result["features"].device.type == "cpu"
    assert torch.isfinite(result["features"]).all()


def test_video_encoder_unbatched_sequence(encoder) -> None:
    result = encoder(torch.randn(4, 3, 32, 32))
    assert result["features"].shape == (4, 512)
    assert torch.isfinite(result["features"]).all()


def test_video_encoder_noncontiguous_batch(encoder) -> None:
    frames = torch.randn(2, 4, 3, 32, 32).transpose(-1, -2)
    assert not frames.is_contiguous()
    result = encoder(frames)
    assert result["features"].shape == (2, 4, 512)
    assert torch.isfinite(result["features"]).all()


def test_video_encoder_rejects_invalid_rank(encoder) -> None:
    with pytest.raises(ValueError):
        encoder(torch.randn(3, 32, 32))


def test_video_transform_handles_single_and_batched_frames() -> None:
    transform = video_module.VideoClipTransform(image_size=(16, 20))
    sequence = transform(torch.randint(0, 256, (4, 3, 32, 40), dtype=torch.uint8))
    batch = transform(torch.randint(0, 256, (2, 4, 3, 32, 40), dtype=torch.uint8))
    assert sequence.shape == (4, 3, 16, 20)
    assert batch.shape == (2, 4, 3, 16, 20)
    assert sequence.dtype == torch.float32 and batch.dtype == torch.float32
    assert torch.isfinite(sequence).all() and torch.isfinite(batch).all()


@pytest.mark.xfail(strict=True, reason="Known gap: VideoEncoder has no empty-frame validation.")
def test_video_encoder_rejects_empty_sequence(encoder) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        encoder(torch.empty(1, 0, 3, 32, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_video_encoder_cuda_when_available(encoder) -> None:
    encoder = encoder.cuda()
    output = encoder(torch.randn(2, 3, 3, 32, 32, device="cuda"))["features"]
    assert output.device.type == "cuda"
    assert output.shape == (2, 3, 512)
    assert torch.isfinite(output).all()


@pytest.mark.xfail(strict=True, reason="Known gap: 5D transform folds with view() without contiguous().")
def test_video_transform_accepts_noncontiguous_batch() -> None:
    transform = video_module.VideoClipTransform(image_size=(16, 16))
    frames = torch.rand(2, 3, 3, 16, 16).transpose(0, 1)
    assert not frames.is_contiguous()
    transformed = transform(frames)
    assert transformed.shape == (3, 2, 3, 16, 16)
