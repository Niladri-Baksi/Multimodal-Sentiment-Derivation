"""Encoder-to-downstream contract tests; backbones are mocked where loading weights is unnecessary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.fusion.alignment import ModalityAligner  # noqa: E402
from sentiment_derivation.fusion.multimodal_fusion import MultimodalFusion  # noqa: E402
from sentiment_derivation.emotion.affective_state import AffectiveStateEstimator  # noqa: E402
from sentiment_derivation.emotion.emotion_classifier import (  # noqa: E402
    EmotionClassifier,
    MultimodalEmotionPipeline,
)


def test_aligner_fusion_emotion_cpu_contract() -> None:
    """Actual downstream modules with synthetic (not dataset or encoder) features."""
    aligner = ModalityAligner()
    fusion = MultimodalFusion(dropout=0.0).eval()
    pipeline = MultimodalEmotionPipeline(
        EmotionClassifier().eval(), AffectiveStateEstimator().eval()
    ).eval()
    video = torch.randn(2, 8, 512)
    audio = torch.randn(2, 16, 128)
    aligned = aligner(video, audio)
    fused = fusion(aligned)
    output = pipeline(fused)
    assert aligned.features["video"].shape == (2, 16, 128)
    assert aligned.features["audio"].shape == (2, 16, 128)
    assert fused["features"].shape == (2, 16, 128)
    assert output["emotion_probabilities"].shape == (2, 16, 8)
    assert output["valence"].shape == (2, 16, 1)
    assert all(value.device.type == "cpu" for value in output.values() if isinstance(value, torch.Tensor))
    assert all(torch.isfinite(value).all() for value in output.values() if isinstance(value, torch.Tensor))
    assert all(value.dtype == torch.float32 for value in output.values() if isinstance(value, torch.Tensor))


def test_missing_optional_modality_is_masked_through_fusion() -> None:
    aligner = ModalityAligner()
    fusion = MultimodalFusion(dropout=0.0).eval()
    video = torch.randn(2, 8, 512)
    audio = torch.randn(2, 16, 128)
    text = torch.randn(2, 12, 128)
    availability = torch.tensor([True, False])
    aligned = aligner(video, audio, text, text_available=availability)
    fused = fusion(aligned, return_attention_weights=True)
    assert aligned.modality_mask.tolist() == [[True, True, True], [True, True, False]]
    assert torch.count_nonzero(aligned.features["text"][1]) == 0
    assert torch.count_nonzero(fused["attention_weights"][1, :, 2]) == 0
    assert fused["features"].shape == (2, 16, 128)
    assert fused["features"].dtype == video.dtype
    assert fused["features"].device == video.device
    assert torch.isfinite(fused["features"]).all()


class DummyResNet(nn.Module):
    """Mock ResNet-like feature backbone; not a pretrained runtime test."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 1000)
        self.conv = nn.Conv2d(3, 512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(-2, -1))


def test_video_encoder_output_to_aligner_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from sentiment_derivation.encoders import video_encoder as video_module
    except Exception as exc:
        pytest.skip(f"Video integration unavailable: {exc}")
    monkeypatch.setattr(video_module.models, "resnet18", lambda **kwargs: DummyResNet())
    encoder = video_module.VideoEncoder(pretrained=False, freeze=False)
    encoded = encoder(torch.rand(2, 3, 3, 32, 32))
    aligned = ModalityAligner()(encoded, torch.randn(2, 16, 128))
    assert encoded["features"].shape == (2, 3, 512)
    assert aligned.features["video"].shape == (2, 16, 128)
    assert aligned.features["video"].dtype == encoded["features"].dtype
    assert aligned.features["video"].device == encoded["features"].device
    assert torch.isfinite(aligned.features["video"]).all()


def test_audio_encoder_output_to_aligner() -> None:
    try:
        from sentiment_derivation.encoders.audio_encoder import AudioEncoder
    except Exception as exc:
        pytest.skip(f"Audio integration unavailable: {exc}")
    encoded = AudioEncoder().eval()(torch.randn(2, 1, 64, 41))
    aligned = ModalityAligner()(torch.randn(2, 8, 512), encoded)
    assert encoded.shape == (2, 16, 128)
    assert aligned.features["audio"].shape == (2, 16, 128)
    assert aligned.features["audio"].device == encoded.device
    assert aligned.features["audio"].dtype == encoded.dtype
    assert torch.isfinite(aligned.features["audio"]).all()
