"""Synthetic three-encoder to alignment and fusion contract test."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.fusion.alignment import ModalityAligner  # noqa: E402
from sentiment_derivation.fusion.multimodal_fusion import MultimodalFusion  # noqa: E402


class _VideoBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(512, 1000)
        self.conv = nn.Conv2d(3, 512, kernel_size=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.conv(frames).mean(dim=(-2, -1))


def test_mocked_video_text_and_cpu_audio_encoders_reach_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sentiment_derivation.encoders import audio_encoder, text_encoder, video_encoder

    monkeypatch.setattr(
        video_encoder,
        "models",
        SimpleNamespace(resnet18=lambda **kwargs: _VideoBackbone()),
    )
    video_model = video_encoder.VideoEncoder(pretrained=False, freeze=False).eval()
    audio_model = audio_encoder.AudioEncoder().eval()

    class _TextBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(dim=24, vocab_size=64)
            self.embedding = nn.Embedding(64, 24)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    monkeypatch.setattr(
        text_encoder.DistilBertModel,
        "from_pretrained",
        classmethod(lambda cls, _name: _TextBackbone()),
    )
    text_model = text_encoder.TextEncoder().eval()

    frames = torch.rand(2, 3, 3, 24, 24)
    mel = torch.randn(2, 1, 64, 40)
    token_ids = torch.randint(0, 64, (2, 12))
    token_mask = torch.ones_like(token_ids)
    token_mask[1] = 0

    with torch.no_grad():
        video_features = video_model(frames)
        audio_features = audio_model(mel)
        text_features = text_model(token_ids, token_mask)
        aligned = ModalityAligner()(
            video_features,
            audio_features,
            text_features,
            audio_available=torch.tensor([True, False]),
            text_available=token_mask.bool().any(dim=1),
        )
        fused = MultimodalFusion(dropout=0.0).eval()(
            aligned, return_attention_weights=True
        )

    assert video_features["features"].shape == (2, 3, 512)
    assert audio_features.shape == (2, 16, 128)
    assert text_features.shape == (2, 16, 128)
    assert aligned.modality_mask.tolist() == [[True, True, True], [True, False, False]]
    assert fused["features"].shape == (2, 16, 128)
    assert torch.count_nonzero(fused["attention_weights"][1, :, 1:]) == 0
    assert torch.allclose(
        fused["attention_weights"].sum(-1), torch.ones(2, 16), atol=1e-6
    )
    assert torch.isfinite(fused["features"]).all()
