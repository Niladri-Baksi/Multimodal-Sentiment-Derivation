"""Behavioral tests for the modality encoders.

Tests use randomly initialized/local backbones so they run without network access
or pretrained checkpoint downloads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sentiment_derivation.encoders.audio_encoder import AudioEncoder
from sentiment_derivation.encoders.video_encoder import VideoEncoder
import sentiment_derivation.encoders.text_encoder as text_encoder_module
from sentiment_derivation.encoders.text_encoder import TextEncoder


@pytest.mark.parametrize("batched", [False, True])
def test_video_encoder_preserves_frame_sequence_and_metadata(batched: bool) -> None:
    torch.manual_seed(7)
    encoder = VideoEncoder(pretrained=False, feature_dim=64, freeze=True).eval()
    frames = torch.randn(2, 3, 3, 64, 64) if batched else torch.randn(3, 3, 64, 64)

    output = encoder(frames)

    expected_shape = (2, 3, 64) if batched else (3, 64)
    assert output["features"].shape == expected_shape
    assert output["metadata"] == {
        "modality": "video",
        "batch_size": 2 if batched else 1,
        "num_frames": 3,
        "feature_dim": 64,
        "is_batched": batched,
        "device": "cpu",
    }
    assert torch.isfinite(output["features"]).all()
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert encoder.backbone.training is False


def test_video_encoder_rejects_invalid_input_rank() -> None:
    encoder = VideoEncoder(pretrained=False)
    with pytest.raises(ValueError, match="expects 4D.*5D"):
        encoder(torch.randn(3, 64, 64))


def test_audio_encoder_returns_finite_temporal_features_and_backpropagates() -> None:
    torch.manual_seed(11)
    encoder = AudioEncoder(target_steps=12, feature_dim=32).train()
    spectrogram = torch.randn(2, 1, 64, 101, requires_grad=True)

    output = encoder(spectrogram)
    assert output.shape == (2, 12, 32)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert spectrogram.grad is not None
    assert torch.isfinite(spectrogram.grad).all()
    assert encoder.conv1[0].weight.grad is not None


class _FakeTextBackbone(nn.Module):
    config = SimpleNamespace(dim=768)

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 768)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_text_encoder_projects_tokens_to_contract_shape_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_encoder_module.DistilBertModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTextBackbone(),
    )
    torch.manual_seed(13)
    encoder = TextEncoder(target_steps=10, feature_dim=24, freeze_backbone=True)
    input_ids = torch.randint(0, 32, (2, 17))
    attention_mask = torch.ones_like(input_ids)

    output = encoder(input_ids, attention_mask, return_dict=True)

    assert set(output) == {"features"}
    assert output["features"].shape == (2, 10, 24)
    assert torch.isfinite(output["features"]).all()
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert any(parameter.requires_grad for parameter in encoder.proj.parameters())

    output["features"].mean().backward()
    assert encoder.proj[0].weight.grad is not None
    assert encoder.backbone.embedding.weight.grad is None


def test_text_encoder_ignores_padded_token_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        text_encoder_module.DistilBertModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTextBackbone(),
    )
    encoder = TextEncoder(target_steps=4, feature_dim=16).eval()
    mask = torch.tensor([[1, 1, 0, 0]])
    first = encoder(torch.tensor([[1, 2, 3, 4]]), mask)
    second = encoder(torch.tensor([[1, 2, 29, 30]]), mask)

    assert torch.allclose(first, second, atol=1e-6)


def test_text_encoder_can_build_from_saved_local_backbone_config() -> None:
    encoder = TextEncoder(
        target_steps=3,
        feature_dim=8,
        pretrained=False,
        backbone_config={
            "vocab_size": 32,
            "max_position_embeddings": 16,
            "sinusoidal_pos_embds": False,
            "n_layers": 1,
            "n_heads": 2,
            "dim": 16,
            "hidden_dim": 32,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
    ).eval()
    output = encoder(torch.tensor([[1, 2, 3, 0]]), torch.tensor([[1, 1, 1, 0]]))

    assert output.shape == (1, 3, 8)
    assert torch.isfinite(output).all()
