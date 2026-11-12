from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import yaml
import pytest
from torch import nn
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.integration.model import MultimodalSentimentModel
from sentiment_derivation.integration.pipeline import export_predictions, load_model_checkpoint
from sentiment_derivation.training.losses import MultitaskLoss
import sentiment_derivation.encoders.text_encoder as text_encoder_module


def _model_config() -> dict:
    with (PROJECT_ROOT / "configs" / "model.yaml").open(encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["video_encoder"]["pretrained"] = False
    return config


class _FakeTextBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=768)
        self.embedding = nn.Embedding(32, 768)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_optional_text_is_encoded_and_masked_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_encoder_module.DistilBertModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTextBackbone(),
    )
    config = _model_config()
    config["text_encoder"]["enabled"] = True
    model = MultimodalSentimentModel.from_config(config).eval()
    frames = torch.rand(2, 2, 3, 32, 32)
    audio = torch.rand(2, 1, 64, 96)
    input_ids = torch.randint(0, 32, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1] = 0

    with torch.no_grad():
        output = model(
            frames,
            audio,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            text_available=torch.tensor([True, False]),
            return_attention_weights=True,
        )

    assert output["sentiment_probabilities"].shape == (2, 3)
    assert output["modality_attention_weights"].shape == (2, 16, 3)
    assert torch.count_nonzero(output["modality_attention_weights"][1, :, 2]) == 0
    assert torch.allclose(
        output["modality_attention_weights"].sum(dim=-1), torch.ones(2, 16), atol=1e-6
    )


def test_optional_text_requires_paired_tensors() -> None:
    model = MultimodalSentimentModel.from_config(_model_config())
    frames = torch.rand(1, 1, 3, 32, 32)
    audio = torch.rand(1, 1, 64, 64)

    with pytest.raises(ValueError, match="provided together"):
        model(frames, audio, text_input_ids=torch.ones(1, 4, dtype=torch.long))


def test_end_to_end_forward_backward_and_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = _model_config()
    model = MultimodalSentimentModel.from_config(config)
    frames = torch.rand(2, 2, 3, 32, 32)
    audio = torch.rand(2, 1, 64, 96)

    outputs = model(frames, audio, return_attention_weights=True)
    assert outputs["emotion_probabilities"].shape == (2, 16, 8)
    assert outputs["valence"].shape == (2, 1)
    assert outputs["sentiment_probabilities"].shape == (2, 3)
    assert outputs["modality_attention_weights"].shape == (2, 16, 3)
    assert torch.allclose(
        outputs["sentiment_probabilities"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    batch = {
        "emotion_labels": torch.randint(0, 2, (2, 8)).float(),
        "valence": torch.tensor([2.0, 5.5]),
        "sentiment_class": torch.tensor([0, 2]),
    }
    losses = MultitaskLoss()(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.sentiment_head.classifier[0].weight.grad is not None
    assert model.audio_encoder.conv1[0].weight.grad is not None

    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 1,
            "model_config": config,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    restored, checkpoint = load_model_checkpoint(checkpoint_path)
    assert checkpoint["epoch"] == 1
    restored_outputs = restored(frames, audio)
    assert restored_outputs["sentiment_probabilities"].shape == (2, 3)

    prediction_path = export_predictions(
        restored,
        [
            {
                "sample_id": ["clip-1", "clip-2"],
                "frames": frames,
                "audio": audio,
            }
        ],
        tmp_path / "predictions.csv",
        device="cpu",
    )
    with prediction_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 2
    assert "emotion_probability_Curiosity" in rows[0]
    assert "predicted_valence" in rows[0]
    assert "sentiment_probability_Negative" in rows[0]
