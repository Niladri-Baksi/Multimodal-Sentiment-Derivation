from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.training.losses import MultitaskLoss
from sentiment_derivation.training.trainer import Trainer


class TinyTrainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = frames.shape[0]
        emotion_logits = self.scale.expand(batch_size, 1, 8)
        sentiment_logits = self.scale * torch.tensor(
            [1.0, 0.0, -1.0],
            device=self.scale.device,
        ).expand(batch_size, 3)
        return {
            "emotion_logits": emotion_logits,
            "emotion_probabilities": emotion_logits.sigmoid(),
            "valence": (4.0 + self.scale).expand(batch_size, 1),
            "sentiment_logits": sentiment_logits,
            "sentiment_probabilities": sentiment_logits.softmax(dim=-1),
        }


def _batches() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "frames": torch.zeros(2, 1),
            "audio": torch.zeros(2, 1),
            "emotion_labels": torch.tensor(
                [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
                dtype=torch.float32,
            ),
            "valence": torch.tensor([2.0, 6.0]),
            "sentiment_class": torch.tensor([0, 2]),
        }
    ]


def test_trainer_runs_validation_and_saves_best_checkpoint(tmp_path: Path) -> None:
    trainer = Trainer(
        TinyTrainableModel(),
        MultitaskLoss(),
        device=torch.device("cpu"),
        learning_rate=0.01,
        model_config={"test": True},
        output_dir=tmp_path,
    )
    history = trainer.fit(_batches(), _batches(), epochs=2, patience=0)

    assert len(history) == 2
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "last_model.pt").is_file()
    assert (tmp_path / "training_history.json").is_file()
    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=True)
    assert checkpoint["model_config"] == {"test": True}
