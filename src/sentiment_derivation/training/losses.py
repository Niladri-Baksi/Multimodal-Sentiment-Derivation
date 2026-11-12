"""Supervised objectives for multi-label emotion, valence, and sentiment."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


class MultitaskLoss(nn.Module):
    """Combine emotion BCE, valence regression, and valence-derived sentiment CE."""

    def __init__(
        self,
        *,
        emotion_weight: float = 1.0,
        valence_weight: float = 1.0,
        sentiment_weight: float = 1.0,
        emotion_pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if min(emotion_weight, valence_weight, sentiment_weight) < 0:
            raise ValueError("Loss weights cannot be negative.")
        if emotion_weight + valence_weight + sentiment_weight == 0:
            raise ValueError("At least one loss weight must be positive.")
        self.emotion_weight = emotion_weight
        self.valence_weight = valence_weight
        self.sentiment_weight = sentiment_weight
        self.register_buffer("emotion_pos_weight", emotion_pos_weight)

    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        emotion_logits = outputs["emotion_logits"]
        if emotion_logits.ndim == 3:
            emotion_logits = emotion_logits.mean(dim=1)
        valence_prediction = outputs["valence"].reshape(-1)
        sentiment_logits = outputs["sentiment_logits"]

        emotion_loss = F.binary_cross_entropy_with_logits(
            emotion_logits,
            batch["emotion_labels"].float(),
            pos_weight=self.emotion_pos_weight,
        )
        valence_loss = F.smooth_l1_loss(
            valence_prediction,
            batch["valence"].float().reshape(-1),
        )
        sentiment_loss = F.cross_entropy(
            sentiment_logits,
            batch["sentiment_class"].long().reshape(-1),
        )
        total = (
            self.emotion_weight * emotion_loss
            + self.valence_weight * valence_loss
            + self.sentiment_weight * sentiment_loss
        )
        return {
            "loss": total,
            "emotion_loss": emotion_loss,
            "valence_loss": valence_loss,
            "sentiment_loss": sentiment_loss,
        }
