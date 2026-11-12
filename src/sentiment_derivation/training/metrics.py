"""Batch-aggregated model metrics used during training and validation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error


def calculate_metrics(
    emotion_probabilities: np.ndarray,
    emotion_targets: np.ndarray,
    sentiment_probabilities: np.ndarray,
    sentiment_targets: np.ndarray,
    valence_predictions: np.ndarray,
    valence_targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    emotion_predictions = emotion_probabilities >= threshold
    sentiment_predictions = sentiment_probabilities.argmax(axis=1)
    return {
        "emotion_macro_f1": float(
            f1_score(emotion_targets, emotion_predictions, average="macro", zero_division=0)
        ),
        "emotion_micro_f1": float(
            f1_score(emotion_targets, emotion_predictions, average="micro", zero_division=0)
        ),
        "sentiment_macro_f1": float(
            f1_score(sentiment_targets, sentiment_predictions, average="macro", zero_division=0)
        ),
        "valence_mae": float(mean_absolute_error(valence_targets, valence_predictions)),
    }


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


class MetricAccumulator:
    """Collect predictions and targets across mini-batches for epoch-level metrics."""

    def __init__(self) -> None:
        self.emotion_probabilities: list[np.ndarray] = []
        self.emotion_targets: list[np.ndarray] = []
        self.sentiment_probabilities: list[np.ndarray] = []
        self.sentiment_targets: list[np.ndarray] = []
        self.valence_predictions: list[np.ndarray] = []
        self.valence_targets: list[np.ndarray] = []

    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        emotion_probabilities = outputs["emotion_probabilities"].mean(dim=1)
        self.emotion_probabilities.append(_as_numpy(emotion_probabilities))
        self.emotion_targets.append(_as_numpy(batch["emotion_labels"]))
        self.sentiment_probabilities.append(_as_numpy(outputs["sentiment_probabilities"]))
        self.sentiment_targets.append(_as_numpy(batch["sentiment_class"]))
        self.valence_predictions.append(_as_numpy(outputs["valence"].reshape(-1)))
        self.valence_targets.append(_as_numpy(batch["valence"].reshape(-1)))

    def compute(self, threshold: float = 0.5) -> dict[str, float]:
        if not self.emotion_targets:
            raise ValueError("Cannot calculate epoch metrics without any batches.")
        return calculate_metrics(
            np.concatenate(self.emotion_probabilities),
            np.concatenate(self.emotion_targets),
            np.concatenate(self.sentiment_probabilities),
            np.concatenate(self.sentiment_targets),
            np.concatenate(self.valence_predictions),
            np.concatenate(self.valence_targets),
            threshold=threshold,
        )
