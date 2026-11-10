"""Sentiment derivation and classification head module for Multimodal Sentiment Derivation.

Reference: AGENTS.md | Canonical Architecture Stage 9 (Sentiment Derivation and Mapping).

This module provides:
1. DEFAULT_SENTIMENT_NAMES: Canonical 3-class sentiment label space ("Negative", "Neutral", "Positive").
2. valence_to_sentiment_label: Ground truth mapping utility deriving sentiment labels from
   continuous EmoReact valence ratings ([1.0, 7.0]) with documented thresholds.
3. SentimentOutput: Standardized container storing predicted categorical sentiments, class probabilities,
   continuous polarity scores, and sentiment-level confidence.
4. SentimentHead: Neural classification and regression head deriving final sentiment and confidence
   from the latent sentiment representation.
5. SentimentDerivationPipeline: Unified pipeline integrating Stage 8 (LatentSentimentMapper)
   and Stage 9 (SentimentHead) into the canonical AGENTS.md contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sentiment_derivation.sentiment.latent_mapper import (
    LatentSentimentMapper,
    LatentSentimentOutput,
)

# Canonical 3-class sentiment label space
DEFAULT_SENTIMENT_NAMES: Tuple[str, ...] = ("Negative", "Neutral", "Positive")
NUM_SENTIMENT_CLASSES: int = len(DEFAULT_SENTIMENT_NAMES)

FeatureInput = Union[torch.Tensor, Mapping[str, Any]]


def valence_to_sentiment_label(
    valence: Union[float, int, Sequence[float], np.ndarray, torch.Tensor],
    neutral_margin: float = 0.25,
    min_valence: float = 1.0,
    max_valence: float = 7.0,
) -> Tuple[Union[int, List[int]], Union[str, List[str]], Union[float, List[float]]]:
    """Derive sentiment categorical labels and continuous scores from EmoReact valence.

    Per AGENTS.md Section 8:
    - EmoReact annotations provide continuous valence on a 1.0 to 7.0 Likert scale
      with neutral midpoint at 4.0.
    - This function maps continuous valence to the standard 3-class sentiment space:
        - Negative (class 0): valence < 4.0 - neutral_margin (e.g. < 3.75)
        - Neutral  (class 1): 4.0 - neutral_margin <= valence <= 4.0 + neutral_margin (e.g. [3.75, 4.25])
        - Positive (class 2): valence > 4.0 + neutral_margin (e.g. > 4.25)
    - Normalized continuous sentiment score is scaled to [-1.0, 1.0]:
        score = (valence - midpoint) / half_range

    Args:
        valence: Continuous valence score(s) in [min_valence, max_valence].
        neutral_margin: Width of neutral deadband around midpoint (default 0.25).
        min_valence: Lower bound of valence scale (default 1.0).
        max_valence: Upper bound of valence scale (default 7.0).

    Returns:
        Tuple of (class_indices, string_labels, normalized_scores).
    """
    midpoint = 0.5 * (min_valence + max_valence)  # 4.0
    half_range = 0.5 * (max_valence - min_valence)  # 3.0

    lower_bound = midpoint - neutral_margin
    upper_bound = midpoint + neutral_margin

    is_scalar = isinstance(valence, (int, float))

    if isinstance(valence, torch.Tensor):
        vals = valence.detach().cpu().numpy().flatten().tolist()
    elif isinstance(valence, np.ndarray):
        vals = valence.flatten().tolist()
    elif isinstance(valence, (list, tuple)):
        vals = list(valence)
    else:
        vals = [float(valence)]

    classes: List[int] = []
    labels: List[str] = []
    scores: List[float] = []

    for v in vals:
        score = float(np.clip((v - midpoint) / half_range, -1.0, 1.0))
        scores.append(score)

        if v < lower_bound:
            classes.append(0)
            labels.append(DEFAULT_SENTIMENT_NAMES[0])  # Negative
        elif v > upper_bound:
            classes.append(2)
            labels.append(DEFAULT_SENTIMENT_NAMES[2])  # Positive
        else:
            classes.append(1)
            labels.append(DEFAULT_SENTIMENT_NAMES[1])  # Neutral

    if is_scalar:
        return classes[0], labels[0], scores[0]
    return classes, labels, scores


@dataclass(frozen=True)
class SentimentOutput:
    """Standardized output container for derived sentiment.

    Per AGENTS.md Stage 9:
    - Provides final sentiment, latent representation, and sentiment confidence.

    Attributes:
        sentiment: Predicted sentiment label(s) as string or list of strings.
        sentiment_class: Predicted categorical class index tensor shaped ``[B]`` (0=Neg, 1=Neu, 2=Pos).
        sentiment_probabilities: Softmax probability distribution shaped ``[B, 3]``.
        sentiment_logits: Unnormalized classification logits shaped ``[B, 3]``.
        sentiment_confidence: Confidence score(s) in [0.0, 1.0] shaped ``[B]`` or float.
        sentiment_score: Continuous polarity score in [-1.0, 1.0] shaped ``[B, 1]``.
        latent_representation: Latent affective representation from Stage 8 shaped ``[B, latent_dim]``.
        sentiment_labels: Canonical ordered names for sentiment classes.
    """

    sentiment: Union[str, List[str]]
    sentiment_class: torch.Tensor
    sentiment_probabilities: torch.Tensor
    sentiment_logits: torch.Tensor
    sentiment_confidence: Union[float, torch.Tensor]
    sentiment_score: torch.Tensor
    latent_representation: torch.Tensor
    sentiment_labels: Tuple[str, ...] = DEFAULT_SENTIMENT_NAMES

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access matching AGENTS.md contracts."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in SentimentOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary matching AGENTS.md contracts."""
        return {
            "sentiment": self.sentiment,
            "sentiment_class": self.sentiment_class,
            "sentiment_probabilities": self.sentiment_probabilities,
            "sentiment_logits": self.sentiment_logits,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_score": self.sentiment_score,
            "latent_representation": self.latent_representation,
            "sentiment_labels": self.sentiment_labels,
        }


class SentimentHead(nn.Module):
    """Neural classification and regression head deriving sentiment and confidence.

    Per AGENTS.md Stage 9:
    - Receives latent representation z in R^{latent_dim} from Stage 8.
    - Derives final categorical sentiment (Negative, Neutral, Positive).
    - Derives continuous sentiment polarity score in [-1.0, 1.0].
    - Estimates sentiment-level prediction confidence in [0.0, 1.0].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = NUM_SENTIMENT_CLASSES,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 1.0,
        confidence_method: str = "margin",
        sentiment_labels: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize SentimentHead.

        Args:
            latent_dim: Dimension of latent sentiment representation from Stage 8 (default 128).
            num_classes: Number of categorical sentiment classes (default 3).
            hidden_dim: Intermediate projection dimension (default 64).
            dropout: Dropout probability for regularization.
            temperature: Calibration temperature for scaling logits before Softmax.
            confidence_method: Metric for sentiment confidence: ``"margin"`` (top1 - top2 prob),
                ``"max_prob"`` (top1 prob), or ``"entropy"`` (1 - normalized entropy).
            sentiment_labels: Sequence of class names (defaults to Negative, Neutral, Positive).
        """
        super().__init__()
        if latent_dim <= 0 or num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("All dimension arguments must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive; got {temperature}.")
        if confidence_method not in ("margin", "max_prob", "entropy"):
            raise ValueError(
                f"confidence_method must be 'margin', 'max_prob', or 'entropy'; got '{confidence_method}'."
            )

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.confidence_method = confidence_method

        if sentiment_labels is not None:
            self.sentiment_labels = tuple(sentiment_labels)
        else:
            self.sentiment_labels = DEFAULT_SENTIMENT_NAMES

        if len(self.sentiment_labels) != num_classes:
            raise ValueError(
                f"Length of sentiment_labels ({len(self.sentiment_labels)}) must equal num_classes ({num_classes})."
            )

        # 1. Intermediate Latent Representation MLP
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Categorical Sentiment Classification Head: [B, num_classes=3]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
        )

        # 3. Continuous Sentiment Polarity Regression Head: [B, 1] in [-1.0, 1.0]
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def _estimate_sentiment_confidence(
        self, probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Estimate sentiment-level prediction confidence in [0.0, 1.0].

        Args:
            probabilities: Softmax probability distribution shaped ``[B, num_classes]``.

        Returns:
            Confidence scores shaped ``[B]``.
        """
        if self.confidence_method == "margin":
            # Margin between top-1 and top-2 probabilities
            if self.num_classes >= 2:
                top2_probs, _ = torch.topk(probabilities, k=2, dim=-1)
                confidence = top2_probs[:, 0] - top2_probs[:, 1]
            else:
                confidence = probabilities[:, 0]
        elif self.confidence_method == "max_prob":
            confidence, _ = torch.max(probabilities, dim=-1)
        elif self.confidence_method == "entropy":
            eps = 1e-7
            p_clamped = torch.clamp(probabilities, min=eps, max=1.0)
            entropy = -(p_clamped * torch.log(p_clamped)).sum(dim=-1)
            max_entropy = np.log(self.num_classes)
            normalized_entropy = entropy / max_entropy
            confidence = 1.0 - torch.clamp(normalized_entropy, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")

        return torch.clamp(confidence, 0.0, 1.0)

    def forward(
        self,
        latent_representation: FeatureInput,
    ) -> SentimentOutput:
        """Derive final sentiment, continuous polarity, and confidence.

        Args:
            latent_representation: Tensor shaped ``[B, latent_dim]`` or ``[B, T, latent_dim]``,
                or a LatentSentimentOutput or dictionary containing ``"latent_representation"``.

        Returns:
            SentimentOutput containing categorical sentiment, probabilities,
            continuous score, and sentiment confidence.
        """
        # Extract tensor if mapping or dataclass was passed
        if hasattr(latent_representation, "latent_representation"):
            z = getattr(latent_representation, "latent_representation")
        elif isinstance(latent_representation, Mapping):
            if "latent_representation" in latent_representation:
                z = latent_representation["latent_representation"]
            elif "features" in latent_representation:
                z = latent_representation["features"]
            else:
                raise ValueError(
                    "Input mapping must contain 'latent_representation' or 'features' key."
                )
        else:
            z = latent_representation

        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected latent_representation to be a torch.Tensor; got {type(z).__name__}.")

        is_unbatched = False
        if z.ndim == 1:
            z = z.unsqueeze(0)  # [1, D]
            is_unbatched = True
        elif z.ndim == 3:
            # If 3D sequence [B, T, D] passed directly, pool across time
            z = z.mean(dim=1)
        elif z.ndim != 2:
            raise ValueError(
                f"Expected latent_representation to have shape [B, D] or [B, T, D]; got {tuple(z.shape)}."
            )

        batch_size, dim = z.shape
        if dim != self.latent_dim:
            raise ValueError(
                f"Expected latent_representation dimension {self.latent_dim}; got {dim}."
            )

        # 1. Project through intermediate encoder
        h = self.encoder(z)

        # 2. Compute classification logits and Softmax probabilities
        logits = self.classifier(h)
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits
        probabilities = F.softmax(scaled_logits, dim=-1)

        # 3. Compute continuous sentiment score: [B, 1] in [-1.0, 1.0]
        score = self.regressor(h)

        # 4. Estimate sentiment confidence: [B] in [0.0, 1.0]
        confidence = self._estimate_sentiment_confidence(probabilities)

        # 5. Extract predicted class indices and string labels
        predicted_classes = torch.argmax(probabilities, dim=-1)  # [B]
        predicted_labels = [self.sentiment_labels[idx.item()] for idx in predicted_classes]

        # Handle unbatched return formatting
        if is_unbatched:
            final_sentiment = predicted_labels[0]
            final_class = predicted_classes.squeeze(0)
            final_prob = probabilities.squeeze(0)
            final_logits = logits.squeeze(0)
            final_conf = confidence.squeeze(0).item()
            final_score = score.squeeze(0)
            final_z = z.squeeze(0)
        else:
            final_sentiment = predicted_labels
            final_class = predicted_classes
            final_prob = probabilities
            final_logits = logits
            final_conf = confidence
            final_score = score
            final_z = z

        return SentimentOutput(
            sentiment=final_sentiment,
            sentiment_class=final_class,
            sentiment_probabilities=final_prob,
            sentiment_logits=final_logits,
            sentiment_confidence=final_conf,
            sentiment_score=final_score,
            latent_representation=final_z,
            sentiment_labels=self.sentiment_labels,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SentimentHead":
        """Instantiate SentimentHead from configuration dictionary."""
        cfg = config.get("sentiment_head", config)
        return cls(
            latent_dim=cfg.get("latent_dim", 128),
            num_classes=cfg.get("num_classes", NUM_SENTIMENT_CLASSES),
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.1),
            temperature=cfg.get("temperature", 1.0),
            confidence_method=cfg.get("confidence_method", "margin"),
            sentiment_labels=cfg.get("sentiment_labels", None),
        )


class SentimentDerivationPipeline(nn.Module):
    """Unified pipeline combining Stage 8 (LatentMapper) and Stage 9 (SentimentHead).

    Per AGENTS.md Section 6:
    Produces the canonical dictionary contract:
    {
        "latent_representation": tensor,
        "sentiment": str or list[str],
        "sentiment_confidence": float or tensor,
        "sentiment_probabilities": tensor,
        "sentiment_score": tensor,
        "temporal_attention": tensor,
    }
    """

    def __init__(
        self,
        latent_mapper: LatentSentimentMapper,
        sentiment_head: SentimentHead,
    ) -> None:
        super().__init__()
        self.latent_mapper = latent_mapper
        self.sentiment_head = sentiment_head

    def forward(
        self,
        gated_features: torch.Tensor,
        emotion_probabilities: torch.Tensor,
        emotion_confidence: torch.Tensor,
        delta_emotion: torch.Tensor,
        persistence: torch.Tensor,
        transition_embedding: Optional[torch.Tensor] = None,
        affective_state: Optional[torch.Tensor] = None,
        valence: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute latent sentiment mapping and final sentiment derivation.

        Returns:
            Dictionary matching AGENTS.md Section 6 canonical contract.
        """
        latent_out = self.latent_mapper(
            gated_features=gated_features,
            emotion_probabilities=emotion_probabilities,
            emotion_confidence=emotion_confidence,
            delta_emotion=delta_emotion,
            persistence=persistence,
            transition_embedding=transition_embedding,
            affective_state=affective_state,
            valence=valence,
        )

        sentiment_out = self.sentiment_head(latent_out.latent_representation)

        return {
            "latent_representation": sentiment_out.latent_representation,
            "latent_sequence": latent_out.latent_sequence,
            "sentiment": sentiment_out.sentiment,
            "sentiment_class": sentiment_out.sentiment_class,
            "sentiment_confidence": sentiment_out.sentiment_confidence,
            "sentiment_probabilities": sentiment_out.sentiment_probabilities,
            "sentiment_logits": sentiment_out.sentiment_logits,
            "sentiment_score": sentiment_out.sentiment_score,
            "temporal_attention_weights": latent_out.temporal_attention_weights,
            "sentiment_labels": sentiment_out.sentiment_labels,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SentimentDerivationPipeline":
        """Instantiate pipeline from a combined configuration mapping."""
        mapper = LatentSentimentMapper.from_config(config)
        head = SentimentHead.from_config(config)
        return cls(latent_mapper=mapper, sentiment_head=head)
