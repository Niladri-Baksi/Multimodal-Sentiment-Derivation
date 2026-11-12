"""Affective state estimation module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 (Parallel Representation)
and EmoReact Continuous Affect Annotations.

This module provides:
1. AffectiveStateOutput: Structured dataclass capturing continuous valence predictions
   and latent affective state representations.
2. AffectiveStateEstimator: Neural network head projecting multimodal/temporal
   embeddings into continuous valence ratings ([1.0, 7.0] Likert scale) and dense
   affective state embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

FeatureInput = Union[torch.Tensor, Mapping[str, Any]]


@dataclass(frozen=True)
class AffectiveStateOutput:
    """Standardized output container for affective state estimation.

    Attributes:
        valence: Continuous valence predictions shaped ``[B, T, 1]`` or ``[B, 1]``.
            In bounded mode, values are strictly within ``[min_valence, max_valence]``.
        affective_state: Dense latent affective state representation shaped
            ``[B, T, affective_dim]`` or ``[B, affective_dim]``.
        raw_valence: Unbounded pre-activation linear valence projection.
        valence_confidence: Emotion-wise/affective certainty shaped like ``valence``,
            normalized to ``[0.0, 1.0]``.
        min_valence: Minimum value of the valence scale (1.0 in EmoReact).
        max_valence: Maximum value of the valence scale (7.0 in EmoReact).
    """

    valence: torch.Tensor
    affective_state: torch.Tensor
    raw_valence: torch.Tensor
    valence_confidence: torch.Tensor
    min_valence: float = 1.0
    max_valence: float = 7.0

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in AffectiveStateOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to a dictionary matching AGENTS.md contracts."""
        return {
            "valence": self.valence,
            "affective_state": self.affective_state,
            "raw_valence": self.raw_valence,
            "valence_confidence": self.valence_confidence,
            "min_valence": self.min_valence,
            "max_valence": self.max_valence,
        }


class AffectiveStateEstimator(nn.Module):
    """Predict continuous affective state (valence) and latent affective embeddings.

    Per AGENTS.md and EmoReact specifications:
    - Input: Fused multimodal sequence ``[B, T, D]`` or pooled vector ``[B, D]``.
    - Output:
      - Continuous valence score on the 1.0 to 7.0 Likert scale.
      - Dense latent affective state embedding ``[B, T, D_affect]`` or ``[B, D_affect]``.
      - Affective confidence estimate normalized to ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        affective_dim: int = 64,
        min_valence: float = 1.0,
        max_valence: float = 7.0,
        dropout: float = 0.1,
        bounded: bool = True,
    ) -> None:
        """Initialize AffectiveStateEstimator.

        Args:
            input_dim: Dimension of incoming multimodal or temporal features.
            hidden_dim: Intermediate projection dimension.
            affective_dim: Dimension of latent affective state embedding.
            min_valence: Lower bound for valence rating (default 1.0 for EmoReact).
            max_valence: Upper bound for valence rating (default 7.0 for EmoReact).
            dropout: Dropout probability for regularization.
            bounded: If True, uses scaled sigmoid to strictly enforce [min, max] range.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or affective_dim <= 0:
            raise ValueError("All dimension arguments must be positive integers.")
        if min_valence >= max_valence:
            raise ValueError(
                f"min_valence ({min_valence}) must be strictly less than max_valence ({max_valence})."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affective_dim = affective_dim
        self.min_valence = float(min_valence)
        self.max_valence = float(max_valence)
        self.bounded = bounded

        # Latent affective state projection
        self.affective_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, affective_dim),
            nn.LayerNorm(affective_dim),
        )

        # Continuous valence regression head
        self.valence_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(affective_dim, 1),
        )

    @staticmethod
    def _extract_tensor(features: FeatureInput) -> Tuple[torch.Tensor, bool]:
        """Validate and extract feature tensor, tracking whether input was unbatched."""
        if isinstance(features, Mapping):
            if "features" not in features:
                raise ValueError(
                    "Input mapping must contain a 'features' key with a torch.Tensor."
                )
            features = features["features"]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Input features must be a torch.Tensor or Mapping, got {type(features).__name__}."
            )

        if not features.is_floating_point():
            raise TypeError("Input features must have a floating-point dtype.")

        is_unbatched = False
        if features.ndim == 1:
            # Unbatched vector [D] -> [1, 1, D]
            features = features.unsqueeze(0).unsqueeze(0)
            is_unbatched = True
        elif features.ndim == 2:
            # Could be [B, D] or [T, D]. Standard convention: treat [B, D] as batched.
            features = features.unsqueeze(1)  # [B, 1, D]
        elif features.ndim != 3:
            raise ValueError(
                f"Input features must have shape [B, T, D], [B, D], or [D]; got {tuple(features.shape)}."
            )

        return features, is_unbatched

    def forward(self, features: FeatureInput) -> AffectiveStateOutput:
        """Estimate affective state and valence from input features.

        Args:
            features: Input tensor shaped ``[B, T, input_dim]``, ``[B, input_dim]``,
                or a dictionary containing ``"features"``.

        Returns:
            AffectiveStateOutput containing valence, affective_state embedding,
            raw_valence, and valence_confidence.
        """
        tensor, is_unbatched = self._extract_tensor(features)
        batch_size, steps, dim = tensor.shape

        if dim != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}; got {dim}."
            )

        # 1. Project into dense affective state embedding: [B, T, affective_dim]
        affective_emb = self.affective_encoder(tensor)

        # 2. Compute raw valence logit: [B, T, 1]
        raw_val = self.valence_head(affective_emb)

        # 3. Apply bounding if configured
        if self.bounded:
            scale = self.max_valence - self.min_valence
            valence = self.min_valence + scale * torch.sigmoid(raw_val)
        else:
            valence = raw_val

        # 4. Compute affective prediction certainty:
        # Distance from neutral midpoint (4.0 in 1-7 scale), normalized to [0, 1]
        midpoint = 0.5 * (self.min_valence + self.max_valence)
        half_range = 0.5 * (self.max_valence - self.min_valence)
        valence_confidence = torch.clamp(
            torch.abs(valence - midpoint) / half_range, 0.0, 1.0
        )

        # If input was 2D [B, D], squeeze step dimension: [B, 1, *] -> [B, *]
        if steps == 1 and not is_unbatched:
            valence = valence.squeeze(1)
            affective_emb = affective_emb.squeeze(1)
            raw_val = raw_val.squeeze(1)
            valence_confidence = valence_confidence.squeeze(1)
        elif is_unbatched:
            valence = valence.squeeze(0).squeeze(0)
            affective_emb = affective_emb.squeeze(0).squeeze(0)
            raw_val = raw_val.squeeze(0).squeeze(0)
            valence_confidence = valence_confidence.squeeze(0).squeeze(0)

        return AffectiveStateOutput(
            valence=valence,
            affective_state=affective_emb,
            raw_valence=raw_val,
            valence_confidence=valence_confidence,
            min_valence=self.min_valence,
            max_valence=self.max_valence,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AffectiveStateEstimator":
        """Instantiate AffectiveStateEstimator from a configuration mapping.

        Looks for an ``affective_state`` section or root-level keys.
        """
        cfg = config.get("affective_state", config)
        return cls(
            input_dim=cfg.get("input_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 128),
            affective_dim=cfg.get("affective_dim", 64),
            min_valence=cfg.get("min_valence", 1.0),
            max_valence=cfg.get("max_valence", 7.0),
            dropout=cfg.get("dropout", 0.1),
            bounded=cfg.get("bounded", True),
        )
