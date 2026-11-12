"""Uncertainty-aware gating network for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 7 (Uncertainty-Aware Gating).

This module provides:
1. UncertaintyGateOutput: Container holding gated representations, gating activation weights
   (for Explainable AI), and gate configuration.
2. UncertaintyAwareGate: Trainable neural gating network evaluating the reliability and contribution
   of emotional states across temporal segments to filter noisy or ambiguous representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UncertaintyGateOutput:
    """Standardized output container for uncertainty-aware gating.

    Attributes:
        gated_features: Modulated representation shaped ``[B, T, D]`` or ``[B, D]``.
        gate_weights: Dynamic gating weights in [0.0, 1.0] shaped like ``gated_features``.
            Essential for Stage 10 XAI to explain which temporal steps were trusted/suppressed.
        temporal_saliency: Step-wise mean gate acceptance score shaped ``[B, T, 1]`` or ``[B, 1]``.
        enabled: Boolean indicating whether gating was active or bypassed (for ablation studies).
    """

    gated_features: torch.Tensor
    gate_weights: torch.Tensor
    temporal_saliency: torch.Tensor
    enabled: bool = True

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in UncertaintyGateOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to a dictionary matching AGENTS.md contracts."""
        return {
            "gated_features": self.gated_features,
            "gate_weights": self.gate_weights,
            "temporal_saliency": self.temporal_saliency,
            "enabled": self.enabled,
        }


class UncertaintyAwareGate(nn.Module):
    """Trainable neural gate modulating representations based on prediction certainty.

    Per AGENTS.md Stage 7:
    - Evaluates the reliability and contribution of cumulative emotional states across
      temporal segments before latent sentiment mapping.
    - Concatenates context representations with emotion probabilities, emotion uncertainties,
      and affective state certainty.
    - Learns dynamic acceptance gate weights g_t in [0.0, 1.0]^D:
          g_t = sigmoid( MLP([h_t, p_t, u_t, c_affect_t]) )
          h_gated_t = g_t * h_t + (1 - g_t) * prior_t
    - Features an explicit `enabled=False` switch for required ablation studies.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_emotions: int = 8,
        valence_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        residual_prior: bool = True,
        enabled: bool = True,
    ) -> None:
        """Initialize UncertaintyAwareGate.

        Args:
            feature_dim: Dimension of input feature sequence (e.g. context embeddings D=128).
            num_emotions: Number of multi-label discrete emotions (default 8).
            valence_dim: Dimension of valence confidence / affect features (default 1).
            hidden_dim: Hidden dimension of the gate projection MLP (default 64).
            dropout: Dropout probability for gate regularization.
            residual_prior: If True, uses convex combination g * h + (1 - g) * prior.
                If False, uses multiplicative filtering g * h.
            enabled: If False, completely bypasses the gate for ablation studies.
        """
        super().__init__()
        if feature_dim <= 0 or num_emotions <= 0 or hidden_dim <= 0:
            raise ValueError("All dimension arguments must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.feature_dim = feature_dim
        self.num_emotions = num_emotions
        self.valence_dim = valence_dim
        self.hidden_dim = hidden_dim
        self.residual_prior = residual_prior
        self.enabled = enabled

        # Total input dimension to gate network:
        # feature_dim (D) + emotion_probabilities (8) + emotion_uncertainty (8) + valence_confidence (1)
        self.gate_input_dim = feature_dim + (2 * num_emotions) + valence_dim

        # Gating network: projects combined evidence into dynamic channel-wise gate weights [0, 1]^D
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Learnable prior/baseline state for temporal slots rejected by the gate
        if residual_prior:
            self.prior = nn.Parameter(torch.zeros(feature_dim))
        else:
            self.register_parameter("prior", None)

    def forward(
        self,
        features: torch.Tensor,
        emotion_probabilities: torch.Tensor,
        emotion_uncertainty: torch.Tensor,
        valence_confidence: Optional[torch.Tensor] = None,
    ) -> UncertaintyGateOutput:
        """Filter and modulate feature representations using certainty evidence.

        Args:
            features: Context or transition features shaped ``[B, T, feature_dim]`` or ``[B, feature_dim]``.
            emotion_probabilities: Predicted emotion probabilities shaped ``[B, T, num_emotions]`` or ``[B, num_emotions]``.
            emotion_uncertainty: Per-emotion uncertainty shaped like ``emotion_probabilities``.
            valence_confidence: Continuous valence certainty shaped ``[B, T, valence_dim]`` or ``[B, valence_dim]``.
                If None, defaults to zeros.

        Returns:
            UncertaintyGateOutput with gated_features and gate_weights.
        """
        if not self.enabled:
            # Ablation bypass: gate is inactive, returning untouched features
            ones_gate = torch.ones_like(features)
            mean_saliency = torch.ones(
                features.shape[:-1] + (1,), dtype=features.dtype, device=features.device
            )
            return UncertaintyGateOutput(
                gated_features=features,
                gate_weights=ones_gate,
                temporal_saliency=mean_saliency,
                enabled=False,
            )

        # Align dimensions: ensure 3D [B, T, *] or 2D [B, *]
        orig_ndim = features.ndim
        if orig_ndim == 1:
            features = features.unsqueeze(0).unsqueeze(0)
            emotion_probabilities = emotion_probabilities.unsqueeze(0).unsqueeze(0)
            emotion_uncertainty = emotion_uncertainty.unsqueeze(0).unsqueeze(0)
            if valence_confidence is not None:
                valence_confidence = valence_confidence.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 2:
            # [B, D]
            features = features.unsqueeze(1)
            emotion_probabilities = emotion_probabilities.unsqueeze(1)
            emotion_uncertainty = emotion_uncertainty.unsqueeze(1)
            if valence_confidence is not None:
                valence_confidence = valence_confidence.unsqueeze(1)

        batch_size, steps, dim = features.shape
        if dim != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}; got {dim} in features."
            )

        # Default valence_confidence to neutral 0.5 certainty if omitted
        if valence_confidence is None:
            valence_confidence = torch.full(
                (batch_size, steps, self.valence_dim),
                0.5,
                dtype=features.dtype,
                device=features.device,
            )
        elif valence_confidence.ndim == 2 and valence_confidence.shape[-1] == 1:
            valence_confidence = valence_confidence.unsqueeze(1)

        # 1. Form composite certainty evidence vector:
        # [features, emotion_probs, emotion_uncertainty, valence_confidence]
        gate_input = torch.cat(
            [features, emotion_probabilities, emotion_uncertainty, valence_confidence],
            dim=-1,
        )

        # 2. Compute dynamic gating weights: g_t in [0.0, 1.0]^D
        gate_weights = self.gate_mlp(gate_input)

        # 3. Compute overall temporal saliency (average gate acceptance score per time step)
        temporal_saliency = torch.mean(gate_weights, dim=-1, keepdim=True)

        # 4. Modulate features
        if self.residual_prior and self.prior is not None:
            # Convex combination: trusted features are preserved, untrusted features fall back to prior
            gated_features = (gate_weights * features) + ((1.0 - gate_weights) * self.prior)
        else:
            # Multiplicative filtering
            gated_features = gate_weights * features

        # Restore original dimension if input was 2D or 1D
        if orig_ndim == 2:
            gated_features = gated_features.squeeze(1)
            gate_weights = gate_weights.squeeze(1)
            temporal_saliency = temporal_saliency.squeeze(1)
        elif orig_ndim == 1:
            gated_features = gated_features.squeeze(0).squeeze(0)
            gate_weights = gate_weights.squeeze(0).squeeze(0)
            temporal_saliency = temporal_saliency.squeeze(0).squeeze(0)

        return UncertaintyGateOutput(
            gated_features=gated_features,
            gate_weights=gate_weights,
            temporal_saliency=temporal_saliency,
            enabled=True,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "UncertaintyAwareGate":
        """Instantiate gate from configuration dictionary."""
        cfg = config.get("uncertainty_gate", config)
        return cls(
            feature_dim=cfg.get("feature_dim", 128),
            num_emotions=cfg.get("num_emotions", 8),
            valence_dim=cfg.get("valence_dim", 1),
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.1),
            residual_prior=cfg.get("residual_prior", True),
            enabled=cfg.get("enabled", True),
        )
