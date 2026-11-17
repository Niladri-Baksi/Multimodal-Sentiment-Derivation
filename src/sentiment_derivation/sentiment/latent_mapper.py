"""Latent sentiment mapping module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 8 (Latent Sentiment Mapping).

This module provides:
1. LatentSentimentOutput: Standardized container storing the evolving latent affective
   representation across sequence steps, pooled latent vector, and temporal attention weights.
2. LatentSentimentMapper: Neural network mapping the multidimensional emotional state,
   emotion-wise confidence, temporal dynamics (delta emotion, persistence), and
   uncertainty-gated context into a unified Latent Affective / Sentiment Representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

FeatureInput = Union[torch.Tensor, Mapping[str, Any]]


@dataclass(frozen=True)
class LatentSentimentOutput:
    """Standardized output container for latent sentiment mapping.

    Per AGENTS.md Stage 8:
    - Captures the latent affective representation and its evolution across time steps (t1..tn).
    - Incorporates current emotional state, emotion-wise confidence, temporal changes,
      emotion persistence, and uncertainty-gated context.

    Attributes:
        latent_sequence: Step-level latent affective representation shaped ``[B, T, latent_dim]``.
        latent_representation: Overall pooled latent representation shaped ``[B, latent_dim]``.
        temporal_attention_weights: Saliency scores over time steps shaped ``[B, T, 1]``
            normalized via Softmax (sum across T is 1.0).
        metadata: Dictionary with dimension and configuration details.
    """

    latent_sequence: torch.Tensor
    latent_representation: torch.Tensor
    temporal_attention_weights: torch.Tensor
    metadata: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in LatentSentimentOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary matching AGENTS.md contracts."""
        return {
            "latent_sequence": self.latent_sequence,
            "latent_representation": self.latent_representation,
            "temporal_attention_weights": self.temporal_attention_weights,
            "metadata": self.metadata,
        }


class LatentSentimentMapper(nn.Module):
    """Map multidimensional emotion dynamics and gated context into latent sentiment.

    Per AGENTS.md Canonical Architecture Stage 8:
    - Inputs:
      1. Gated Context Features: uncertainty-aware representation from Stage 7 [B, T, D_context].
      2. Emotion Probabilities: multi-label emotional state [B, T, K].
      3. Emotion Confidence: emotion-wise reliability estimates [B, T, K].
      4. Delta Emotion: temporal velocity/transitions [B, T, K].
      5. Emotion Persistence: smoothed temporal accumulation [B, T, K].
      6. Transition Embedding: composite transition features from Stage 6 [B, T, D_trans] (optional).
      7. Affective State / Valence: continuous affect representation [B, T, D_affect] or [B, T, 1] (optional).
    - Processing:
      - Projects disentangled streams (emotion dynamics, context, transition, affect) into
        a unified latent affective representation z_t in R^{latent_dim}.
      - Applies learnable temporal attention pooling across T steps to compute the overall
        summary latent vector z_pooled in R^{latent_dim}.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        transition_dim: int = 128,
        num_emotions: int = 8,
        affective_dim: int = 64,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        temporal_pooling: str = "attention",
    ) -> None:
        """Initialize LatentSentimentMapper.

        Args:
            feature_dim: Dimension of gated context features from Stage 7 (default 128).
            transition_dim: Dimension of composite transition embeddings from Stage 6 (default 128).
            num_emotions: Number of multi-label discrete emotions (default 8).
            affective_dim: Dimension of continuous affective state embeddings (default 64).
            latent_dim: Output dimension of latent sentiment representations (default 128).
            hidden_dim: Intermediate projection dimension for feature streams (default 64).
            dropout: Dropout probability for regularization.
            temporal_pooling: Sequence aggregation strategy: ``"attention"`` (default),
                ``"mean"``, or ``"last"``.
        """
        super().__init__()
        if (
            feature_dim <= 0
            or transition_dim <= 0
            or num_emotions <= 0
            or latent_dim <= 0
            or hidden_dim <= 0
        ):
            raise ValueError("All dimension arguments must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")
        if temporal_pooling not in ("attention", "mean", "last"):
            raise ValueError(
                f"temporal_pooling must be 'attention', 'mean', or 'last'; got '{temporal_pooling}'."
            )

        self.feature_dim = feature_dim
        self.transition_dim = transition_dim
        self.num_emotions = num_emotions
        self.affective_dim = affective_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.temporal_pooling = temporal_pooling

        # 1. Stream 1: Emotion Dynamics [probabilities (K) + confidence (K) + delta (K) + persistence (K)] = 4*K
        emotion_dynamics_dim = num_emotions * 4
        self.emotion_stream_proj = nn.Sequential(
            nn.Linear(emotion_dynamics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Stream 2: Uncertainty-Gated Context
        self.context_stream_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Stream 3: Composite Transition Representation
        self.transition_stream_proj = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Stream 4: Affective State / Valence
        self.affect_stream_proj = nn.Sequential(
            nn.Linear(affective_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Multimodal Latent Fusion MLP
        # Combined dimension: hidden_dim (emotion) + hidden_dim (context) + hidden_dim (transition) + hidden_dim//2 (affect)
        combined_dim = hidden_dim * 3 + (hidden_dim // 2)
        self.latent_fusion = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 6. Temporal Attention Pooling (learnable attention over sequence steps)
        if temporal_pooling == "attention":
            self.temporal_attention = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
        else:
            self.temporal_attention = None

    @staticmethod
    def _ensure_3d(tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Ensure input tensor has shape [B, T, D], tracking if original was 2D or 1D."""
        if tensor.ndim == 1:
            return tensor.unsqueeze(0).unsqueeze(0), True
        if tensor.ndim == 2:
            return tensor.unsqueeze(1), True
        if tensor.ndim == 3:
            return tensor, False
        raise ValueError(
            f"Input tensor must have 1, 2, or 3 dimensions; got shape {tuple(tensor.shape)}."
        )

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
    ) -> LatentSentimentOutput:
        """Map emotional dynamics, transitions, and context to latent sentiment.

        Args:
            gated_features: Uncertainty-gated representation shaped ``[B, T, feature_dim]`` or ``[B, feature_dim]``.
            emotion_probabilities: Predicted probabilities shaped ``[B, T, num_emotions]``.
            emotion_confidence: Emotion-wise confidence shaped ``[B, T, num_emotions]``.
            delta_emotion: Temporal velocity/transitions shaped ``[B, T, num_emotions]``.
            persistence: Emotion persistence shaped ``[B, T, num_emotions]``.
            transition_embedding: Transition features shaped ``[B, T, transition_dim]``.
                If None, zeros are substituted.
            affective_state: Dense affective state shaped ``[B, T, affective_dim]``.
                If None and valence is provided, valence is projected or zeros substituted.
            valence: Continuous valence ratings shaped ``[B, T, 1]``.

        Returns:
            LatentSentimentOutput containing latent_sequence, latent_representation,
            and temporal_attention_weights.
        """
        # Ensure all inputs are 3D: [B, T, D]
        gated_3d, is_unbatched = self._ensure_3d(gated_features)
        batch_size, steps, feat_dim = gated_3d.shape
        device = gated_3d.device

        if feat_dim != self.feature_dim:
            raise ValueError(
                f"Expected gated_features feature_dim={self.feature_dim}; got {feat_dim}."
            )

        prob_3d, _ = self._ensure_3d(emotion_probabilities)
        conf_3d, _ = self._ensure_3d(emotion_confidence)
        delta_3d, _ = self._ensure_3d(delta_emotion)
        pers_3d, _ = self._ensure_3d(persistence)

        for name, t in (
            ("emotion_probabilities", prob_3d),
            ("emotion_confidence", conf_3d),
            ("delta_emotion", delta_3d),
            ("persistence", pers_3d),
        ):
            if t.shape[-1] != self.num_emotions:
                raise ValueError(
                    f"Expected {name} to have num_emotions={self.num_emotions}; got {t.shape[-1]}."
                )

        # 1. Project Emotion Dynamics Stream: [B, T, 4 * num_emotions] -> [B, T, hidden_dim]
        emotion_dynamics = torch.cat([prob_3d, conf_3d, delta_3d, pers_3d], dim=-1)
        h_emotion = self.emotion_stream_proj(emotion_dynamics)

        # 2. Project Gated Context Stream: [B, T, feature_dim] -> [B, T, hidden_dim]
        h_context = self.context_stream_proj(gated_3d)

        # 3. Project Transition Stream: [B, T, transition_dim] -> [B, T, hidden_dim]
        if transition_embedding is not None:
            trans_3d, _ = self._ensure_3d(transition_embedding)
            if trans_3d.shape[-1] != self.transition_dim:
                raise ValueError(
                    f"Expected transition_embedding dimension {self.transition_dim}; got {trans_3d.shape[-1]}."
                )
        else:
            trans_3d = torch.zeros(batch_size, steps, self.transition_dim, device=device)
        h_transition = self.transition_stream_proj(trans_3d)

        # 4. Project Affective State Stream: [B, T, affective_dim] -> [B, T, hidden_dim // 2]
        if affective_state is not None:
            affect_3d, _ = self._ensure_3d(affective_state)
            if affect_3d.shape[-1] != self.affective_dim:
                # If scalar valence [B, T, 1] was passed as affective_state, pad or adapt
                if affect_3d.shape[-1] == 1:
                    affect_3d = affect_3d.expand(-1, -1, self.affective_dim)
                else:
                    raise ValueError(
                        f"Expected affective_state dimension {self.affective_dim}; got {affect_3d.shape[-1]}."
                    )
        elif valence is not None:
            val_3d, _ = self._ensure_3d(valence)
            affect_3d = val_3d.expand(-1, -1, self.affective_dim)
        else:
            affect_3d = torch.zeros(batch_size, steps, self.affective_dim, device=device)
        h_affect = self.affect_stream_proj(affect_3d)

        # 5. Fuse Streams into Latent Sequence: [B, T, latent_dim]
        combined = torch.cat([h_emotion, h_context, h_transition, h_affect], dim=-1)
        latent_seq = self.latent_fusion(combined)

        # 6. Temporal Pooling to compute summary latent representation [B, latent_dim]
        if steps == 1 or self.temporal_pooling == "last":
            pooled_latent = latent_seq[:, -1, :]
            temporal_weights = torch.ones(batch_size, steps, 1, device=device)
        elif self.temporal_pooling == "mean":
            pooled_latent = latent_seq.mean(dim=1)
            temporal_weights = torch.full((batch_size, steps, 1), 1.0 / steps, device=device)
        elif self.temporal_pooling == "attention":
            attn_scores = self.temporal_attention(latent_seq)  # [B, T, 1]
            temporal_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]
            pooled_latent = (latent_seq * temporal_weights).sum(dim=1)  # [B, latent_dim]
        else:
            pooled_latent = latent_seq.mean(dim=1)
            temporal_weights = torch.full((batch_size, steps, 1), 1.0 / steps, device=device)

        # Adjust dimensions if input was originally 2D or 1D
        if is_unbatched and steps == 1:
            latent_seq_out = latent_seq.squeeze(1)
            if latent_seq_out.shape[0] == 1:
                latent_seq_out = latent_seq_out.squeeze(0)
                pooled_latent = pooled_latent.squeeze(0)
                temporal_weights = temporal_weights.squeeze(0).squeeze(0)
        else:
            latent_seq_out = latent_seq

        return LatentSentimentOutput(
            latent_sequence=latent_seq_out,
            latent_representation=pooled_latent,
            temporal_attention_weights=temporal_weights,
            metadata={
                "feature_dim": self.feature_dim,
                "transition_dim": self.transition_dim,
                "num_emotions": self.num_emotions,
                "latent_dim": self.latent_dim,
                "temporal_pooling": self.temporal_pooling,
            },
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LatentSentimentMapper":
        """Instantiate LatentSentimentMapper from configuration dictionary."""
        cfg = config.get("latent_mapper", config)
        return cls(
            feature_dim=cfg.get("feature_dim", 128),
            transition_dim=cfg.get("transition_dim", 128),
            num_emotions=cfg.get("num_emotions", 8),
            affective_dim=cfg.get("affective_dim", 64),
            latent_dim=cfg.get("latent_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 64),
            dropout=cfg.get("dropout", 0.1),
            temporal_pooling=cfg.get("temporal_pooling", "attention"),
        )
