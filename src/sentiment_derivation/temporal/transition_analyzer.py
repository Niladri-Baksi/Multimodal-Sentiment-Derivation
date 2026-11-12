"""Temporal emotion transition analysis module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 5 & Stage 6.

This module provides:
1. TransitionOutput: Standardized container storing transitional embeddings, emotion velocity
   (delta emotion), persistence scores, and temporal attention weights.
2. TransitionAnalyzer: Neural module modeling emotion dynamics over time:
   - Delta Emotion (velocity/difference across consecutive time steps)
   - Persistence (sustained vs transient emotional states)
   - Temporal Attention (saliency of emotional transition inflection points)
   - Disentangled composite transition representation (Emotion, Affect, Confidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransitionOutput:
    """Standardized output container for temporal emotion transition analysis.

    Per AGENTS.md Stage 5 & Stage 6:
    - Produces transitional embeddings BEFORE final sentiment derivation.
    - Incorporates Emotion, Affective State, and Confidence streams.

    Attributes:
        transition_embedding: Dense composite transition representation shaped
            ``[B, T, transition_dim]`` or ``[B, transition_dim]``.
        delta_emotion: First-order temporal difference / velocity of emotion probabilities
            shaped ``[B, T, num_emotions]``, where delta_emotion[:, 0, :] == 0.0.
            Values in [-1.0, 1.0] indicate rising (>0) or fading (<0) emotions.
        persistence: Smoothed accumulation tracking sustained vs. transient emotions
            shaped ``[B, T, num_emotions]`` with values in [0.0, 1.0].
        temporal_attention_weights: Saliency scores over time steps shaped ``[B, T, 1]``
            normalized via Softmax (sum across T is 1.0).
        pooled_transition: Attention-weighted sequence summary shaped ``[B, transition_dim]``.
        delta_valence: Optional first-order temporal difference of continuous valence
            shaped ``[B, T, 1]``.
        emotion_names: Optional tuple of emotion class names.
    """

    transition_embedding: torch.Tensor
    delta_emotion: torch.Tensor
    persistence: torch.Tensor
    temporal_attention_weights: torch.Tensor
    pooled_transition: torch.Tensor
    delta_valence: Optional[torch.Tensor] = None
    emotion_names: Optional[Tuple[str, ...]] = None

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in TransitionOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary matching AGENTS.md contracts."""
        d: Dict[str, Any] = {
            "transition_embedding": self.transition_embedding,
            "delta_emotion": self.delta_emotion,
            "persistence": self.persistence,
            "temporal_attention_weights": self.temporal_attention_weights,
            "pooled_transition": self.pooled_transition,
            "emotion_names": self.emotion_names,
        }
        if self.delta_valence is not None:
            d["delta_valence"] = self.delta_valence
        return d


class TransitionAnalyzer(nn.Module):
    """Analyze temporal emotion evolution, velocity, persistence, and transition saliency.

    Per AGENTS.md Canonical Architecture:
    - Input:
      - emotion_probabilities: [B, T, K]
      - context_embeddings: [B, T, D_context]
      - affective_state / valence: [B, T, D_affect] or [B, T, 1]
      - confidence: [B, T, K]
    - Computes:
      1. Delta Emotion (velocity): E_t - E_{t-1}
      2. Persistence: Exponential Moving Average (EMA) of emotional activation
      3. Temporal Attention: Weights salient transition moments across T steps
      4. Disentangled projection & composite transition representation [B, T, D_trans]
    """

    def __init__(
        self,
        num_emotions: int = 8,
        context_dim: int = 128,
        affective_dim: int = 64,
        transition_dim: int = 128,
        hidden_dim: int = 64,
        persistence_alpha: float = 0.7,
        learnable_persistence: bool = True,
        dropout: float = 0.1,
        emotion_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize TransitionAnalyzer.

        Args:
            num_emotions: Number of multi-label emotion classes (default 8).
            context_dim: Dimension of temporal context features from Context Branch (default 128).
            affective_dim: Dimension of latent affective state embedding (default 64).
            transition_dim: Dimension of output composite transition representation (default 128).
            hidden_dim: Intermediate projection dimension (default 64).
            persistence_alpha: Initial decay factor for persistence accumulation in [0, 1].
            learnable_persistence: If True, optimizes per-emotion persistence alpha.
            dropout: Dropout probability for regularizing projections.
            emotion_names: Optional names for emotion classes.
        """
        super().__init__()
        if num_emotions <= 0 or context_dim <= 0 or transition_dim <= 0:
            raise ValueError("All dimension arguments must be positive integers.")
        if not 0.0 <= persistence_alpha <= 1.0:
            raise ValueError(f"persistence_alpha must be in [0, 1]; got {persistence_alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.num_emotions = num_emotions
        self.context_dim = context_dim
        self.affective_dim = affective_dim
        self.transition_dim = transition_dim
        self.hidden_dim = hidden_dim
        self.learnable_persistence = learnable_persistence

        if emotion_names is None:
            self.emotion_names = tuple(f"emotion_{i}" for i in range(num_emotions))
        else:
            if len(emotion_names) != num_emotions:
                raise ValueError(
                    f"emotion_names length ({len(emotion_names)}) must match num_emotions ({num_emotions})."
                )
            self.emotion_names = tuple(emotion_names)

        # 1. Persistence smoothing parameter
        # Parameterized via sigmoid to guarantee alpha_k strictly in (0, 1)
        init_logit = float(torch.logit(torch.tensor(persistence_alpha)))
        if learnable_persistence:
            self.persistence_logits = nn.Parameter(
                torch.full((num_emotions,), init_logit, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "persistence_logits",
                torch.full((num_emotions,), init_logit, dtype=torch.float32),
            )

        # 2. Emotion Dynamics Stream: projects [E_t, Delta E_t, Persistence_t] -> 64-dim
        # Input dimension: 8 (current) + 8 (delta) + 8 (persistence) = 24
        emo_dyn_input_dim = num_emotions * 3
        self.emotion_dynamics_proj = nn.Sequential(
            nn.Linear(emo_dyn_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # 3. Affective Dynamics Stream: projects [Affect_t, Delta Valence_t] -> 32-dim
        # Input dimension: affective_dim (64) + 1 (valence) + 1 (delta_valence) = 66
        affect_dyn_input_dim = affective_dim + 2
        self.affect_dynamics_proj = nn.Sequential(
            nn.Linear(affect_dyn_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Confidence Stream: projects [Confidence_t, Uncertainty_t] -> 32-dim
        # Input dimension: 8 (confidence) + 8 (uncertainty) = 16
        conf_dyn_input_dim = num_emotions * 2
        self.conf_dynamics_proj = nn.Sequential(
            nn.Linear(conf_dyn_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Composite Transition Projection: combines Context + Emotion + Affect + Conf
        # Dimensions: context_dim (128) + 64 (emo) + 32 (aff) + 32 (conf) = 256
        composite_input_dim = context_dim + hidden_dim + (hidden_dim // 2) * 2
        self.composite_proj = nn.Sequential(
            nn.Linear(composite_input_dim, transition_dim),
            nn.LayerNorm(transition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transition_dim, transition_dim),
            nn.LayerNorm(transition_dim),
        )

        # 6. Temporal Attention Mechanism: scores transition saliency across T steps
        self.temporal_attention = nn.Sequential(
            nn.Linear(transition_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    @property
    def persistence_alpha(self) -> torch.Tensor:
        """Return per-emotion persistence smoothing factor in (0, 1)."""
        return torch.sigmoid(self.persistence_logits)

    def compute_delta(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute first-order temporal difference along sequence dimension (dim=1).

        For sequence [x_0, x_1, ..., x_{T-1}]:
            delta_t = x_t - x_{t-1} for t >= 1
            delta_0 = 0.0

        Args:
            tensor: Input tensor shaped ``[B, T, D]``.

        Returns:
            Delta tensor shaped identically to input.
        """
        if tensor.ndim < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions; got {tensor.ndim}.")
        steps = tensor.shape[1]
        if steps <= 1:
            return torch.zeros_like(tensor)

        diff = tensor[:, 1:] - tensor[:, :-1]
        zero_pad = torch.zeros(
            (tensor.shape[0], 1) + tensor.shape[2:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([zero_pad, diff], dim=1)

    def compute_persistence(self, emotion_probs: torch.Tensor) -> torch.Tensor:
        """Compute exponential moving average persistence across sequence steps.

        P_0 = E_0
        P_t = alpha * P_{t-1} + (1 - alpha) * E_t

        Args:
            emotion_probs: Probabilities shaped ``[B, T, num_emotions]``.

        Returns:
            Persistence tensor shaped ``[B, T, num_emotions]`` in [0.0, 1.0].
        """
        batch_size, steps, num_emotions = emotion_probs.shape
        alpha = self.persistence_alpha.to(emotion_probs.device)  # [num_emotions]

        persistence_steps = []
        p_prev = emotion_probs[:, 0, :]  # P_0 = E_0
        persistence_steps.append(p_prev)

        for t in range(1, steps):
            e_t = emotion_probs[:, t, :]
            # P_t = alpha * P_{t-1} + (1 - alpha) * E_t
            p_t = alpha * p_prev + (1.0 - alpha) * e_t
            persistence_steps.append(p_t)
            p_prev = p_t

        return torch.stack(persistence_steps, dim=1)

    def forward(
        self,
        emotion_probabilities: torch.Tensor,
        context_embeddings: torch.Tensor,
        affective_state: Optional[torch.Tensor] = None,
        valence: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> TransitionOutput:
        """Compute transitional representation and dynamics.

        Args:
            emotion_probabilities: Predicted probabilities shaped ``[B, T, num_emotions]`` or ``[B, num_emotions]``.
            context_embeddings: Temporal context features shaped ``[B, T, context_dim]`` or ``[B, context_dim]``.
            affective_state: Latent affective embeddings shaped ``[B, T, affective_dim]``.
            valence: Continuous valence ratings shaped ``[B, T, 1]``.
            confidence: Emotion prediction confidence shaped ``[B, T, num_emotions]``.
            uncertainty: Emotion prediction uncertainty shaped ``[B, T, num_emotions]``.

        Returns:
            TransitionOutput dataclass conforming to AGENTS.md Stage 6.
        """
        # Ensure 3D tensors: [B, T, *]
        orig_ndim = emotion_probabilities.ndim
        if orig_ndim == 2:
            emotion_probabilities = emotion_probabilities.unsqueeze(1)
            context_embeddings = context_embeddings.unsqueeze(1)
            if affective_state is not None:
                affective_state = affective_state.unsqueeze(1)
            if valence is not None:
                valence = valence.unsqueeze(1)
            if confidence is not None:
                confidence = confidence.unsqueeze(1)
            if uncertainty is not None:
                uncertainty = uncertainty.unsqueeze(1)

        batch_size, steps, k_dim = emotion_probabilities.shape
        device = emotion_probabilities.device

        # 1. Compute Emotion Dynamics (Delta Emotion & Persistence)
        delta_emotion = self.compute_delta(emotion_probabilities)
        persistence = self.compute_persistence(emotion_probabilities)

        # Project emotion dynamics: [B, T, 24] -> [B, T, hidden_dim=64]
        emo_dyn_input = torch.cat([emotion_probabilities, delta_emotion, persistence], dim=-1)
        emo_emb = self.emotion_dynamics_proj(emo_dyn_input)

        # 2. Compute Affective Dynamics
        if affective_state is None:
            affective_state = torch.zeros(batch_size, steps, self.affective_dim, device=device)
        if valence is None:
            valence = torch.full((batch_size, steps, 1), 4.0, device=device)

        delta_val = self.compute_delta(valence)
        affect_dyn_input = torch.cat([affective_state, valence, delta_val], dim=-1)
        affect_emb = self.affect_dynamics_proj(affect_dyn_input)

        # 3. Compute Confidence Dynamics
        if confidence is None:
            # Fallback to margin confidence: 2 * |p - 0.5|
            confidence = torch.clamp(2.0 * torch.abs(emotion_probabilities - 0.5), 0.0, 1.0)
        if uncertainty is None:
            uncertainty = torch.clamp(1.0 - confidence, 0.0, 1.0)

        conf_dyn_input = torch.cat([confidence, uncertainty], dim=-1)
        conf_emb = self.conf_dynamics_proj(conf_dyn_input)

        # 4. Form Composite Transition Representation
        # Concatenate: [Context (128), Emo (64), Affect (32), Conf (32)] -> 256 -> 128
        composite_input = torch.cat([context_embeddings, emo_emb, affect_emb, conf_emb], dim=-1)
        transition_emb = self.composite_proj(composite_input)  # [B, T, transition_dim]

        # 5. Temporal Attention across steps: scores salient transition moments
        # Attention scores: [B, T, 1]
        attn_logits = self.temporal_attention(transition_emb)
        attn_weights = F.softmax(attn_logits, dim=1)  # Softmax across T

        # Pooled transition summary: sum_t (alpha_t * h_t) -> [B, transition_dim]
        pooled_transition = torch.sum(attn_weights * transition_emb, dim=1)

        # Restore original 2D shape if input was un-sequenced
        if orig_ndim == 2:
            transition_emb = transition_emb.squeeze(1)
            delta_emotion = delta_emotion.squeeze(1)
            persistence = persistence.squeeze(1)
            attn_weights = attn_weights.squeeze(1)
            delta_val = delta_val.squeeze(1)

        return TransitionOutput(
            transition_embedding=transition_emb,
            delta_emotion=delta_emotion,
            persistence=persistence,
            temporal_attention_weights=attn_weights,
            pooled_transition=pooled_transition,
            delta_valence=delta_val,
            emotion_names=self.emotion_names,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TransitionAnalyzer":
        """Instantiate analyzer from configuration dictionary."""
        cfg = config.get("temporal", config)
        return cls(
            num_emotions=cfg.get("num_emotions", 8),
            context_dim=cfg.get("context_dim", 128),
            affective_dim=cfg.get("affective_dim", 64),
            transition_dim=cfg.get("transition_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 64),
            persistence_alpha=cfg.get("persistence_alpha", 0.7),
            learnable_persistence=cfg.get("learnable_persistence", True),
            dropout=cfg.get("dropout", 0.1),
            emotion_names=cfg.get("emotion_names", None),
        )
