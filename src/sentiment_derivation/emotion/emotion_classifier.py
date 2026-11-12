"""Multi-label emotion classification and parallel representation module.

Reference: AGENTS.md | Canonical Architecture Stage 4 (Parallel Representation:
Emotion Branch & Context Branch) and EmoReact Multi-Label Annotations.

This module provides:
1. DEFAULT_EMOTION_NAMES: The 8 canonical discrete emotions from EmoReact.
2. EmotionOutput: Structured dataclass capturing multi-label logits, probabilities,
   per-emotion confidence, per-emotion uncertainty, and parallel branch embeddings.
3. EmotionClassifier: Multi-label classifier maintaining logically distinct
   Emotion and Context branches, with separate per-emotion confidence estimation.
4. MultimodalEmotionPipeline: Composite module uniting EmotionClassifier and
   AffectiveStateEstimator into the canonical AGENTS.md Section 6 output interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from sentiment_derivation.emotion.affective_state import (
    AffectiveStateEstimator,
    AffectiveStateOutput,
)

# Canonical EmoReact discrete emotions strictly ordered by column indices 0..7
DEFAULT_EMOTION_NAMES: Tuple[str, ...] = (
    "Curiosity",     # Col 0
    "Uncertainty",   # Col 1
    "Excitement",    # Col 2
    "Happiness",     # Col 3
    "Surprise",      # Col 4
    "Disgust",       # Col 5
    "Fear",          # Col 6
    "Frustration",   # Col 7
)

NUM_EMOTIONS: int = len(DEFAULT_EMOTION_NAMES)
FeatureInput = Union[torch.Tensor, Mapping[str, Any]]


@dataclass(frozen=True)
class EmotionOutput:
    """Standardized output container for the Emotion and Context branches.

    Attributes:
        logits: Unnormalized classification logits shaped ``[B, T, 8]`` or ``[B, 8]``.
        probabilities: Multi-label sigmoid probabilities in ``[0.0, 1.0]``.
        emotion_confidence: Separate confidence scores for each emotion in ``[0.0, 1.0]``.
            Maintains per-emotion granularity without collapsing into a single scalar.
        emotion_uncertainty: Predictive uncertainty for each emotion in ``[0.0, 1.0]``,
            defined mathematically as ``1.0 - emotion_confidence``.
        emotion_embeddings: Intermediate representations from the Emotion Branch,
            shaped ``[B, T, emotion_dim]`` or ``[B, emotion_dim]``.
        context_embeddings: Parallel representations from the Context Branch,
            shaped ``[B, T, context_dim]`` or ``[B, context_dim]``.
        emotion_names: Canonical ordered names for the 8 emotion classes.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    emotion_confidence: torch.Tensor
    emotion_uncertainty: torch.Tensor
    emotion_embeddings: torch.Tensor
    context_embeddings: torch.Tensor
    emotion_names: Tuple[str, ...] = DEFAULT_EMOTION_NAMES

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access matching AGENTS.md Section 6 contracts."""
        if hasattr(self, key):
            return getattr(self, key)
        # Compatibility aliases for canonical AGENTS.md keys
        if key == "emotion_probabilities":
            return self.probabilities
        raise KeyError(f"Key '{key}' not found in EmotionOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary for pipeline integration."""
        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "emotion_probabilities": self.probabilities,
            "emotion_confidence": self.emotion_confidence,
            "emotion_uncertainty": self.emotion_uncertainty,
            "emotion_embeddings": self.emotion_embeddings,
            "context_embeddings": self.context_embeddings,
            "emotion_names": self.emotion_names,
        }


class EmotionClassifier(nn.Module):
    """Multi-label emotion classification head with parallel representation.

    Fulfills the core architectural requirements of AGENTS.md:
    1. Emotion and Context branches remain logically distinct.
    2. Multi-label emotion predictions are generated across 8 discrete classes.
    3. Individual emotion probabilities are strictly preserved.
    4. Prediction confidence and predictive uncertainty are estimated independently
       for each individual emotion, satisfying the mathematical relation:
       confidence_k = 1.0 - uncertainty_k.
    """

    def __init__(
        self,
        input_dim: int = 128,
        emotion_dim: int = 128,
        context_dim: int = 128,
        num_emotions: int = NUM_EMOTIONS,
        dropout: float = 0.1,
        temperature: float = 1.0,
        confidence_method: str = "margin",
        emotion_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize EmotionClassifier.

        Args:
            input_dim: Dimension of input features from fusion or temporal encoder.
            emotion_dim: Dimension of intermediate Emotion Branch representation.
            context_dim: Dimension of intermediate Context Branch representation.
            num_emotions: Number of multi-label target emotions (default 8).
            dropout: Dropout probability for regularization.
            temperature: Calibration temperature for scaling logits before sigmoid.
            confidence_method: Metric for confidence estimation; either ``"margin"``
                (|2p - 1|) or ``"entropy"`` (1 - normalized binary entropy).
            emotion_names: Sequence of emotion class names (defaults to EmoReact 8).
        """
        super().__init__()
        if input_dim <= 0 or emotion_dim <= 0 or context_dim <= 0 or num_emotions <= 0:
            raise ValueError("All dimension arguments must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive; got {temperature}.")
        if confidence_method not in ("margin", "entropy"):
            raise ValueError(
                f"confidence_method must be 'margin' or 'entropy'; got '{confidence_method}'."
            )

        self.input_dim = input_dim
        self.emotion_dim = emotion_dim
        self.context_dim = context_dim
        self.num_emotions = num_emotions
        self.temperature = float(temperature)
        self.confidence_method = confidence_method
        if emotion_names is not None:
            self.emotion_names = tuple(emotion_names)
        elif num_emotions == NUM_EMOTIONS:
            self.emotion_names = DEFAULT_EMOTION_NAMES
        else:
            self.emotion_names = tuple(f"Emotion_{i}" for i in range(num_emotions))

        if len(self.emotion_names) != num_emotions:
            raise ValueError(
                f"Length of emotion_names ({len(self.emotion_names)}) must equal num_emotions ({num_emotions})."
            )

        # 1. EMOTION BRANCH: Specialized representation for affective classification
        self.emotion_branch = nn.Sequential(
            nn.Linear(input_dim, emotion_dim),
            nn.LayerNorm(emotion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emotion_dim, emotion_dim),
            nn.LayerNorm(emotion_dim),
        )

        # 2. CONTEXT BRANCH: Logically distinct parallel context representation
        self.context_branch = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

        # 3. MULTI-LABEL EMOTION HEAD
        self.emotion_head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emotion_dim, num_emotions),
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
            # Batched vector [B, D] -> [B, 1, D]
            features = features.unsqueeze(1)
        elif features.ndim != 3:
            raise ValueError(
                f"Input features must have shape [B, T, D], [B, D], or [D]; got {tuple(features.shape)}."
            )

        return features, is_unbatched

    def _estimate_confidence_and_uncertainty(
        self, probabilities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute independent per-emotion confidence and predictive uncertainty.

        For each emotion k in 0..7:
        - Margin method:
            confidence_k = |2 * p_k - 1| in [0, 1]
            uncertainty_k = 1.0 - confidence_k
        - Entropy method:
            H(p_k) = -[p_k * log2(p_k) + (1 - p_k) * log2(1 - p_k)] in [0, 1]
            uncertainty_k = H(p_k)
            confidence_k = 1.0 - uncertainty_k
        """
        if self.confidence_method == "margin":
            # Margin distance from decision threshold (p=0.5):
            # p=0.5 -> confidence=0.0 (maximal uncertainty)
            # p=0.0 or 1.0 -> confidence=1.0 (certain prediction)
            confidence = torch.abs(2.0 * probabilities - 1.0)
            uncertainty = 1.0 - confidence
        elif self.confidence_method == "entropy":
            eps = 1e-7
            p_clamped = torch.clamp(probabilities, min=eps, max=1.0 - eps)
            # Normalized binary entropy (divided by ln(2) so max value is 1.0)
            binary_entropy = -(
                p_clamped * torch.log(p_clamped)
                + (1.0 - p_clamped) * torch.log(1.0 - p_clamped)
            ) / 0.6931471805599453
            uncertainty = torch.clamp(binary_entropy, 0.0, 1.0)
            confidence = 1.0 - uncertainty
        else:
            raise ValueError(f"Unknown confidence method: {self.confidence_method}")

        return confidence, uncertainty

    def forward(self, features: FeatureInput) -> EmotionOutput:
        """Forward pass for emotion classification and parallel representation.

        Args:
            features: Input tensor shaped ``[B, T, input_dim]``, ``[B, input_dim]``,
                or a dictionary containing ``"features"``.

        Returns:
            EmotionOutput containing multi-label logits, probabilities, per-emotion
            confidence, per-emotion uncertainty, emotion embeddings, and context embeddings.
        """
        tensor, is_unbatched = self._extract_tensor(features)
        batch_size, steps, dim = tensor.shape

        if dim != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}; got {dim}."
            )

        # 1. Parallel Representation: Emotion Branch and Context Branch
        emotion_emb = self.emotion_branch(tensor)
        context_emb = self.context_branch(tensor)

        # 2. Multi-label Emotion Head: Raw Logits [B, T, num_emotions]
        logits = self.emotion_head(emotion_emb)

        # 3. Probabilities via Sigmoid with optional temperature calibration
        scaled_logits = logits / self.temperature if self.temperature != 1.0 else logits
        probabilities = torch.sigmoid(scaled_logits)

        # 4. Independent Emotion-wise Confidence and Uncertainty
        confidence, uncertainty = self._estimate_confidence_and_uncertainty(probabilities)

        # Squeeze temporal dimension if input was [B, D] or [D]
        if steps == 1 and not is_unbatched:
            logits = logits.squeeze(1)
            probabilities = probabilities.squeeze(1)
            confidence = confidence.squeeze(1)
            uncertainty = uncertainty.squeeze(1)
            emotion_emb = emotion_emb.squeeze(1)
            context_emb = context_emb.squeeze(1)
        elif is_unbatched:
            logits = logits.squeeze(0).squeeze(0)
            probabilities = probabilities.squeeze(0).squeeze(0)
            confidence = confidence.squeeze(0).squeeze(0)
            uncertainty = uncertainty.squeeze(0).squeeze(0)
            emotion_emb = emotion_emb.squeeze(0).squeeze(0)
            context_emb = context_emb.squeeze(0).squeeze(0)

        return EmotionOutput(
            logits=logits,
            probabilities=probabilities,
            emotion_confidence=confidence,
            emotion_uncertainty=uncertainty,
            emotion_embeddings=emotion_emb,
            context_embeddings=context_emb,
            emotion_names=self.emotion_names,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EmotionClassifier":
        """Instantiate EmotionClassifier from configuration mapping.

        Looks for an ``emotion_classifier`` section or root-level keys.
        """
        cfg = config.get("emotion_classifier", config)
        return cls(
            input_dim=cfg.get("input_dim", 128),
            emotion_dim=cfg.get("emotion_dim", 128),
            context_dim=cfg.get("context_dim", 128),
            num_emotions=cfg.get("num_emotions", NUM_EMOTIONS),
            dropout=cfg.get("dropout", 0.1),
            temperature=cfg.get("temperature", 1.0),
            confidence_method=cfg.get("confidence_method", "margin"),
            emotion_names=cfg.get("emotion_names", DEFAULT_EMOTION_NAMES),
        )


class MultimodalEmotionPipeline(nn.Module):
    """Unified pipeline combining EmotionClassifier and AffectiveStateEstimator.

    Produces the canonical output dictionary specified by AGENTS.md Section 6:
        {
            "affective_state": tensor,
            "valence": tensor,
            "emotion_probabilities": tensor,
            "emotion_confidence": tensor,
            "emotion_uncertainty": tensor,
            ...
        }
    """

    def __init__(
        self,
        emotion_classifier: EmotionClassifier,
        affective_estimator: AffectiveStateEstimator,
    ) -> None:
        super().__init__()
        self.emotion_classifier = emotion_classifier
        self.affective_estimator = affective_estimator

    def forward(self, features: FeatureInput) -> Dict[str, Any]:
        """Compute parallel emotion and affective state representations.

        Args:
            features: Input tensor ``[B, T, D]``, ``[B, D]``, or dictionary with ``"features"``.

        Returns:
            Dictionary matching AGENTS.md Section 6 canonical contract.
        """
        emotion_out = self.emotion_classifier(features)
        affective_out = self.affective_estimator(features)

        return {
            "affective_state": affective_out.affective_state,
            "valence": affective_out.valence,
            "raw_valence": affective_out.raw_valence,
            "valence_confidence": affective_out.valence_confidence,
            "emotion_probabilities": emotion_out.probabilities,
            "emotion_logits": emotion_out.logits,
            "emotion_confidence": emotion_out.emotion_confidence,
            "emotion_uncertainty": emotion_out.emotion_uncertainty,
            "emotion_embeddings": emotion_out.emotion_embeddings,
            "context_embeddings": emotion_out.context_embeddings,
            "emotion_names": emotion_out.emotion_names,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MultimodalEmotionPipeline":
        """Instantiate pipeline from a combined configuration mapping."""
        classifier = EmotionClassifier.from_config(config)
        estimator = AffectiveStateEstimator.from_config(config)
        return cls(emotion_classifier=classifier, affective_estimator=estimator)
