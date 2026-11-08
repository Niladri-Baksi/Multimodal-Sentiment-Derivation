"""Certainty and uncertainty estimation module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 & Phase 6.

This module provides:
1. UncertaintyOutput: Standardized container storing emotion-wise confidence and uncertainty
   tensors, preserving per-emotion categories without premature scalar collapse.
2. UncertaintyEstimator: Module computing aleatoric uncertainty (Margin, Shannon Binary Entropy)
   and optional epistemic uncertainty via Monte Carlo (MC) Dropout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UncertaintyOutput:
    """Standardized output container for emotion certainty and uncertainty estimation.

    Per AGENTS.md Section 6 and Architectural Constraint 4:
    - Emotion-wise confidence must be maintained separately for each category.
    - Do not replace per-emotion confidence with a single overall scalar.

    Attributes:
        confidence: Per-emotion certainty shaped ``[B, T, K]`` or ``[B, K]``, normalized to [0.0, 1.0].
            Higher values indicate higher certainty (extreme probability near 0 or 1).
        uncertainty: Per-emotion uncertainty shaped like ``confidence``, normalized to [0.0, 1.0].
            Higher values indicate higher ambiguity (probability near 0.5 decision boundary).
        method: Estimation method used ('margin', 'entropy', or 'mc_dropout').
        epistemic_uncertainty: Optional model parameter uncertainty (from MC-Dropout variance).
        aleatoric_uncertainty: Optional data noise uncertainty (from average entropy).
        emotion_names: Optional tuple of emotion class names.
    """

    confidence: torch.Tensor
    uncertainty: torch.Tensor
    method: str = "entropy"
    epistemic_uncertainty: Optional[torch.Tensor] = None
    aleatoric_uncertainty: Optional[torch.Tensor] = None
    emotion_names: Optional[Tuple[str, ...]] = None

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in UncertaintyOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary matching AGENTS.md contracts."""
        d: Dict[str, Any] = {
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "method": self.method,
            "emotion_names": self.emotion_names,
        }
        if self.epistemic_uncertainty is not None:
            d["epistemic_uncertainty"] = self.epistemic_uncertainty
        if self.aleatoric_uncertainty is not None:
            d["aleatoric_uncertainty"] = self.aleatoric_uncertainty
        return d


class UncertaintyEstimator(nn.Module):
    """Estimate emotion-wise confidence and uncertainty from predicted probabilities.

    Supports three scientifically established formulations:
    1. **Margin** (Default in Phase 5 baseline):
       - Confidence: c_k = 2 * |p_k - 0.5| in [0, 1]
       - Uncertainty: u_k = 1.0 - c_k in [0, 1]
    2. **Binary Entropy** (Normalized Shannon Binary Information Entropy):
       - For independent binary emotion predictions:
         H_k(p) = - [ p * log2(p + eps) + (1 - p) * log2(1 - p + eps) ] in [0, 1]
       - Uncertainty: u_k = H_k(p)
       - Confidence: c_k = 1.0 - u_k
    3. **MC-Dropout** (Epistemic and Aleatoric decomposition):
       - Epistemic variance: Var_T(p^(t))
       - Aleatoric entropy: E_T[H(p^(t))]
    """

    SUPPORTED_METHODS = ("entropy", "margin", "mc_dropout")

    def __init__(
        self,
        method: str = "entropy",
        num_emotions: int = 8,
        emotion_names: Optional[Sequence[str]] = None,
        eps: float = 1e-7,
        mc_samples: int = 10,
    ) -> None:
        """Initialize UncertaintyEstimator.

        Args:
            method: Method for uncertainty calculation ('entropy', 'margin', or 'mc_dropout').
            num_emotions: Number of multi-label emotion targets (default 8).
            emotion_names: Optional sequence of category names.
            eps: Numerical stability constant for log computations.
            mc_samples: Number of forward passes when using MC-Dropout.
        """
        super().__init__()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods: {self.SUPPORTED_METHODS}"
            )
        if num_emotions <= 0:
            raise ValueError("num_emotions must be a positive integer.")
        if eps <= 0.0 or eps >= 0.1:
            raise ValueError("eps must be in range (0.0, 0.1).")

        self.method = method
        self.num_emotions = num_emotions
        self.eps = eps
        self.mc_samples = mc_samples

        if emotion_names is None:
            self.emotion_names = tuple(f"emotion_{i}" for i in range(num_emotions))
        else:
            if len(emotion_names) != num_emotions:
                raise ValueError(
                    f"emotion_names length ({len(emotion_names)}) must match num_emotions ({num_emotions})."
                )
            self.emotion_names = tuple(emotion_names)

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute normalized binary Shannon entropy: -[p log2(p) + (1-p) log2(1-p)].

        When p = 0.5, entropy = 1.0 (maximal uncertainty).
        When p = 0.0 or 1.0, entropy = 0.0 (minimal uncertainty).
        """
        clamped_p = torch.clamp(probs, self.eps, 1.0 - self.eps)
        entropy = -(
            clamped_p * torch.log2(clamped_p) + (1.0 - clamped_p) * torch.log2(1.0 - clamped_p)
        )
        return torch.clamp(entropy, 0.0, 1.0)

    def _compute_margin(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute margin confidence: c = 2 * |p - 0.5| and uncertainty: u = 1 - c."""
        confidence = torch.clamp(2.0 * torch.abs(probs - 0.5), 0.0, 1.0)
        uncertainty = torch.clamp(1.0 - confidence, 0.0, 1.0)
        return confidence, uncertainty

    def forward(
        self,
        probabilities: torch.Tensor,
        model_fn: Optional[Callable[[], torch.Tensor]] = None,
    ) -> UncertaintyOutput:
        """Compute per-emotion confidence and uncertainty tensors.

        Args:
            probabilities: Emotion probabilities shaped ``[B, T, num_emotions]``,
                ``[B, num_emotions]``, or ``[num_emotions]``.
            model_fn: Optional callable performing a stochastic forward pass (for MC-Dropout).

        Returns:
            UncertaintyOutput container with confidence and uncertainty tensors.
        """
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError(
                f"probabilities must be a torch.Tensor; got {type(probabilities).__name__}."
            )
        if probabilities.shape[-1] != self.num_emotions:
            raise ValueError(
                f"Expected last dimension {self.num_emotions}; got {probabilities.shape[-1]}."
            )

        if self.method == "margin":
            conf, uncert = self._compute_margin(probabilities)
            return UncertaintyOutput(
                confidence=conf,
                uncertainty=uncert,
                method="margin",
                emotion_names=self.emotion_names,
            )

        elif self.method == "entropy":
            uncert = self._compute_entropy(probabilities)
            conf = torch.clamp(1.0 - uncert, 0.0, 1.0)
            return UncertaintyOutput(
                confidence=conf,
                uncertainty=uncert,
                method="entropy",
                emotion_names=self.emotion_names,
            )

        elif self.method == "mc_dropout":
            if model_fn is None:
                # Fallback to single-pass entropy if model_fn is not provided
                uncert = self._compute_entropy(probabilities)
                conf = torch.clamp(1.0 - uncert, 0.0, 1.0)
                return UncertaintyOutput(
                    confidence=conf,
                    uncertainty=uncert,
                    method="entropy",
                    emotion_names=self.emotion_names,
                )

            # Perform MC-Dropout sampling
            samples = []
            for _ in range(self.mc_samples):
                samples.append(model_fn())
            # Stack into shape: [S, B, T, K] or [S, B, K]
            stacked = torch.stack(samples, dim=0)

            # Epistemic uncertainty: predictive variance across stochastic passes
            epistemic = torch.var(stacked, dim=0, unbiased=False)
            # Normalize variance: maximum variance of Bernoulli is 0.25 (at p=0.5)
            epistemic_norm = torch.clamp(epistemic / 0.25, 0.0, 1.0)

            # Aleatoric uncertainty: expected binary entropy across stochastic passes
            sample_entropies = torch.stack([self._compute_entropy(s) for s in samples], dim=0)
            aleatoric = torch.mean(sample_entropies, dim=0)

            # Total uncertainty: combination of aleatoric and epistemic
            total_uncertainty = torch.clamp(0.5 * (epistemic_norm + aleatoric), 0.0, 1.0)
            total_confidence = torch.clamp(1.0 - total_uncertainty, 0.0, 1.0)

            return UncertaintyOutput(
                confidence=total_confidence,
                uncertainty=total_uncertainty,
                method="mc_dropout",
                epistemic_uncertainty=epistemic_norm,
                aleatoric_uncertainty=aleatoric,
                emotion_names=self.emotion_names,
            )

        raise RuntimeError(f"Unexpected method state: {self.method}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "UncertaintyEstimator":
        """Instantiate estimator from configuration dictionary."""
        cfg = config.get("uncertainty", config)
        return cls(
            method=cfg.get("method", "entropy"),
            num_emotions=cfg.get("num_emotions", 8),
            emotion_names=cfg.get("emotion_names", None),
            eps=cfg.get("eps", 1e-7),
            mc_samples=cfg.get("mc_samples", 10),
        )
