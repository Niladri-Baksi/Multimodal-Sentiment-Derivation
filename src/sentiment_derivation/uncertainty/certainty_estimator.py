"""Certainty and uncertainty estimation module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 & Phase 6.

This module provides:
1. UncertaintyOutput: Standardized container storing emotion-wise confidence and uncertainty
   tensors, preserving per-emotion categories without premature scalar collapse.
2. UncertaintyEstimator: Module computing per-label probability ambiguity scores and optional
   stochastic variance/entropy proxies from Monte Carlo (MC) Dropout.
"""

from __future__ import annotations

import math
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
        epistemic_uncertainty: Optional normalized stochastic-pass variance proxy.
        aleatoric_uncertainty: Optional mean per-pass binary entropy proxy.
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
    """Compute emotion-wise probability ambiguity scores from predicted probabilities.

    Supports:
    1. **Margin** (Default in Phase 5 baseline):
       - Confidence: c_k = 2 * |p_k - 0.5| in [0, 1]
       - Uncertainty: u_k = 1.0 - c_k in [0, 1]
    2. **Binary Entropy** (Normalized Shannon Binary Information Entropy):
       - For independent binary emotion predictions:
         H_k(p) = - [ p * log2(p + eps) + (1 - p) * log2(1 - p + eps) ] in [0, 1]
       - Uncertainty: u_k = H_k(p)
       - Confidence: c_k = 1.0 - u_k
    3. **MC-Dropout summary** (stochastic proxies):
       - Normalized variance proxy: min(Var_S(p^(s)) / 0.25, 1)
       - Mean entropy proxy: E_S[H(p^(s))]
       - Reported total is their equal-weight average; it is a heuristic, not a calibrated
         or exact epistemic/aleatoric decomposition.
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
        if not isinstance(num_emotions, int) or isinstance(num_emotions, bool) or num_emotions <= 0:
            raise ValueError("num_emotions must be a positive integer.")
        if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.1:
            raise ValueError("eps must be in range (0.0, 0.1).")
        if not isinstance(mc_samples, int) or isinstance(mc_samples, bool):
            raise ValueError("mc_samples must be an integer.")
        if mc_samples <= 0 or (method == "mc_dropout" and mc_samples < 2):
            raise ValueError("mc_samples must be positive and at least 2 for mc_dropout.")

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
        calculation_probs = (
            probs.float() if probs.dtype in (torch.float16, torch.bfloat16) else probs
        )
        clamped_p = torch.clamp(calculation_probs, self.eps, 1.0 - self.eps)
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
        probabilities = self._validate_probabilities(probabilities, "probabilities")
        if self.method == "mc_dropout" and model_fn is None:
            raise ValueError("model_fn is required when method='mc_dropout'.")

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
            # Perform MC-Dropout sampling. The callback is responsible for
            # enabling stochastic model behavior (for example, dropout).
            assert model_fn is not None
            samples = []
            for _ in range(self.mc_samples):
                sample = self._validate_probabilities(model_fn(), "model_fn output")
                if sample.shape != probabilities.shape:
                    raise ValueError("model_fn outputs must match probabilities shape.")
                if sample.device != probabilities.device or sample.dtype != probabilities.dtype:
                    raise ValueError("model_fn outputs must match probabilities device and dtype.")
                samples.append(sample)
            stacked = torch.stack(samples, dim=0)

            # Epistemic uncertainty proxy: normalized variance across stochastic passes.
            epistemic = torch.var(stacked, dim=0, unbiased=False)
            epistemic_norm = torch.clamp(epistemic / 0.25, 0.0, 1.0)

            # Aleatoric uncertainty proxy: mean binary entropy across passes.
            sample_entropies = torch.stack([self._compute_entropy(s) for s in samples], dim=0)
            aleatoric = torch.mean(sample_entropies, dim=0)

            # Heuristic equal-weight average, not a calibrated total-uncertainty identity.
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

    def _validate_probabilities(self, probabilities: torch.Tensor, name: str) -> torch.Tensor:
        """Validate a batched or unbatched multi-label probability tensor."""
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if probabilities.ndim not in (1, 2, 3):
            raise ValueError(f"{name} must have rank 1, 2, or 3; got {probabilities.ndim}.")
        if probabilities.shape[-1] != self.num_emotions:
            raise ValueError(
                f"Expected last dimension {self.num_emotions}; got {probabilities.shape[-1]}."
            )
        if any(size == 0 for size in probabilities.shape):
            raise ValueError(f"{name} dimensions must be non-empty.")
        if not probabilities.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype.")
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"{name} contains NaN or infinite values.")
        if bool(torch.any((probabilities < 0.0) | (probabilities > 1.0))):
            raise ValueError(f"{name} values must lie in [0, 1].")
        return probabilities

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
