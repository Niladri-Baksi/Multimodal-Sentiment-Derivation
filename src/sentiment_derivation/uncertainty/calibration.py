"""Multi-label temperature calibration module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 & Phase 6 (Certainty & Uncertainty Mechanism).

This module provides:
1. MultiLabelTemperatureCalibrator: PyTorch module that learns per-emotion temperature
   scaling parameters T_k > 0 strictly on the validation set using binary cross-entropy (NLL).
2. compute_multilabel_ece: Function calculating Expected Calibration Error (ECE) per emotion
   and overall macro-ECE.
3. compute_reliability_curve: Computes binned accuracies and confidences for plotting
   reliability diagrams (used in Explainable AI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationMetrics:
    """Container for multi-label calibration evaluation metrics.

    Attributes:
        per_emotion_ece: Dictionary mapping emotion name (or index) to ECE in [0.0, 1.0].
        macro_ece: Average ECE across all emotions.
        per_emotion_brier: Brier score per emotion.
        macro_brier: Average Brier score across all emotions.
        temperatures: Dictionary of learned optimal temperatures per emotion.
    """

    per_emotion_ece: Dict[str, float]
    macro_ece: float
    per_emotion_brier: Dict[str, float]
    macro_brier: float
    temperatures: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "per_emotion_ece": self.per_emotion_ece,
            "macro_ece": self.macro_ece,
            "per_emotion_brier": self.per_emotion_brier,
            "macro_brier": self.macro_brier,
            "temperatures": self.temperatures,
        }


def compute_binary_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) for a single binary classification task.

    Bins predictions into `n_bins` equal-width intervals in [0.0, 1.0], computing:
        ECE = sum_b ( |acc(b) - conf(b)| * ( |b| / N ) )

    Args:
        probs: 1D array of predicted probabilities in [0.0, 1.0].
        labels: 1D binary array of ground truth labels in {0, 1}.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts).
    """
    probs = np.asarray(probs, dtype=np.float64).flatten()
    labels = np.asarray(labels, dtype=np.float64).flatten()

    if len(probs) != len(labels):
        raise ValueError(f"probs ({len(probs)}) and labels ({len(labels)}) must have equal length.")
    if len(probs) == 0:
        return 0.0, np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    total_samples = len(probs)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        # Include upper boundary in the last bin
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)

        count = np.sum(mask)
        bin_counts[i] = count

        if count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            bin_accs[i] = bin_acc
            bin_confs[i] = bin_conf
            ece += (count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece), bin_accs, bin_confs, bin_counts


def compute_multilabel_ece(
    probs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    emotion_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute per-emotion ECE and overall macro-ECE across all multi-label targets.

    Args:
        probs: Array or Tensor shaped [N, num_emotions] or [N, T, num_emotions].
        labels: Array or Tensor shaped [N, num_emotions] or [N, T, num_emotions].
        emotion_names: Optional list of emotion class names.
        n_bins: Number of calibration bins (default 10).

    Returns:
        Dictionary containing:
        - "macro_ece": average ECE across all emotions.
        - "per_emotion_ece": dict of emotion_name -> ECE.
        - "curves": dict of emotion_name -> (bin_accs, bin_confs, bin_counts).
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Flatten temporal dimension if 3D: [B, T, K] -> [B * T, K]
    if probs.ndim == 3:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])
    elif probs.ndim == 1:
        probs = probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)

    num_emotions = probs.shape[1]
    if emotion_names is None:
        emotion_names = [f"emotion_{i}" for i in range(num_emotions)]
    elif len(emotion_names) != num_emotions:
        raise ValueError(
            f"Length of emotion_names ({len(emotion_names)}) does not match num_emotions ({num_emotions})."
        )

    per_emotion_ece: Dict[str, float] = {}
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for k in range(num_emotions):
        name = emotion_names[k]
        ece, accs, confs, counts = compute_binary_ece(probs[:, k], labels[:, k], n_bins=n_bins)
        per_emotion_ece[name] = ece
        curves[name] = {
            "bin_accuracies": accs,
            "bin_confidences": confs,
            "bin_counts": counts,
        }

    macro_ece = float(np.mean(list(per_emotion_ece.values())))

    return {
        "macro_ece": macro_ece,
        "per_emotion_ece": per_emotion_ece,
        "curves": curves,
    }


def compute_brier_score(
    probs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    emotion_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute multi-label Brier score: mean squared error between probabilities and labels.

    Args:
        probs: Predictions in [0.0, 1.0], shaped [N, K] or [N, T, K].
        labels: Binary targets in {0, 1}, shaped [N, K] or [N, T, K].
        emotion_names: Optional list of emotion names.

    Returns:
        Dictionary containing "macro_brier" and "per_emotion_brier".
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if probs.ndim == 3:
        probs = probs.reshape(-1, probs.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

    num_emotions = probs.shape[1]
    if emotion_names is None:
        emotion_names = [f"emotion_{i}" for i in range(num_emotions)]

    per_emotion_brier: Dict[str, float] = {}
    for k in range(num_emotions):
        name = emotion_names[k]
        mse = float(np.mean((probs[:, k] - labels[:, k]) ** 2))
        per_emotion_brier[name] = mse

    macro_brier = float(np.mean(list(per_emotion_brier.values())))
    return {
        "macro_brier": macro_brier,
        "per_emotion_brier": per_emotion_brier,
    }


class MultiLabelTemperatureCalibrator(nn.Module):
    """Post-hoc multi-label calibration via learned temperature scaling parameters.

    In multi-label emotion derivation (8 independent binary sigmoids), a single global
    temperature is suboptimal because some emotions (e.g. Happiness) may be overconfident
    while rare ones (e.g. Fear) may be underconfident.

    This module maintains a learnable vector of positive temperatures T_k > 0:
        p_calibrated_k = sigmoid(z_k / T_k)

    Temperatures are optimized strictly on the **Validation set** (never on the Test set)
    by minimizing Binary Cross-Entropy Loss via L-BFGS or Adam.
    """

    def __init__(
        self,
        num_emotions: int = 8,
        emotion_names: Optional[Sequence[str]] = None,
        initial_temperature: float = 1.0,
    ) -> None:
        """Initialize MultiLabelTemperatureCalibrator.

        Args:
            num_emotions: Number of multi-label emotion categories (default 8).
            emotion_names: Optional names for the emotion categories.
            initial_temperature: Initial value for each temperature T_k (default 1.0).
        """
        super().__init__()
        if num_emotions <= 0:
            raise ValueError("num_emotions must be a positive integer.")
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be strictly positive.")

        self.num_emotions = num_emotions
        if emotion_names is None:
            self.emotion_names = tuple(f"emotion_{i}" for i in range(num_emotions))
        else:
            if len(emotion_names) != num_emotions:
                raise ValueError(
                    f"emotion_names length ({len(emotion_names)}) must match num_emotions ({num_emotions})."
                )
            self.emotion_names = tuple(emotion_names)

        # Log-temperatures parameter to guarantee T_k = exp(log_T_k) > 0
        init_log_t = float(np.log(initial_temperature))
        self.log_temperatures = nn.Parameter(
            torch.full((num_emotions,), init_log_t, dtype=torch.float32)
        )

    @property
    def temperatures(self) -> torch.Tensor:
        """Return positive temperature values T_k = exp(log_T_k)."""
        return torch.exp(self.log_temperatures)

    def get_temperatures_dict(self) -> Dict[str, float]:
        """Return learned temperatures as a human-readable mapping."""
        temps = self.temperatures.detach().cpu().numpy()
        return {name: float(temps[i]) for i, name in enumerate(self.emotion_names)}

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale unnormalized logits by learned temperatures: z_cal = z / T.

        Args:
            logits: Pre-activation logits shaped [..., num_emotions].

        Returns:
            Calibrated logits shaped identically to input.
        """
        if logits.shape[-1] != self.num_emotions:
            raise ValueError(
                f"Expected last dimension {self.num_emotions}; got {logits.shape[-1]}."
            )
        # Broadcast across batch and time dimensions: [..., num_emotions] / [num_emotions]
        temps = self.temperatures.to(logits.device)
        return logits / temps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Calibrate logits and return calibrated probabilities in [0.0, 1.0].

        Args:
            logits: Pre-activation logits shaped [..., num_emotions].

        Returns:
            Calibrated sigmoid probabilities shaped identically to input.
        """
        calibrated_logits = self.calibrate_logits(logits)
        return torch.sigmoid(calibrated_logits)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> CalibrationMetrics:
        """Fit optimal temperatures on the validation set using L-BFGS optimizer.

        Minimizes Binary Cross-Entropy (NLL) with respect to log_temperatures.

        Args:
            val_logits: Uncalibrated validation logits shaped [N, num_emotions] or [N, T, num_emotions].
            val_labels: Ground truth binary targets in {0, 1} shaped identically to val_logits.
            lr: Learning rate for L-BFGS optimizer (default 0.01).
            max_iter: Maximum optimization iterations (default 100).
            verbose: If True, prints optimization progress.

        Returns:
            CalibrationMetrics summary containing ECE and Brier scores.
        """
        self.train()
        device = self.log_temperatures.device

        # Flatten any temporal dimension: [B, T, K] -> [B * T, K]
        v_logits = val_logits.detach().to(device=device, dtype=torch.float32)
        v_labels = val_labels.detach().to(device=device, dtype=torch.float32)

        if v_logits.ndim > 2:
            v_logits = v_logits.reshape(-1, self.num_emotions)
            v_labels = v_labels.reshape(-1, self.num_emotions)

        bce_criterion = nn.BCEWithLogitsLoss()

        # Compute uncalibrated metrics
        with torch.no_grad():
            uncal_probs = torch.sigmoid(v_logits).cpu().numpy()
            labels_np = v_labels.cpu().numpy()
            uncal_ece = compute_multilabel_ece(uncal_probs, labels_np, self.emotion_names)["macro_ece"]

        # Optimize using L-BFGS
        optimizer = torch.optim.LBFGS([self.log_temperatures], lr=lr, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = v_logits / torch.exp(self.log_temperatures)
            loss = bce_criterion(scaled_logits, v_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        # Compute post-calibration metrics
        with torch.no_grad():
            cal_probs = self.forward(v_logits).cpu().numpy()
            cal_ece_dict = compute_multilabel_ece(cal_probs, labels_np, self.emotion_names)
            cal_brier_dict = compute_brier_score(cal_probs, labels_np, self.emotion_names)

        metrics = CalibrationMetrics(
            per_emotion_ece=cal_ece_dict["per_emotion_ece"],
            macro_ece=cal_ece_dict["macro_ece"],
            per_emotion_brier=cal_brier_dict["per_emotion_brier"],
            macro_brier=cal_brier_dict["macro_brier"],
            temperatures=self.get_temperatures_dict(),
        )

        if verbose:
            print(f"[Temperature Calibration] Macro-ECE: {uncal_ece:.4f} -> {metrics.macro_ece:.4f}")
            print(f"[Temperature Calibration] Optimal Temperatures: {metrics.temperatures}")

        return metrics

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MultiLabelTemperatureCalibrator":
        """Instantiate calibrator from configuration dictionary."""
        cfg = config.get("uncertainty", config)
        return cls(
            num_emotions=cfg.get("num_emotions", 8),
            emotion_names=cfg.get("emotion_names", None),
            initial_temperature=cfg.get("initial_temperature", 1.0),
        )
