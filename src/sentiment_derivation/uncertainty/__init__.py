"""Uncertainty, calibration, and gating package for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 & Stage 7 (Phase 6).

Implements:
1. MultiLabelTemperatureCalibrator, compute_multilabel_ece, compute_brier_score.
2. UncertaintyEstimator, UncertaintyOutput.
3. UncertaintyAwareGate, UncertaintyGateOutput.
"""

from sentiment_derivation.uncertainty.calibration import (
    CalibrationMetrics,
    MultiLabelTemperatureCalibrator,
    compute_binary_ece,
    compute_brier_score,
    compute_multilabel_ece,
)
from sentiment_derivation.uncertainty.certainty_estimator import (
    UncertaintyEstimator,
    UncertaintyOutput,
)
from sentiment_derivation.uncertainty.uncertainty_gate import (
    UncertaintyAwareGate,
    UncertaintyGateOutput,
)

__all__ = [
    "CalibrationMetrics",
    "MultiLabelTemperatureCalibrator",
    "compute_binary_ece",
    "compute_brier_score",
    "compute_multilabel_ece",
    "UncertaintyEstimator",
    "UncertaintyOutput",
    "UncertaintyAwareGate",
    "UncertaintyGateOutput",
]
