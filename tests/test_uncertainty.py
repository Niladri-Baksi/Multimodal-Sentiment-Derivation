"""Unit tests for Phase 6 Uncertainty, Calibration, and Gating modules.

Tests:
1. MultiLabelTemperatureCalibrator and calibration metrics (ECE, Brier score).
2. UncertaintyEstimator (Margin, Shannon Entropy, MC-Dropout).
3. UncertaintyAwareGate (Forward pass, gradient flow, residual gating, ablation toggle).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sentiment_derivation.uncertainty import (
    CalibrationMetrics,
    MultiLabelTemperatureCalibrator,
    UncertaintyAwareGate,
    UncertaintyEstimator,
    UncertaintyGateOutput,
    UncertaintyOutput,
    compute_binary_ece,
    compute_brier_score,
    compute_multilabel_ece,
)


# ============================================================================
# 1. Calibration Tests
# ============================================================================

def test_compute_binary_ece():
    """Verify binary ECE computation on known synthetic cases."""
    # Perfect calibration: predicted 0.8 with 80% accuracy
    probs = np.array([0.8] * 10)
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  # mean = 0.8
    ece, accs, confs, counts = compute_binary_ece(probs, labels, n_bins=10)
    assert np.isclose(ece, 0.0, atol=1e-5)

    # Extreme miscalibration: predicted 0.9 with 0% accuracy
    probs_bad = np.array([0.95] * 10)
    labels_bad = np.array([0] * 10)
    ece_bad, _, _, _ = compute_binary_ece(probs_bad, labels_bad, n_bins=10)
    assert np.isclose(ece_bad, 0.95, atol=1e-5)


def test_compute_multilabel_ece_and_brier():
    """Verify multi-label ECE and Brier score across 8 emotions."""
    num_samples = 100
    num_emotions = 8
    probs = np.random.uniform(0.1, 0.9, size=(num_samples, num_emotions))
    labels = (probs > 0.5).astype(np.float32)

    ece_res = compute_multilabel_ece(probs, labels)
    assert "macro_ece" in ece_res
    assert "per_emotion_ece" in ece_res
    assert 0.0 <= ece_res["macro_ece"] <= 1.0
    assert len(ece_res["per_emotion_ece"]) == num_emotions

    brier_res = compute_brier_score(probs, labels)
    assert "macro_brier" in brier_res
    assert 0.0 <= brier_res["macro_brier"] <= 1.0
    assert len(brier_res["per_emotion_brier"]) == num_emotions


def test_temperature_calibrator_optimization():
    """Verify that MultiLabelTemperatureCalibrator learns T_k > 0 and scales logits."""
    calibrator = MultiLabelTemperatureCalibrator(num_emotions=8)
    assert torch.all(calibrator.temperatures > 0.0)
    assert calibrator.temperatures.shape == (8,)

    # Synthetic overconfident logits
    torch.manual_seed(42)
    val_logits = torch.randn(200, 8) * 4.0
    val_labels = torch.randint(0, 2, (200, 8)).float()

    metrics = calibrator.fit(val_logits, val_labels, max_iter=25)
    assert isinstance(metrics, CalibrationMetrics)
    assert torch.all(calibrator.temperatures > 0.0)

    # Check forward pass shapes: 3D [B, T, K]
    test_logits = torch.randn(4, 16, 8)
    cal_probs = calibrator(test_logits)
    assert cal_probs.shape == (4, 16, 8)
    assert torch.all(cal_probs >= 0.0) and torch.all(cal_probs <= 1.0)


# ============================================================================
# 2. Uncertainty Estimator Tests
# ============================================================================

def test_uncertainty_estimator_margin():
    """Verify Margin uncertainty computation: c = 2*|p - 0.5|, u = 1 - c."""
    estimator = UncertaintyEstimator(method="margin", num_emotions=8)

    probs = torch.tensor([[0.5, 1.0, 0.0, 0.25, 0.75, 0.5, 0.1, 0.9]])
    out = estimator(probs)

    assert isinstance(out, UncertaintyOutput)
    assert out.confidence.shape == probs.shape
    assert out.uncertainty.shape == probs.shape

    # At p = 0.5: confidence = 0.0, uncertainty = 1.0
    assert torch.isclose(out.confidence[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(out.uncertainty[0, 0], torch.tensor(1.0), atol=1e-5)

    # At p = 1.0 or 0.0: confidence = 1.0, uncertainty = 0.0
    assert torch.isclose(out.confidence[0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out.uncertainty[0, 1], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(out.confidence[0, 2], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(out.uncertainty[0, 2], torch.tensor(0.0), atol=1e-5)


def test_uncertainty_estimator_entropy():
    """Verify Shannon Binary Entropy uncertainty computation."""
    estimator = UncertaintyEstimator(method="entropy", num_emotions=8)

    probs = torch.tensor([[0.5, 0.999, 0.001, 0.5, 0.2, 0.8, 0.5, 0.5]])
    out = estimator(probs)

    assert isinstance(out, UncertaintyOutput)
    # At p = 0.5: entropy is maximum (1.0), confidence is 0.0
    assert torch.isclose(out.uncertainty[0, 0], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(out.confidence[0, 0], torch.tensor(0.0), atol=1e-3)

    # At p near 1 or 0: entropy is near 0.0, confidence is near 1.0
    assert out.uncertainty[0, 1] < 0.05
    assert out.confidence[0, 1] > 0.95


def test_uncertainty_estimator_temporal_3d():
    """Verify tensor shapes are preserved for sequences [B, T=16, 8]."""
    estimator = UncertaintyEstimator(method="entropy", num_emotions=8)
    probs_3d = torch.rand(4, 16, 8)

    out = estimator(probs_3d)
    assert out.confidence.shape == (4, 16, 8)
    assert out.uncertainty.shape == (4, 16, 8)
    assert torch.all(out.confidence >= 0.0) and torch.all(out.confidence <= 1.0)
    assert torch.all(out.uncertainty >= 0.0) and torch.all(out.uncertainty <= 1.0)

    # Dictionary access
    d = out.to_dict()
    assert "confidence" in d and "uncertainty" in d


# ============================================================================
# 3. Uncertainty-Aware Gate Tests
# ============================================================================

def test_uncertainty_gate_forward_shapes():
    """Verify UncertaintyAwareGate forward pass shapes on 3D sequence and 2D pooled tensors."""
    gate = UncertaintyAwareGate(
        feature_dim=128,
        num_emotions=8,
        valence_dim=1,
        hidden_dim=64,
        residual_prior=True,
    )

    # 3D Sequence input: [B=2, T=16, 128]
    features = torch.randn(2, 16, 128)
    probs = torch.rand(2, 16, 8)
    uncertainty = torch.rand(2, 16, 8)
    val_conf = torch.rand(2, 16, 1)

    out = gate(features, probs, uncertainty, val_conf)
    assert isinstance(out, UncertaintyGateOutput)
    assert out.gated_features.shape == (2, 16, 128)
    assert out.gate_weights.shape == (2, 16, 128)
    assert out.temporal_saliency.shape == (2, 16, 1)
    assert torch.all(out.gate_weights >= 0.0) and torch.all(out.gate_weights <= 1.0)

    # 2D Batched vector: [B=3, 128]
    f_2d = torch.randn(3, 128)
    p_2d = torch.rand(3, 8)
    u_2d = torch.rand(3, 8)
    v_2d = torch.rand(3, 1)

    out_2d = gate(f_2d, p_2d, u_2d, v_2d)
    assert out_2d.gated_features.shape == (3, 128)
    assert out_2d.gate_weights.shape == (3, 128)
    assert out_2d.temporal_saliency.shape == (3, 1)


def test_uncertainty_gate_gradient_flow():
    """Verify gradients propagate back through features and gate weights."""
    gate = UncertaintyAwareGate(feature_dim=64, num_emotions=8, valence_dim=1, hidden_dim=32)

    features = torch.randn(2, 16, 64, requires_grad=True)
    probs = torch.rand(2, 16, 8, requires_grad=True)
    uncertainty = torch.rand(2, 16, 8, requires_grad=True)
    val_conf = torch.rand(2, 16, 1, requires_grad=True)

    out = gate(features, probs, uncertainty, val_conf)
    loss = out.gated_features.sum()
    loss.backward()

    assert features.grad is not None and torch.any(features.grad != 0.0)
    assert probs.grad is not None and torch.any(probs.grad != 0.0)
    assert uncertainty.grad is not None and torch.any(uncertainty.grad != 0.0)
    assert val_conf.grad is not None and torch.any(val_conf.grad != 0.0)


def test_uncertainty_gate_ablation_bypass():
    """Verify enabled=False completely bypasses gating for ablation studies."""
    gate = UncertaintyAwareGate(feature_dim=128, enabled=False)

    features = torch.randn(2, 16, 128)
    probs = torch.rand(2, 16, 8)
    uncertainty = torch.rand(2, 16, 8)

    out = gate(features, probs, uncertainty)
    assert out.enabled is False
    # Features must match input identically
    assert torch.allclose(out.gated_features, features)
    # Gate weights must be identity (ones)
    assert torch.all(out.gate_weights == 1.0)


@pytest.mark.parametrize(
    "bad,exception,match",
    [
        (torch.empty(0, 8), ValueError, "non-empty"),
        (torch.ones(2, 7), ValueError, "last dimension"),
        (torch.ones(2, 8, dtype=torch.long), TypeError, "floating-point"),
        (torch.full((2, 8), float("nan")), ValueError, "finite"),
        (torch.full((2, 8), 1.1), ValueError, "\\[0, 1\\]"),
        (torch.empty(1, 2, 3, 8, 1), ValueError, "rank"),
    ],
)
def test_uncertainty_estimator_rejects_invalid_probability_contract(bad, exception, match):
    with pytest.raises(exception, match=match):
        UncertaintyEstimator()(bad)


def test_uncertainty_estimator_requires_callback_for_mc_dropout():
    estimator = UncertaintyEstimator(method="mc_dropout", mc_samples=3)
    with pytest.raises(ValueError, match="model_fn is required"):
        estimator(torch.full((1, 8), 0.5))


def test_mc_dropout_uncertainty_formula_and_sample_contract():
    estimator = UncertaintyEstimator(method="mc_dropout", num_emotions=2, mc_samples=2)
    samples = iter((torch.tensor([[0.25, 0.25]]), torch.tensor([[0.75, 0.75]])))
    out = estimator(torch.full((1, 2), 0.5), model_fn=lambda: next(samples))
    binary_entropy = -(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75))
    expected_epistemic = torch.tensor([[0.25, 0.25]])
    expected_total = (expected_epistemic + binary_entropy) / 2.0
    assert out.method == "mc_dropout"
    assert torch.allclose(out.epistemic_uncertainty, expected_epistemic)
    assert torch.allclose(out.aleatoric_uncertainty, torch.full((1, 2), binary_entropy))
    assert torch.allclose(out.uncertainty, expected_total)
    assert torch.allclose(out.confidence, 1.0 - expected_total)


def test_ece_rejects_invalid_values_and_empty_input():
    with pytest.raises(ValueError, match="at least one sample"):
        compute_binary_ece(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="n_bins"):
        compute_binary_ece(np.array([0.5]), np.array([1]), n_bins=0)
    with pytest.raises(ValueError, match="binary"):
        compute_binary_ece(np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        compute_binary_ece(np.array([1.5]), np.array([1]))


def test_multilabel_metrics_known_values_and_shape_contract():
    probabilities = np.array([[0.2, 0.8], [0.8, 0.2]])
    labels = np.array([[0, 1], [1, 0]])
    ece = compute_multilabel_ece(probabilities, labels, n_bins=2)
    brier = compute_brier_score(probabilities, labels)
    assert ece["macro_ece"] == pytest.approx(0.2)
    assert brier["macro_brier"] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_multilabel_ece(probabilities, labels[:, :1])
    with pytest.raises(ValueError, match="non-empty"):
        compute_brier_score(np.empty((0, 2)), np.empty((0, 2)))
    with pytest.raises(ValueError, match="binary"):
        compute_brier_score(probabilities, np.full(labels.shape, 0.5))


def test_calibrator_round_trip_and_invalid_validation_data():
    source = MultiLabelTemperatureCalibrator(num_emotions=2)
    with torch.no_grad():
        source.log_temperatures.copy_(torch.tensor([0.2, -0.3]))
    restored = MultiLabelTemperatureCalibrator(num_emotions=2)
    restored.load_state_dict(source.state_dict())
    assert torch.allclose(restored.temperatures, source.temperatures)
    with pytest.raises(ValueError, match="non-empty"):
        source.fit(torch.empty(0, 2), torch.empty(0, 2))
    with pytest.raises(ValueError, match="binary"):
        source.fit(torch.zeros(3, 2), torch.full((3, 2), 0.5))


def test_gate_exact_residual_formula_and_saliency():
    gate = UncertaintyAwareGate(feature_dim=4, num_emotions=2, hidden_dim=3, dropout=0.0)
    gate.eval()
    with torch.no_grad():
        for parameter in gate.gate_mlp.parameters():
            parameter.zero_()
    features = torch.arange(8.0).reshape(2, 1, 4)
    probabilities = torch.full((2, 1, 2), 0.5)
    uncertainty = torch.full((2, 1, 2), 0.5)
    valence_confidence = torch.full((2, 1, 1), 0.5)
    out = gate(features, probabilities, uncertainty, valence_confidence)
    assert torch.allclose(out.gate_weights, torch.full_like(features, 0.5))
    assert torch.allclose(out.gated_features, features * 0.5)
    assert torch.allclose(out.temporal_saliency, torch.full((2, 1, 1), 0.5))


def test_gate_handles_saturation_and_rejects_invalid_inputs():
    gate = UncertaintyAwareGate(feature_dim=4, num_emotions=2, hidden_dim=3, dropout=0.0)
    features = torch.ones(1, 2, 4)
    probabilities = torch.full((1, 2, 2), 0.5)
    uncertainty = torch.full((1, 2, 2), 0.5)
    with torch.no_grad():
        gate.gate_mlp[-2].weight.zero_()
        gate.gate_mlp[-2].bias.fill_(-100.0)
    suppressed = gate(features, probabilities, uncertainty)
    assert torch.equal(suppressed.gate_weights, torch.zeros_like(features))
    assert torch.equal(suppressed.gated_features, torch.zeros_like(features))
    with pytest.raises(ValueError, match="shape"):
        gate(features, torch.ones(1, 2, 3), uncertainty)
    with pytest.raises(ValueError, match="finite"):
        gate(features * float("nan"), probabilities, uncertainty)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        gate(features, probabilities, torch.full_like(uncertainty, 2.0))


def test_gate_rejects_empty_sequence_and_preserves_dtype_contract():
    gate = UncertaintyAwareGate(feature_dim=4, num_emotions=2, dropout=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        gate(torch.empty(2, 0, 4), torch.empty(2, 0, 2), torch.empty(2, 0, 2))
    out = gate(
        torch.ones(2, 3, 4, dtype=torch.float64),
        torch.full((2, 3, 2), 0.5, dtype=torch.float64),
        torch.full((2, 3, 2), 0.5, dtype=torch.float64),
    )
    assert out.gated_features.dtype == torch.float32
