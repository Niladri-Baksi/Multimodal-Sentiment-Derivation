"""Unit tests for Phase 7 Temporal Encoder and Transition Analyzer modules.

Tests:
1. TemporalEncoder (BiLSTM forward pass, residual connections, pooled representations).
2. TransitionAnalyzer (Delta Emotion velocity, Persistence, Temporal Attention, and
   composite Transition Representation).
"""

from __future__ import annotations

import pytest
import torch

from sentiment_derivation.temporal import (
    TemporalEncoder,
    TemporalEncoderOutput,
    TransitionAnalyzer,
    TransitionOutput,
)


# ============================================================================
# 1. Temporal Encoder Tests
# ============================================================================

def test_temporal_encoder_forward_3d():
    """Verify TemporalEncoder processes 3D sequence [B, T=16, D=128]."""
    encoder = TemporalEncoder(
        input_dim=128,
        hidden_dim=128,
        num_layers=1,
        bidirectional=True,
        residual=True,
    )

    inputs = torch.randn(2, 16, 128)
    out = encoder(inputs)

    assert isinstance(out, TemporalEncoderOutput)
    assert out.features.shape == (2, 16, 128)
    assert out.pooled_features.shape == (2, 128)
    assert out.last_hidden.shape == (2, 128)

    # Dictionary access
    d = out.to_dict()
    assert "features" in d and "pooled_features" in d


def test_temporal_encoder_forward_2d():
    """Verify TemporalEncoder handles 2D pooled vector [B, D=128]."""
    encoder = TemporalEncoder(input_dim=128, hidden_dim=128)
    inputs = torch.randn(4, 128)

    out = encoder(inputs)
    assert out.features.shape == (4, 128)
    assert out.pooled_features.shape == (4, 128)


def test_temporal_encoder_gradient_flow():
    """Verify gradients propagate through BiLSTM and residual connections."""
    encoder = TemporalEncoder(input_dim=64, hidden_dim=64)
    inputs = torch.randn(2, 8, 64, requires_grad=True)

    out = encoder(inputs)
    loss = out.features.sum()
    loss.backward()

    assert inputs.grad is not None and torch.any(inputs.grad != 0.0)
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."


# ============================================================================
# 2. Transition Analyzer Tests
# ============================================================================

def test_transition_analyzer_delta_emotion():
    """Verify Delta Emotion (velocity) computation and boundary condition."""
    analyzer = TransitionAnalyzer(num_emotions=8, context_dim=64, transition_dim=64)

    # Controlled emotion probabilities across 3 timesteps
    # t=0: all 0.2
    # t=1: all 0.5 (delta should be +0.3)
    # t=2: all 0.1 (delta should be -0.4)
    probs = torch.tensor([
        [[0.2] * 8, [0.5] * 8, [0.1] * 8]
    ])  # [1, 3, 8]

    delta = analyzer.compute_delta(probs)
    assert delta.shape == (1, 3, 8)

    # Boundary: delta at t=0 must be exactly 0
    assert torch.all(delta[:, 0, :] == 0.0)

    # t=1 delta: 0.5 - 0.2 = 0.3
    assert torch.allclose(delta[:, 1, :], torch.tensor(0.3), atol=1e-5)

    # t=2 delta: 0.1 - 0.5 = -0.4
    assert torch.allclose(delta[:, 2, :], torch.tensor(-0.4), atol=1e-5)


def test_transition_analyzer_persistence():
    """Verify Persistence exponential moving average smoothing."""
    analyzer = TransitionAnalyzer(num_emotions=8, context_dim=64, transition_dim=64, persistence_alpha=0.5)

    probs = torch.rand(2, 16, 8)
    persistence = analyzer.compute_persistence(probs)

    assert persistence.shape == (2, 16, 8)
    # P_0 must equal E_0
    assert torch.allclose(persistence[:, 0, :], probs[:, 0, :])
    # Persistence values must be in [0, 1]
    assert torch.all(persistence >= 0.0) and torch.all(persistence <= 1.0)


def test_transition_analyzer_temporal_attention():
    """Verify Temporal Attention weights sum to 1.0 across sequence steps."""
    analyzer = TransitionAnalyzer(num_emotions=8, context_dim=128, transition_dim=128)

    probs = torch.rand(2, 16, 8)
    context = torch.randn(2, 16, 128)
    affect = torch.randn(2, 16, 64)
    valence = torch.full((2, 16, 1), 4.5)

    out = analyzer(
        emotion_probabilities=probs,
        context_embeddings=context,
        affective_state=affect,
        valence=valence,
    )

    assert isinstance(out, TransitionOutput)
    assert out.temporal_attention_weights.shape == (2, 16, 1)

    # Softmax condition: sum across time steps (dim=1) must equal 1.0
    attn_sum = torch.sum(out.temporal_attention_weights, dim=1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    # Output dimensions
    assert out.transition_embedding.shape == (2, 16, 128)
    assert out.pooled_transition.shape == (2, 128)
    assert out.delta_emotion.shape == (2, 16, 8)
    assert out.persistence.shape == (2, 16, 8)


def test_transition_analyzer_gradient_flow():
    """Verify gradient flow across all inputs and transition projections."""
    analyzer = TransitionAnalyzer(num_emotions=8, context_dim=64, affective_dim=32, transition_dim=64)

    probs = torch.rand(2, 8, 8, requires_grad=True)
    context = torch.randn(2, 8, 64, requires_grad=True)
    affect = torch.randn(2, 8, 32, requires_grad=True)
    valence = torch.rand(2, 8, 1, requires_grad=True)

    out = analyzer(
        emotion_probabilities=probs,
        context_embeddings=context,
        affective_state=affect,
        valence=valence,
    )

    loss = out.transition_embedding.sum() + out.pooled_transition.sum()
    loss.backward()

    assert probs.grad is not None and torch.any(probs.grad != 0.0)
    assert context.grad is not None and torch.any(context.grad != 0.0)
    assert affect.grad is not None and torch.any(affect.grad != 0.0)
    assert valence.grad is not None and torch.any(valence.grad != 0.0)
