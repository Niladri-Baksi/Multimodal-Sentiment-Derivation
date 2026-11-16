"""Synthetic Phase 4/5/6 uncertainty-flow and gradient integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.emotion import (  # noqa: E402
    AffectiveStateEstimator,
    EmotionClassifier,
    MultimodalEmotionPipeline,
)
from sentiment_derivation.fusion.alignment import ModalityAligner  # noqa: E402
from sentiment_derivation.fusion.multimodal_fusion import MultimodalFusion  # noqa: E402
from sentiment_derivation.uncertainty import (  # noqa: E402
    UncertaintyAwareGate,
    UncertaintyEstimator,
)


def test_phase4_to_phase6_data_flow_and_gradients() -> None:
    aligner = ModalityAligner()
    fusion = MultimodalFusion(dropout=0.0)
    emotion_classifier = EmotionClassifier(dropout=0.0)
    affect_estimator = AffectiveStateEstimator(dropout=0.0)
    phase5 = MultimodalEmotionPipeline(emotion_classifier, affect_estimator)
    phase6_estimator = UncertaintyEstimator(method="margin", num_emotions=8)
    gate = UncertaintyAwareGate(dropout=0.0)

    video = torch.randn(2, 5, 512, requires_grad=True)
    audio = torch.randn(2, 12, 128, requires_grad=True)
    text = torch.randn(2, 9, 128, requires_grad=True)

    aligned = aligner(video, audio, text)
    fused = fusion(aligned)
    outputs = phase5(fused)
    uncertainty = phase6_estimator(outputs["emotion_probabilities"])
    gated = gate(
        outputs["context_embeddings"],
        outputs["emotion_probabilities"],
        uncertainty.uncertainty,
        outputs["valence_confidence"],
    )

    assert fused["features"].shape == (2, 16, 128)
    assert outputs["emotion_probabilities"].shape == (2, 16, 8)
    assert outputs["emotion_confidence"].shape == (2, 16, 8)
    assert outputs["emotion_uncertainty"].shape == (2, 16, 8)
    assert outputs["valence_confidence"].shape == (2, 16, 1)
    assert torch.allclose(uncertainty.confidence, outputs["emotion_confidence"], atol=1e-6)
    assert torch.allclose(uncertainty.uncertainty, outputs["emotion_uncertainty"], atol=1e-6)
    assert gated.gated_features.shape == (2, 16, 128)
    assert gated.gate_weights.shape == (2, 16, 128)
    assert gated.temporal_saliency.shape == (2, 16, 1)
    assert torch.isfinite(gated.gated_features).all()

    gated.gated_features.square().mean().backward()
    for value in (video, audio, text):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert emotion_classifier.emotion_head[2].weight.grad is not None
    assert affect_estimator.valence_head[2].weight.grad is not None
    assert gate.gate_mlp[0].weight.grad is not None
