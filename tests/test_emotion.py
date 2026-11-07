"""Unit and integration tests for Phase 5 Affective State & Emotion Derivation.

Tests the Emotion Branch, Context Branch, independent emotion-wise confidence
estimation, continuous valence affective state estimation, and end-to-end
integration with Phase 4 Multimodal Fusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.emotion import (  # noqa: E402
    DEFAULT_EMOTION_NAMES,
    NUM_EMOTIONS,
    AffectiveStateEstimator,
    AffectiveStateOutput,
    EmotionClassifier,
    EmotionOutput,
    MultimodalEmotionPipeline,
)
from sentiment_derivation.fusion.alignment import (  # noqa: E402
    ModalityAligner,
)
from sentiment_derivation.fusion.multimodal_fusion import (  # noqa: E402
    MultimodalFusion,
)


@pytest.fixture
def emotion_modules() -> tuple[EmotionClassifier, AffectiveStateEstimator]:
    classifier = EmotionClassifier(
        input_dim=128,
        emotion_dim=128,
        context_dim=128,
        num_emotions=8,
        dropout=0.0,
    )
    estimator = AffectiveStateEstimator(
        input_dim=128,
        hidden_dim=128,
        affective_dim=64,
        min_valence=1.0,
        max_valence=7.0,
        dropout=0.0,
    )
    return classifier, estimator


def test_emotion_classifier_sequence_input_shapes_and_ranges(emotion_modules) -> None:
    classifier, _ = emotion_modules
    features = torch.randn(2, 16, 128)

    output = classifier(features)

    assert isinstance(output, EmotionOutput)
    assert output.logits.shape == (2, 16, 8)
    assert output.probabilities.shape == (2, 16, 8)
    assert output.emotion_confidence.shape == (2, 16, 8)
    assert output.emotion_uncertainty.shape == (2, 16, 8)
    assert output.emotion_embeddings.shape == (2, 16, 128)
    assert output.context_embeddings.shape == (2, 16, 128)
    assert output.emotion_names == DEFAULT_EMOTION_NAMES
    assert len(output.emotion_names) == 8

    # Multi-label probabilities must strictly reside in [0, 1]
    assert (output.probabilities >= 0.0).all() and (output.probabilities <= 1.0).all()

    # Confidences and uncertainties must reside in [0, 1]
    assert (output.emotion_confidence >= 0.0).all() and (output.emotion_confidence <= 1.0).all()
    assert (output.emotion_uncertainty >= 0.0).all() and (output.emotion_uncertainty <= 1.0).all()


def test_emotion_confidence_and_uncertainty_mathematical_relationship(emotion_modules) -> None:
    classifier, _ = emotion_modules
    features = torch.randn(4, 16, 128)

    output = classifier(features)

    # Mathematical identity: confidence_k + uncertainty_k == 1.0
    sum_identity = output.emotion_confidence + output.emotion_uncertainty
    assert torch.allclose(sum_identity, torch.ones_like(sum_identity), atol=1e-6)

    # Per-emotion confidence must be maintained separately (not a single scalar)
    assert output.emotion_confidence.shape[-1] == NUM_EMOTIONS
    assert not torch.allclose(
        output.emotion_confidence[0, 0, 0], output.emotion_confidence[0, 0, 1]
    )


def test_emotion_confidence_margin_properties() -> None:
    classifier = EmotionClassifier(input_dim=8, emotion_dim=8, context_dim=8, num_emotions=2)

    # Synthetic probabilities
    probs = torch.tensor([[[0.5, 1.0], [0.0, 0.25]]])
    conf, uncert = classifier._estimate_confidence_and_uncertainty(probs)

    # When p = 0.5 (maximum ambiguity), confidence is 0.0 and uncertainty is 1.0
    assert torch.isclose(conf[0, 0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(uncert[0, 0, 0], torch.tensor(1.0), atol=1e-6)

    # When p = 1.0 (certain), confidence is 1.0 and uncertainty is 0.0
    assert torch.isclose(conf[0, 0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(uncert[0, 0, 1], torch.tensor(0.0), atol=1e-6)

    # When p = 0.0 (certain absence), confidence is 1.0 and uncertainty is 0.0
    assert torch.isclose(conf[0, 1, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(uncert[0, 1, 0], torch.tensor(0.0), atol=1e-6)

    # When p = 0.25, |2*0.25 - 1| = 0.5
    assert torch.isclose(conf[0, 1, 1], torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(uncert[0, 1, 1], torch.tensor(0.5), atol=1e-6)


def test_emotion_confidence_entropy_method() -> None:
    classifier = EmotionClassifier(
        input_dim=8,
        emotion_dim=8,
        context_dim=8,
        num_emotions=2,
        confidence_method="entropy",
    )
    probs = torch.tensor([[[0.5, 0.9999]]])
    conf, uncert = classifier._estimate_confidence_and_uncertainty(probs)

    # At p=0.5, binary entropy is maximal (1.0), so uncertainty is 1.0 and confidence is 0.0
    assert torch.isclose(uncert[0, 0, 0], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(conf[0, 0, 0], torch.tensor(0.0), atol=1e-4)

    # At p near 1.0, entropy approaches 0.0, so uncertainty is 0.0 and confidence is 1.0
    assert uncert[0, 0, 1] < 0.05
    assert conf[0, 0, 1] > 0.95


def test_parallel_branches_produce_distinct_representations(emotion_modules) -> None:
    classifier, _ = emotion_modules
    features = torch.randn(2, 16, 128)

    output = classifier(features)

    # Emotion branch and Context branch must be distinct representations per AGENTS.md Constraint 3
    assert not torch.allclose(output.emotion_embeddings, output.context_embeddings)


def test_emotion_classifier_batched_vector_and_unbatched_inputs(emotion_modules) -> None:
    classifier, _ = emotion_modules

    # 2D Batched Vector [B=3, D=128]
    vec_input = torch.randn(3, 128)
    out_2d = classifier(vec_input)
    assert out_2d.logits.shape == (3, 8)
    assert out_2d.probabilities.shape == (3, 8)
    assert out_2d.emotion_confidence.shape == (3, 8)
    assert out_2d.emotion_embeddings.shape == (3, 128)
    assert out_2d.context_embeddings.shape == (3, 128)

    # 1D Unbatched Vector [D=128]
    unbatched = torch.randn(128)
    out_1d = classifier(unbatched)
    assert out_1d.logits.shape == (8,)
    assert out_1d.probabilities.shape == (8,)
    assert out_1d.emotion_confidence.shape == (8,)


def test_affective_state_estimator_bounds_and_shapes(emotion_modules) -> None:
    _, estimator = emotion_modules
    features = torch.randn(3, 16, 128)

    out = estimator(features)

    assert isinstance(out, AffectiveStateOutput)
    assert out.valence.shape == (3, 16, 1)
    assert out.affective_state.shape == (3, 16, 64)
    assert out.raw_valence.shape == (3, 16, 1)
    assert out.valence_confidence.shape == (3, 16, 1)

    # Valence must be strictly within [1.0, 7.0]
    assert (out.valence >= 1.0).all() and (out.valence <= 7.0).all()
    assert (out.valence_confidence >= 0.0).all() and (out.valence_confidence <= 1.0).all()


def test_affective_state_estimator_batched_vector_and_unbatched(emotion_modules) -> None:
    _, estimator = emotion_modules

    # 2D Batched Vector [B=2, D=128]
    vec_input = torch.randn(2, 128)
    out_2d = estimator(vec_input)
    assert out_2d.valence.shape == (2, 1)
    assert out_2d.affective_state.shape == (2, 64)
    assert (out_2d.valence >= 1.0).all() and (out_2d.valence <= 7.0).all()

    # 1D Unbatched Vector [D=128]
    unbatched = torch.randn(128)
    out_1d = estimator(unbatched)
    assert out_1d.valence.shape == (1,)
    assert out_1d.affective_state.shape == (64,)
    assert (out_1d.valence >= 1.0).all() and (out_1d.valence <= 7.0).all()


def test_multimodal_emotion_pipeline_contract_and_keys(emotion_modules) -> None:
    classifier, estimator = emotion_modules
    pipeline = MultimodalEmotionPipeline(classifier, estimator)
    features = torch.randn(2, 16, 128)

    result = pipeline(features)

    expected_keys = {
        "affective_state",
        "valence",
        "raw_valence",
        "valence_confidence",
        "emotion_probabilities",
        "emotion_logits",
        "emotion_confidence",
        "emotion_uncertainty",
        "emotion_embeddings",
        "context_embeddings",
        "emotion_names",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["affective_state"].shape == (2, 16, 64)
    assert result["valence"].shape == (2, 16, 1)
    assert result["emotion_probabilities"].shape == (2, 16, 8)
    assert result["emotion_confidence"].shape == (2, 16, 8)
    assert result["emotion_uncertainty"].shape == (2, 16, 8)
    assert result["emotion_names"] == DEFAULT_EMOTION_NAMES


def test_end_to_end_fusion_to_emotion_derivation_pipeline() -> None:
    # 1. Align modalities (Phase 4)
    aligner = ModalityAligner(
        video_dim=512, audio_dim=128, text_dim=128, common_dim=128, target_steps=16
    )
    fusion = MultimodalFusion(feature_dim=128, num_heads=4, dropout=0.0)

    # 2. Emotion Pipeline (Phase 5)
    classifier = EmotionClassifier(input_dim=128, emotion_dim=128, context_dim=128, num_emotions=8)
    estimator = AffectiveStateEstimator(input_dim=128, affective_dim=64, min_valence=1.0, max_valence=7.0)
    pipeline = MultimodalEmotionPipeline(classifier, estimator)

    # Synthetic multimodal inputs
    video = torch.randn(2, 8, 512)
    audio = torch.randn(2, 16, 128)
    text = torch.randn(2, 16, 128)

    aligned = aligner(video, audio, text)
    fused = fusion(aligned)
    emotion_results = pipeline(fused)

    assert emotion_results["affective_state"].shape == (2, 16, 64)
    assert emotion_results["valence"].shape == (2, 16, 1)
    assert emotion_results["emotion_probabilities"].shape == (2, 16, 8)
    assert emotion_results["emotion_confidence"].shape == (2, 16, 8)
    assert emotion_results["emotion_uncertainty"].shape == (2, 16, 8)
    assert (emotion_results["valence"] >= 1.0).all() and (emotion_results["valence"] <= 7.0).all()


def test_gradients_reach_emotion_and_affective_parameters() -> None:
    classifier = EmotionClassifier(input_dim=128, emotion_dim=64, context_dim=64, num_emotions=8)
    estimator = AffectiveStateEstimator(input_dim=128, affective_dim=32)
    features = torch.randn(2, 16, 128, requires_grad=True)

    emotion_out = classifier(features)
    affective_out = estimator(features)

    loss = (
        emotion_out.logits.sum()
        + emotion_out.context_embeddings.sum()
        + affective_out.valence.sum()
    )
    loss.backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert classifier.emotion_branch[0].weight.grad is not None
    assert classifier.context_branch[0].weight.grad is not None
    assert classifier.emotion_head[2].weight.grad is not None
    assert estimator.affective_encoder[0].weight.grad is not None
    assert estimator.valence_head[2].weight.grad is not None


def test_from_config_instantiation() -> None:
    config = {
        "emotion_classifier": {
            "input_dim": 128,
            "emotion_dim": 64,
            "context_dim": 64,
            "num_emotions": 8,
            "dropout": 0.05,
            "temperature": 1.2,
            "confidence_method": "margin",
        },
        "affective_state": {
            "input_dim": 128,
            "hidden_dim": 64,
            "affective_dim": 32,
            "min_valence": 1.0,
            "max_valence": 7.0,
            "dropout": 0.05,
            "bounded": True,
        },
    }

    classifier = EmotionClassifier.from_config(config)
    estimator = AffectiveStateEstimator.from_config(config)
    pipeline = MultimodalEmotionPipeline.from_config(config)

    assert classifier.emotion_dim == 64
    assert classifier.temperature == 1.2
    assert estimator.affective_dim == 32
    assert estimator.max_valence == 7.0
    assert isinstance(pipeline, MultimodalEmotionPipeline)


def test_invalid_shapes_and_parameters_raise_meaningful_errors() -> None:
    classifier = EmotionClassifier(input_dim=128)
    estimator = AffectiveStateEstimator(input_dim=128)

    # Wrong feature dimension
    with pytest.raises(ValueError, match="dimension"):
        classifier(torch.randn(2, 16, 64))
    with pytest.raises(ValueError, match="dimension"):
        estimator(torch.randn(2, 16, 64))

    # Invalid tensor dimensions (4D)
    with pytest.raises(ValueError, match="shape"):
        classifier(torch.randn(2, 16, 128, 2))
    with pytest.raises(ValueError, match="shape"):
        estimator(torch.randn(2, 16, 128, 2))

    # Invalid config arguments
    with pytest.raises(ValueError, match="strictly positive"):
        EmotionClassifier(input_dim=128, temperature=-1.0)
    with pytest.raises(ValueError, match="confidence_method"):
        EmotionClassifier(input_dim=128, confidence_method="invalid_method")
    with pytest.raises(ValueError, match="min_valence"):
        AffectiveStateEstimator(min_valence=7.0, max_valence=1.0)


def test_cpu_execution(emotion_modules) -> None:
    classifier, estimator = emotion_modules
    features = torch.randn(2, 16, 128)

    emotion_out = classifier(features)
    affective_out = estimator(features)

    assert emotion_out.logits.device.type == "cpu"
    assert affective_out.valence.device.type == "cpu"
    assert torch.isfinite(emotion_out.logits).all()
    assert torch.isfinite(affective_out.valence).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_execution_when_available(emotion_modules) -> None:
    classifier, estimator = (m.cuda() for m in emotion_modules)
    features = torch.randn(2, 16, 128).cuda()

    emotion_out = classifier(features)
    affective_out = estimator(features)

    assert emotion_out.logits.device.type == "cuda"
    assert affective_out.valence.device.type == "cuda"
