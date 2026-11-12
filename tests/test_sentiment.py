"""Unit and integration tests for Phase 8 & Phase 9 Sentiment Derivation modules.

Tests:
1. LatentSentimentMapper (Stream projections, multimodal latent fusion, temporal attention pooling).
2. valence_to_sentiment_label (Ground truth mapping from continuous Likert valence to sentiment).
3. SentimentHead (Categorical sentiment, continuous polarity, and sentiment-level confidence).
4. SentimentDerivationPipeline & full end-to-end integration across all 9 canonical stages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.emotion import (  # noqa: E402
    AffectiveStateEstimator,
    EmotionClassifier,
)
from sentiment_derivation.fusion.alignment import (  # noqa: E402
    ModalityAligner,
)
from sentiment_derivation.fusion.multimodal_fusion import (  # noqa: E402
    MultimodalFusion,
)
from sentiment_derivation.sentiment import (  # noqa: E402
    DEFAULT_SENTIMENT_NAMES,
    NUM_SENTIMENT_CLASSES,
    LatentSentimentMapper,
    LatentSentimentOutput,
    SentimentDerivationPipeline,
    SentimentHead,
    SentimentOutput,
    valence_to_sentiment_label,
)
from sentiment_derivation.temporal import (  # noqa: E402
    TemporalEncoder,
    TransitionAnalyzer,
)
from sentiment_derivation.uncertainty import (  # noqa: E402
    UncertaintyAwareGate,
)


# ============================================================================
# 1. Latent Sentiment Mapper Tests
# ============================================================================

def test_latent_sentiment_mapper_sequence_forward():
    """Verify LatentSentimentMapper processes 3D sequence inputs correctly."""
    mapper = LatentSentimentMapper(
        feature_dim=128,
        transition_dim=128,
        num_emotions=8,
        affective_dim=64,
        latent_dim=128,
        hidden_dim=64,
        temporal_pooling="attention",
    )

    batch_size = 2
    steps = 16

    gated_features = torch.randn(batch_size, steps, 128)
    emotion_probs = torch.sigmoid(torch.randn(batch_size, steps, 8))
    emotion_conf = torch.rand(batch_size, steps, 8)
    delta_emotion = torch.clamp(torch.randn(batch_size, steps, 8), -1.0, 1.0)
    persistence = torch.rand(batch_size, steps, 8)
    transition_emb = torch.randn(batch_size, steps, 128)
    affective_state = torch.randn(batch_size, steps, 64)

    out = mapper(
        gated_features=gated_features,
        emotion_probabilities=emotion_probs,
        emotion_confidence=emotion_conf,
        delta_emotion=delta_emotion,
        persistence=persistence,
        transition_embedding=transition_emb,
        affective_state=affective_state,
    )

    assert isinstance(out, LatentSentimentOutput)
    assert out.latent_sequence.shape == (batch_size, steps, 128)
    assert out.latent_representation.shape == (batch_size, 128)
    assert out.temporal_attention_weights.shape == (batch_size, steps, 1)

    # Attention weights must sum to 1.0 across sequence steps
    attn_sums = out.temporal_attention_weights.sum(dim=1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)

    # Dictionary access
    d = out.to_dict()
    assert "latent_representation" in d and "latent_sequence" in d


def test_latent_sentiment_mapper_pooling_modes():
    """Verify LatentSentimentMapper with mean and last pooling strategies."""
    for mode in ("mean", "last"):
        mapper = LatentSentimentMapper(
            feature_dim=64,
            transition_dim=64,
            num_emotions=4,
            affective_dim=32,
            latent_dim=64,
            temporal_pooling=mode,
        )

        b, t = 3, 8
        out = mapper(
            gated_features=torch.randn(b, t, 64),
            emotion_probabilities=torch.rand(b, t, 4),
            emotion_confidence=torch.rand(b, t, 4),
            delta_emotion=torch.randn(b, t, 4),
            persistence=torch.rand(b, t, 4),
            transition_embedding=torch.randn(b, t, 64),
        )
        assert out.latent_representation.shape == (b, 64)
        assert out.latent_sequence.shape == (b, t, 64)


def test_latent_sentiment_mapper_optional_inputs():
    """Verify mapper handles omitted optional transition and affective streams."""
    mapper = LatentSentimentMapper(
        feature_dim=128,
        transition_dim=128,
        num_emotions=8,
        affective_dim=64,
        latent_dim=128,
    )

    b, t = 2, 8
    # Omit transition_embedding, affective_state, and valence
    out = mapper(
        gated_features=torch.randn(b, t, 128),
        emotion_probabilities=torch.rand(b, t, 8),
        emotion_confidence=torch.rand(b, t, 8),
        delta_emotion=torch.randn(b, t, 8),
        persistence=torch.rand(b, t, 8),
        transition_embedding=None,
        affective_state=None,
        valence=None,
    )
    assert out.latent_representation.shape == (b, 128)
    assert torch.isfinite(out.latent_representation).all()


def test_latent_sentiment_mapper_gradient_flow():
    """Verify gradients propagate through all streams and attention pooling."""
    mapper = LatentSentimentMapper(
        feature_dim=32,
        transition_dim=32,
        num_emotions=4,
        affective_dim=16,
        latent_dim=32,
        hidden_dim=16,
    )

    gated = torch.randn(2, 6, 32, requires_grad=True)
    probs = torch.rand(2, 6, 4, requires_grad=True)
    conf = torch.rand(2, 6, 4, requires_grad=True)
    delta = torch.randn(2, 6, 4, requires_grad=True)
    pers = torch.rand(2, 6, 4, requires_grad=True)
    trans = torch.randn(2, 6, 32, requires_grad=True)

    out = mapper(gated, probs, conf, delta, pers, trans)
    loss = out.latent_representation.sum()
    loss.backward()

    assert gated.grad is not None and torch.isfinite(gated.grad).all()
    assert probs.grad is not None and torch.isfinite(probs.grad).all()
    assert mapper.temporal_attention[0].weight.grad is not None


# ============================================================================
# 2. Valence to Sentiment Label Mapping Tests
# ============================================================================

def test_valence_to_sentiment_label_scalars():
    """Verify categorical classification and score calculation on known scalar valence values."""
    # Negative: valence < 3.75 (neutral_margin=0.25)
    cls_neg, label_neg, score_neg = valence_to_sentiment_label(2.0, neutral_margin=0.25)
    assert cls_neg == 0
    assert label_neg == "Negative"
    assert score_neg < 0.0

    # Neutral: 3.75 <= valence <= 4.25
    cls_neu, label_neu, score_neu = valence_to_sentiment_label(4.0, neutral_margin=0.25)
    assert cls_neu == 1
    assert label_neu == "Neutral"
    assert score_neu == 0.0

    # Positive: valence > 4.25
    cls_pos, label_pos, score_pos = valence_to_sentiment_label(6.5, neutral_margin=0.25)
    assert cls_pos == 2
    assert label_pos == "Positive"
    assert score_pos > 0.0


def test_valence_to_sentiment_label_arrays_and_tensors():
    """Verify batch processing on arrays and PyTorch tensors."""
    val_tensor = torch.tensor([1.5, 3.8, 4.0, 4.2, 5.5])
    classes, labels, scores = valence_to_sentiment_label(val_tensor, neutral_margin=0.25)

    assert len(classes) == 5
    assert classes == [0, 1, 1, 1, 2]
    assert labels == ["Negative", "Neutral", "Neutral", "Neutral", "Positive"]
    for s in scores:
        assert -1.0 <= s <= 1.0


# ============================================================================
# 3. Sentiment Head Tests
# ============================================================================

def test_sentiment_head_batched_forward():
    """Verify SentimentHead produces 3-class distributions, scores, and confidences."""
    head = SentimentHead(
        latent_dim=128,
        num_classes=3,
        hidden_dim=64,
        confidence_method="margin",
    )

    latent = torch.randn(4, 128)
    out = head(latent)

    assert isinstance(out, SentimentOutput)
    assert out.sentiment_logits.shape == (4, 3)
    assert out.sentiment_probabilities.shape == (4, 3)
    assert out.sentiment_class.shape == (4,)
    assert out.sentiment_confidence.shape == (4,)
    assert out.sentiment_score.shape == (4, 1)
    assert len(out.sentiment) == 4

    # Probabilities must sum to 1.0
    prob_sums = out.sentiment_probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    # Confidence must reside in [0.0, 1.0]
    assert (out.sentiment_confidence >= 0.0).all() and (out.sentiment_confidence <= 1.0).all()

    # Continuous score must reside in [-1.0, 1.0]
    assert (out.sentiment_score >= -1.0).all() and (out.sentiment_score <= 1.0).all()

    # Labels must match classes
    for i in range(4):
        pred_idx = out.sentiment_class[i].item()
        assert out.sentiment[i] == DEFAULT_SENTIMENT_NAMES[pred_idx]


def test_sentiment_head_confidence_methods():
    """Verify all supported confidence estimation methods."""
    for method in ("margin", "max_prob", "entropy"):
        head = SentimentHead(latent_dim=32, confidence_method=method)
        latent = torch.randn(3, 32)
        out = head(latent)
        assert (out.sentiment_confidence >= 0.0).all()
        assert (out.sentiment_confidence <= 1.0).all()


def test_sentiment_head_unbatched_forward():
    """Verify unbatched 1D vector [D=128] returns single scalar/string types."""
    head = SentimentHead(latent_dim=64)
    vec = torch.randn(64)
    out = head(vec)

    assert isinstance(out.sentiment, str)
    assert isinstance(out.sentiment_confidence, float)
    assert out.sentiment_class.ndim == 0
    assert 0.0 <= out.sentiment_confidence <= 1.0


def test_sentiment_head_gradient_flow():
    """Verify gradients propagate to classification and regression heads."""
    head = SentimentHead(latent_dim=64, hidden_dim=32)
    latent = torch.randn(2, 64, requires_grad=True)

    out = head(latent)
    loss = out.sentiment_logits.sum() + out.sentiment_score.sum()
    loss.backward()

    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert head.classifier[0].weight.grad is not None
    assert head.regressor[0].weight.grad is not None


# ============================================================================
# 4. Pipeline & End-to-End Integration Tests
# ============================================================================

def test_sentiment_derivation_pipeline():
    """Verify SentimentDerivationPipeline integrates Stage 8 and Stage 9 into canonical contract."""
    mapper = LatentSentimentMapper(feature_dim=64, latent_dim=64, num_emotions=8)
    head = SentimentHead(latent_dim=64, num_classes=3)
    pipeline = SentimentDerivationPipeline(latent_mapper=mapper, sentiment_head=head)

    b, t = 2, 8
    res = pipeline(
        gated_features=torch.randn(b, t, 64),
        emotion_probabilities=torch.rand(b, t, 8),
        emotion_confidence=torch.rand(b, t, 8),
        delta_emotion=torch.randn(b, t, 8),
        persistence=torch.rand(b, t, 8),
    )

    required_keys = {
        "latent_representation",
        "latent_sequence",
        "sentiment",
        "sentiment_class",
        "sentiment_confidence",
        "sentiment_probabilities",
        "sentiment_logits",
        "sentiment_score",
        "temporal_attention_weights",
        "sentiment_labels",
    }
    assert required_keys.issubset(set(res.keys()))
    assert res["latent_representation"].shape == (b, 64)
    assert res["latent_sequence"].shape == (b, t, 64)
    assert res["sentiment_probabilities"].shape == (b, 3)
    assert res["sentiment_score"].shape == (b, 1)


def test_full_pipeline_alignment_to_sentiment_derivation():
    """Verify complete end-to-end forward chain across all 9 canonical stages:
    ModalityAligner -> MultimodalFusion -> TemporalEncoder -> EmotionClassifier + AffectiveStateEstimator
    -> TransitionAnalyzer -> UncertaintyAwareGate -> LatentSentimentMapper -> SentimentHead.
    """
    # 1. Modality Alignment (Stage 3)
    aligner = ModalityAligner(video_dim=512, audio_dim=128, text_dim=128, common_dim=128, target_steps=16)
    fusion = MultimodalFusion(feature_dim=128, num_heads=4, dropout=0.0)

    # 2. Temporal BiLSTM (Stage 4)
    temporal_enc = TemporalEncoder(input_dim=128, hidden_dim=128)

    # 3. Emotion and Affective Heads (Stage 4 Parallel Representation)
    emotion_cls = EmotionClassifier(input_dim=128, emotion_dim=128, context_dim=128, num_emotions=8)
    affect_est = AffectiveStateEstimator(input_dim=128, affective_dim=64)

    # 4. Transition Analysis (Stage 5 & Stage 6)
    trans_analyzer = TransitionAnalyzer(num_emotions=8, context_dim=128, affective_dim=64, transition_dim=128)

    # 5. Uncertainty-Aware Gate (Stage 7)
    gate = UncertaintyAwareGate(feature_dim=128, num_emotions=8, valence_dim=1)

    # 6. Latent Sentiment Mapper (Stage 8)
    mapper = LatentSentimentMapper(feature_dim=128, transition_dim=128, num_emotions=8, latent_dim=128)

    # 7. Sentiment Head (Stage 9)
    head = SentimentHead(latent_dim=128, num_classes=3)

    # Execute end-to-end forward pass
    video = torch.randn(2, 8, 512)
    audio = torch.randn(2, 16, 128)
    text = torch.randn(2, 16, 128)

    aligned = aligner(video, audio, text)
    fused = fusion(aligned)
    temp_out = temporal_enc(fused["features"])

    emotion_out = emotion_cls(temp_out.features)
    affect_out = affect_est(temp_out.features)

    trans_out = trans_analyzer(
        emotion_probabilities=emotion_out.probabilities,
        context_embeddings=emotion_out.context_embeddings,
        affective_state=affect_out.affective_state,
        valence=affect_out.valence,
        confidence=emotion_out.emotion_confidence,
        uncertainty=emotion_out.emotion_uncertainty,
    )

    gated_out = gate(
        features=emotion_out.context_embeddings,
        emotion_probabilities=emotion_out.probabilities,
        emotion_uncertainty=emotion_out.emotion_uncertainty,
        valence_confidence=affect_out.valence_confidence,
    )

    latent_out = mapper(
        gated_features=gated_out.gated_features,
        emotion_probabilities=emotion_out.probabilities,
        emotion_confidence=emotion_out.emotion_confidence,
        delta_emotion=trans_out.delta_emotion,
        persistence=trans_out.persistence,
        transition_embedding=trans_out.transition_embedding,
        affective_state=affect_out.affective_state,
        valence=affect_out.valence,
    )

    sentiment_out = head(latent_out.latent_representation)

    # Assertions on final sentiment predictions
    assert sentiment_out.sentiment_probabilities.shape == (2, 3)
    assert sentiment_out.sentiment_confidence.shape == (2,)
    assert sentiment_out.sentiment_score.shape == (2, 1)
    assert len(sentiment_out.sentiment) == 2
    for s in sentiment_out.sentiment:
        assert s in DEFAULT_SENTIMENT_NAMES


def test_from_config_instantiation():
    """Verify instantiation from model.yaml dictionary."""
    config = {
        "latent_mapper": {
            "feature_dim": 128,
            "transition_dim": 128,
            "num_emotions": 8,
            "affective_dim": 64,
            "latent_dim": 128,
            "hidden_dim": 64,
            "temporal_pooling": "attention",
        },
        "sentiment_head": {
            "latent_dim": 128,
            "num_classes": 3,
            "hidden_dim": 64,
            "dropout": 0.05,
            "temperature": 1.1,
            "confidence_method": "margin",
        },
    }

    mapper = LatentSentimentMapper.from_config(config)
    head = SentimentHead.from_config(config)
    pipeline = SentimentDerivationPipeline.from_config(config)

    assert mapper.latent_dim == 128
    assert head.temperature == 1.1
    assert isinstance(pipeline, SentimentDerivationPipeline)
