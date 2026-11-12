"""End-to-end video/audio model for emotion and sentiment prediction."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from sentiment_derivation.encoders.audio_encoder import AudioEncoder
from sentiment_derivation.encoders.text_encoder import TextEncoder
from sentiment_derivation.encoders.video_encoder import VideoEncoder
from sentiment_derivation.emotion.affective_state import AffectiveStateEstimator
from sentiment_derivation.emotion.emotion_classifier import EmotionClassifier
from sentiment_derivation.fusion.alignment import ModalityAligner
from sentiment_derivation.fusion.multimodal_fusion import MultimodalFusion
from sentiment_derivation.sentiment.latent_mapper import LatentSentimentMapper
from sentiment_derivation.sentiment.sentiment_head import SentimentHead
from sentiment_derivation.temporal.temporal_encoder import TemporalEncoder
from sentiment_derivation.temporal.transition_analyzer import TransitionAnalyzer
from sentiment_derivation.uncertainty.certainty_estimator import UncertaintyEstimator
from sentiment_derivation.uncertainty.uncertainty_gate import UncertaintyAwareGate


class MultimodalSentimentModel(nn.Module):
    """Run the configured video/audio-to-sentiment architecture end to end."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        video_cfg = config.get("video_encoder", {})
        text_cfg = config.get("text_encoder", {})
        alignment_cfg = config.get("modality_alignment", {})

        self.video_encoder = VideoEncoder(
            pretrained=video_cfg.get("pretrained", True),
            weights=video_cfg.get("weights", "DEFAULT"),
            feature_dim=video_cfg.get("feature_dim", 512),
            freeze=video_cfg.get("freeze_backbone", True),
            apply_transforms=video_cfg.get("apply_transforms", False),
            image_size=tuple(video_cfg.get("image_size", (224, 224))),
        )
        self.audio_encoder = AudioEncoder(
            target_steps=alignment_cfg.get("target_steps", 16),
            feature_dim=alignment_cfg.get("audio_input_dim", 128),
        )
        self.text_encoder = (
            TextEncoder(
                target_steps=alignment_cfg.get("target_steps", 16),
                feature_dim=alignment_cfg.get("text_input_dim", 128),
                model_name=text_cfg.get("model_name", "distilbert-base-uncased"),
                freeze_backbone=text_cfg.get("freeze_backbone", True),
                pretrained=text_cfg.get("pretrained", True),
                backbone_config=text_cfg.get("backbone_config"),
            )
            if text_cfg.get("enabled", False)
            else None
        )
        self.aligner = ModalityAligner.from_config(config)
        self.fusion = MultimodalFusion.from_config(config)
        self.temporal_encoder = TemporalEncoder.from_config(config)
        self.emotion_classifier = EmotionClassifier.from_config(config)
        self.affective_estimator = AffectiveStateEstimator.from_config(config)
        self.uncertainty_estimator = UncertaintyEstimator.from_config(config)
        self.transition_analyzer = TransitionAnalyzer.from_config(config)
        self.uncertainty_gate = UncertaintyAwareGate.from_config(config)
        self.latent_mapper = LatentSentimentMapper.from_config(config)
        self.sentiment_head = SentimentHead.from_config(config)

    def forward(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        *,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        text_available: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> dict[str, Any]:
        """Predict from video/audio and optional tokenized text inputs."""
        if (text_input_ids is None) != (text_attention_mask is None):
            raise ValueError("text_input_ids and text_attention_mask must be provided together.")
        text_features = None
        if text_input_ids is not None and text_attention_mask is not None:
            if self.text_encoder is None:
                raise ValueError(
                    "Text input was provided but text_encoder.enabled is false in model config."
                )
            if not isinstance(text_input_ids, torch.Tensor) or not isinstance(
                text_attention_mask, torch.Tensor
            ):
                raise TypeError("Text IDs and attention masks must be torch.Tensor instances.")
            if text_input_ids.ndim != 2 or text_attention_mask.shape != text_input_ids.shape:
                raise ValueError("Text IDs and attention mask must share shape [B, sequence_length].")
            if text_input_ids.shape[1] == 0:
                raise ValueError("Text token sequences must contain at least one position.")
            token_available = text_attention_mask.to(dtype=torch.bool).any(dim=1)
            if text_available is None:
                text_available = token_available
            elif not isinstance(text_available, torch.Tensor):
                raise TypeError("text_available must be a torch.Tensor.")
            elif text_available.dtype != torch.bool or text_available.shape != token_available.shape:
                raise ValueError("text_available must be a boolean tensor with shape [B].")
            elif text_available.device != text_input_ids.device:
                raise ValueError("text_available must be on the same device as the text tokens.")
            elif bool((text_available & ~token_available).any()):
                raise ValueError("text_available cannot be true for a sample with an empty attention mask.")
            # Avoid all-masked transformer rows, whose attention can become NaN.
            safe_attention_mask = text_attention_mask.clone()
            empty_rows = ~token_available
            if bool(empty_rows.any()):
                safe_attention_mask[empty_rows, 0] = 1
            text_features = self.text_encoder(text_input_ids, safe_attention_mask)
        video_features = self.video_encoder(frames, return_dict=False)
        audio_features = self.audio_encoder(audio, return_dict=False)
        return self.forward_features(
            video_features,
            audio_features,
            text_features=text_features,
            text_available=text_available,
            return_attention_weights=return_attention_weights,
        )

    def forward_features(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        *,
        text_features: torch.Tensor | None = None,
        text_available: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> dict[str, Any]:
        """Run the trainable multimodal and temporal stages on encoded features."""
        aligned = self.aligner(
            video_features=video_features,
            audio_features=audio_features,
            text_features=text_features,
            text_available=text_available,
        )
        fused = self.fusion(aligned, return_attention_weights=return_attention_weights)
        temporal = self.temporal_encoder(fused["features"])

        emotion = self.emotion_classifier(temporal.features)
        affect = self.affective_estimator(temporal.features)
        certainty = self.uncertainty_estimator(emotion.probabilities)
        transition = self.transition_analyzer(
            emotion_probabilities=emotion.probabilities,
            context_embeddings=emotion.context_embeddings,
            affective_state=affect.affective_state,
            valence=affect.valence,
            confidence=certainty.confidence,
            uncertainty=certainty.uncertainty,
        )
        gated = self.uncertainty_gate(
            features=emotion.context_embeddings,
            emotion_probabilities=emotion.probabilities,
            emotion_uncertainty=certainty.uncertainty,
            valence_confidence=affect.valence_confidence,
        )
        latent = self.latent_mapper(
            gated_features=gated.gated_features,
            emotion_probabilities=emotion.probabilities,
            emotion_confidence=certainty.confidence,
            delta_emotion=transition.delta_emotion,
            persistence=transition.persistence,
            transition_embedding=transition.transition_embedding,
            affective_state=affect.affective_state,
            valence=affect.valence,
        )
        sentiment = self.sentiment_head(latent.latent_representation)

        results: dict[str, Any] = {
            "emotion_logits": emotion.logits,
            "emotion_probabilities": emotion.probabilities,
            "emotion_confidence": certainty.confidence,
            "emotion_uncertainty": certainty.uncertainty,
            "emotion_names": emotion.emotion_names,
            "affective_state": affect.affective_state,
            "valence_sequence": affect.valence,
            "valence": affect.valence.mean(dim=1),
            "valence_confidence": affect.valence_confidence,
            "context_embeddings": emotion.context_embeddings,
            "transition_embedding": transition.transition_embedding,
            "delta_emotion": transition.delta_emotion,
            "persistence": transition.persistence,
            "transition_attention_weights": transition.temporal_attention_weights,
            "gate_weights": gated.gate_weights,
            "latent_sequence": latent.latent_sequence,
            "latent_representation": latent.latent_representation,
            "sentiment_logits": sentiment.sentiment_logits,
            "sentiment_probabilities": sentiment.sentiment_probabilities,
            "sentiment_class": sentiment.sentiment_class,
            "sentiment": sentiment.sentiment,
            "sentiment_confidence": sentiment.sentiment_confidence,
            "sentiment_score": sentiment.sentiment_score,
            "sentiment_labels": sentiment.sentiment_labels,
        }
        if return_attention_weights:
            results["modality_attention_weights"] = fused["attention_weights"]
            results["modality_order"] = fused["modality_order"]
        return results

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MultimodalSentimentModel":
        return cls(config)


def build_model(config: Mapping[str, Any]) -> MultimodalSentimentModel:
    """Construct the production model from the model YAML mapping."""
    return MultimodalSentimentModel.from_config(config)
