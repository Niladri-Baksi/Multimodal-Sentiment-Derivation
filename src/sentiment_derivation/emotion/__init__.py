"""Emotion and affective state derivation package.

Implements Stage 4 Parallel Representation (Emotion Branch & Context Branch)
and continuous affective state estimation per AGENTS.md.
"""

from sentiment_derivation.emotion.affective_state import (
    AffectiveStateEstimator,
    AffectiveStateOutput,
)
from sentiment_derivation.emotion.emotion_classifier import (
    DEFAULT_EMOTION_NAMES,
    NUM_EMOTIONS,
    EmotionClassifier,
    EmotionOutput,
    MultimodalEmotionPipeline,
)

__all__ = [
    "AffectiveStateEstimator",
    "AffectiveStateOutput",
    "DEFAULT_EMOTION_NAMES",
    "NUM_EMOTIONS",
    "EmotionClassifier",
    "EmotionOutput",
    "MultimodalEmotionPipeline",
]

