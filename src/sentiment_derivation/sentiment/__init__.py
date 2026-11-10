"""Latent mapping and sentiment derivation package for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 8 & Stage 9 (Phase 9 & Phase 10).

Implements:
1. LatentSentimentMapper, LatentSentimentOutput: Maps emotional dynamics, transitions,
   and uncertainty-gated context to a unified latent affective representation.
2. SentimentHead, SentimentOutput, DEFAULT_SENTIMENT_NAMES, NUM_SENTIMENT_CLASSES:
   Derives categorical sentiment, continuous polarity score, and sentiment confidence.
3. valence_to_sentiment_label: Ground truth mapping utility deriving sentiment labels from
   continuous EmoReact valence ratings ([1.0, 7.0]) with documented thresholds.
4. SentimentDerivationPipeline: Unified pipeline integrating Stage 8 and Stage 9.
"""

from sentiment_derivation.sentiment.latent_mapper import (
    LatentSentimentMapper,
    LatentSentimentOutput,
)
from sentiment_derivation.sentiment.sentiment_head import (
    DEFAULT_SENTIMENT_NAMES,
    NUM_SENTIMENT_CLASSES,
    SentimentDerivationPipeline,
    SentimentHead,
    SentimentOutput,
    valence_to_sentiment_label,
)

__all__ = [
    "DEFAULT_SENTIMENT_NAMES",
    "NUM_SENTIMENT_CLASSES",
    "LatentSentimentMapper",
    "LatentSentimentOutput",
    "SentimentDerivationPipeline",
    "SentimentHead",
    "SentimentOutput",
    "valence_to_sentiment_label",
]
