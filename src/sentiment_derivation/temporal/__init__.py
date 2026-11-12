"""Temporal representation and transition analysis package.

Reference: AGENTS.md | Canonical Architecture Stage 4, Stage 5, & Stage 6.

Implements:
1. TemporalEncoder, TemporalEncoderOutput: BiLSTM sequence encoder.
2. TransitionAnalyzer, TransitionOutput: Emotion evolution, velocity (delta emotion),
   persistence, and transition representation.
"""

from sentiment_derivation.temporal.temporal_encoder import (
    TemporalEncoder,
    TemporalEncoderOutput,
)
from sentiment_derivation.temporal.transition_analyzer import (
    TransitionAnalyzer,
    TransitionOutput,
)

__all__ = [
    "TemporalEncoder",
    "TemporalEncoderOutput",
    "TransitionAnalyzer",
    "TransitionOutput",
]
