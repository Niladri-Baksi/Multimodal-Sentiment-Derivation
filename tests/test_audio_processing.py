from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.encoders.audio_encoder import AudioPreprocessor


def test_unreadable_audio_fails_instead_of_returning_silent_features() -> None:
    preprocessor = AudioPreprocessor(duration_sec=0.1)
    with patch(
        "sentiment_derivation.encoders.audio_encoder.torchaudio.load",
        side_effect=RuntimeError("decoder failed"),
    ):
        with pytest.raises(RuntimeError, match="decoder failed"):
            preprocessor.process_file("unreadable.wav")
