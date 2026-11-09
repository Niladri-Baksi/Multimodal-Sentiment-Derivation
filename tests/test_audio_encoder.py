"""Audio unit tests. Audio decode behavior is mocked; CNN tests use the real module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from sentiment_derivation.encoders import audio_encoder as audio_module
except Exception as exc:
    pytest.skip(f"Audio encoder runtime unavailable: {exc}", allow_module_level=True)


@pytest.fixture
def encoder():
    return audio_module.AudioEncoder(target_steps=16, feature_dim=128).eval()


@pytest.mark.parametrize("batch_size", [1, 3])
def test_audio_encoder_batch_shapes_dtype_finite_cpu(encoder, batch_size: int) -> None:
    mel = torch.randn(batch_size, 1, 64, 41, dtype=torch.float32)
    with torch.no_grad():
        features = encoder(mel)
    assert features.shape == (batch_size, 16, 128)
    assert features.dtype == torch.float32
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()


def test_audio_encoder_return_dict(encoder) -> None:
    output = encoder(torch.randn(1, 1, 64, 41), return_dict=True)
    assert output["features"].shape == (1, 16, 128)


def test_audio_preprocessor_valid_mocked_decode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mocked decode validates preprocessing math, not torchaudio file decoding."""
    waveform = torch.randn(2, 1600)
    monkeypatch.setattr(audio_module.torchaudio, "load", lambda _: (waveform, 16000))
    preprocessor = audio_module.AudioPreprocessor(
        sample_rate=16000, duration_sec=0.1, n_mels=64, n_fft=400, hop_length=160
    )
    features = preprocessor.process_file(str(tmp_path / "mock.wav"))
    assert features.ndim == 3 and features.shape[:2] == (1, 64)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


@pytest.mark.xfail(strict=True, reason="Known bug: load exceptions are replaced with silent audio.")
def test_audio_preprocessor_surfaces_missing_or_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio_module.torchaudio, "load", lambda _: (_ for _ in ()).throw(RuntimeError("decode failed")))
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises(RuntimeError, match="decode failed"):
        preprocessor.process_file(str(tmp_path / "corrupt.wav"))


@pytest.mark.xfail(strict=True, reason="Known bug: a missing path is converted to silent audio.")
def test_audio_preprocessor_rejects_missing_path(tmp_path: Path) -> None:
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        preprocessor.process_file(str(tmp_path / "missing.wav"))


@pytest.mark.xfail(strict=True, reason="Known bug: corrupt-file decode errors are converted to silent audio.")
def test_audio_preprocessor_rejects_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wav file")
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        preprocessor.process_file(str(corrupt))


@pytest.mark.xfail(strict=True, reason="Known gap: AudioEncoder does not validate empty batch/time dimensions.")
def test_audio_encoder_rejects_empty_batch(encoder) -> None:
    with pytest.raises(ValueError, match="empty"):
        encoder(torch.empty(0, 1, 64, 41))


def test_audio_encoder_rejects_wrong_rank(encoder) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        encoder(torch.randn(1, 64, 41))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_audio_encoder_cuda_when_available(encoder) -> None:
    output = encoder.cuda()(torch.randn(2, 1, 64, 41, device="cuda"))
    assert output.device.type == "cuda"
    assert output.shape == (2, 16, 128)
    assert torch.isfinite(output).all()
