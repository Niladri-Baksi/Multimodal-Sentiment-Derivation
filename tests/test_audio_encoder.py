"""Audio unit tests. Audio decode behavior is mocked; CNN tests use the real module."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from sentiment_derivation.encoders import audio_encoder as audio_module


class FakeMelSpectrogram:
    def __init__(self, *, n_mels: int, hop_length: int, **_kwargs) -> None:
        self.n_mels = n_mels
        self.hop_length = hop_length

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        steps = max(1, waveform.shape[-1] // self.hop_length)
        image = waveform[:, None, None, :]
        return torch.nn.functional.interpolate(
            image, size=(self.n_mels, steps), mode="bilinear", align_corners=False
        ).squeeze(1).abs()


class FakeResample:
    def __init__(self, *, orig_freq: int, new_freq: int) -> None:
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        new_length = max(1, round(waveform.shape[-1] * self.new_freq / self.orig_freq))
        return torch.nn.functional.interpolate(
            waveform[:, None, :], size=new_length, mode="linear", align_corners=False
        ).squeeze(1)


def mock_audio_io(monkeypatch: pytest.MonkeyPatch, loader) -> None:
    monkeypatch.setattr(audio_module, "torchaudio", SimpleNamespace(load=loader))
    monkeypatch.setattr(
        audio_module,
        "T",
        SimpleNamespace(MelSpectrogram=FakeMelSpectrogram, Resample=FakeResample),
    )


@pytest.fixture
def encoder():
    return audio_module.AudioEncoder(target_steps=16, feature_dim=128).eval()


@pytest.mark.parametrize("batch_size,time_steps", [(1, 4), (3, 41), (2, 103)])
def test_audio_encoder_batch_shapes_dtype_finite_cpu(encoder, batch_size: int, time_steps: int) -> None:
    mel = torch.randn(batch_size, 1, 64, time_steps, dtype=torch.float32)
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
    mock_audio_io(monkeypatch, lambda _: (waveform, 16000))
    preprocessor = audio_module.AudioPreprocessor(
        sample_rate=16000, duration_sec=0.1, n_mels=64, n_fft=400, hop_length=160
    )
    features = preprocessor.process_file(str(tmp_path / "mock.wav"))
    assert features.ndim == 3 and features.shape[:2] == (1, 64)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_audio_preprocessor_surfaces_missing_or_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_audio_io(
        monkeypatch,
        lambda _: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises(RuntimeError, match="decode failed"):
        preprocessor.process_file(str(tmp_path / "corrupt.wav"))


def test_audio_preprocessor_rejects_missing_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_audio_io(
        monkeypatch,
        lambda _: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        preprocessor.process_file(str(tmp_path / "missing.wav"))


def test_audio_preprocessor_rejects_corrupt_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a wav file")
    mock_audio_io(
        monkeypatch,
        lambda _: (_ for _ in ()).throw(RuntimeError("invalid wav")),
    )
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        preprocessor.process_file(str(corrupt))


def test_audio_encoder_rejects_empty_batch(encoder) -> None:
    with pytest.raises(ValueError, match="empty"):
        encoder(torch.empty(0, 1, 64, 41))


def test_audio_encoder_rejects_wrong_rank(encoder) -> None:
    with pytest.raises(ValueError, match="expects"):
        encoder(torch.randn(1, 64, 41))


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 41), (1, 2, 64, 41), (1, 1, 64, 3)],
)
def test_audio_encoder_rejects_invalid_channel_or_time_dimensions(encoder, shape) -> None:
    with pytest.raises(ValueError, match="expects"):
        encoder(torch.randn(*shape))


def test_audio_encoder_rejects_non_finite_input(encoder) -> None:
    mel = torch.zeros(1, 1, 64, 41)
    mel[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinite"):
        encoder(mel)


def test_audio_preprocessor_handles_silent_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_audio_io(monkeypatch, lambda _: (torch.zeros(1, 1600), 16000))
    preprocessor = audio_module.AudioPreprocessor(
        sample_rate=16000, duration_sec=0.1, n_mels=64, n_fft=400, hop_length=160
    )
    features = preprocessor.process_file("silent.wav")
    assert features.shape[0:2] == (1, 64)
    assert torch.isfinite(features).all()


def test_audio_preprocessor_rejects_empty_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_audio_io(monkeypatch, lambda _: (torch.empty(1, 0), 16000))
    preprocessor = audio_module.AudioPreprocessor(duration_sec=0.1)
    with pytest.raises(ValueError, match="channels and samples"):
        preprocessor.process_file("empty.wav")


def test_audio_preprocessor_resamples_stereo_input(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_audio_io(monkeypatch, lambda _: (torch.randn(2, 800), 8000))
    preprocessor = audio_module.AudioPreprocessor(
        sample_rate=16000, duration_sec=0.1, n_mels=64, n_fft=400, hop_length=160
    )
    features = preprocessor.process_file("stereo.wav")
    assert features.shape[0:2] == (1, 64)
    assert torch.isfinite(features).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_audio_encoder_cuda_when_available(encoder) -> None:
    output = encoder.cuda()(torch.randn(2, 1, 64, 41, device="cuda"))
    assert output.device.type == "cuda"
    assert output.shape == (2, 16, 128)
    assert torch.isfinite(output).all()
