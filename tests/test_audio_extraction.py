"""Synthetic tests for FFmpeg audio extraction and WAV cache handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import wave

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

import sentiment_derivation.data.preprocessing as preprocessing


def _write_valid_wav(path: Path, sample_rate: int = 16000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * channels * 8)


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"synthetic video placeholder")
    return path


@pytest.fixture
def ffmpeg_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preprocessing.shutil, "which", lambda _binary: "fake-ffmpeg")


def test_extract_audio_success_writes_expected_wav(
    tmp_path: Path,
    video_file: Path,
    ffmpeg_available: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "audio" / "clip.wav"
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        _write_valid_wav(output, sample_rate=22050, channels=2)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(preprocessing.subprocess, "run", fake_run)

    assert preprocessing.extract_audio_from_video(
        video_file, output, target_sample_rate=22050, channels=2
    )
    assert preprocessing._wav_is_usable(output, 22050, 2)
    assert commands[0][commands[0].index("-ar") + 1] == "22050"
    assert commands[0][commands[0].index("-ac") + 1] == "2"


def test_extract_audio_nonzero_ffmpeg_result_removes_partial_output(
    tmp_path: Path,
    video_file: Path,
    ffmpeg_available: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "clip.wav"

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        output.write_bytes(b"partial output")
        return SimpleNamespace(returncode=1, stderr="synthetic ffmpeg failure")

    monkeypatch.setattr(preprocessing.subprocess, "run", fake_run)

    assert not preprocessing.extract_audio_from_video(video_file, output)
    assert not output.exists()


@pytest.mark.parametrize("output_mode", ["missing", "empty", "malformed"])
def test_extract_audio_rejects_missing_empty_or_malformed_output(
    tmp_path: Path,
    video_file: Path,
    ffmpeg_available: None,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: str,
) -> None:
    output = tmp_path / "clip.wav"

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        if output_mode == "empty":
            output.write_bytes(b"")
        elif output_mode == "malformed":
            output.write_bytes(b"not a wave file")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(preprocessing.subprocess, "run", fake_run)

    assert not preprocessing.extract_audio_from_video(video_file, output)
    assert not output.exists()


def test_extract_audio_exception_removes_partial_output(
    tmp_path: Path,
    video_file: Path,
    ffmpeg_available: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "clip.wav"

    def fake_run(_command: list[str], **_kwargs: object) -> None:
        output.write_bytes(b"partial output")
        raise OSError("synthetic process launch failure")

    monkeypatch.setattr(preprocessing.subprocess, "run", fake_run)

    assert not preprocessing.extract_audio_from_video(video_file, output)
    assert not output.exists()


def _build_manifest_with_audio(
    tmp_path: Path,
    audio_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_extraction: Callable[..., SimpleNamespace],
) -> dict:
    video_dir = tmp_path / "videos"
    video_dir.mkdir(exist_ok=True)
    video_file = video_dir / "clip.mp4"
    video_file.write_bytes(b"synthetic video placeholder")
    annotations = tmp_path / "labels.text"
    annotations.write_text("0,0,0,1,0,0,0,0,5.0\n", encoding="utf-8")
    monkeypatch.setattr(
        preprocessing,
        "inspect_media_file",
        lambda _path: {
            "valid": True,
            "width": 16,
            "height": 16,
            "fps": 25.0,
            "frame_count": 8,
            "duration_seconds": 0.32,
            "file_size_bytes": video_file.stat().st_size,
            "error": None,
        },
    )
    monkeypatch.setattr(preprocessing.shutil, "which", lambda _binary: "fake-ffmpeg")
    monkeypatch.setattr(preprocessing.subprocess, "run", run_extraction)

    return preprocessing.build_processed_manifest(
        dataset_root=tmp_path,
        annotation_files={"train": annotations},
        video_dirs={"train": video_dir},
        output_audio_dir=audio_dir,
        extract_audio=True,
        target_sample_rate=16000,
        audio_channels=1,
    )


def test_valid_cached_wav_is_reused_without_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio"
    cached = audio_dir / "train" / "clip.wav"
    _write_valid_wav(cached)

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("FFmpeg must not run for a valid cached WAV")

    manifest = _build_manifest_with_audio(
        tmp_path, audio_dir, monkeypatch, unexpected_run
    )

    assert manifest["samples"]["train"][0]["audio_path"] == str(cached)
    assert preprocessing._wav_is_usable(cached, 16000, 1)


def test_invalid_cached_wav_is_removed_before_extraction_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio"
    cached = audio_dir / "train" / "clip.wav"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"stale malformed cache")
    calls = 0

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert not cached.exists(), "invalid cache should be removed before retry"
        _write_valid_wav(cached)
        return SimpleNamespace(returncode=0, stderr="")

    manifest = _build_manifest_with_audio(tmp_path, audio_dir, monkeypatch, fake_run)

    assert calls == 1
    assert manifest["samples"]["train"][0]["audio_path"] == str(cached)
    assert preprocessing._wav_is_usable(cached, 16000, 1)


def test_extract_audio_ffmpeg_unavailable(
    tmp_path: Path,
    video_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "clip.wav"

    monkeypatch.setattr(
        preprocessing.shutil,
        "which",
        lambda _binary: None,
    )

    def unexpected_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("FFmpeg should not be launched")

    monkeypatch.setattr(preprocessing.subprocess, "run", unexpected_run)

    assert not preprocessing.extract_audio_from_video(video_file, output)


def test_wav_is_usable_rejects_truncated_audio(
    tmp_path: Path,
) -> None:
    output = tmp_path / "truncated.wav"

    _write_valid_wav(output)

    # Keep the WAV header but remove part of its audio data.
    data = output.read_bytes()
    output.write_bytes(data[:-4])

    assert not preprocessing._wav_is_usable(output, 16000, 1)


def test_extract_audio_cleanup_failure_is_handled(
    tmp_path: Path,
    video_file: Path,
    ffmpeg_available: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "clip.wav"

    def fake_run(
        *_command: list[str],
        **_kwargs: object,
    ) -> SimpleNamespace:
        output.write_bytes(b"partial output")
        return SimpleNamespace(
            returncode=1,
            stderr="synthetic FFmpeg failure",
        )

    monkeypatch.setattr(preprocessing.subprocess, "run", fake_run)

    original_unlink = Path.unlink

    def fail_output_unlink(
        self: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if self == output:
            raise OSError("synthetic permission error")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_output_unlink)

    assert not preprocessing.extract_audio_from_video(video_file, output)
