from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.data.dataset import EmoReactDataset


def test_windows_absolute_manifest_paths_fail_with_rebuild_guidance(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": {
                    "train": [
                        {
                            "sample_id": "clip",
                            "video_path": r"C:\dataset\clip.mp4",
                            "audio_path": r"C:\dataset\clip.wav",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    dataset = EmoReactDataset(manifest_path, "train", num_frames=1)

    with pytest.raises(FileNotFoundError, match="Windows absolute path"):
        dataset[0]


@pytest.mark.parametrize("valence", [float("nan"), float("inf"), 0.5, 7.5])
def test_dataset_rejects_invalid_manifest_valence(tmp_path: Path, valence: float) -> None:
    video_path = tmp_path / "clip.mp4"
    audio_path = tmp_path / "clip.wav"
    video_path.touch()
    audio_path.touch()
    labels = {
        "Curiosity": 0,
        "Uncertainty": 0,
        "Excitement": 0,
        "Happiness": 0,
        "Surprise": 0,
        "Disgust": 0,
        "Fear": 0,
        "Frustration": 0,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": {
                    "train": [
                        {
                            "sample_id": "clip",
                            "video_path": str(video_path),
                            "audio_path": str(audio_path),
                            "emotion_labels": labels,
                            "other_original_annotations": {"valence": valence},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    dataset = EmoReactDataset(manifest_path, "train", num_frames=1)

    with pytest.raises(ValueError, match="valence must be finite and in"):
        dataset[0]


def test_dataset_tokenizes_optional_text_and_marks_missing_transcripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "clip.mp4"
    audio_path = tmp_path / "clip.wav"
    video_path.touch()
    audio_path.touch()
    labels = {name: 0 for name in (
        "Curiosity", "Uncertainty", "Excitement", "Happiness",
        "Surprise", "Disgust", "Fear", "Frustration",
    )}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"samples": {"train": [{
            "sample_id": "clip",
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "emotion_labels": labels,
            "other_original_annotations": {"valence": 4.0},
            "text": None,
        }]}}),
        encoding="utf-8",
    )

    class FakeTextPreprocessor:
        def __init__(self, **_kwargs) -> None:
            pass

        def process_text(self, text: str) -> dict[str, torch.Tensor]:
            assert text == ""
            return {
                "input_ids": torch.tensor([1, 2, 0]),
                "attention_mask": torch.tensor([1, 1, 0]),
            }

    monkeypatch.setattr("sentiment_derivation.data.dataset.TextPreprocessor", FakeTextPreprocessor)
    monkeypatch.setattr(
        "sentiment_derivation.data.dataset.AudioPreprocessor.process_file",
        lambda _self, _path: torch.zeros(1, 64, 8),
    )
    dataset = EmoReactDataset(manifest_path, "train", num_frames=1, include_text=True)
    monkeypatch.setattr(dataset, "_read_video", lambda _path, _sample_id: torch.zeros(1, 3, 8, 8))

    sample = dataset[0]

    assert sample["input_ids"].tolist() == [1, 2, 0]
    assert sample["attention_mask"].tolist() == [1, 1, 0]
    assert sample["text_available"].item() is False
