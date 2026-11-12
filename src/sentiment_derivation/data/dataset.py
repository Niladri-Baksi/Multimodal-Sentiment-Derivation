"""Manifest-backed dataset for video/audio model training and evaluation."""

from __future__ import annotations

import math
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sentiment_derivation.data.preprocessing import DEFAULT_EMOTION_NAMES, load_manifest
from sentiment_derivation.data.transforms import VideoClipTransform
from sentiment_derivation.encoders.audio_encoder import AudioPreprocessor
from sentiment_derivation.encoders.text_encoder import TextPreprocessor
from sentiment_derivation.sentiment.sentiment_head import valence_to_sentiment_label


class EmoReactDataset(Dataset[dict[str, Any]]):
    """Load decoded video frames, extracted audio, and labels from a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        num_frames: int = 16,
        image_size: tuple[int, int] = (224, 224),
        audio_duration_sec: float = 4.0,
        train: bool = False,
        include_text: bool = False,
        text_model_name: str = "distilbert-base-uncased",
        text_max_length: int = 48,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}; expected train, val, or test.")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root_dir = self.manifest_path.parent
        manifest = load_manifest(self.manifest_path)
        samples = manifest.get("samples", {}).get(split)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Manifest has no samples for split {split!r}.")
        self.samples = samples
        self.split = split
        self.num_frames = num_frames
        self.transform = VideoClipTransform(image_size=image_size, is_train=train)
        self.audio_preprocessor = AudioPreprocessor(duration_sec=audio_duration_sec)
        self.text_preprocessor = (
            TextPreprocessor(model_name=text_model_name, max_length=text_max_length)
            if include_text
            else None
        )
        self.emotion_names = tuple(DEFAULT_EMOTION_NAMES)

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, value: Any, kind: str, sample_id: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            action = (
                "Run scripts/prepare_data.py --extract-audio"
                if kind == "audio"
                else "Check the manifest video_path"
            )
            raise FileNotFoundError(f"Sample {sample_id!r} has no {kind} path. {action}.")
        path = Path(value).expanduser()
        if PureWindowsPath(value).is_absolute() and not path.is_absolute():
            raise FileNotFoundError(
                f"Sample {sample_id!r} {kind} path is a Windows absolute path unavailable "
                f"on this platform: {value}. Rebuild the manifest from the local dataset."
            )
        if not path.is_absolute():
            path = self.root_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Sample {sample_id!r} {kind} file does not exist: {path}. "
                "Check the manifest paths and preprocessing output."
            )
        return path

    def _read_video(self, path: Path, sample_id: str) -> torch.Tensor:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video for sample {sample_id!r}: {path}")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise RuntimeError(f"Video has no decodable frames for sample {sample_id!r}: {path}")
            indices = np.linspace(0, frame_count - 1, self.num_frames).round().astype(int)
            frames: list[np.ndarray] = []
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(
                        f"Failed decoding frame {frame_index} for sample {sample_id!r}: {path}"
                    )
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        return self.transform(frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: Mapping[str, Any] = self.samples[index]
        sample_id = str(sample.get("sample_id", f"{self.split}_{index}"))
        video_path = self._resolve_path(sample.get("video_path"), "video", sample_id)
        audio_path = self._resolve_path(sample.get("audio_path"), "audio", sample_id)

        labels = sample.get("emotion_labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"Sample {sample_id!r} has no emotion_labels mapping.")
        missing_labels = [name for name in self.emotion_names if name not in labels]
        if missing_labels:
            raise ValueError(
                f"Sample {sample_id!r} is missing emotion labels: {', '.join(missing_labels)}"
            )
        emotion_targets = torch.tensor(
            [float(labels[name]) for name in self.emotion_names], dtype=torch.float32
        )
        if not bool(torch.all((emotion_targets == 0) | (emotion_targets == 1))):
            raise ValueError(f"Sample {sample_id!r} emotion targets must be binary.")

        annotations = sample.get("other_original_annotations", {})
        try:
            valence = float(annotations["valence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Sample {sample_id!r} has invalid or missing valence.") from error
        if not math.isfinite(valence) or not 1.0 <= valence <= 7.0:
            raise ValueError(
                f"Sample {sample_id!r} valence must be finite and in [1.0, 7.0]; "
                f"received {valence}."
            )
        sentiment_class, _, _ = valence_to_sentiment_label(valence)

        try:
            audio = self.audio_preprocessor.process_file(str(audio_path))
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Failed processing audio for sample {sample_id!r}: {audio_path}") from error

        result = {
            "sample_id": sample_id,
            "frames": self._read_video(video_path, sample_id),
            "audio": audio,
            "emotion_labels": emotion_targets,
            "valence": torch.tensor(valence, dtype=torch.float32),
            "sentiment_class": torch.tensor(sentiment_class, dtype=torch.long),
        }
        if self.text_preprocessor is not None:
            text = sample.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"Sample {sample_id!r} text must be a string or null.")
            text_available = isinstance(text, str) and bool(text.strip())
            tokens = self.text_preprocessor.process_text(text if text_available else "")
            result.update(tokens)
            result["text_available"] = torch.tensor(text_available, dtype=torch.bool)
        return result
