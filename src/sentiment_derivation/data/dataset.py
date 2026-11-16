"""Manifest-backed dataset for encoder-ready multimodal samples."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the local runtime environment
    cv2 = None  # type: ignore[assignment]

from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    _UNKNOWN_ANNOTATION_VIDEO_ASSOCIATION,
    _normalize_recorded_annotation_video_association,
    extract_sample_frames,
    load_manifest,
)

logger = logging.getLogger(__name__)

def _normalize_association_provenance(
    value: Any, *, context: str
) -> dict[str, Optional[str]]:
    """Validate provenance without inferring verification from missing fields.

    Legacy manifests with no policy are represented explicitly as unknown and
    unverified. This package can preserve a caller declaration, but cannot
    establish independently verified dataset provenance.
    """
    return _normalize_recorded_annotation_video_association(value, context=context)


class ManifestDataset(Dataset[dict[str, Any]]):
    """Read one manifest split and prepare sample inputs for modality encoders.

    Video frames are decoded or loaded from optional pre-extracted frame paths,
    transformed to ``[T,C,H,W]``, and labels are returned as tensors. Audio is
    converted to log-mel features by an injected processor or the project's
    ``AudioPreprocessor`` when an audio path is present. Text is preserved as
    a transcript unless an optional text preprocessor is supplied.

    Invalid required video samples raise at access time by default. Set
    ``invalid_sample_policy="skip"`` to exclude records whose video is absent
    or marked invalid by preprocessing metadata.
    """

    def __init__(
        self,
        manifest: Union[str, Path, Mapping[str, Any]],
        split: str,
        num_frames: int = 16,
        image_size: tuple[int, int] = (224, 224),
        video_transform: Optional[Callable[[list[np.ndarray]], torch.Tensor]] = None,
        audio_preprocessor: Optional[Any] = None,
        text_preprocessor: Optional[Any] = None,
        invalid_sample_policy: str = "raise",
    ) -> None:
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if invalid_sample_policy not in {"raise", "skip"}:
            raise ValueError("invalid_sample_policy must be 'raise' or 'skip'.")

        if isinstance(manifest, Mapping):
            manifest_data = dict(manifest)
        else:
            manifest_data = load_manifest(manifest)
        samples_by_split = manifest_data.get("samples")
        if not isinstance(samples_by_split, Mapping):
            raise ValueError("Manifest must contain a 'samples' mapping by split.")
        if split not in samples_by_split:
            raise ValueError(f"Manifest has no {split!r} split.")
        if split not in {"train", "val", "validation", "test"}:
            raise ValueError(f"Unsupported split name: {split!r}.")

        manifest_association = (
            _normalize_association_provenance(
                manifest_data["annotation_video_association"],
                context="Manifest-level",
            )
            if "annotation_video_association" in manifest_data
            else dict(_UNKNOWN_ANNOTATION_VIDEO_ASSOCIATION)
        )

        self._validate_split_leakage(samples_by_split)
        for split_name, split_records in samples_by_split.items():
            if not isinstance(split_records, list):
                continue  # _validate_split_leakage reports malformed split records.
            for index, record in enumerate(split_records):
                if isinstance(record, Mapping) and "annotation_video_association" in record:
                    _normalize_association_provenance(
                        record["annotation_video_association"],
                        context=f"Sample {index} in split {split_name!r}",
                    )
        records = samples_by_split[split]
        if not isinstance(records, list):
            raise ValueError(f"Manifest split {split!r} must be a list of samples.")
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        emotion_names = manifest_data.get("emotion_names", DEFAULT_EMOTION_NAMES)
        if not isinstance(emotion_names, list) or any(
            not isinstance(name, str) for name in emotion_names
        ):
            raise ValueError("Manifest emotion_names must be a list of strings.")
        self.emotion_names = list(emotion_names)
        if not self.emotion_names or len(self.emotion_names) != len(set(self.emotion_names)):
            raise ValueError("Manifest emotion_names must be a non-empty list of unique labels.")
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.invalid_sample_policy = invalid_sample_policy
        self.skipped_samples: list[str] = []
        self.samples: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"Sample at index {index} in {split!r} is not a mapping.")
            association = (
                _normalize_association_provenance(
                    record["annotation_video_association"],
                    context=f"Sample {index} in split {split!r}",
                )
                if "annotation_video_association" in record
                else dict(manifest_association)
            )
            record_split = record.get("split")
            if record_split is not None and record_split != split:
                raise ValueError(
                    f"Sample {record['sample_id']!r} is stored under {split!r} "
                    f"but declares split {record_split!r}."
                )
            if invalid_sample_policy == "skip" and not self._has_valid_video(record):
                self.skipped_samples.append(record["sample_id"])
                continue
            normalized_record = dict(record)
            normalized_record["annotation_video_association"] = association
            self.samples.append(normalized_record)

        if video_transform is None:
            # Import torchvision-backed transforms only when a Dataset instance
            # needs the default transform.
            try:
                from sentiment_derivation.data.transforms import VideoClipTransform
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "torchvision is required for the default video transform; "
                    "provide video_transform explicitly if using a custom transform."
                ) from exc

            self.video_transform = VideoClipTransform(image_size=image_size)
        else:
            self.video_transform = video_transform

        if self.skipped_samples:
            logger.warning(
                "Skipped %d invalid video sample(s) in split %s: %s",
                len(self.skipped_samples),
                split,
                ", ".join(self.skipped_samples[:10]),
            )

    @staticmethod
    def _validate_split_leakage(samples_by_split: Mapping[str, Any]) -> None:
        owners: dict[str, str] = {}
        seen_in_split: set[tuple[str, str]] = set()
        for split_name, records in samples_by_split.items():
            if not isinstance(records, list):
                raise ValueError(f"Manifest split {split_name!r} must be a list of samples.")
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"Sample record at index {index} in split {split_name!r} must be a mapping."
                    )
                if "sample_id" not in record:
                    raise ValueError(
                        f"Sample at index {index} in split {split_name!r} is missing sample_id."
                    )
                sample_id = record["sample_id"]
                if not isinstance(sample_id, str):
                    raise ValueError(
                        f"sample_id at index {index} in split {split_name!r} must be a string."
                    )
                if not sample_id.strip():
                    raise ValueError(
                        f"sample_id at index {index} in split {split_name!r} must contain "
                        "at least one non-whitespace character."
                    )
                split_key = (str(split_name), sample_id)
                if split_key in seen_in_split:
                    raise ValueError(
                        f"Sample ID {sample_id!r} is duplicated within split {split_name!r}."
                    )
                seen_in_split.add(split_key)
                prior_split = owners.get(sample_id)
                if prior_split is not None and prior_split != split_name:
                    raise ValueError(
                        f"Sample ID {sample_id!r} occurs in both {prior_split!r} "
                        f"and {split_name!r}; split leakage is possible."
                    )
                owners[sample_id] = str(split_name)

    @staticmethod
    def _has_valid_video(sample: Mapping[str, Any]) -> bool:
        video_path = sample.get("video_path")
        metadata = sample.get("metadata") or {}
        availability = sample.get("modality_availability") or {}
        if not isinstance(metadata, Mapping) or not isinstance(availability, Mapping):
            return False
        path_is_valid = bool(
            video_path
            and Path(video_path).is_file()
            and metadata.get("valid", True)
            and availability.get("video", True)
        )
        frame_paths = sample.get("video_frame_paths") or []
        frames_are_valid = bool(frame_paths) and all(Path(path).is_file() for path in frame_paths)
        return path_is_valid or frames_are_valid

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frames(self, sample: Mapping[str, Any]) -> list[np.ndarray]:
        paths = sample.get("video_frame_paths") or []
        if paths:
            if cv2 is None:
                raise ImportError("OpenCV is required to load pre-extracted image frames.")
            frames: list[np.ndarray] = []
            for frame_path in paths:
                frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise ValueError(
                        f"Unreadable frame for sample {sample.get('sample_id')!r}: {frame_path}"
                    )
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            video_path = sample.get("video_path")
            if not video_path or not Path(video_path).is_file():
                raise FileNotFoundError(
                    f"Video file is missing for sample {sample.get('sample_id')!r}: {video_path}"
                )
            frames = extract_sample_frames(video_path, num_frames=self.num_frames)

        if not frames:
            raise ValueError(f"No readable video frames for sample {sample.get('sample_id')!r}.")

        # Ensure fixed temporal length even if decoding had to omit corrupt frames.
        if len(frames) != self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[int(index)] for index in indices]
        return frames

    def _process_audio(self, audio_path: str) -> torch.Tensor:
        if self.audio_preprocessor is None:
            try:
                from sentiment_derivation.encoders.audio_encoder import AudioPreprocessor
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "Audio is present but torchaudio preprocessing is unavailable; "
                    "provide audio_preprocessor explicitly."
                ) from exc
            self.audio_preprocessor = AudioPreprocessor()

        processor = self.audio_preprocessor
        result = (
            processor.process_file(audio_path)
            if hasattr(processor, "process_file")
            else processor(audio_path)
        )
        if not isinstance(result, torch.Tensor) or result.ndim != 3:
            raise ValueError("Audio preprocessing must return a Tensor shaped [1,M,T].")
        if result.shape[0] != 1 or result.shape[1] <= 0 or result.shape[2] <= 0:
            raise ValueError("Audio features must have shape [1,M,T] with positive dimensions.")
        return result.to(dtype=torch.float32)

    def _process_text(self, text: Optional[str]) -> Optional[dict[str, torch.Tensor]]:
        if text is None or self.text_preprocessor is None:
            return None
        processor = self.text_preprocessor
        result = (
            processor.process_text(text)
            if hasattr(processor, "process_text")
            else processor(text)
        )
        if not isinstance(result, Mapping) or not {
            "input_ids", "attention_mask"
        }.issubset(result):
            raise ValueError("Text preprocessing must return input_ids and attention_mask.")
        return {
            "input_ids": torch.as_tensor(result["input_ids"], dtype=torch.long).reshape(-1),
            "attention_mask": torch.as_tensor(
                result["attention_mask"], dtype=torch.long
            ).reshape(-1),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        sample_id = sample["sample_id"]
        if not self._has_valid_video(sample):
            if self.invalid_sample_policy == "skip":
                raise RuntimeError("Invalid-sample filtering invariant was violated.")
            raise ValueError(f"Sample {sample_id!r} has no valid video input.")

        frames = self._read_frames(sample)
        video = self.video_transform(frames)
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError("video_transform must return a Tensor shaped [T,C,H,W].")
        if video.shape[0] != self.num_frames or video.shape[1] != 3:
            raise ValueError(
                f"video_transform returned {tuple(video.shape)}; expected "
                f"[{self.num_frames},3,H,W]."
            )

        audio_path = sample.get("audio_path")
        availability = sample.get("modality_availability") or {}
        audio_available = bool(
            availability.get("audio", True)
            and audio_path
            and Path(audio_path).is_file()
            and Path(audio_path).stat().st_size > 0
        )
        audio = self._process_audio(str(audio_path)) if audio_available else None

        text = sample.get("text")
        text = str(text) if text is not None else None
        text_available = bool(
            availability.get("text", True) and text and text.strip()
        )
        text_features = self._process_text(text) if text_available else None

        labels = sample.get("emotion_labels")
        if not isinstance(labels, Mapping) or any(name not in labels for name in self.emotion_names):
            raise ValueError(f"Sample {sample_id!r} is missing one or more emotion labels.")
        try:
            label_values = [float(labels[name]) for name in self.emotion_names]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Sample {sample_id!r} has non-numeric emotion labels.") from exc
        if any(value not in (0.0, 1.0) for value in label_values):
            raise ValueError(f"Sample {sample_id!r} has non-binary emotion labels.")

        annotations = sample.get("other_original_annotations") or {}
        try:
            valence = float(annotations["valence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Sample {sample_id!r} is missing a numeric valence label.") from exc
        if not np.isfinite(valence) or not 1.0 <= valence <= 7.0:
            raise ValueError(f"Sample {sample_id!r} has valence outside finite [1, 7].")

        return {
            "sample_id": sample_id,
            "split": str(sample.get("split", self.split)),
            "video": video.to(dtype=torch.float32),
            "audio": audio,
            "text": text,
            "text_features": text_features,
            "emotion_labels": torch.tensor(label_values, dtype=torch.float32),
            "valence": torch.tensor(valence, dtype=torch.float32),
            "modality_availability": {
                "video": True,
                "audio": audio_available,
                "text": text_available,
                "video_frames": bool(sample.get("video_frame_paths")),
                "visual_features": bool(
                    isinstance(sample.get("visual_feature_paths"), Mapping)
                    and sample["visual_feature_paths"].get("available", False)
                ),
            },
            "video_path": sample.get("video_path"),
            "audio_path": audio_path if audio_available else None,
            "visual_feature_paths": sample.get("visual_feature_paths") or {},
            "metadata": sample.get("metadata") or {},
            "annotation_video_association": dict(
                sample["annotation_video_association"]
            ),
            "annotation_video_association_status": sample[
                "annotation_video_association"
            ]["verification_status"],
            "annotation_video_association_evidence": sample[
                "annotation_video_association"
            ]["evidence"],
        }
