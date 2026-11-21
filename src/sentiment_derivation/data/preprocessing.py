"""Preprocessing utilities for EmoReact dataset.

Provides reusable, modular, and beginner-friendly functions to:
1. Load and validate multi-label emotion and affective annotations.
2. Match sample identifiers with corresponding video and feature files.
3. Inspect and validate video and audio media files.
4. Extract standardized audio waveforms and uniform video frames.
5. Preserve original annotations without fabricating labels.
6. Generate a standardized, reproducible dataset manifest.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the local runtime environment
    cv2 = None  # type: ignore[assignment]

# Configure module-level logger
logger = logging.getLogger(__name__)



def _wav_is_usable(path: Path, sample_rate: int, channels: int) -> bool:
    """Check that a WAV contains complete, readable audio frames."""
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_count = wav_file.getnframes()

            if (
                frame_count <= 0
                or wav_file.getframerate() != sample_rate
                or wav_file.getnchannels() != channels
                or wav_file.getcomptype() != "NONE"
            ):
                return False

            expected_bytes = (
                frame_count
                * wav_file.getnchannels()
                * wav_file.getsampwidth()
            )

            audio_data = wav_file.readframes(frame_count)

            return len(audio_data) == expected_bytes

    except (EOFError, OSError, wave.Error):
        return False


# Canonical EmoReact emotion labels (ordered strictly by column index 0..7)
DEFAULT_EMOTION_NAMES: List[str] = [
    "Curiosity",     # Col 0
    "Uncertainty",   # Col 1
    "Excitement",    # Col 2
    "Happiness",     # Col 3
    "Surprise",      # Col 4
    "Disgust",       # Col 5
    "Fear",          # Col 6
    "Frustration",   # Col 7
]

_ANNOTATION_VIDEO_ASSOCIATION_METHOD = "sorted_video_stem_position"
_UNKNOWN_ANNOTATION_VIDEO_ASSOCIATION = {
    "method": "unknown",
    "verification_status": "unverified",
    "evidence": None,
}


def _normalize_recorded_annotation_video_association(
    value: Any, *, context: str
) -> Dict[str, Any]:
    """Validate provenance already recorded in a manifest.

    An omitted policy is handled by the caller as legacy unknown provenance.
    An explicitly present empty or partial mapping is malformed. Independent
    ``verified`` status is never a valid recorded value; preprocessing can only
    record a caller declaration as ``caller_declared_verified``.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} annotation/video provenance must be a mapping.")
    if not value:
        raise ValueError(
            f"{context} annotation/video provenance cannot be empty; omit it only "
            "for a legacy manifest."
        )

    required_keys = {"method", "verification_status", "evidence"}
    unknown_keys = set(value) - required_keys
    missing_keys = required_keys - set(value)
    if unknown_keys or missing_keys:
        details = []
        if missing_keys:
            details.append(f"missing fields {sorted(missing_keys)}")
        if unknown_keys:
            details.append(f"unsupported fields {sorted(map(str, unknown_keys))}")
        raise ValueError(
            f"{context} annotation/video provenance has " + " and ".join(details) + "."
        )

    method = value["method"]
    status = value["verification_status"]
    evidence = value["evidence"]
    if not isinstance(method, str) or method not in {
        "unknown",
        _ANNOTATION_VIDEO_ASSOCIATION_METHOD,
    }:
        raise ValueError(f"{context} annotation/video provenance has an unsupported method.")
    if not isinstance(status, str) or status not in {
        "unverified",
        "caller_declared_verified",
    }:
        raise ValueError(
            f"{context} annotation/video provenance status must be 'unverified' or "
            "'caller_declared_verified'; independent verification is not accepted."
        )
    if evidence is not None and (
        not isinstance(evidence, str) or not evidence.strip()
    ):
        raise ValueError(
            f"{context} annotation/video provenance evidence must be a non-empty string."
        )
    if status == "caller_declared_verified" and evidence is None:
        raise ValueError(
            f"{context} caller-declared verification requires a non-empty evidence reference."
        )
    return {
        "method": method,
        "verification_status": status,
        "evidence": evidence,
    }


def _normalize_manifest_association_provenance(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate manifest structure and normalize association provenance.

    Missing provenance is the sole legacy exception: the manifest defaults to
    unknown/unverified and samples without an explicit policy inherit that
    manifest policy. Every split and record must still have a valid structure.
    Explicit sample provenance takes precedence over the manifest policy.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest JSON root must be an object.")

    samples_by_split = manifest.get("samples")
    if not isinstance(samples_by_split, Mapping):
        raise ValueError("Manifest 'samples' must be an object mapping splits to lists.")

    manifest_policy = (
        _normalize_recorded_annotation_video_association(
            manifest["annotation_video_association"], context="Manifest-level"
        )
        if "annotation_video_association" in manifest
        else dict(_UNKNOWN_ANNOTATION_VIDEO_ASSOCIATION)
    )

    normalized_splits: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, records in samples_by_split.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("Manifest split names must be non-empty strings.")
        if not isinstance(records, list):
            raise ValueError(
                f"Manifest split {split_name!r} must contain a list of records."
            )
        normalized_records: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            context = f"Sample {index} in split {split_name!r}"
            if not isinstance(record, Mapping):
                raise ValueError(f"{context} record must be an object.")
            normalized_record = dict(record)
            if "annotation_video_association" in record:
                normalized_record["annotation_video_association"] = (
                    _normalize_recorded_annotation_video_association(
                        record["annotation_video_association"], context=context
                    )
                )
            else:
                normalized_record["annotation_video_association"] = dict(
                    manifest_policy
                )
            normalized_records.append(normalized_record)

        normalized_splits[split_name] = normalized_records

    normalized_manifest = dict(manifest)
    normalized_manifest["annotation_video_association"] = manifest_policy
    normalized_manifest["samples"] = normalized_splits
    return normalized_manifest


def _normalize_annotation_video_association_policy(
    association_policy: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate and normalize caller-supplied annotation/video provenance.

    The function records an explicit caller attestation; it cannot independently
    prove that a dataset's annotation row order matches its video filename order.
    """
    if association_policy is None:
        association_policy = {}
    if not isinstance(association_policy, Mapping):
        raise TypeError("association_policy must be a mapping when provided.")

    allowed_keys = {"method", "verification_status", "evidence"}
    unknown_keys = [key for key in association_policy if key not in allowed_keys]
    if unknown_keys:
        raise ValueError(
            "association_policy contains unsupported keys: "
            f"{sorted((repr(key) for key in unknown_keys))}."
        )

    method = association_policy.get("method", _ANNOTATION_VIDEO_ASSOCIATION_METHOD)
    if method != _ANNOTATION_VIDEO_ASSOCIATION_METHOD:
        raise ValueError(
            "Only the implemented association method "
            f"'{_ANNOTATION_VIDEO_ASSOCIATION_METHOD}' is supported."
        )

    status = association_policy.get("verification_status", "unverified")
    if not isinstance(status, str) or status not in {"unverified", "verified"}:
        raise ValueError(
            "association_policy.verification_status must be 'unverified' or 'verified'."
        )

    evidence = association_policy.get("evidence")
    if evidence is not None and (
        not isinstance(evidence, str) or not evidence.strip()
    ):
        raise ValueError(
            "association_policy.evidence must be a non-empty string when provided."
        )
    if status == "verified" and evidence is None:
        raise ValueError(
            "A verified association policy requires a non-empty evidence reference."
        )

    # Do not label a caller assertion as independently verified by this code.
    recorded_status = (
        "caller_declared_verified" if status == "verified" else "unverified"
    )
    return {
        "method": _ANNOTATION_VIDEO_ASSOCIATION_METHOD,
        "verification_status": recorded_status,
        "evidence": evidence,
    }


def load_annotations(
    file_path: Union[str, Path],
    emotion_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load and validate multi-label emotion annotations from an EmoReact label file.

    Each row in the annotation file contains 9 comma-separated values:
      - Columns 0 to 7: Binary emotion indicators (0.0 or 1.0)
      - Column 8: Continuous valence rating (typically 1.0 to 7.0)

    Args:
        file_path: Path to the annotation text file (e.g. train_labels.text).
        emotion_names: List of emotion names corresponding to columns 0..7.
            Defaults to DEFAULT_EMOTION_NAMES.

    Returns:
        List of dictionaries containing parsed emotion labels and valence.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a row has an unexpected column count or invalid values.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found at: {path}")

    if emotion_names is None:
        emotion_names = DEFAULT_EMOTION_NAMES

    expected_emotions_count = len(emotion_names)
    records: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Preserve empty cells so malformed rows cannot shift columns.
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != expected_emotions_count + 1:
                raise ValueError(
                    f"Line {line_num} in '{path.name}' has {len(parts)} columns, "
                    f"expected {expected_emotions_count + 1}."
                )

            try:
                # Parse binary emotion indicators
                emotions: Dict[str, int] = {}
                for idx, name in enumerate(emotion_names):
                    val = float(parts[idx])
                    if val not in (0.0, 1.0):
                        raise ValueError(
                            f"Line {line_num}, column {idx} ('{name}') has non-binary value: {val}"
                        )
                    emotions[name] = int(val)

                # Parse continuous valence
                valence = float(parts[expected_emotions_count])
                if not math.isfinite(valence) or not 1.0 <= valence <= 7.0:
                    raise ValueError(
                        f"Line {line_num}, valence must be finite and within [1, 7]; "
                        f"got {valence}."
                    )
            except ValueError as e:
                raise ValueError(
                    f"Failed parsing numerical values on line {line_num} of '{path.name}': {e}"
                ) from e

            records.append({
                "emotion_labels": emotions,
                "valence": valence,
            })

    return records


def get_split_sample_ids(video_dir: Union[str, Path]) -> List[str]:
    """Retrieve sorted sample IDs from the video directory for a split.

    Only valid video files (.mp4) are indexed. Temporary system files like
    'Thumbs.db' or hidden files are safely ignored.

    Args:
        video_dir: Directory containing raw video files for a split.

    Returns:
        Sorted list of sample IDs (filename stems, e.g. "80'sTOYS105_2").

    Raises:
        FileNotFoundError: If video_dir does not exist.
    """
    dir_path = Path(video_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Video directory not found at: {dir_path}")

    # Sorting gives deterministic positional pairing; it does not verify that
    # the annotation rows use this same ordering.
    video_files = sorted([
        f.stem for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    ])
    if len(video_files) != len(set(video_files)):
        raise ValueError(
            f"Video directory contains duplicate sample IDs (case-sensitive stems): {dir_path}"
        )

    return video_files


def match_annotations_with_media(
    sample_ids: List[str],
    annotations: List[Dict[str, Any]],
    *,
    association_policy: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Pair annotation rows with video IDs by sorted position.

    Annotation rows do not contain IDs, so this positional operation cannot
    establish identity by itself. Each returned record carries provenance
    metadata. By default it is explicitly marked unverified. A caller may
    supply a policy with a verification evidence reference; that is recorded
    as a caller declaration and is not independently checked here.

    Args:
        sample_ids: Video filename stems in the intended pairing order.
        annotations: List of annotation dictionaries loaded from label file.
        association_policy: Optional mapping with ``method`` equal to
            ``sorted_video_stem_position``, ``verification_status`` equal to
            ``unverified`` (default) or ``verified``, and an evidence reference
            required for ``verified``. A verified caller policy is recorded as
            ``caller_declared_verified`` to avoid implying independent proof.

    Returns:
        List of combined sample dictionaries.

    Raises:
        ValueError: If IDs are missing/invalid/duplicated, counts differ, or the
            association policy is malformed.
    """
    normalized_policy = _normalize_annotation_video_association_policy(
        association_policy
    )
    for index, sample_id in enumerate(sample_ids):
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"Video sample ID at position {index} must be a non-empty string."
            )
    if len(sample_ids) != len(annotations):
        raise ValueError(
            f"Count mismatch: {len(sample_ids)} sample IDs vs "
            f"{len(annotations)} annotation rows."
        )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique to avoid ambiguous annotation matching.")

    matched: List[Dict[str, Any]] = []
    for sample_id, ann in zip(sample_ids, annotations):
        item = {
            "sample_id": sample_id,
            "emotion_labels": ann["emotion_labels"],
            "valence": ann["valence"],
            "annotation_video_association": dict(normalized_policy),
        }
        matched.append(item)

    return matched



def inspect_media_file(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect video metadata and verify that at least one frame is readable."""

    path = Path(video_path)

    result = {
        "valid": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "file_size_bytes": 0,
        "error": None,
    }

    if not path.is_file():
        result["error"] = f"File does not exist: {path}"
        return result

    file_size = path.stat().st_size
    result["file_size_bytes"] = file_size

    if cv2 is None:
        result["error"] = (
            "OpenCV is unavailable; video readability could not be inspected"
        )
        return result

    cap = None

    try:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            result["error"] = "Failed to open video with OpenCV"
            return result

        width_value = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_value = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Reject invalid or non-finite metadata.
        if not all(
            math.isfinite(value)
            for value in (
                width_value,
                height_value,
                fps,
                frame_count_value,
            )
        ):
            result["error"] = "Video contains non-finite decoder metadata"
            return result

        width = int(width_value)
        height = int(height_value)
        frame_count = int(frame_count_value)

        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            result["error"] = "Video contains invalid dimensions, FPS, or frame count"
            return result

        # Metadata alone does not prove that the video is readable.
        # Attempt to decode at least one actual frame.
        frame_read, frame = cap.read()

        if not frame_read or frame is None:
            result["error"] = "Video decoder could not read a frame"
            return result

        duration = frame_count / fps

        result.update({
            "valid": True,
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
            "error": None,
        })

        return result

    except Exception as exc:
        result["error"] = f"Video inspection failed: {exc}"
        return result

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                logger.warning(
                    "Could not release video capture for '%s'.",
                    path,
                    exc_info=True,
                )



def extract_audio_from_video(
    video_path: Union[str, Path],
    output_wav_path: Union[str, Path],
    target_sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_binary: str = "ffmpeg",
) -> bool:
    """Extract a standardized WAV audio file from an MP4 video clip using FFmpeg.

    Args:
        video_path: Source MP4 video path.
        output_wav_path: Destination WAV audio path.
        target_sample_rate: Target sampling rate in Hz (default: 16000 Hz).
        channels: Target number of audio channels (default: 1 = mono).
        ffmpeg_binary: FFmpeg command or binary name.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    v_path = Path(video_path)
    out_path = Path(output_wav_path)

    if not v_path.is_file():
        logger.error(f"Cannot extract audio: video file missing: {v_path}")
        return False
    if (
        isinstance(target_sample_rate, bool)
        or not isinstance(target_sample_rate, int)
        or target_sample_rate <= 0
        or isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels <= 0
    ):
        raise ValueError("target_sample_rate and channels must be positive integers.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg is available
    if not shutil.which(ffmpeg_binary):
        logger.warning(f"FFmpeg binary '{ffmpeg_binary}' not found on system PATH.")
        return False

    cmd = [
        ffmpeg_binary,
        "-y",                     # Overwrite output without prompting
        "-v", "error",            # Suppress verbose banner and logging
        "-i", str(v_path),        # Input video file
        "-vn",                    # Disable video recording
        "-acodec", "pcm_s16le",   # Standard 16-bit PCM uncompressed WAV
        "-ar", str(target_sample_rate),
        "-ac", str(channels),
        str(out_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0 and _wav_is_usable(out_path, target_sample_rate, channels):
            return True
        logger.warning(
            f"FFmpeg audio extraction failed for '{v_path.name}': {res.stderr}"
        )
        if out_path.exists():
            out_path.unlink()
        return False
    except Exception as e:
        logger.error(f"Exception during FFmpeg execution for '{v_path.name}': {e}")
        try:
            out_path.unlink(missing_ok=True)
        except OSError as cleanup_error:
            logger.warning(
                f"Could not remove partial audio output '{out_path}': {cleanup_error}"
            )
        return False


def extract_sample_frames(
    video_path: Union[str, Path],
    num_frames: int = 16,
) -> List[np.ndarray]:
    """Uniformly sample a fixed number of RGB frames from a video clip.

    Args:
        video_path: Path to the MP4 video.
        num_frames: Number of evenly spaced frames to extract.

    Returns:
        List of RGB image arrays of shape (H, W, 3) as uint8.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer.")
    v_path = Path(video_path)
    if not v_path.is_file():
        return []
    if cv2 is None:
        raise ImportError("OpenCV is required to decode video frames.")

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        cap.release()
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames: List[np.ndarray] = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame_bgr = cap.read()
            if ret and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        return frames
    finally:
        cap.release()


def verify_visual_features(
    sample_id: str,
    feat_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Verify presence of OpenFace pre-extracted visual feature files for a sample.

    OpenFace produces three feature files per sample:
      - <sample_id>.params.txt (facial landmarks, pose, shape)
      - <sample_id>_au.txt (Action Units)
      - <sample_id>_gaze.txt (Gaze coordinates and vectors)

    Args:
        sample_id: Sample identifier stem.
        feat_dir: Directory containing OpenFace feature files for the split.

    Returns:
        Dictionary indicating availability and paths to each feature file.
    """
    d_path = Path(feat_dir)
    if not d_path.is_dir():
        return {
            "available": False,
            "params_path": None,
            "au_path": None,
            "gaze_path": None,
        }

    params_p = d_path / f"{sample_id}.params.txt"
    au_p = d_path / f"{sample_id}_au.txt"
    gaze_p = d_path / f"{sample_id}_gaze.txt"

    has_params = params_p.is_file()
    has_au = au_p.is_file()
    has_gaze = gaze_p.is_file()

    available = has_params and has_au and has_gaze

    return {
        "available": available,
        "params_path": str(params_p) if has_params else None,
        "au_path": str(au_p) if has_au else None,
        "gaze_path": str(gaze_p) if has_gaze else None,
    }


def build_standardized_sample(
    sample_id: str,
    split: str,
    video_path: Union[str, Path],
    emotion_labels: Dict[str, int],
    valence: float,
    audio_path: Optional[Union[str, Path]] = None,
    text: Optional[str] = None,
    visual_features: Optional[Dict[str, Any]] = None,
    media_metadata: Optional[Dict[str, Any]] = None,
    video_frame_paths: Optional[List[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """Construct a canonical, standardized sample dictionary.

    Follows the architectural contract defined in AGENTS.md.
    Preserves multi-label emotion indicators and continuous valence.

    Args:
        sample_id: Unique clip identifier.
        split: Partition identifier ('train', 'val', or 'test').
        video_path: Path to the raw video file.
        emotion_labels: Dict of 8 discrete binary emotion labels.
        valence: Continuous valence rating (1.0 to 7.0).
        audio_path: Optional path to extracted standardized WAV audio file.
        text: Optional transcript text (None if not provided in dataset).
        visual_features: Optional OpenFace feature paths.
        media_metadata: Optional dictionary with video duration, fps, resolution.

    Returns:
        Standardized sample dictionary.
    """
    v_path_str = str(video_path) if video_path else None
    a_path_str = str(audio_path) if audio_path else None

    # Check existence on disk
    video_exists = bool(v_path_str and Path(v_path_str).is_file())
    audio_file = Path(a_path_str) if a_path_str else None
    audio_exists = bool(audio_file and audio_file.is_file() and audio_file.stat().st_size > 0)
    text_exists = bool(text is not None and len(text.strip()) > 0)
    feat_available = bool(visual_features and visual_features.get("available", False))
    video_valid = bool(media_metadata.get("valid", True)) if media_metadata else True
    frame_paths = [str(path) for path in (video_frame_paths or [])]

    sample_dict: Dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "video_path": v_path_str,
        "video_frame_paths": frame_paths,
        "audio_path": a_path_str,
        "text": text,
        "emotion_labels": dict(emotion_labels),
        "other_original_annotations": {
            "valence": float(valence),
        },
        "modality_availability": {
            "video": video_exists and video_valid,
            "video_frames": bool(frame_paths),
            "audio": audio_exists,
            "text": text_exists,
            "visual_features": feat_available,
        },
        "visual_feature_paths": visual_features if visual_features else {
            "available": False,
            "params_path": None,
            "au_path": None,
            "gaze_path": None,
        },
        "metadata": media_metadata if media_metadata else {},
    }

    return sample_dict


def build_processed_manifest(
    dataset_root: Union[str, Path],
    annotation_files: Dict[str, Union[str, Path]],
    video_dirs: Dict[str, Union[str, Path]],
    visual_feature_dirs: Optional[Dict[str, Union[str, Path]]] = None,
    output_audio_dir: Optional[Union[str, Path]] = None,
    extract_audio: bool = False,
    extract_frames: bool = False,
    output_frames_dir: Optional[Union[str, Path]] = None,
    num_frames: int = 16,
    emotion_names: Optional[List[str]] = None,
    target_sample_rate: int = 16000,
    audio_channels: int = 1,
    association_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute the complete dataset validation and preprocessing pipeline.

    Processes all splits, matches annotations with media files, validates file
    integrity, extracts audio if requested, and generates a structured manifest.

    Args:
        dataset_root: Root directory of the dataset.
        annotation_files: Mapping of split name to label file path.
        video_dirs: Mapping of split name to video directory.
        visual_feature_dirs: Optional mapping of split name to OpenFace feature dir.
        output_audio_dir: Directory where extracted WAV files should be saved.
        extract_audio: Whether to extract WAV audio from videos.
        extract_frames: Whether to extract uniform video frames.
        output_frames_dir: Directory where extracted frame images should be saved.
        num_frames: Number of frames to sample if extract_frames is True.
        emotion_names: Ordered emotion label names.
        target_sample_rate: Sample rate for extracted WAV audio.
        audio_channels: Number of output audio channels.
        association_policy: Optional caller declaration for the positional
            annotation/video association. Defaults to unverified. A verified
            declaration requires an evidence reference and is recorded as
            caller-declared, not independently confirmed.

    Returns:
        Dictionary containing manifest header, split samples, and dataset statistics.
    """
    if emotion_names is None:
        emotion_names = DEFAULT_EMOTION_NAMES

    normalized_association_policy = (
        _normalize_annotation_video_association_policy(association_policy)
    )
    if normalized_association_policy["verification_status"] == "unverified":
        logger.warning(
            "Annotation/video pairing uses sorted video stems and annotation "
            "row positions; the association is unverified."
        )

    if extract_audio and output_audio_dir is None:
        raise ValueError("output_audio_dir is required when extract_audio is enabled.")
    if extract_frames and output_frames_dir is None:
        raise ValueError("output_frames_dir is required when extract_frames is enabled.")
    if extract_frames and (
        isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0
    ):
        raise ValueError("num_frames must be a positive integer when frame extraction is enabled.")

    manifest_samples: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    corrupted_or_missing: List[Dict[str, Any]] = []
    total_processed = 0

    # Ensure output directories exist if extraction is enabled
    if extract_audio and output_audio_dir:
        Path(output_audio_dir).mkdir(parents=True, exist_ok=True)
    if extract_frames and output_frames_dir:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    for split_key in ["train", "val", "test"]:
        ann_path = annotation_files.get(split_key)
        v_dir = video_dirs.get(split_key)
        f_dir = visual_feature_dirs.get(split_key) if visual_feature_dirs else None

        if not ann_path or not v_dir:
            logger.warning(f"Split '{split_key}' paths not fully configured; skipping.")
            continue

        ann_path = Path(ann_path)
        v_dir = Path(v_dir)

        # 1. Load annotations
        annotations = load_annotations(ann_path, emotion_names=emotion_names)

        # 2. Retrieve sorted sample IDs
        sample_ids = get_split_sample_ids(v_dir)
        video_paths_by_id = {
            path.stem: path
            for path in v_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        }

        # 3. Match annotations with sample IDs
        matched = match_annotations_with_media(
            sample_ids,
            annotations,
            association_policy=association_policy,
        )

        # 4. Process each sample
        for item in matched:
            sid = item["sample_id"]
            video_file = video_paths_by_id[sid]

            # Inspect video
            media_info = inspect_media_file(video_file)
            if not media_info["valid"]:
                corrupted_or_missing.append({
                    "sample_id": sid,
                    "split": split_key,
                    "issue": media_info.get("error", "Invalid video"),
                })

            # Check visual features
            feat_info = None
            if f_dir:
                feat_info = verify_visual_features(sid, f_dir)

            # Audio extraction if enabled
            audio_path = None
            if extract_audio and output_audio_dir:
                audio_dest = Path(output_audio_dir) / split_key / f"{sid}.wav"
                if not _wav_is_usable(audio_dest, target_sample_rate, audio_channels):
                    if audio_dest.is_file():
                        audio_dest.unlink()
                    audio_ok = extract_audio_from_video(
                        video_file,
                        audio_dest,
                        target_sample_rate=target_sample_rate,
                        channels=audio_channels,
                    )
                else:
                    audio_ok = True
                if audio_ok:
                    audio_path = str(audio_dest)
                else:
                    corrupted_or_missing.append({
                        "sample_id": sid,
                        "split": split_key,
                        "issue": "Audio extraction failed",
                    })

            # Frame extraction if enabled
            frame_paths: List[str] = []
            if extract_frames and output_frames_dir:
                frame_subdir = Path(output_frames_dir) / split_key / sid
                frame_subdir.mkdir(parents=True, exist_ok=True)
                frames = extract_sample_frames(video_file, num_frames=num_frames)
                for f_idx, frame in enumerate(frames):
                    frame_path = frame_subdir / f"frame_{f_idx:02d}.jpg"
                    if not frame_path.is_file() or frame_path.stat().st_size == 0:
                        written = cv2.imwrite(
                            str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        )
                        if not written:
                            continue
                    frame_paths.append(str(frame_path))
                if not frame_paths:
                    corrupted_or_missing.append({
                        "sample_id": sid,
                        "split": split_key,
                        "issue": "Video frame extraction failed",
                    })

            # Build standardized sample
            sample = build_standardized_sample(
                sample_id=sid,
                split=split_key,
                video_path=video_file,
                emotion_labels=item["emotion_labels"],
                valence=item["valence"],
                audio_path=audio_path,
                text=None,  # Not provided in EmoReact V1.0
                visual_features=feat_info,
                media_metadata=media_info,
                video_frame_paths=frame_paths,
            )
            sample["annotation_video_association"] = dict(
                item["annotation_video_association"]
            )

            manifest_samples[split_key].append(sample)
            total_processed += 1

    # Aggregate dataset statistics
    stats = compute_manifest_statistics(manifest_samples, emotion_names=emotion_names)

    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "dataset_name": "EmoReact_V_1.0",
        "dataset_root": str(dataset_root),
        "emotion_names": emotion_names,
        "valence_range": [1.0, 7.0],
        "annotation_video_association": normalized_association_policy,
        "statistics": stats,
        "corrupted_or_missing": corrupted_or_missing,
        "samples": manifest_samples,
    }

    return manifest


def compute_manifest_statistics(
    manifest_samples: Dict[str, List[Dict[str, Any]]],
    emotion_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute summary counts, emotion frequencies, and valence distribution across splits.

    Args:
        manifest_samples: Dictionary mapping split names to lists of sample dictionaries.
        emotion_names: List of emotion categories.

    Returns:
        Summary statistics dictionary.
    """
    if emotion_names is None:
        emotion_names = DEFAULT_EMOTION_NAMES

    all_samples: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}

    for split_name, samples in manifest_samples.items():
        split_counts[split_name] = len(samples)
        all_samples.extend(samples)

    total_samples = len(all_samples)
    if total_samples == 0:
        return {
            "total_samples": 0,
            "split_counts": split_counts,
            "emotion_counts": {e: 0 for e in emotion_names},
            "valence": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
        }

    # Emotion frequencies
    emotion_counts: Dict[str, int] = {e: 0 for e in emotion_names}
    valences: List[float] = []
    active_labels_per_sample: List[int] = []

    for s in all_samples:
        labels = s.get("emotion_labels", {})
        active = 0
        for e in emotion_names:
            if labels.get(e, 0) == 1:
                emotion_counts[e] += 1
                active += 1
        active_labels_per_sample.append(active)

        other = s.get("other_original_annotations", {})
        if "valence" in other:
            valences.append(float(other["valence"]))

    mean_valence = float(np.mean(valences)) if valences else 0.0
    std_valence = float(np.std(valences)) if valences else 0.0
    min_valence = float(np.min(valences)) if valences else 0.0
    max_valence = float(np.max(valences)) if valences else 0.0

    # Co-occurrence distribution (how many samples have k active labels)
    co_occurrence: Dict[int, int] = {}
    for k in range(len(emotion_names) + 1):
        co_occurrence[k] = int(sum(1 for a in active_labels_per_sample if a == k))

    # Modality availability
    modality_counts: Dict[str, int] = {"video": 0, "audio": 0, "text": 0, "visual_features": 0}
    for s in all_samples:
        avail = s.get("modality_availability", {})
        for mod, val in avail.items():
            if val:
                modality_counts[mod] = modality_counts.get(mod, 0) + 1

    # Video profile statistics (resolutions & FPS)
    resolutions: Dict[str, int] = {}
    fps_counts: Dict[str, int] = {}
    for s in all_samples:
        meta = s.get("metadata", {})
        w = meta.get("width", 0)
        h = meta.get("height", 0)
        if w > 0 and h > 0:
            res_str = f"{w}x{h}"
            resolutions[res_str] = resolutions.get(res_str, 0) + 1
        fps_val = meta.get("fps")
        if fps_val is not None and fps_val > 0:
            fps_str = str(round(fps_val, 3))
            fps_counts[fps_str] = fps_counts.get(fps_str, 0) + 1

    # Train split specific emotion distribution and BCE pos_weight ratios
    train_samples = manifest_samples.get("train", [])
    train_total = len(train_samples)
    train_emotion_counts: Dict[str, int] = {e: 0 for e in emotion_names}
    for s in train_samples:
        labels = s.get("emotion_labels", {})
        for e in emotion_names:
            if labels.get(e, 0) == 1:
                train_emotion_counts[e] += 1

    train_emotion_stats: Dict[str, Dict[str, Any]] = {}
    for e in emotion_names:
        c = train_emotion_counts[e]
        train_emotion_stats[e] = {
            "count": c,
            "percentage": round(c / train_total * 100, 2) if train_total > 0 else 0.0,
            "pos_weight": round((train_total - c) / max(c, 1), 4) if train_total > 0 else 1.0,
        }

    return {
        "total_samples": total_samples,
        "split_counts": split_counts,
        "emotion_frequencies": {
            e: {
                "count": count,
                "percentage": round(count / total_samples * 100, 2),
                "pos_weight": round((total_samples - count) / max(count, 1), 4),
            }
            for e, count in emotion_counts.items()
        },
        "train_split_emotion_stats": train_emotion_stats,
        "valence_statistics": {
            "min": round(min_valence, 4),
            "max": round(max_valence, 4),
            "mean": round(mean_valence, 4),
            "std": round(std_valence, 4),
        },
        "active_emotions_histogram": co_occurrence,
        "modality_availability": modality_counts,
        "video_profile": {
            "resolutions": resolutions,
            "fps_distribution": fps_counts,
        },
    }


def save_manifest(
    manifest: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """Validate and serialize a dataset manifest to a JSON file.

    Present association provenance must use the complete recorded schema.
    Omitted provenance remains compatible with legacy manifests and is
    materialized as unknown/unverified when loaded.

    Args:
        manifest: Manifest dictionary.
        output_path: Destination JSON file path.
    """
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping.")
    normalized_manifest = _normalize_manifest_association_provenance(manifest)
    # Serialize before touching the destination so invalid/non-serializable
    # input cannot create directories or leave a partial manifest behind.
    serialized = json.dumps(normalized_manifest, indent=2)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out_path.parent,
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
        os.replace(temporary_path, out_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    logger.info(f"Manifest successfully saved to: {out_path}")


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and normalize a processed dataset manifest from JSON.

    Legacy manifests without association provenance receive explicit
    unknown/unverified policy at manifest and sample level. Present empty,
    partial, malformed, or independently ``verified`` provenance is rejected.

    Args:
        manifest_path: Path to the JSON manifest.

    Returns:
        Manifest dictionary.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If the JSON root or present provenance is invalid.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, Mapping):
        raise ValueError("Manifest JSON root must be an object.")
    return _normalize_manifest_association_provenance(loaded)

