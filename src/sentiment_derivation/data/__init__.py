"""Data loading, preprocessing, and dataset management."""

from typing import Any

from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    load_annotations,
    get_split_sample_ids,
    match_annotations_with_media,
    inspect_media_file,
    extract_audio_from_video,
    extract_sample_frames,
    verify_visual_features,
    build_standardized_sample,
    build_processed_manifest,
    save_manifest,
    load_manifest,
)

__all__ = [
    "DEFAULT_EMOTION_NAMES",
    "load_annotations",
    "get_split_sample_ids",
    "match_annotations_with_media",
    "inspect_media_file",
    "extract_audio_from_video",
    "extract_sample_frames",
    "verify_visual_features",
    "build_standardized_sample",
    "build_processed_manifest",
    "save_manifest",
    "load_manifest",
    "ManifestDataset",
    "MODALITY_ORDER",
    "collate_manifest_samples",
    "create_dataloader",
]


def __getattr__(name: str) -> Any:
    """Load torch-backed data APIs only when a caller requests them."""
    if name == "ManifestDataset":
        from sentiment_derivation.data.dataset import ManifestDataset

        return ManifestDataset
    if name in {"MODALITY_ORDER", "collate_manifest_samples", "create_dataloader"}:
        from sentiment_derivation.data import dataloader

        return getattr(dataloader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

