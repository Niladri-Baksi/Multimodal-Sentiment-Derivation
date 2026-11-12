"""Data loading, preprocessing, and dataset management."""

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
from sentiment_derivation.data.dataset import EmoReactDataset
from sentiment_derivation.data.dataloader import create_dataloader, emotion_positive_weights

__all__ = [
    "EmoReactDataset",
    "create_dataloader",
    "emotion_positive_weights",
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
]
