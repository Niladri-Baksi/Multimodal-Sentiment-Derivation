"""Execution script for dataset validation and preprocessing pipeline.

Main entry point to:
1. Load configuration from `configs/data.yaml`.
2. Validate presence of raw dataset files across Train, Validation, and Test splits.
3. Match annotations with corresponding media and OpenFace visual features.
4. Optionally extract standardized WAV audio and video frames.
5. Generate and persist `data/processed/processed_manifest.json` and summary report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.utils.config import (  # noqa: E402
    load_config as load_shared_config,
    validate_required_sections,
)
from sentiment_derivation.utils.logger import configure_logging  # noqa: E402
from sentiment_derivation.utils.seed import set_seed  # noqa: E402
from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    build_processed_manifest,
    save_manifest,
)

logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Dataset Exploration and Preprocessing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "data.yaml"),
        help="Path to data configuration YAML file",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract standardized WAV audio files from MP4 videos",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract uniform RGB video frames from MP4 videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for processed data",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load data YAML through shared validation and require its main sections."""
    cfg = load_shared_config(config_path)
    validate_required_sections(cfg, ["dataset", "labels", "modalities", "output"])
    logger.info(f"Loaded configuration from: {config_path}")
    return cfg


def _configured_dataset_path(
    configured_path: str,
    configured_root: str,
    resolved_root: Path,
) -> Path:
    """Resolve a configured dataset member under the selected root candidate."""
    path = Path(configured_path)
    root = Path(configured_root)
    if path.is_absolute():
        comparison_root = root if root.is_absolute() else PROJECT_ROOT / root
        try:
            relative = path.relative_to(comparison_root)
        except ValueError:
            return path
        return resolved_root / relative
    try:
        relative = path.relative_to(root)
    except ValueError:
        project_candidate = PROJECT_ROOT / path
        if root.is_absolute() and not project_candidate.exists():
            return resolved_root / path
        return project_candidate
    return resolved_root / relative


def _project_path(value: str, fallback: str) -> Path:
    """Resolve configured output paths relative to the repository root."""
    path = Path(value or fallback)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    """Main execution function."""
    configure_logging("prepare_data", level="INFO")
    args = parse_args()
    cfg = load_config(args.config)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Phase 2 Dataset Preprocessing Pipeline")
    logger.info("=" * 60)

    # 1. Locate dataset directories (resolving candidates for Colab / Local compatibility)
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("The 'dataset' configuration section must be a mapping.")
    raw_root_setting = dataset_cfg.get("root_dir", "datasets/EmoReact_V_1.0")
    if not isinstance(raw_root_setting, str):
        raise ValueError("dataset.root_dir must be a string path.")
    root_dir = Path(raw_root_setting) if Path(raw_root_setting).is_absolute() else PROJECT_ROOT / raw_root_setting

    if not root_dir.is_dir():
        candidates = dataset_cfg.get("root_dir_candidates", [])
        for cand in candidates:
            cand_p = Path(cand) if Path(cand).is_absolute() else PROJECT_ROOT / cand
            if cand_p.is_dir():
                root_dir = cand_p
                break

    if not root_dir.is_dir():
        logger.error(f"Dataset root directory not found at: {root_dir}")
        sys.exit(1)
    logger.info(f"Dataset root directory verified: {root_dir}")

    configured_seed = dataset_cfg.get("random_seed")
    if configured_seed is not None:
        set_seed(configured_seed)

    # Build split paths relative to resolved dataset root
    def resolve_split_paths(section: str, fallback_suffixes: Dict[str, str]) -> Dict[str, str]:
        configured = dataset_cfg.get(section, {})
        if not isinstance(configured, dict):
            raise ValueError(f"dataset.{section} must be a mapping by split.")
        return {
            split: str(
                _configured_dataset_path(str(configured[split]), raw_root_setting, root_dir)
                if split in configured
                else root_dir / fallback_suffixes[split]
            )
            for split in ("train", "val", "test")
        }

    ann_files = resolve_split_paths(
        "annotation_files",
        {"train": "Labels/train_labels.text", "val": "Labels/val_labels.text", "test": "Labels/test_labels.text"},
    )
    video_dirs = resolve_split_paths(
        "video_dirs",
        {"train": "Data/Train", "val": "Data/Validation", "test": "Data/Test"},
    )
    feat_dirs = resolve_split_paths(
        "visual_feature_dirs",
        {
            "train": "Visual_features/Train_feat",
            "val": "Visual_features/Val_feat",
            "test": "Visual_features/Test_feat",
        },
    )

    # 2. Configure output paths
    out_cfg = cfg.get("output", {})
    if not isinstance(out_cfg, dict):
        raise ValueError("The 'output' configuration section must be a mapping.")
    if args.output_dir:
        processed_dir = Path(args.output_dir)
        manifest_path = processed_dir / "processed_manifest.json"
        summary_path = processed_dir / "preprocessing_summary.json"
        audio_dir = processed_dir / "audio"
        frames_dir = processed_dir / "frames"
    else:
        processed_dir = _project_path(out_cfg.get("processed_dir", "data/processed"), "data/processed")
        manifest_path = _project_path(
            out_cfg.get("manifest_path", str(processed_dir / "processed_manifest.json")),
            str(processed_dir / "processed_manifest.json"),
        )
        summary_path = _project_path(
            out_cfg.get("summary_path", str(processed_dir / "preprocessing_summary.json")),
            str(processed_dir / "preprocessing_summary.json"),
        )
        audio_dir = _project_path(out_cfg.get("audio_output_dir", str(processed_dir / "audio")), str(processed_dir / "audio"))
        frames_dir = _project_path(out_cfg.get("frames_output_dir", str(processed_dir / "frames")), str(processed_dir / "frames"))

    # Extraction options
    modalities_cfg = cfg.get("modalities", {})
    if not isinstance(modalities_cfg, dict):
        raise ValueError("The 'modalities' configuration section must be a mapping.")
    audio_cfg = modalities_cfg.get("audio", {})
    video_cfg = modalities_cfg.get("video", {})
    if not isinstance(audio_cfg, dict) or not isinstance(video_cfg, dict):
        raise ValueError("The 'modalities.audio' and 'modalities.video' sections must be mappings.")
    extract_audio = args.extract_audio or bool(
        audio_cfg.get("enabled", False) and audio_cfg.get("extract_from_video", False)
    )
    extract_frames = args.extract_frames or bool(video_cfg.get("extract_frames", False))
    num_frames = video_cfg.get("num_frames", 16)
    target_sample_rate = audio_cfg.get("target_sample_rate", 16000)
    audio_channels = audio_cfg.get("channels", 1)
    labels_cfg = cfg.get("labels", {})
    if not isinstance(labels_cfg, dict):
        raise ValueError("The 'labels' configuration section must be a mapping.")
    emotion_names = labels_cfg.get("emotions", DEFAULT_EMOTION_NAMES)

    logger.info(f"Extract Audio  : {extract_audio}")
    logger.info(f"Extract Frames : {extract_frames}")
    logger.info(f"Target Manifest: {manifest_path}")

    # 3. Execute preprocessing pipeline
    manifest = build_processed_manifest(
        dataset_root=root_dir,
        annotation_files=ann_files,
        video_dirs=video_dirs,
        visual_feature_dirs=feat_dirs,
        output_audio_dir=audio_dir if extract_audio else None,
        extract_audio=extract_audio,
        extract_frames=extract_frames,
        output_frames_dir=frames_dir if extract_frames else None,
        num_frames=num_frames,
        emotion_names=emotion_names,
        target_sample_rate=target_sample_rate,
        audio_channels=audio_channels,
    )

    expected_sizes = dataset_cfg.get("expected_split_sizes", {})
    if not isinstance(expected_sizes, dict):
        raise ValueError("dataset.expected_split_sizes must be a mapping by split.")
    actual_sizes = manifest["statistics"].get("split_counts", {})
    size_mismatches = {
        split: (expected, actual_sizes.get(split, 0))
        for split, expected in expected_sizes.items()
        if split in {"train", "val", "test"}
        and actual_sizes.get(split, 0) != expected
    }
    if "total" in expected_sizes and manifest["statistics"].get("total_samples") != expected_sizes["total"]:
        size_mismatches["total"] = (
            expected_sizes["total"],
            manifest["statistics"].get("total_samples", 0),
        )
    if size_mismatches:
        raise ValueError(f"Dataset split counts differ from configured expectations: {size_mismatches}")

    # 4. Save manifest
    save_manifest(manifest, manifest_path)

    # 5. Save summary report
    stats = manifest.get("statistics", {})
    issues = manifest.get("corrupted_or_missing", [])

    summary = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "total_samples": stats.get("total_samples", 0),
        "split_counts": stats.get("split_counts", {}),
        "emotion_frequencies": stats.get("emotion_frequencies", {}),
        "train_split_emotion_stats": stats.get("train_split_emotion_stats", {}),
        "valence_statistics": stats.get("valence_statistics", {}),
        "active_emotions_histogram": stats.get("active_emotions_histogram", {}),
        "video_profile": stats.get("video_profile", {}),
        "modality_availability": stats.get("modality_availability", {}),
        "corrupted_or_missing_count": len(issues),
        "issues": issues,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Preprocessing summary saved to: {summary_path}")

    # 6. Print CLI Summary
    logger.info("=" * 60)
    logger.info("Preprocessing Pipeline Complete!")
    logger.info(f"Total Samples Processed: {stats.get('total_samples')}")
    for split_name, count in stats.get("split_counts", {}).items():
        logger.info(f"  - {split_name.capitalize()}: {count} samples")
    logger.info(f"Corrupted / Invalid Samples: {len(issues)}")
    logger.info(f"Elapsed Time: {summary['elapsed_seconds']}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
