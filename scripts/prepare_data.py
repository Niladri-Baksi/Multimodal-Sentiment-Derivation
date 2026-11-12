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
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure project root src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    build_processed_manifest,
    save_manifest,
)

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Dataset Exploration & Preprocessing Execution Script"
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
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {path}")
    return cfg


def main() -> None:
    """Main execution function."""
    args = parse_args()
    cfg = load_config(args.config)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting Phase 1 Preprocessing Pipeline")
    logger.info("=" * 60)

    # 1. Locate dataset directories (resolving candidates for Colab / Local compatibility)
    dataset_cfg = cfg.get("dataset", {})
    raw_root_setting = dataset_cfg.get("root_dir", "datasets/EmoReact_V_1.0")
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

    # Build split paths relative to resolved dataset root
    ann_files = {
        "train": str(root_dir / "Labels" / "train_labels.text"),
        "val": str(root_dir / "Labels" / "val_labels.text"),
        "test": str(root_dir / "Labels" / "test_labels.text"),
    }
    video_dirs = {
        "train": str(root_dir / "Data" / "Train"),
        "val": str(root_dir / "Data" / "Validation"),
        "test": str(root_dir / "Data" / "Test"),
    }
    feat_dirs = {
        "train": str(root_dir / "Visual_features" / "Train_feat"),
        "val": str(root_dir / "Visual_features" / "Val_feat"),
        "test": str(root_dir / "Visual_features" / "Test_feat"),
    }

    # 2. Configure output paths
    out_cfg = cfg.get("output", {})
    processed_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / out_cfg.get("processed_dir", "data/processed")
    )
    manifest_path = processed_dir / "processed_manifest.json"
    summary_path = processed_dir / "preprocessing_summary.json"
    audio_dir = processed_dir / "audio"
    frames_dir = processed_dir / "frames"

    # Extraction options
    extract_audio = args.extract_audio or cfg.get("preprocessing", {}).get("extract_audio", False)
    extract_frames = args.extract_frames or cfg.get("preprocessing", {}).get("extract_frames", False)
    num_frames = cfg.get("modalities", {}).get("video", {}).get("num_frames", 16)
    emotion_names = cfg.get("labels", {}).get("emotions", DEFAULT_EMOTION_NAMES)

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
    )

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
