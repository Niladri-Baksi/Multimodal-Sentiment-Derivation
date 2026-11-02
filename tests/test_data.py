"""Unit and integration tests for Phase 1 data preprocessing.

Tests:
1. Annotation loading and numerical validation.
2. Error handling for malformed or missing label files.
3. Sample ID discovery and filtering (ignoring system files like Thumbs.db).
4. Alignment between sample IDs and multi-label annotations.
5. Media file inspection and corruption detection.
6. Standardized sample dictionary contract compliance.
7. Manifest generation, statistics calculation, and serialization.
8. Real dataset integrity verification (when dataset is present).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest

from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    build_processed_manifest,
    build_standardized_sample,
    compute_manifest_statistics,
    get_split_sample_ids,
    inspect_media_file,
    load_annotations,
    load_manifest,
    match_annotations_with_media,
    save_manifest,
    verify_visual_features,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for isolated test cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_load_valid_annotations(temp_workspace: Path):
    """Test loading valid 9-column annotation rows."""
    label_file = temp_workspace / "valid_labels.text"
    content = (
        "0,0,0,1,0,0,0,0,3.3333\n"
        "1,1,1,0,1,0,0,0,5.0000\n"
        "0,0,0,0,0,0,0,0,2.6667\n"
    )
    label_file.write_text(content, encoding="utf-8")

    records = load_annotations(label_file)
    assert len(records) == 3

    # Check first record
    assert records[0]["emotion_labels"]["Happiness"] == 1
    assert records[0]["emotion_labels"]["Curiosity"] == 0
    assert pytest.approx(records[0]["valence"], 0.0001) == 3.3333

    # Check second record (multi-label)
    assert records[1]["emotion_labels"]["Curiosity"] == 1
    assert records[1]["emotion_labels"]["Uncertainty"] == 1
    assert records[1]["emotion_labels"]["Excitement"] == 1
    assert records[1]["emotion_labels"]["Surprise"] == 1
    assert pytest.approx(records[1]["valence"], 0.0001) == 5.0000

    # Check third record (neutral / 0 active emotions)
    assert sum(records[2]["emotion_labels"].values()) == 0
    assert pytest.approx(records[2]["valence"], 0.0001) == 2.6667


def test_load_invalid_column_count(temp_workspace: Path):
    """Test that rows with incorrect column count raise ValueError."""
    label_file = temp_workspace / "bad_columns.text"
    # Only 5 columns instead of 9
    label_file.write_text("0,1,0,0,3.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns"):
        load_annotations(label_file)


def test_load_non_binary_emotion(temp_workspace: Path):
    """Test that non-binary emotion values (e.g. 2.0) raise ValueError."""
    label_file = temp_workspace / "non_binary.text"
    # Column 0 has value 2 instead of 0 or 1
    label_file.write_text("2,0,0,0,0,0,0,0,4.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-binary value"):
        load_annotations(label_file)


def test_load_nonexistent_file(temp_workspace: Path):
    """Test that non-existent annotation path raises FileNotFoundError."""
    missing_path = temp_workspace / "non_existent.text"
    with pytest.raises(FileNotFoundError):
        load_annotations(missing_path)


def test_get_split_sample_ids(temp_workspace: Path):
    """Test that sample IDs are extracted sorted from MP4s and ignore non-mp4 files."""
    video_dir = temp_workspace / "videos"
    video_dir.mkdir()

    # Create dummy video files and non-video files
    (video_dir / "clip_B.mp4").write_text("dummy", encoding="utf-8")
    (video_dir / "clip_A.mp4").write_text("dummy", encoding="utf-8")
    (video_dir / "clip_C.MP4").write_text("dummy", encoding="utf-8")
    (video_dir / "Thumbs.db").write_text("dummy", encoding="utf-8")
    (video_dir / "notes.txt").write_text("dummy", encoding="utf-8")

    sample_ids = get_split_sample_ids(video_dir)
    assert sample_ids == ["clip_A", "clip_B", "clip_C"]


def test_match_annotations_with_media_success():
    """Test successful 1:1 mapping between sample IDs and annotations."""
    sample_ids = ["clip_01", "clip_02"]
    annotations = [
        {"emotion_labels": {"Happiness": 1}, "valence": 5.0},
        {"emotion_labels": {"Curiosity": 1}, "valence": 4.0},
    ]

    matched = match_annotations_with_media(sample_ids, annotations)
    assert len(matched) == 2
    assert matched[0]["sample_id"] == "clip_01"
    assert matched[0]["valence"] == 5.0
    assert matched[1]["sample_id"] == "clip_02"


def test_match_annotations_with_media_mismatch():
    """Test that count mismatch between IDs and annotations raises ValueError."""
    sample_ids = ["clip_01"]
    annotations = [
        {"emotion_labels": {"Happiness": 1}, "valence": 5.0},
        {"emotion_labels": {"Curiosity": 1}, "valence": 4.0},
    ]

    with pytest.raises(ValueError, match="Count mismatch"):
        match_annotations_with_media(sample_ids, annotations)


def test_inspect_media_file_missing(temp_workspace: Path):
    """Test media inspection returns invalid status for missing file."""
    res = inspect_media_file(temp_workspace / "missing.mp4")
    assert res["valid"] is False
    assert "does not exist" in res["error"]


def test_build_standardized_sample():
    """Test standardized sample schema according to AGENTS.md contract."""
    sample = build_standardized_sample(
        sample_id="test_01",
        split="train",
        video_path="path/to/vid.mp4",
        emotion_labels={"Happiness": 1, "Surprise": 0},
        valence=4.5,
        audio_path=None,
        text=None,
        visual_features=None,
        media_metadata={"width": 1280, "height": 720},
    )

    assert sample["sample_id"] == "test_01"
    assert sample["split"] == "train"
    assert sample["emotion_labels"]["Happiness"] == 1
    assert sample["other_original_annotations"]["valence"] == 4.5
    assert sample["text"] is None
    assert "modality_availability" in sample
    assert sample["modality_availability"]["text"] is False
    assert sample["metadata"]["width"] == 1280


def test_build_processed_manifest_mock(temp_workspace: Path):
    """Test building a manifest with a small isolated mock dataset."""
    root = temp_workspace / "dataset"
    train_vdir = root / "Data" / "Train"
    train_vdir.mkdir(parents=True)
    labels_dir = root / "Labels"
    labels_dir.mkdir(parents=True)

    # 2 mock samples
    (train_vdir / "clip_01.mp4").write_text("dummy", encoding="utf-8")
    (train_vdir / "clip_02.mp4").write_text("dummy", encoding="utf-8")

    label_content = (
        "0,0,0,1,0,0,0,0,5.0\n"
        "1,0,0,0,1,0,0,0,4.0\n"
    )
    label_file = labels_dir / "train_labels.text"
    label_file.write_text(label_content, encoding="utf-8")

    manifest = build_processed_manifest(
        dataset_root=root,
        annotation_files={"train": label_file},
        video_dirs={"train": train_vdir},
        extract_audio=False,
        extract_frames=False,
    )

    assert manifest["dataset_name"] == "EmoReact_V_1.0"
    assert len(manifest["samples"]["train"]) == 2
    assert manifest["statistics"]["total_samples"] == 2
    assert manifest["statistics"]["emotion_frequencies"]["Happiness"]["count"] == 1
    assert manifest["statistics"]["emotion_frequencies"]["Curiosity"]["count"] == 1
    assert "pos_weight" in manifest["statistics"]["emotion_frequencies"]["Happiness"]
    assert "train_split_emotion_stats" in manifest["statistics"]
    assert "video_profile" in manifest["statistics"]


def test_save_and_load_manifest(temp_workspace: Path):
    """Test saving and loading manifest to JSON."""
    manifest_data = {
        "manifest_version": "1.0",
        "dataset_name": "TestDataset",
        "samples": {"train": [{"sample_id": "s1"}]},
    }
    out_file = temp_workspace / "test_manifest.json"
    save_manifest(manifest_data, out_file)
    assert out_file.is_file()

    loaded = load_manifest(out_file)
    assert loaded["dataset_name"] == "TestDataset"
    assert loaded["samples"]["train"][0]["sample_id"] == "s1"


def test_real_dataset_integrity():
    """Verify integrity of real EmoReact dataset in the workspace."""
    dataset_root = Path("datasets/EmoReact_V_1.0")
    if not dataset_root.is_dir():
        pytest.skip("EmoReact dataset not found at default location.")

    splits = [
        ("Train", "Data/Train", "Labels/train_labels.text", 432),
        ("Validation", "Data/Validation", "Labels/val_labels.text", 303),
        ("Test", "Data/Test", "Labels/test_labels.text", 367),
    ]

    for name, v_rel, l_rel, expected_count in splits:
        v_dir = dataset_root / v_rel
        l_file = dataset_root / l_rel

        assert v_dir.is_dir(), f"Missing video directory: {v_dir}"
        assert l_file.is_file(), f"Missing label file: {l_file}"

        ids = get_split_sample_ids(v_dir)
        anns = load_annotations(l_file)

        assert len(ids) == expected_count, f"{name}: expected {expected_count} videos, got {len(ids)}"
        assert len(anns) == expected_count, f"{name}: expected {expected_count} labels, got {len(anns)}"
