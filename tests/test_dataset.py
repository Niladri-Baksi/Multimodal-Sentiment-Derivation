"""Tests for manifest-backed dataset and Phase 3 batch contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.data.dataloader import collate_manifest_samples, create_dataloader
from sentiment_derivation.data.dataset import ManifestDataset
from sentiment_derivation.data.preprocessing import (
    DEFAULT_EMOTION_NAMES,
    build_processed_manifest,
)


def _sample(sample_id: str, video_path: Path, split: str = "train", **overrides: Any) -> dict[str, Any]:
    sample = {
        "sample_id": sample_id,
        "split": split,
        "video_path": str(video_path),
        "video_frame_paths": [],
        "audio_path": None,
        "text": None,
        "emotion_labels": {name: 0 for name in DEFAULT_EMOTION_NAMES},
        "other_original_annotations": {"valence": 4.0},
        "modality_availability": {"video": True, "audio": False, "text": False},
        "metadata": {"valid": True, "fps": 25.0},
    }
    sample.update(overrides)
    return sample


def _manifest(samples: list[dict[str, Any]], split: str = "train") -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "emotion_names": DEFAULT_EMOTION_NAMES,
        "samples": {split: samples, **{other: [] for other in ("train", "val", "test") if other != split}},
    }


def _frame_transform(frames: list[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0 for frame in frames]
    )


@pytest.fixture
def mock_video_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    from sentiment_derivation.data import dataset as dataset_module

    monkeypatch.setattr(
        dataset_module,
        "extract_sample_frames",
        lambda path, num_frames: [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(num_frames)],
    )


def test_manifest_dataset_preserves_contract_and_encoder_video_shape(
    tmp_path: Path, mock_video_decode: None
) -> None:
    video_path = tmp_path / "one.mp4"
    video_path.write_bytes(b"fixture")
    labels = dict.fromkeys(DEFAULT_EMOTION_NAMES, 0)
    labels["Happiness"] = 1
    sample = _sample("one", video_path, emotion_labels=labels)

    dataset = ManifestDataset(
        _manifest([sample]),
        "train",
        num_frames=3,
        video_transform=_frame_transform,
    )
    item = dataset[0]

    assert item["sample_id"] == "one"
    assert item["split"] == "train"
    assert item["video"].shape == (3, 3, 8, 8)
    assert item["video"].dtype == torch.float32
    assert item["emotion_labels"].shape == (8,)
    assert item["emotion_labels"][DEFAULT_EMOTION_NAMES.index("Happiness")] == 1
    assert item["valence"].item() == 4.0
    assert item["modality_availability"]["video"] is True
    assert item["modality_availability"]["audio"] is False
    assert item["modality_availability"]["text"] is False
    assert item["annotation_video_association"] == {
        "method": "unknown",
        "verification_status": "unverified",
        "evidence": None,
    }
    assert item["annotation_video_association_status"] == "unverified"
    assert item["annotation_video_association_evidence"] is None


def test_collation_batches_missing_audio_and_optional_text(
    tmp_path: Path, mock_video_decode: None
) -> None:
    video_paths = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    audio_path = tmp_path / "one.wav"
    for path in video_paths:
        path.write_bytes(b"video fixture")
    audio_path.write_bytes(b"audio fixture")

    samples = [
        _sample(
            "one",
            video_paths[0],
            text="hello",
            audio_path=str(audio_path),
            modality_availability={"video": True, "audio": True, "text": True},
        ),
        _sample("two", video_paths[1]),
    ]

    class AudioProcessor:
        def process_file(self, path: str) -> torch.Tensor:
            return torch.ones((1, 64, 7), dtype=torch.float64)

    class TextProcessor:
        def process_text(self, text: str) -> dict[str, torch.Tensor]:
            return {
                "input_ids": torch.tensor([101, 202]),
                "attention_mask": torch.ones(2, dtype=torch.long),
            }

    dataset = ManifestDataset(
        _manifest(samples),
        "train",
        num_frames=2,
        video_transform=_frame_transform,
        audio_preprocessor=AudioProcessor(),
        text_preprocessor=TextProcessor(),
    )
    batch = collate_manifest_samples([dataset[0], dataset[1]])

    assert batch["video"].shape == (2, 2, 3, 8, 8)
    assert batch["audio"].shape == (2, 1, 64, 7)
    assert batch["audio"].dtype == torch.float32
    assert torch.count_nonzero(batch["audio"][1]) == 0
    assert batch["input_ids"].shape == (2, 2)
    assert torch.equal(batch["attention_mask"][1], torch.zeros(2, dtype=torch.long))
    assert batch["emotion_labels"].shape == (2, 8)
    assert batch["valence"].shape == (2,)
    assert batch["modality_order"] == ("video", "audio", "text")
    assert batch["modality_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert batch["annotation_video_association_status"] == ["unverified", "unverified"]

    longer_item = dict(dataset[0])
    longer_item["sample_id"] = "three"
    longer_item["audio"] = torch.ones((1, 64, 11))
    variable_batch = collate_manifest_samples([dataset[0], dataset[1], longer_item])
    assert variable_batch["audio"].shape == (3, 1, 64, 11)
    assert torch.count_nonzero(variable_batch["audio"][0, ..., 7:]) == 0


def test_validation_loader_keeps_manifest_order_and_training_seed_repeats(
    tmp_path: Path, mock_video_decode: None
) -> None:
    video_paths = [tmp_path / f"{i}.mp4" for i in range(8)]
    for path in video_paths:
        path.write_bytes(b"fixture")
    train_samples = [_sample(str(i), path) for i, path in enumerate(video_paths)]
    train_data = ManifestDataset(
        _manifest(train_samples), "train", num_frames=1, video_transform=_frame_transform
    )

    def order(seed: int) -> list[str]:
        return [sample_id for batch in create_dataloader(train_data, batch_size=2, seed=seed)
                for sample_id in batch["sample_id"]]

    assert order(9) == order(9)
    assert order(9) != [str(i) for i in range(8)]

    val_samples = [_sample(str(i), path, split="val") for i, path in enumerate(video_paths)]
    val_data = ManifestDataset(
        _manifest(val_samples, split="val"), "val", num_frames=1, video_transform=_frame_transform
    )
    val_order = [sample_id for batch in create_dataloader(val_data, batch_size=3, seed=5)
                 for sample_id in batch["sample_id"]]
    assert val_order == [str(i) for i in range(8)]
    with pytest.raises(ValueError, match="preserve manifest ordering"):
        create_dataloader(val_data, shuffle=True)


def test_invalid_video_can_be_skipped_or_reported(tmp_path: Path) -> None:
    video_path = tmp_path / "corrupt.mp4"
    video_path.write_bytes(b"not a video")
    sample = _sample(
        "corrupt",
        video_path,
        modality_availability={"video": False, "audio": False, "text": False},
        metadata={"valid": False},
    )
    manifest = _manifest([sample])

    skipped = ManifestDataset(
        manifest,
        "train",
        invalid_sample_policy="skip",
        video_transform=_frame_transform,
    )
    assert len(skipped) == 0
    assert skipped.skipped_samples == ["corrupt"]
    with pytest.raises(ValueError, match="empty 'train' split"):
        create_dataloader(skipped)

    raised = ManifestDataset(manifest, "train", video_transform=_frame_transform)
    with pytest.raises(ValueError, match="no valid video"):
        raised[0]


def test_split_leakage_duplicate_and_empty_batch_are_rejected(tmp_path: Path) -> None:
    video_path = tmp_path / "one.mp4"
    video_path.write_bytes(b"fixture")
    leaked = _manifest([_sample("same", video_path)])
    leaked["samples"]["val"] = [_sample("same", video_path, split="val")]
    with pytest.raises(ValueError, match="split leakage"):
        ManifestDataset(leaked, "train", video_transform=_frame_transform)

    duplicate = _manifest([_sample("same", video_path), _sample("same", video_path)])
    with pytest.raises(ValueError, match="duplicated within split"):
        ManifestDataset(duplicate, "train", video_transform=_frame_transform)
    with pytest.raises(ValueError, match="empty sample batch"):
        collate_manifest_samples([])


@pytest.mark.parametrize(
    ("sample_id", "remove_id", "message"),
    [
        (None, True, "missing sample_id"),
        (None, False, "must be a string"),
        ("", False, "non-whitespace character"),
        (" \t ", False, "non-whitespace character"),
        (17, False, "must be a string"),
        (True, False, "must be a string"),
    ],
)
def test_manifest_rejects_invalid_sample_ids(sample_id, remove_id, message) -> None:
    record = _sample("valid-placeholder", Path("unused.mp4"))
    if remove_id:
        record.pop("sample_id")
    else:
        record["sample_id"] = sample_id

    with pytest.raises(ValueError, match=message) as exc_info:
        ManifestDataset(_manifest([record]), "train", video_transform=_frame_transform)

    assert "index 0" in str(exc_info.value)
    assert "train" in str(exc_info.value)


def test_invalid_sample_id_in_unselected_split_is_rejected() -> None:
    manifest = _manifest([_sample("train-id", Path("train.mp4"))])
    invalid_validation_record = _sample("placeholder", Path("val.mp4"), split="val")
    invalid_validation_record["sample_id"] = None
    manifest["samples"]["val"] = [invalid_validation_record]

    with pytest.raises(ValueError, match="sample_id.*val.*string") as exc_info:
        ManifestDataset(manifest, "train", video_transform=_frame_transform)

    assert "index 0" in str(exc_info.value)


def test_valid_sample_id_is_preserved_through_item_and_collation(
    tmp_path: Path, mock_video_decode: None
) -> None:
    ids = ["  stable:ID_01  ", "clip/B"]
    samples = []
    for index, sample_id in enumerate(ids):
        video_path = tmp_path / f"{index}.mp4"
        video_path.write_bytes(b"video fixture")
        samples.append(_sample(sample_id, video_path))
    dataset = ManifestDataset(
        _manifest(samples), "train", num_frames=1, video_transform=_frame_transform
    )

    items = [dataset[index] for index in range(len(dataset))]
    batch = collate_manifest_samples(items)

    assert [item["sample_id"] for item in items] == ids
    assert batch["sample_id"] == ids


def test_dataset_and_collator_preserve_mixed_association_provenance(
    tmp_path: Path, mock_video_decode: None
) -> None:
    policies = [
        {
            "method": "sorted_video_stem_position",
            "verification_status": "unverified",
            "evidence": None,
        },
        {
            "method": "sorted_video_stem_position",
            "verification_status": "caller_declared_verified",
            "evidence": "protocol.md#split-validation",
        },
    ]
    records = []
    for index, policy in enumerate(policies):
        video_path = tmp_path / f"provenance-{index}.mp4"
        video_path.write_bytes(b"video fixture")
        records.append(
            _sample(
                f"provenance-{index}",
                video_path,
                annotation_video_association=policy,
            )
        )
    manifest = _manifest(records)
    manifest["annotation_video_association"] = policies[0]
    dataset = ManifestDataset(
        manifest, "train", num_frames=2, video_transform=_frame_transform
    )

    items = [dataset[index] for index in range(2)]
    batch = collate_manifest_samples(items)

    assert [item["annotation_video_association"] for item in items] == policies
    assert batch["annotation_video_association"] == policies
    assert batch["annotation_video_association_status"] == [
        "unverified",
        "caller_declared_verified",
    ]
    assert batch["annotation_video_association_evidence"] == [
        None,
        "protocol.md#split-validation",
    ]
    # Provenance is metadata only; encoder-facing shapes and mask order remain stable.
    assert batch["video"].shape == (2, 2, 3, 8, 8)
    assert batch["audio"] is None
    assert batch["modality_mask"].shape == (2, 3)
    assert batch["modality_order"] == ("video", "audio", "text")
    assert batch["modality_mask"].tolist() == [[True, False, False]] * 2


@pytest.mark.parametrize(
    "provenance",
    [
        {"method": "sorted_video_stem_position", "verification_status": "caller_declared_verified", "evidence": None},
        {"method": "sorted_video_stem_position", "verification_status": "caller_declared_verified", "evidence": "  "},
        {"method": "sorted_video_stem_position", "verification_status": "caller_declared_verified", "evidence": 42},
        {"method": "sorted_video_stem_position", "verification_status": "verified", "evidence": "source"},
        {},
    ],
)
def test_dataset_rejects_missing_or_invalid_association_evidence(provenance) -> None:
    record = _sample("invalid-provenance", Path("unused.mp4"))
    record["annotation_video_association"] = provenance
    with pytest.raises(ValueError, match="provenance|evidence"):
        ManifestDataset(
            _manifest([record]), "train", video_transform=_frame_transform
        )


def test_dataset_uses_manifest_policy_when_sample_policy_is_legacy_missing() -> None:
    record = _sample("inherited-policy", Path("unused.mp4"))
    manifest = _manifest([record])
    manifest["annotation_video_association"] = {
        "method": "sorted_video_stem_position",
        "verification_status": "caller_declared_verified",
        "evidence": "protocol.md#ordering",
    }

    dataset = ManifestDataset(
        manifest, "train", video_transform=_frame_transform
    )

    assert dataset.samples[0]["annotation_video_association"] == manifest[
        "annotation_video_association"
    ]


@pytest.mark.parametrize(
    ("manifest_status", "sample_status", "sample_evidence"),
    [
        ("unverified", "caller_declared_verified", "sample-source.md#verified"),
        ("caller_declared_verified", "unverified", None),
    ],
)
def test_explicit_sample_provenance_overrides_manifest_policy(
    manifest_status: str, sample_status: str, sample_evidence: str | None
) -> None:
    manifest_evidence = "manifest-source.md#verified" if manifest_status == "caller_declared_verified" else None
    manifest = _manifest([_sample("conflicting", Path("unused.mp4"))])
    manifest["annotation_video_association"] = {
        "method": "sorted_video_stem_position",
        "verification_status": manifest_status,
        "evidence": manifest_evidence,
    }
    manifest["samples"]["train"][0]["annotation_video_association"] = {
        "method": "sorted_video_stem_position",
        "verification_status": sample_status,
        "evidence": sample_evidence,
    }

    dataset = ManifestDataset(manifest, "train", video_transform=_frame_transform)

    assert dataset.samples[0]["annotation_video_association"] == manifest[
        "samples"
    ]["train"][0]["annotation_video_association"]


def test_manifest_producer_to_dataset_sample_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sentiment_derivation.data import preprocessing as preprocessing_module
    from sentiment_derivation.data import dataset as dataset_module

    video_dir = tmp_path / "Data" / "Train"
    video_dir.mkdir(parents=True)
    video_path = video_dir / "clip.MP4"
    video_path.write_bytes(b"synthetic media placeholder")
    annotations = tmp_path / "train_labels.text"
    annotations.write_text("0,0,0,1,0,0,0,0,5.5\n", encoding="utf-8")
    monkeypatch.setattr(
        preprocessing_module,
        "inspect_media_file",
        lambda path: {
            "valid": True,
            "width": 8,
            "height": 8,
            "fps": 24.0,
            "frame_count": 3,
            "duration_seconds": 0.125,
            "file_size_bytes": video_path.stat().st_size,
            "error": None,
        },
    )
    monkeypatch.setattr(
        dataset_module,
        "extract_sample_frames",
        lambda path, num_frames: [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(num_frames)],
    )

    manifest = build_processed_manifest(
        dataset_root=tmp_path,
        annotation_files={"train": annotations},
        video_dirs={"train": video_dir},
    )
    dataset = ManifestDataset(
        manifest, "train", num_frames=2, video_transform=_frame_transform
    )
    item = dataset[0]

    assert item["sample_id"] == "clip"
    assert item["video"].shape == (2, 3, 8, 8)
    assert item["emotion_labels"][DEFAULT_EMOTION_NAMES.index("Happiness")] == 1
    assert item["valence"].item() == 5.5
