"""Batch collation and reproducible DataLoader construction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from sentiment_derivation.data.dataset import (
    ManifestDataset,
    _normalize_association_provenance,
)

MODALITY_ORDER = ("video", "audio", "text")


def collate_manifest_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate sample records into encoder-ready batched tensors.

    Video output is ``[B,T,C,H,W]``. Audio, when present in at least one item,
    is zero-padded to ``[B,1,M,T_audio]``; absent audio items are zeros and
    indicated by ``modality_mask[:,1]``. Optional text tokens are padded and
    returned as top-level ``input_ids`` and ``attention_mask``. When no text
    processor was supplied to the Dataset, raw transcripts are retained in
    ``text`` and both token fields are ``None``.

    Args:
        samples: Non-empty sequence of output dictionaries from
            :class:`ManifestDataset`.

    Returns:
        A batch mapping with inputs, labels, metadata, and a boolean modality
        mask ordered as video, audio, text.

    Raises:
        ValueError: If the input is empty or required sample tensor shapes do
            not agree.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample batch.")

    try:
        video = torch.stack([sample["video"] for sample in samples], dim=0)
        emotion_labels = torch.stack([sample["emotion_labels"] for sample in samples], dim=0)
        valence = torch.stack([sample["valence"] for sample in samples], dim=0)
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Samples have missing or incompatible required tensors: {exc}") from exc

    audio_items = [sample.get("audio") for sample in samples]
    audio_availability: list[bool] = []
    for index, sample in enumerate(samples):
        availability = sample.get("modality_availability")
        if not isinstance(availability, Mapping):
            raise ValueError(f"Sample at batch index {index} needs modality_availability mapping.")
        available = availability.get("audio")
        if not isinstance(available, (bool, np.bool_)):
            raise ValueError(
                f"Sample at batch index {index} needs a boolean audio availability flag."
            )
        if bool(available) != (audio_items[index] is not None):
            raise ValueError(
                f"Sample at batch index {index} has inconsistent audio tensor and availability flag."
            )
        audio_availability.append(bool(available))

    present_audio = [item for item in audio_items if item is not None]
    audio: Optional[torch.Tensor] = None
    if present_audio:
        if any(
            not isinstance(item, torch.Tensor)
            or item.ndim != 3
            or item.shape[0] != 1
            or item.shape[1] <= 0
            or item.shape[2] <= 0
            or not item.is_floating_point()
            or not bool(torch.isfinite(item).all())
            for item in present_audio
        ):
            raise ValueError(
                "Per-sample audio tensors must be finite floating tensors shaped [1,M,T] "
                "with positive dimensions."
            )
        mel_bands = present_audio[0].shape[1]
        if any(item.shape[1] != mel_bands for item in present_audio):
            raise ValueError("All present audio samples must have the same mel-band dimension.")
        max_audio_steps = max(item.shape[2] for item in present_audio)
        padded_audio: list[torch.Tensor] = []
        for item in audio_items:
            if item is None:
                padded_audio.append(present_audio[0].new_zeros((1, mel_bands, max_audio_steps)))
            else:
                padded_audio.append(torch.nn.functional.pad(
                    item, (0, max_audio_steps - item.shape[2])
                ))
        audio = torch.stack(padded_audio, dim=0)

    token_records = [sample.get("text_features") for sample in samples]
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    if any(tokens is not None for tokens in token_records):
        padded_ids = []
        padded_masks = []
        for tokens in token_records:
            if tokens is None:
                padded_ids.append(torch.empty(0, dtype=torch.long))
                padded_masks.append(torch.empty(0, dtype=torch.long))
            else:
                padded_ids.append(tokens["input_ids"].to(dtype=torch.long))
                padded_masks.append(tokens["attention_mask"].to(dtype=torch.long))
        input_ids = pad_sequence(padded_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(padded_masks, batch_first=True, padding_value=0)
        if input_ids.shape[1] == 0:
            input_ids = torch.zeros((len(samples), 1), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
        if input_ids.shape != attention_mask.shape:
            raise ValueError("Text input_ids and attention_mask must have matching batch shapes.")

    modality_mask = torch.tensor(
        [
            [
                bool(sample["modality_availability"].get("video", False)),
                audio_availability[index],
                bool(sample["modality_availability"].get("text", False)),
            ]
            for index, sample in enumerate(samples)
        ],
        dtype=torch.bool,
    )

    association_records = [
        _normalize_association_provenance(
            sample.get("annotation_video_association", {
                "method": "unknown",
                "verification_status": "unverified",
                "evidence": None,
            }),
            context=f"Sample at batch index {index}",
        )
        for index, sample in enumerate(samples)
    ]

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "video": video,
        "audio": audio,
        "text": [sample.get("text") for sample in samples],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "emotion_labels": emotion_labels,
        "valence": valence,
        "modality_mask": modality_mask,
        "modality_order": MODALITY_ORDER,
        "video_path": [sample.get("video_path") for sample in samples],
        "audio_path": [sample.get("audio_path") for sample in samples],
        "visual_feature_paths": [sample.get("visual_feature_paths", {}) for sample in samples],
        "metadata": [sample.get("metadata", {}) for sample in samples],
        "annotation_video_association": association_records,
        "annotation_video_association_status": [
            item["verification_status"] for item in association_records
        ],
        "annotation_video_association_evidence": [
            item["evidence"] for item in association_records
        ],
    }


def _seed_data_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker from PyTorch's worker seed."""
    del worker_id  # PyTorch incorporates the worker ID into initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: ManifestDataset,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    seed: Optional[int] = None,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    """Create a batched loader with split-safe ordering and optional seeding.

    Shuffling defaults on for the training split and off for validation/test.
    Explicitly requesting shuffle for a non-training split is rejected to
    protect deterministic evaluation order.

    Args:
        dataset: A :class:`ManifestDataset` instance.
        batch_size: Number of samples per batch.
        shuffle: Override the training-default shuffle behavior.
        seed: Optional seed for reproducible sample ordering and worker RNGs.
        num_workers: Number of subprocess data workers.
        drop_last: Whether to discard an incomplete final batch.
        pin_memory: Whether to pin output tensors for accelerator transfer.

    Returns:
        A PyTorch DataLoader using :func:`collate_manifest_samples`.

    Raises:
        ValueError: If batch size, seed, worker count, split ordering, or
            dataset length is invalid.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if shuffle is not None and not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool or None.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    if len(dataset) == 0:
        raise ValueError(f"Cannot create a DataLoader for an empty {dataset.split!r} split.")

    should_shuffle = dataset.split == "train" if shuffle is None else shuffle
    if dataset.split != "train" and should_shuffle:
        raise ValueError("Validation and test DataLoaders must preserve manifest ordering.")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_manifest_samples,
        generator=generator,
        worker_init_fn=_seed_data_worker,
    )
