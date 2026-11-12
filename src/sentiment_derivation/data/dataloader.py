"""DataLoader construction and training-label statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sentiment_derivation.data.dataset import EmoReactDataset
from sentiment_derivation.data.preprocessing import DEFAULT_EMOTION_NAMES, load_manifest


def create_dataloader(
    manifest_path: str | Path,
    split: str,
    *,
    batch_size: int,
    num_frames: int = 16,
    image_size: tuple[int, int] = (224, 224),
    audio_duration_sec: float = 4.0,
    num_workers: int = 0,
    shuffle: bool = False,
    include_text: bool = False,
    text_model_name: str = "distilbert-base-uncased",
    text_max_length: int = 48,
) -> DataLoader[dict[str, Any]]:
    """Create a deterministic DataLoader; only the training loader should shuffle."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    dataset = EmoReactDataset(
        manifest_path,
        split,
        num_frames=num_frames,
        image_size=image_size,
        audio_duration_sec=audio_duration_sec,
        train=shuffle,
        include_text=include_text,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def emotion_positive_weights(
    manifest_path: str | Path,
    split: str = "train",
) -> torch.Tensor:
    """Return BCE positive-class weights, computed from manifest labels."""
    manifest = load_manifest(manifest_path)
    samples = manifest.get("samples", {}).get(split)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Manifest has no samples for split {split!r}.")
    positives = torch.zeros(len(DEFAULT_EMOTION_NAMES), dtype=torch.float32)
    for sample in samples:
        labels = sample.get("emotion_labels", {})
        try:
            values = torch.tensor(
                [float(labels[name]) for name in DEFAULT_EMOTION_NAMES],
                dtype=torch.float32,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid emotion labels in sample {sample.get('sample_id', '<unknown>')!r}."
            ) from error
        if not bool(torch.all((values == 0) | (values == 1))):
            raise ValueError("Emotion labels must contain only 0 or 1.")
        positives += values
    negatives = len(samples) - positives
    if bool((positives == 0).any()):
        absent = [
            DEFAULT_EMOTION_NAMES[index]
            for index, count in enumerate(positives.tolist())
            if count == 0
        ]
        raise ValueError(
            f"Cannot compute positive weights; no positive labels for: {', '.join(absent)}."
        )
    return negatives / positives
