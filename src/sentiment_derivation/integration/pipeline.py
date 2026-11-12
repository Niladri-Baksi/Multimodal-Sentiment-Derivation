"""Checkpoint loading and batch prediction utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from sentiment_derivation.integration.model import MultimodalSentimentModel


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[MultimodalSentimentModel, dict[str, Any]]:
    """Load a model checkpoint without downloading encoder weights."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError(
            "Checkpoint must contain model_config and model_state_dict; "
            "train a checkpoint with scripts/train.py."
        )
    config = dict(model_config)
    config["video_encoder"] = dict(config.get("video_encoder", {}))
    config["video_encoder"]["pretrained"] = False
    if config.get("text_encoder", {}).get("enabled", False):
        config["text_encoder"] = dict(config["text_encoder"])
        config["text_encoder"]["pretrained"] = False
    model = MultimodalSentimentModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, dict(checkpoint)


def export_predictions(
    model: MultimodalSentimentModel,
    loader: DataLoader[dict[str, Any]],
    output_path: str | Path,
    *,
    device: torch.device | str,
) -> Path:
    """Run a model on every sample and export evaluator-compatible probabilities."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    emotion_names = tuple(model.emotion_classifier.emotion_names)
    sentiment_names = tuple(model.sentiment_head.sentiment_labels)
    fieldnames = (
        ["sample_id"]
        + [f"emotion_probability_{name}" for name in emotion_names]
        + ["predicted_valence"]
        + [f"sentiment_probability_{name}" for name in sentiment_names]
    )
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    frames = batch["frames"].to(device, non_blocking=True)
                    audio = batch["audio"].to(device, non_blocking=True)
                    model_inputs: dict[str, torch.Tensor] = {}
                    if "input_ids" in batch or "attention_mask" in batch:
                        if "input_ids" not in batch or "attention_mask" not in batch:
                            raise ValueError(
                                "Text batches must provide input_ids and attention_mask together."
                            )
                        model_inputs = {
                            "text_input_ids": batch["input_ids"].to(device, non_blocking=True),
                            "text_attention_mask": batch["attention_mask"].to(
                                device, non_blocking=True
                            ),
                        }
                        if "text_available" in batch:
                            model_inputs["text_available"] = batch["text_available"].to(
                                device, non_blocking=True
                            )
                    output = model(frames, audio, **model_inputs)
                    emotion_probabilities = output["emotion_probabilities"].mean(dim=1).cpu()
                    sentiment_probabilities = output["sentiment_probabilities"].cpu()
                    valences = output["valence"].reshape(-1).cpu()
                    for index, sample_id in enumerate(batch["sample_id"]):
                        row: dict[str, Any] = {"sample_id": sample_id}
                        row.update(
                            {
                                f"emotion_probability_{name}": float(emotion_probabilities[index, column])
                                for column, name in enumerate(emotion_names)
                            }
                        )
                        row["predicted_valence"] = float(valences[index])
                        row.update(
                            {
                                f"sentiment_probability_{name}": float(sentiment_probabilities[index, column])
                                for column, name in enumerate(sentiment_names)
                            }
                        )
                        writer.writerow(row)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination
