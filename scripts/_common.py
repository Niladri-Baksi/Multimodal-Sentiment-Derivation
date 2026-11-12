"""Shared configuration and runtime helpers for command-line scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file_obj:
        value = yaml.safe_load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return value


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
    return device


def manifest_path(data_config: dict[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    configured = data_config.get("output", {}).get(
        "manifest_path", "data/processed/processed_manifest.json"
    )
    path = Path(configured).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def loader_options(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    video_config = data_config.get("modalities", {}).get("video", {})
    encoder_config = model_config.get("video_encoder", {})
    text_config = model_config.get("text_encoder", {})
    return {
        "num_frames": training_config.get(
            "num_frames", video_config.get("num_frames", 16)
        ),
        "image_size": tuple(encoder_config.get("image_size", (224, 224))),
        "audio_duration_sec": training_config.get("audio_duration_sec", 4.0),
        "num_workers": training_config.get("num_workers", 0),
        "include_text": bool(text_config.get("enabled", False)),
        "text_model_name": text_config.get("model_name", "distilbert-base-uncased"),
        "text_max_length": text_config.get("max_length", 48),
    }
