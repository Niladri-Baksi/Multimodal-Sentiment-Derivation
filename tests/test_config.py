"""Unit tests for shared YAML configuration utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.utils.config import (  # noqa: E402
    load_config,
    validate_required_sections,
)


def test_load_config_from_string_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  name: example\n  seed: 7\n", encoding="utf-8")

    assert load_config(str(config_path)) == {"dataset": {"name": "example", "seed": 7}}


def test_load_config_accepts_path_object(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  width: 128\n", encoding="utf-8")

    assert load_config(config_path) == {"model": {"width": 128}}


def test_load_config_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(missing_path)


def test_load_config_directory_raises_is_a_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="not a file"):
        load_config(tmp_path)


def test_load_config_rejects_empty_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty or resolves to null"):
        load_config(config_path)


def test_load_config_rejects_invalid_yaml_with_path(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("dataset: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML.*invalid.yaml"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must have a mapping.*list"):
        load_config(config_path)


def test_load_config_rejects_non_string_root_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "non_string_key.yaml"
    config_path.write_text("1: value\nname: example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="string keys at the YAML root"):
        load_config(config_path)


def test_validate_required_sections_accepts_present_sections() -> None:
    validate_required_sections(
        {"dataset": {}, "model": {"width": 128}}, ["dataset", "model"]
    )


def test_validate_required_sections_lists_missing_sections() -> None:
    with pytest.raises(ValueError, match="training, experiment"):
        validate_required_sections({"dataset": {}}, ["training", "experiment"])


def test_validate_required_sections_rejects_non_mapping_input() -> None:
    with pytest.raises(TypeError, match="config must be a mapping"):
        validate_required_sections(["dataset"], ["dataset"])  # type: ignore[arg-type]
