"""Shared YAML configuration loading and lightweight section validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file whose root must be a mapping.

    Args:
        config_path: Filesystem path to the YAML configuration.

    Returns:
        The parsed YAML mapping as a dictionary.

    Raises:
        FileNotFoundError: If the path does not identify an existing file.
        IsADirectoryError: If the path identifies a directory.
        ValueError: If the YAML is empty, malformed, has a non-mapping root, or
            contains non-string keys at the root.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file '{path}': {exc}") from exc

    if config is None:
        raise ValueError(
            f"Configuration file '{path}' is empty or resolves to null; "
            "expected a YAML mapping."
        )
    if not isinstance(config, Mapping):
        raise ValueError(
            f"Configuration file '{path}' must have a mapping at its YAML root; "
            f"got {type(config).__name__}."
        )
    non_string_keys = [key for key in config if not isinstance(key, str)]
    if non_string_keys:
        keys = ", ".join(repr(key) for key in non_string_keys)
        raise ValueError(
            f"Configuration file '{path}' must use string keys at the YAML root; "
            f"got non-string key(s): {keys}."
        )

    return dict(config)


def validate_required_sections(
    config: Mapping[str, Any], required_sections: Sequence[str]
) -> None:
    """Check that top-level configuration sections are present.

    This checks key presence only. It deliberately does not impose a schema on
    section contents or treat present-but-null values as missing.

    Args:
        config: Parsed configuration mapping.
        required_sections: Top-level section names that must exist.

    Raises:
        TypeError: If ``config`` is not a mapping.
        ValueError: If one or more required section keys are absent.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")

    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing required configuration sections: {missing}")
