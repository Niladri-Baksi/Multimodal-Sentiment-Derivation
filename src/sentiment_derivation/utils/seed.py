"""Reproducibility helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed available random number generators used by the project.

    Seeds Python's ``random`` module, NumPy, PyTorch's CPU generator, and all
    CUDA generators when CUDA is available. PyTorch is imported lazily. If it
    is unavailable, Python and NumPy will have been seeded before an
    ``ImportError`` is raised, so the caller is not led to believe that all
    project generators were initialized.

    When ``deterministic`` is true, deterministic PyTorch algorithms are
    required and cuDNN benchmarking is disabled. These settings can reduce
    performance or cause operations without deterministic implementations to
    raise errors. Full CUDA reproducibility may also require setting
    ``CUBLAS_WORKSPACE_CONFIG`` before starting Python.

    Args:
        seed: Integer seed in the inclusive range 0 through 2**32 - 1.
        deterministic: Whether to enable deterministic PyTorch algorithms.

    Raises:
        TypeError: If ``seed`` is not an integer or ``deterministic`` is not a
            boolean.
        ValueError: If ``seed`` is outside the supported range.
        ImportError: If PyTorch is unavailable after Python and NumPy are
            seeded.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer (bool is not accepted).")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 2**32 - 1 inclusive.")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a bool.")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ImportError(
                "PyTorch is required to seed the project's PyTorch generators."
            ) from exc
        raise

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
