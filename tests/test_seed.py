"""Tests for reproducibility utilities."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.utils.seed import set_seed  # noqa: E402


def test_python_random_reproducibility() -> None:
    set_seed(1234)
    first = [random.random() for _ in range(5)]

    set_seed(1234)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_numpy_random_reproducibility() -> None:
    set_seed(4321)
    first = np.random.random(5)

    set_seed(4321)
    second = np.random.random(5)

    np.testing.assert_array_equal(first, second)


def test_torch_cpu_reproducibility_when_available() -> None:
    torch = pytest.importorskip("torch")

    set_seed(9876)
    first = torch.rand(5, device="cpu")
    set_seed(9876)
    second = torch.rand(5, device="cpu")

    assert torch.equal(first, second)


def test_cuda_reproducibility_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    set_seed(2468)
    first = torch.rand(5, device="cuda")
    set_seed(2468)
    second = torch.rand(5, device="cuda")

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("invalid_seed", "exception"),
    [
        (True, TypeError),
        ("42", TypeError),
        (-1, ValueError),
        (2**32, ValueError),
    ],
)
def test_invalid_seed_is_rejected(invalid_seed: object, exception: type[Exception]) -> None:
    with pytest.raises(exception, match="seed"):
        set_seed(invalid_seed)  # type: ignore[arg-type]


def test_deterministic_mode_configures_torch_and_restores_settings() -> None:
    torch = pytest.importorskip("torch")
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        set_seed(1357, deterministic=True)

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
