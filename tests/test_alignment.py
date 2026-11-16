"""Focused contract tests for Phase 4 modality alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.fusion.alignment import ModalityAligner  # noqa: E402


def test_resamples_different_sequence_lengths_to_configured_steps() -> None:
    aligner = ModalityAligner(
        video_dim=2, audio_dim=3, text_dim=4, common_dim=5, target_steps=7
    )
    aligned = aligner(
        torch.randn(2, 3, 2), torch.randn(2, 11, 3), torch.randn(2, 5, 4)
    )
    assert all(value.shape == (2, 7, 5) for value in aligned.features.values())
    assert aligned.modality_mask.tolist() == [[True, True, True]] * 2
    assert aligned.alignment_method == "sequence_position_linear_resampling"
    assert aligned.timestamp_synchronized is False


def test_audio_absent_and_mixed_audio_rows_are_explicitly_masked() -> None:
    aligner = ModalityAligner()
    video = torch.randn(2, 4, 512)
    audio = torch.randn(2, 9, 128)
    aligned = aligner(video, audio, audio_available=torch.tensor([True, False]))

    assert aligned.modality_mask[:, 1].tolist() == [True, False]
    assert torch.count_nonzero(aligned.features["audio"][1]) == 0
    no_audio = aligner(video, None)
    assert not no_audio.modality_mask[:, 1].any()
    assert torch.count_nonzero(no_audio.features["audio"]) == 0


@pytest.mark.parametrize(
    "mask,exception,match",
    [
        (torch.tensor([[True], [False]]), ValueError, "shape"),
        (torch.tensor([1, 0]), TypeError, "boolean"),
        (torch.tensor([True]), ValueError, "shape"),
    ],
)
def test_audio_availability_mask_contract(mask, exception, match) -> None:
    aligner = ModalityAligner()
    with pytest.raises(exception, match=match):
        aligner(torch.randn(2, 4, 512), torch.randn(2, 9, 128), audio_available=mask)


def test_optional_audio_with_mask_preserves_video_gradients() -> None:
    aligner = ModalityAligner()
    video = torch.randn(2, 4, 512, requires_grad=True)
    aligned = aligner(video, None)
    aligned.features["video"].square().mean().backward()
    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert aligner.projections["video"].weight.grad is not None
