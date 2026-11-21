"""Regression tests for missing-audio BatchNorm isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.data.dataloader import collate_manifest_samples  # noqa: E402
from sentiment_derivation.encoders.audio_encoder import AudioEncoder  # noqa: E402
from sentiment_derivation.fusion.alignment import ModalityAligner  # noqa: E402


def _sample(sample_id: str, audio: torch.Tensor | None, available: bool) -> dict:
    return {
        "sample_id": sample_id,
        "split": "train",
        "video": torch.zeros(1, 3, 8, 8),
        "audio": audio,
        "text": None,
        "text_features": None,
        "emotion_labels": torch.zeros(8),
        "valence": torch.tensor(4.0),
        "modality_availability": {
            "video": True,
            "audio": available,
            "text": False,
        },
        "video_path": "video.mp4",
        "audio_path": "audio.wav" if available else None,
        "metadata": {},
    }


def _batchnorm_state(model: AudioEncoder) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    ]


def test_mixed_batch_excludes_missing_rows_from_batchnorm_and_preserves_gradients() -> None:
    torch.manual_seed(17)
    encoder = AudioEncoder().train()
    reference = AudioEncoder().train()
    reference.load_state_dict(encoder.state_dict())
    present = torch.randn(1, 1, 64, 16, requires_grad=True)
    mixed = torch.cat((present, torch.zeros_like(present)), dim=0).detach()
    mixed.requires_grad_()
    mask = torch.tensor([True, False])

    actual = encoder(mixed, audio_available=mask)
    expected = reference(present.detach())

    assert torch.allclose(actual[:1], expected, atol=1e-6, rtol=1e-5)
    assert torch.count_nonzero(actual[1]) == 0
    for actual_state, expected_state in zip(_batchnorm_state(encoder), _batchnorm_state(reference)):
        for actual_value, expected_value in zip(actual_state, expected_state):
            assert torch.equal(actual_value, expected_value)

    actual[0].square().mean().backward()
    assert mixed.grad is not None
    assert torch.count_nonzero(mixed.grad[0]) > 0
    assert torch.count_nonzero(mixed.grad[1]) == 0


def test_all_present_mask_matches_unmasked_encoder() -> None:
    torch.manual_seed(23)
    masked = AudioEncoder().train()
    unmasked = AudioEncoder().train()
    unmasked.load_state_dict(masked.state_dict())
    audio = torch.randn(2, 1, 64, 16)

    actual = masked(audio, audio_available=torch.tensor([True, True]))
    expected = unmasked(audio)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for actual_state, expected_state in zip(_batchnorm_state(masked), _batchnorm_state(unmasked)):
        for actual_value, expected_value in zip(actual_state, expected_state):
            assert torch.equal(actual_value, expected_value)


def test_all_missing_audio_skips_encoder_batchnorm_and_aligns_as_missing() -> None:
    samples = [_sample("one", None, False), _sample("two", None, False)]
    batch = collate_manifest_samples(samples)
    assert batch["audio"] is None
    assert batch["modality_mask"].tolist() == [[True, False, False]] * 2

    encoder = AudioEncoder().train()
    before = _batchnorm_state(encoder)
    audio_features = encoder(batch["audio"], audio_available=batch["modality_mask"][:, 1])
    assert audio_features.shape == (2, 16, 128)
    assert torch.count_nonzero(audio_features) == 0
    for old, new in zip(before, _batchnorm_state(encoder)):
        for old_value, new_value in zip(old, new):
            assert torch.equal(old_value, new_value)

    aligned = ModalityAligner()(
        torch.randn(2, 4, 512),
        audio_features,
        audio_available=batch["modality_mask"][:, 1],
    )
    assert aligned.modality_mask[:, 1].tolist() == [False, False]
    assert torch.count_nonzero(aligned.features["audio"]) == 0


@pytest.mark.parametrize(
    "audio,mask,exception,match",
    [
        (None, torch.tensor([True]), ValueError, "cannot mark rows present"),
        (torch.zeros(1, 1, 64, 8), torch.tensor([False, True]), ValueError, "shape"),
        (torch.zeros(1, 1, 64, 8), torch.tensor([1]), TypeError, "boolean"),
    ],
)
def test_encoder_rejects_inconsistent_availability_contract(audio, mask, exception, match) -> None:
    with pytest.raises(exception, match=match):
        AudioEncoder()(audio, audio_available=mask)


@pytest.mark.parametrize(
    "audio,available",
    [(None, True), (torch.ones(1, 64, 8), False)],
)
def test_collator_rejects_audio_mask_tensor_disagreement(audio, available) -> None:
    with pytest.raises(ValueError, match="inconsistent audio tensor"):
        collate_manifest_samples([_sample("sample", audio, available)])


def test_collator_preserves_mixed_audio_row_order_and_mask() -> None:
    batch = collate_manifest_samples(
        [_sample("present", torch.ones(1, 64, 8), True), _sample("missing", None, False)]
    )
    assert batch["sample_id"] == ["present", "missing"]
    assert batch["modality_order"] == ("video", "audio", "text")
    assert batch["modality_mask"][:, 1].tolist() == [True, False]
    assert torch.count_nonzero(batch["audio"][1]) == 0
