"""Unit tests for Phase 4 modality alignment and attention fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_derivation.fusion.alignment import (  # noqa: E402
    AlignedModalities,
    ModalityAligner,
)
from sentiment_derivation.fusion.multimodal_fusion import (  # noqa: E402
    MultimodalFusion,
)


@pytest.fixture
def modules() -> tuple[ModalityAligner, MultimodalFusion]:
    aligner = ModalityAligner(target_steps=16, common_dim=128)
    fusion = MultimodalFusion(feature_dim=128, num_heads=4, dropout=0.0)
    return aligner, fusion


def _features(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 8, 512),
        torch.randn(batch_size, 16, 128),
        torch.randn(batch_size, 16, 128),
    )


def test_video_audio_fusion_preserves_batch_time_and_dimension(modules) -> None:
    aligner, fusion = modules
    video, audio, _ = _features()

    aligned = aligner(video, audio)
    result = fusion(aligned)

    assert result["features"].shape == (2, 16, 128)
    assert aligned.features["video"].shape == (2, 16, 128)
    assert aligned.features["audio"].shape == (2, 16, 128)
    assert aligned.features["text"].shape == (2, 16, 128)
    assert torch.equal(aligned.modality_mask, torch.tensor([[True, True, False]] * 2))
    assert aligned.timestamp_synchronized is False


def test_video_audio_text_fusion_and_attention_weights(modules) -> None:
    aligner, fusion = modules
    video, audio, text = _features()

    aligned = aligner(video, audio, text)
    result = fusion(aligned, return_attention_weights=True)

    assert result["features"].shape == (2, 16, 128)
    assert result["attention_weights"].shape == (2, 16, 3)
    assert result["modality_order"] == ("video", "audio", "text")
    assert torch.allclose(
        result["attention_weights"].sum(dim=-1), torch.ones(2, 16), atol=1e-6
    )


def test_missing_text_is_masked_from_fusion(modules) -> None:
    aligner, fusion = modules
    video, audio, text = _features()
    text_available = torch.tensor([True, False])

    aligned = aligner(video, audio, text, text_available=text_available)
    result = fusion(aligned, return_attention_weights=True)

    assert aligned.modality_mask[:, 2].tolist() == [True, False]
    assert torch.count_nonzero(aligned.features["text"][1]) == 0
    assert torch.count_nonzero(result["attention_weights"][1, :, 2]) == 0
    assert result["features"].shape == (2, 16, 128)


def test_encoder_output_dictionary_and_unbatched_video_are_accepted(modules) -> None:
    aligner, fusion = modules
    video, audio, _ = _features(batch_size=1)

    aligned = aligner({"features": video.squeeze(0)}, {"features": audio})
    result = fusion(aligned)

    assert result["features"].shape == (1, 16, 128)


def test_sequence_resampling_is_order_preserving_and_not_timestamp_claim(modules) -> None:
    del modules
    aligner = ModalityAligner(
        video_dim=1, audio_dim=1, text_dim=1, common_dim=1, target_steps=5
    )
    with torch.no_grad():
        for projection in aligner.projections.values():
            projection.weight.fill_(1.0)
            projection.bias.zero_()
    video = torch.tensor([[[0.0], [1.0]]])
    audio = torch.arange(5.0).reshape(1, 5, 1)

    aligned = aligner(video, audio)

    assert torch.allclose(
        aligned.features["video"].flatten(), torch.linspace(0.0, 1.0, 5)
    )
    assert aligned.alignment_method == "sequence_position_linear_resampling"
    assert aligned.timestamp_synchronized is False


@pytest.mark.parametrize(
    "video,audio,text,match",
    [
        (torch.randn(2, 8, 512, 1), torch.randn(2, 16, 128), None, "shape"),
        (torch.randn(2, 8, 511), torch.randn(2, 16, 128), None, "dimension"),
        (torch.randn(2, 8, 512), torch.randn(3, 16, 128), None, "batch sizes"),
        (torch.randn(2, 8, 512), torch.randn(2, 16, 128), torch.randn(3, 16, 128), "Text batch"),
    ],
)
def test_invalid_feature_shapes_raise_meaningful_errors(video, audio, text, match) -> None:
    aligner = ModalityAligner()
    with pytest.raises(ValueError, match=match):
        aligner(video, audio, text)


def test_text_mask_validation_and_required_modalities(modules) -> None:
    aligner, fusion = modules
    video, audio, text = _features()

    with pytest.raises(ValueError, match="shape"):
        aligner(video, audio, text, text_available=torch.ones(2, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="text_features is None"):
        aligner(video, audio, text_features=None, text_available=torch.ones(2, dtype=torch.bool))

    aligned = aligner(video, audio)
    broken = AlignedModalities(
        features=aligned.features,
        modality_mask=torch.tensor([[False, True, False], [True, True, False]]),
    )
    with pytest.raises(ValueError, match="Video and audio"):
        fusion(broken)


def test_gradients_reach_projection_and_fusion_parameters(modules) -> None:
    aligner, fusion = modules
    video, audio, text = _features()
    video.requires_grad_()
    audio.requires_grad_()
    text.requires_grad_()

    result = fusion(aligner(video, audio, text))["features"]
    result.square().mean().backward()

    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert aligner.projections["video"].weight.grad is not None
    assert fusion.cross_modal_attention.in_proj_weight.grad is not None


def test_cpu_execution(modules) -> None:
    aligner, fusion = modules
    video, audio, text = _features()

    output = fusion(aligner(video, audio, text))["features"]

    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_execution_when_available(modules) -> None:
    aligner, fusion = (module.cuda() for module in modules)
    video, audio, text = (value.cuda() for value in _features())

    output = fusion(aligner(video, audio, text))["features"]

    assert output.device.type == "cuda"
    assert output.shape == (2, 16, 128)
