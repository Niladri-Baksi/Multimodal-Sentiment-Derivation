"""Project modality features and resample sequences to a common length.

The current EmoReact preprocessing and encoder outputs do not carry shared
per-frame/per-token timestamps. This module therefore aligns sequence *positions*
by linear interpolation; it does not claim timestamp-based synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


MODALITY_ORDER: Tuple[str, ...] = ("video", "audio", "text")
FeatureInput = Union[torch.Tensor, Mapping[str, Any]]


@dataclass(frozen=True)
class AlignedModalities:
    """Aligned feature sequences and per-sample modality availability.

    Attributes:
        features: Video, audio and text tensors, each shaped ``[B, T, D]``.
            Text is a zero tensor for unavailable samples and is masked out.
        modality_mask: Boolean tensor shaped ``[B, 3]`` in ``MODALITY_ORDER``.
        modality_order: The fixed modality order used by the mask and fusion.
        alignment_method: Sequence-position resampling method used.
        timestamp_synchronized: Always false for the currently available data.
    """

    features: Dict[str, torch.Tensor]
    modality_mask: torch.Tensor
    modality_order: Tuple[str, ...] = MODALITY_ORDER
    alignment_method: str = "sequence_position_linear_resampling"
    timestamp_synchronized: bool = False


class ModalityAligner(nn.Module):
    """Project encoder outputs to a shared size and a common step count.

    Inputs may be feature tensors or encoder output dictionaries containing a
    ``"features"`` tensor. Batched tensors use ``[B, T, D]``; an unbatched
    video sequence ``[T, D]`` is also accepted and treated as batch size one.

    The default contract maps video/audio/text dimensions 512/128/128 to
    ``common_dim=128`` and resamples each sequence to ``target_steps=16``.
    Resampling preserves order and sequence shape, but does not establish
    real-time correspondence between modalities.
    """

    def __init__(
        self,
        video_dim: int = 512,
        audio_dim: int = 128,
        text_dim: int = 128,
        common_dim: int = 128,
        target_steps: int = 16,
    ) -> None:
        super().__init__()
        dimensions = {
            "video": video_dim,
            "audio": audio_dim,
            "text": text_dim,
            "common": common_dim,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"Feature dimensions must be positive; got {dimensions}.")
        if target_steps <= 0:
            raise ValueError("target_steps must be a positive integer.")

        self.input_dims = {name: dimensions[name] for name in MODALITY_ORDER}
        self.common_dim = common_dim
        self.target_steps = target_steps
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, common_dim)
                for name, input_dim in self.input_dims.items()
            }
        )

    @staticmethod
    def _features_tensor(value: FeatureInput, modality: str) -> torch.Tensor:
        """Extract and validate a feature tensor from an encoder output."""
        if isinstance(value, Mapping):
            if "features" not in value:
                raise ValueError(
                    f"{modality} encoder output mapping must contain a 'features' key."
                )
            value = value["features"]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{modality} features must be a torch.Tensor.")
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(
                f"{modality} features must have shape [B, T, D] or [T, D]; "
                f"received {tuple(value.shape)}."
            )
        if value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError(f"{modality} batch and sequence dimensions must be non-empty.")
        if not value.is_floating_point():
            raise TypeError(f"{modality} features must have a floating-point dtype.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{modality} features contain NaN or infinite values.")
        return value

    @staticmethod
    def _availability_mask(
        value: Optional[torch.Tensor],
        *,
        name: str,
        batch_size: int,
        reference: torch.Tensor,
        default: bool,
    ) -> torch.Tensor:
        if value is None:
            return torch.full(
                (batch_size,), default, dtype=torch.bool, device=reference.device
            )
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean torch.Tensor.")
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [{batch_size}].")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on the feature device ({reference.device}).")
        return value

    def _project_and_resample(
        self, value: torch.Tensor, modality: str
    ) -> torch.Tensor:
        expected_dim = self.input_dims[modality]
        if value.shape[-1] != expected_dim:
            raise ValueError(
                f"{modality} feature dimension must be {expected_dim}; "
                f"received {value.shape[-1]}."
            )
        projection = self.projections[modality]
        if value.device != projection.weight.device:
            raise ValueError(
                f"{modality} features are on {value.device}; aligner is on "
                f"{projection.weight.device}."
            )
        if value.dtype != projection.weight.dtype:
            raise ValueError(
                f"{modality} feature dtype is {value.dtype}; aligner expects "
                f"{projection.weight.dtype}."
            )
        projected = projection(value)
        if projected.shape[1] != self.target_steps:
            projected = F.interpolate(
                projected.transpose(1, 2),
                size=self.target_steps,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        return projected

    def forward(
        self,
        video_features: FeatureInput,
        audio_features: Optional[FeatureInput],
        text_features: Optional[FeatureInput] = None,
        *,
        audio_available: Optional[torch.Tensor] = None,
        text_available: Optional[torch.Tensor] = None,
    ) -> AlignedModalities:
        """Return projected, position-resampled features and availability mask.

        ``audio_features`` and ``audio_available`` support an absent audio
        stream and mixed-availability batches. When features are supplied and
        no mask is given, audio is considered available for every row for
        backward compatibility. Masked rows receive an explicit zero aligned
        tensor and a false availability bit; those zeros are not valid audio
        features. Video remains required. Text is optional.
        """
        raw = {"video": self._features_tensor(video_features, "video")}
        batch_size = raw["video"].shape[0]
        reference = raw["video"]
        if reference.device != self.projections["video"].weight.device:
            raise ValueError("Video features and aligner must be on the same device.")
        if reference.dtype != self.projections["video"].weight.dtype:
            raise ValueError("Video feature dtype must match the aligner parameter dtype.")

        audio_mask = self._availability_mask(
            audio_available,
            name="audio_available",
            batch_size=batch_size,
            reference=reference,
            default=audio_features is not None,
        )
        if audio_features is None:
            if bool(audio_mask.any()):
                raise ValueError(
                    "audio_available cannot mark audio present when audio_features is None."
                )
            aligned_audio = reference.new_zeros(
                (batch_size, self.target_steps, self.common_dim)
            )
        else:
            audio = self._features_tensor(audio_features, "audio")
            if audio.shape[0] != batch_size:
                raise ValueError(
                    f"Video and audio batch sizes must match; got {batch_size} and "
                    f"{audio.shape[0]}."
                )
            if audio.device != reference.device or audio.dtype != reference.dtype:
                raise ValueError("Audio features must match video device and dtype.")
            aligned_audio = self._project_and_resample(audio, "audio")
            aligned_audio = aligned_audio.masked_fill(~audio_mask[:, None, None], 0.0)

        if text_features is not None:
            text = self._features_tensor(text_features, "text")
            if text.shape[0] != batch_size:
                raise ValueError(
                    f"Text batch size must be {batch_size}; received {text.shape[0]}."
                )
            if text.device != reference.device or text.dtype != reference.dtype:
                raise ValueError("Text features must match video device and dtype.")
            raw["text"] = text

        text_mask = self._availability_mask(
            text_available,
            name="text_available",
            batch_size=batch_size,
            reference=reference,
            default=text_features is not None,
        )
        if text_features is None:
            if bool(text_mask.any()):
                raise ValueError(
                    "text_available cannot mark text present when text_features is None."
                )

        aligned: Dict[str, torch.Tensor] = {}
        aligned["video"] = self._project_and_resample(raw["video"], "video")
        aligned["audio"] = aligned_audio
        if "text" in raw:
            aligned_text = self._project_and_resample(raw["text"], "text")
            aligned_text = aligned_text.masked_fill(~text_mask[:, None, None], 0.0)
            aligned["text"] = aligned_text
        else:
            aligned["text"] = reference.new_zeros(
                (batch_size, self.target_steps, self.common_dim)
            )

        modality_mask = torch.stack(
            (
                torch.ones(batch_size, dtype=torch.bool, device=reference.device),
                audio_mask,
                text_mask,
            ),
            dim=1,
        )
        return AlignedModalities(features=aligned, modality_mask=modality_mask)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ModalityAligner":
        """Build an aligner from the ``modality_alignment`` YAML section."""
        alignment_cfg = config.get("modality_alignment", config)
        return cls(
            video_dim=alignment_cfg.get("video_input_dim", 512),
            audio_dim=alignment_cfg.get("audio_input_dim", 128),
            text_dim=alignment_cfg.get("text_input_dim", 128),
            common_dim=alignment_cfg.get("common_dim", 128),
            target_steps=alignment_cfg.get("target_steps", 16),
        )
