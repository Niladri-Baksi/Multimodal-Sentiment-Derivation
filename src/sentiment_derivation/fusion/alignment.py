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
        if value.shape[1] == 0:
            raise ValueError(f"{modality} sequence must contain at least one step.")
        if not value.is_floating_point():
            raise TypeError(f"{modality} features must have a floating-point dtype.")
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
        projected = self.projections[modality](value)
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
        audio_features: FeatureInput,
        text_features: Optional[FeatureInput] = None,
        *,
        text_available: Optional[torch.Tensor] = None,
    ) -> AlignedModalities:
        """Return projected, position-resampled features and availability mask.

        ``text_available`` may mark transcript availability per batch row. If
        omitted, supplied text features are considered available for every row.
        Video and audio are required for this project's AV baseline.
        """
        raw = {
            "video": self._features_tensor(video_features, "video"),
            "audio": self._features_tensor(audio_features, "audio"),
        }
        if raw["video"].shape[0] != raw["audio"].shape[0]:
            raise ValueError(
                "Video and audio batch sizes must match; got "
                f"{raw['video'].shape[0]} and {raw['audio'].shape[0]}."
            )
        batch_size = raw["video"].shape[0]
        reference = raw["video"]
        for modality, value in raw.items():
            if value.device != reference.device:
                raise ValueError(f"{modality} features must be on {reference.device}.")
            if value.dtype != reference.dtype:
                raise ValueError("All supplied modality features must have the same dtype.")

        if text_features is not None:
            text = self._features_tensor(text_features, "text")
            if text.shape[0] != batch_size:
                raise ValueError(
                    f"Text batch size must be {batch_size}; received {text.shape[0]}."
                )
            if text.device != reference.device or text.dtype != reference.dtype:
                raise ValueError("Text features must match video device and dtype.")
            raw["text"] = text

        if text_available is not None:
            if not isinstance(text_available, torch.Tensor):
                raise TypeError("text_available must be a boolean torch.Tensor.")
            if text_available.shape != (batch_size,):
                raise ValueError(
                    f"text_available must have shape [{batch_size}]; "
                    f"received {tuple(text_available.shape)}."
                )
            if text_available.dtype != torch.bool:
                raise TypeError("text_available must have boolean dtype.")
            if text_available.device != reference.device:
                raise ValueError("text_available must be on the feature device.")
            if text_features is None and bool(text_available.any()):
                raise ValueError(
                    "text_available cannot mark text present when text_features is None."
                )

        aligned: Dict[str, torch.Tensor] = {}
        for modality in ("video", "audio"):
            aligned[modality] = self._project_and_resample(raw[modality], modality)

        text_mask = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
        if "text" in raw:
            text_mask.fill_(True)
            if text_available is not None:
                text_mask = text_available
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
                torch.ones(batch_size, dtype=torch.bool, device=reference.device),
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
