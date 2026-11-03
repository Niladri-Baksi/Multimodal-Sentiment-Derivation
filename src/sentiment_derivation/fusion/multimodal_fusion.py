"""Lightweight cross-modal attention fusion over aligned feature sequences."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import nn

from sentiment_derivation.fusion.alignment import AlignedModalities, MODALITY_ORDER


class MultimodalFusion(nn.Module):
    """Fuse modality tokens independently at each aligned sequence position.

    A learned query attends over the available video/audio/optional-text tokens
    at each of ``T`` positions. The module does not mix information across time;
    its output keeps shape ``[B, T, D]`` for downstream temporal modelling.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        if num_heads <= 0 or feature_dim % num_heads != 0:
            raise ValueError(
                "num_heads must be positive and divide feature_dim; "
                f"received num_heads={num_heads}, feature_dim={feature_dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.fusion_query = nn.Parameter(torch.empty(1, 1, feature_dim))
        nn.init.normal_(self.fusion_query, mean=0.0, std=feature_dim**-0.5)
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(feature_dim)

        feedforward_dim = hidden_dim or feature_dim * 2
        if feedforward_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer when provided.")
        self.feedforward = nn.Sequential(
            nn.Linear(feature_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feature_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        aligned: AlignedModalities,
        *,
        return_attention_weights: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, Union[torch.Tensor, tuple[str, ...]]]]:
        """Fuse the aligned modalities and preserve their temporal dimension.

        Args:
            aligned: Output from :class:`ModalityAligner`.
            return_attention_weights: Include per-position modality weights
                shaped ``[B, T, 3]`` in the returned mapping when true.

        Returns:
            A mapping containing ``features`` shaped ``[B, T, feature_dim]``.
            Optional ``attention_weights`` follow ``aligned.modality_order``.
        """
        if not isinstance(aligned, AlignedModalities):
            raise TypeError("aligned must be an AlignedModalities instance.")
        if aligned.modality_order != MODALITY_ORDER:
            raise ValueError(
                f"Expected modality order {MODALITY_ORDER}; "
                f"received {aligned.modality_order}."
            )
        if set(aligned.features) != set(MODALITY_ORDER):
            raise ValueError(
                f"Aligned features must contain exactly {MODALITY_ORDER}."
            )

        modality_mask = aligned.modality_mask
        if not isinstance(modality_mask, torch.Tensor) or modality_mask.dtype != torch.bool:
            raise TypeError("modality_mask must be a boolean torch.Tensor.")
        if modality_mask.ndim != 2 or modality_mask.shape[1] != len(MODALITY_ORDER):
            raise ValueError(
                "modality_mask must have shape [B, 3]; "
                f"received {tuple(modality_mask.shape)}."
            )
        if not bool(modality_mask[:, :2].all()):
            raise ValueError("Video and audio must be available for every sample.")

        features = [aligned.features[name] for name in MODALITY_ORDER]
        reference = features[0]
        if reference.ndim != 3:
            raise ValueError("Each aligned modality must have shape [B, T, D].")
        batch_size, steps, feature_dim = reference.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Aligned feature dimension must be {self.feature_dim}; received {feature_dim}."
            )
        for name, value in zip(MODALITY_ORDER, features):
            if value.ndim != 3 or value.shape != reference.shape:
                raise ValueError(
                    f"{name} aligned features must have shape {tuple(reference.shape)}; "
                    f"received {tuple(value.shape)}."
                )
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("All aligned features must share device and dtype.")
        if modality_mask.shape[0] != batch_size:
            raise ValueError(
                f"modality_mask batch size must be {batch_size}; "
                f"received {modality_mask.shape[0]}."
            )
        if modality_mask.device != reference.device:
            raise ValueError("modality_mask must be on the feature device.")

        # [B, T, M, D] -> [B*T, M, D]: attention is across modalities only.
        modality_tokens = torch.stack(features, dim=2).reshape(
            batch_size * steps, len(MODALITY_ORDER), self.feature_dim
        )
        key_padding_mask = (
            ~modality_mask[:, None, :]
            .expand(batch_size, steps, len(MODALITY_ORDER))
            .reshape(batch_size * steps, len(MODALITY_ORDER))
        )
        query = self.fusion_query.expand(batch_size * steps, -1, -1)
        attended, attention_weights = self.cross_modal_attention(
            query,
            modality_tokens,
            modality_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention_weights,
            average_attn_weights=True,
        )
        fused = self.attention_norm(query + attended)
        fused = self.output_norm(fused + self.feedforward(fused))
        fused = fused.reshape(batch_size, steps, self.feature_dim)

        result: Dict[str, Any] = {"features": fused}
        if return_attention_weights:
            result["attention_weights"] = attention_weights.reshape(
                batch_size, steps, len(MODALITY_ORDER)
            )
            result["modality_order"] = aligned.modality_order
        return result

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MultimodalFusion":
        """Build fusion from the ``multimodal_fusion`` YAML section."""
        fusion_cfg = config.get("multimodal_fusion", config)
        alignment_cfg = config.get("modality_alignment", {})
        return cls(
            feature_dim=fusion_cfg.get(
                "feature_dim", alignment_cfg.get("common_dim", 128)
            ),
            num_heads=fusion_cfg.get("num_heads", 4),
            hidden_dim=fusion_cfg.get("hidden_dim"),
            dropout=fusion_cfg.get("dropout", 0.1),
        )
