"""Temporal sequence encoder module for Multimodal Sentiment Derivation system.

Reference: AGENTS.md | Canonical Architecture Stage 4 (Temporal Encoding).

This module provides:
1. TemporalEncoderOutput: Container holding bidirectional temporal feature sequence
   and pooled sequence representations.
2. TemporalEncoder: Bidirectional LSTM network modeling temporal context across aligned
   multimodal sequence steps (T=16).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TemporalEncoderOutput:
    """Standardized output container for temporal encoding.

    Attributes:
        features: BiLSTM encoded temporal sequence shaped ``[B, T, output_dim]``.
        pooled_features: Mean-pooled sequence summary shaped ``[B, output_dim]``.
        last_hidden: Final timestep representation shaped ``[B, output_dim]``.
    """

    features: torch.Tensor
    pooled_features: torch.Tensor
    last_hidden: torch.Tensor

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in TemporalEncoderOutput.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary matching AGENTS.md contracts."""
        return {
            "features": self.features,
            "pooled_features": self.pooled_features,
            "last_hidden": self.last_hidden,
        }


class TemporalEncoder(nn.Module):
    """Bidirectional LSTM encoder modeling temporal dependencies across sequence steps.

    Per AGENTS.md Stage 4:
    - Input: Aligned fused multimodal features ``[B, T=16, input_dim]`` or ``[B, input_dim]``.
    - Output: Bidirectional context embeddings ``[B, T, output_dim]``.
    - Supports residual skip-connection and LayerNorm stabilization.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
        residual: bool = True,
    ) -> None:
        """Initialize TemporalEncoder.

        Args:
            input_dim: Dimension of incoming fused multimodal representations (default 128).
            hidden_dim: Output dimension of the temporal encoder (default 128).
            num_layers: Number of stacked LSTM layers (default 1).
            dropout: Dropout probability between LSTM layers and residual projection.
            bidirectional: If True, uses bidirectional LSTM.
            residual: If True and input_dim == hidden_dim, adds residual connection.
        """
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("All dimension and layer arguments must be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.residual = residual

        # For bidirectional LSTM, each direction has hidden_dim // 2 units
        if bidirectional:
            if hidden_dim % 2 != 0:
                raise ValueError(f"hidden_dim ({hidden_dim}) must be even for bidirectional LSTM.")
            lstm_hidden_size = hidden_dim // 2
        else:
            lstm_hidden_size = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional linear projection if input_dim != hidden_dim for residual matching
        if input_dim != hidden_dim and residual:
            self.residual_proj: Optional[nn.Linear] = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        features: Union[torch.Tensor, Mapping[str, Any]],
    ) -> TemporalEncoderOutput:
        """Encode sequence of multimodal features into bidirectional temporal representations.

        Args:
            features: Tensor shaped ``[B, T, input_dim]``, ``[B, input_dim]``,
                or dictionary containing ``"features"``.

        Returns:
            TemporalEncoderOutput with features, pooled_features, and last_hidden.
        """
        if isinstance(features, Mapping):
            if "features" not in features:
                raise ValueError("Input mapping must contain a 'features' key.")
            features = features["features"]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected features to be torch.Tensor; got {type(features).__name__}."
            )

        orig_ndim = features.ndim
        if orig_ndim == 1:
            # [D] -> [1, 1, D]
            features = features.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 2:
            # [B, D] -> [B, 1, D]
            features = features.unsqueeze(1)
        elif orig_ndim != 3:
            raise ValueError(f"Expected 1D, 2D, or 3D input; got shape {tuple(features.shape)}.")

        batch_size, steps, dim = features.shape
        if dim != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}; got {dim}."
            )

        # 1. BiLSTM forward pass: [B, T, hidden_dim]
        lstm_out, _ = self.lstm(features)

        # 2. Residual connection and LayerNorm
        if self.residual:
            if self.residual_proj is not None:
                res = self.residual_proj(features)
            else:
                res = features
            encoded = self.norm(self.dropout(lstm_out) + res)
        else:
            encoded = self.norm(self.dropout(lstm_out))

        # 3. Pooling representations
        pooled = torch.mean(encoded, dim=1)  # [B, hidden_dim]
        last_hidden = encoded[:, -1, :]      # [B, hidden_dim]

        # Squeeze sequence dimension if original input was 2D or 1D
        if orig_ndim == 2:
            encoded = encoded.squeeze(1)
        elif orig_ndim == 1:
            encoded = encoded.squeeze(0).squeeze(0)
            pooled = pooled.squeeze(0)
            last_hidden = last_hidden.squeeze(0)

        return TemporalEncoderOutput(
            features=encoded,
            pooled_features=pooled,
            last_hidden=last_hidden,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TemporalEncoder":
        """Instantiate encoder from configuration dictionary."""
        cfg = config.get("temporal", config)
        return cls(
            input_dim=cfg.get("input_dim", 128),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.1),
            bidirectional=cfg.get("bidirectional", True),
            residual=cfg.get("residual", True),
        )
