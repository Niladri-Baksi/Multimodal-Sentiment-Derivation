"""
Text Encoder and Preprocessing Module for Multimodal Sentiment Derivation System.
Reference: AGENTS.md | Canonical Architecture Stage 2 (Text Modality)

This module provides:
1. TextPreprocessor: Tokenizes text transcripts using HuggingFace DistilBERT tokenizer.
2. TextEncoder: Frozen DistilBERT model projecting contextual token representations
   into the canonical sequence shape [Batch, T=16, D=128].
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class TextPreprocessor:
    """
    Tokenizes text transcripts using DistilBERT tokenizer with padding and truncation.
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 48,
    ):
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single text string into input_ids and attention_mask tensors.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must contain at least one non-whitespace character.")
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class TextEncoder(nn.Module):
    """
    Pretrained DistilBERT text encoder mapping tokenized transcripts into
    time-aligned sequence representations [Batch, T=16, D_text=128].
    
    Contract (AGENTS.md Section 6):
        Input:  input_ids [B, seq_len], attention_mask [B, seq_len]
        Output: [Batch, T=16, D_text=128]
    """
    def __init__(
        self,
        target_steps: int = 16,
        feature_dim: int = 128,
        model_name: str = "distilbert-base-uncased",
        freeze_backbone: bool = True,
    ):
        super(TextEncoder, self).__init__()
        if not isinstance(target_steps, int) or target_steps <= 0:
            raise ValueError("target_steps must be a positive integer.")
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        self.target_steps = target_steps
        self.feature_dim = feature_dim

        # Pretrained DistilBERT backbone
        self.backbone = DistilBertModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # DistilBERT's hidden width is configurable across model variants.
        hidden_dim = getattr(getattr(self.backbone, "config", None), "dim", 768)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # Adaptive temporal pooler to ensure T=16 steps aligned with Video and Audio
        self.temporal_pool = nn.AdaptiveAvgPool1d(target_steps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: Tensor of shape [B, seq_len]
            attention_mask: Tensor of shape [B, seq_len]
            return_dict: If True, returns dict {"features": [B, 16, 128]}
        Returns:
            Tensor of shape [B, 16, 128] or dict {"features": [B, 16, 128]}
        """
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise TypeError("input_ids and attention_mask must be torch.Tensor objects.")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both have shape [B, L].")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have matching shapes.")
        if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
            raise ValueError("Text encoder batch and sequence dimensions must be non-empty.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype (int32 or int64).")
        if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
            raise TypeError("attention_mask must use a boolean or integer dtype.")
        if input_ids.device != attention_mask.device:
            raise ValueError("input_ids and attention_mask must be on the same device.")
        if torch.any(input_ids < 0):
            raise ValueError("input_ids must contain non-negative token IDs.")
        vocab_size = getattr(getattr(self.backbone, "config", None), "vocab_size", None)
        if vocab_size is not None and torch.any(input_ids >= vocab_size):
            raise ValueError(f"input_ids contain a token ID outside vocabulary size {vocab_size}.")
        if torch.any((attention_mask != 0) & (attention_mask != 1)):
            raise ValueError("attention_mask values must be binary (0 or 1).")
        parameter = next(self.backbone.parameters(), None)
        if parameter is not None and input_ids.device != parameter.device:
            raise ValueError(
                f"Text inputs are on {input_ids.device}; encoder is on {parameter.device}."
            )

        # Backbone forward pass: [B, seq_len, H]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or hidden_states.shape[:2] != input_ids.shape
        ):
            raise ValueError("Text backbone must return hidden states shaped [B, L, H].")

        # Project 768 -> 128: [B, seq_len, 128]
        proj_features = self.proj(hidden_states)

        # Temporal pooling to target_steps: [B, 128, seq_len] -> [B, 128, 16] -> [B, 16, 128]
        mask = attention_mask.to(dtype=proj_features.dtype).unsqueeze(-1)
        weighted_features = proj_features * mask
        pooled_sum = self.temporal_pool(weighted_features.transpose(1, 2))
        pooled_weight = self.temporal_pool(mask.transpose(1, 2))
        aligned = (pooled_sum / pooled_weight.clamp_min(1e-8)).transpose(1, 2)
        aligned = aligned.masked_fill(pooled_weight.transpose(1, 2) == 0, 0.0)

        if return_dict:
            return {"features": aligned}
        return aligned


if __name__ == "__main__":
    # Self-test when executed directly
    print("Running TextEncoder self-test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextEncoder(target_steps=16, feature_dim=128).to(device)

    # Synthetic input batch: [Batch=4, seq_len=48]
    synthetic_ids = torch.randint(0, 1000, (4, 48)).to(device)
    synthetic_mask = torch.ones((4, 48)).to(device)

    output = model(synthetic_ids, synthetic_mask)

    print(f"Device: {device}")
    print(f"Input IDs Shape: {synthetic_ids.shape}")
    print(f"Output Shape:    {output.shape}")
    assert output.shape == torch.Size([4, 16, 128]), "Shape mismatch!"
    print("TextEncoder self-test PASSED successfully! Output matches [B, 16, 128].")
