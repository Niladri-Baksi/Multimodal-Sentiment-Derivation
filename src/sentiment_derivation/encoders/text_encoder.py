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
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single text string into input_ids and attention_mask tensors.
        """
        encoded = self.tokenizer(
            str(text),
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
        self.target_steps = target_steps
        self.feature_dim = feature_dim

        # Pretrained DistilBERT backbone
        self.backbone = DistilBertModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection: 768 -> feature_dim (128)
        self.proj = nn.Sequential(
            nn.Linear(768, feature_dim),
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
        # Backbone forward pass: [B, seq_len, 768]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Project 768 -> 128: [B, seq_len, 128]
        proj_features = self.proj(hidden_states)

        # Temporal pooling to target_steps: [B, 128, seq_len] -> [B, 128, 16] -> [B, 16, 128]
        aligned = self.temporal_pool(proj_features.transpose(1, 2)).transpose(1, 2)

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
