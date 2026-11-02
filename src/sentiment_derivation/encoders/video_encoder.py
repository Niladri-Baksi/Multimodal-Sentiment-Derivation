"""Video feature encoder module for Multimodal Emotion-to-Sentiment Derivation.

Extracts frame-level visual embeddings using a pretrained ResNet-18 backbone,
preserving the temporal sequence dimension for subsequent temporal analysis (Stage 4).

Interface compliance:
- Input shape: (B, T, C, H, W) or (T, C, H, W)
- Output shape: (B, T, 512) or (T, 512)
- Output dictionary:
    {
        "features": torch.Tensor,
        "metadata": {
            "modality": "video",
            "batch_size": int,
            "num_frames": int,
            "feature_dim": int,
            "device": str
        }
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

from sentiment_derivation.data.transforms import VideoClipTransform

logger = logging.getLogger(__name__)


class VideoEncoder(nn.Module):
    """Visual feature encoder using ResNet-18 backbone.

    Processes video clips frame-by-frame while preserving the temporal dimension T
    to supply sequential representations for downstream temporal models (BiLSTM).

    Attributes:
        backbone (nn.Module): Pretrained ResNet-18 feature extractor without classifier.
        feature_dim (int): Output embedding dimension per frame (default: 512).
        freeze_backbone (bool): Whether backbone weights are frozen.
        transform (Optional[VideoClipTransform]): Optional internal spatial transform.
    """

    def __init__(
        self,
        pretrained: bool = True,
        weights: Optional[Union[str, models.ResNet18_Weights]] = "DEFAULT",
        feature_dim: int = 512,
        freeze: bool = True,
        apply_transforms: bool = False,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize the VideoEncoder.

        Args:
            pretrained: Whether to load pretrained ImageNet weights.
            weights: Specific torchvision weights enum or string identifier (default: "DEFAULT").
            feature_dim: Target visual feature dimension per frame (default: 512).
            freeze: Whether to freeze backbone weights (default: True per AGENTS.md).
            apply_transforms: If True, applies resizing and ImageNet normalization internally.
            image_size: Target image dimensions if apply_transforms is True.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze
        self.apply_transforms = apply_transforms

        # 1. Initialize ResNet-18 backbone
        if pretrained:
            if isinstance(weights, str) and weights.upper() == "DEFAULT":
                resnet_weights = models.ResNet18_Weights.DEFAULT
            else:
                resnet_weights = weights
            self.backbone = models.resnet18(weights=resnet_weights)
        else:
            self.backbone = models.resnet18(weights=None)

        # 2. Extract 512-dim features: replace the 1000-class classification head with Identity
        backbone_out_features = self.backbone.fc.in_features  # 512 for ResNet-18
        self.backbone.fc = nn.Identity()

        # 3. Optional linear projection if custom feature_dim is requested
        if feature_dim != backbone_out_features:
            self.projection = nn.Linear(backbone_out_features, feature_dim)
        else:
            self.projection = nn.Identity()

        # 4. Optional built-in frame transformation module
        if apply_transforms:
            self.transform: Optional[VideoClipTransform] = VideoClipTransform(image_size=image_size)
        else:
            self.transform = None

        # 5. Freeze parameters if requested (standard baseline for initial experiments)
        if freeze:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all backbone parameters to prevent gradient updates."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.freeze_backbone = True
        logger.info("VideoEncoder backbone parameters have been frozen.")

    def unfreeze_encoder(self, stages: Optional[List[str]] = None) -> None:
        """Unfreeze backbone parameters for selective fine-tuning.

        Args:
            stages: Optional list of sub-module names to unfreeze (e.g. ['layer4']).
                If None, unfreezes all backbone parameters.
        """
        if stages is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.freeze_backbone = False
            logger.info("All VideoEncoder backbone parameters have been unfrozen.")
        else:
            for name, child in self.backbone.named_children():
                if name in stages:
                    for param in child.parameters():
                        param.requires_grad = True
                    logger.info(f"Unfrozen backbone sub-module: {name}")

    def forward(
        self,
        frames: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Extract frame-level visual embeddings while preserving temporal order.

        Args:
            frames: Input video frame tensor.
                Expected shapes:
                - 5D Tensor: (batch_size, num_frames, channels, height, width) [B, T, C, H, W]
                - 4D Tensor: (num_frames, channels, height, width) [T, C, H, W]
            return_dict: Whether to return a standardized contract dictionary
                or the raw embedding tensor.

        Returns:
            If return_dict is True:
                Dictionary with "features" tensor of shape (B, T, feature_dim) or (T, feature_dim)
                and "metadata" with execution information.
            If return_dict is False:
                Tensor of shape (B, T, feature_dim) or (T, feature_dim).

        Raises:
            ValueError: If input tensor does not have 4 or 5 dimensions.
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(frames)}")

        input_dim = frames.ndim
        if input_dim not in (4, 5):
            raise ValueError(
                f"VideoEncoder expects 4D (T, C, H, W) or 5D (B, T, C, H, W) tensor, got {input_dim}D tensor."
            )

        # Apply optional internal spatial transforms (resizing, ImageNet normalization)
        if self.transform is not None:
            frames = self.transform(frames)

        # Handle 4D input (single video sequence without batch dimension): (T, C, H, W)
        if input_dim == 4:
            num_frames, c, h, w = frames.shape
            batch_size = 1
            is_batched = False

            # Extract features per frame: (T, 512)
            features = self.backbone(frames)
            features = self.projection(features)

        # Handle 5D input (batched video sequences): (B, T, C, H, W)
        else:
            batch_size, num_frames, c, h, w = frames.shape
            is_batched = True

            # Fold batch and time into a single 2D batch of images: (B * T, C, H, W)
            folded_frames = frames.contiguous().view(batch_size * num_frames, c, h, w)

            # Pass through ResNet-18 backbone: (B * T, 512)
            frame_features = self.backbone(folded_frames)
            frame_features = self.projection(frame_features)

            # Unfold back to preserve temporal dimensions: (B, T, 512)
            features = frame_features.view(batch_size, num_frames, self.feature_dim)

        if not return_dict:
            return features

        output_dict: Dict[str, Any] = {
            "features": features,
            "metadata": {
                "modality": "video",
                "batch_size": batch_size if is_batched else 1,
                "num_frames": num_frames,
                "feature_dim": self.feature_dim,
                "is_batched": is_batched,
                "device": str(frames.device),
            },
        }

        return output_dict

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> VideoEncoder:
        """Construct a VideoEncoder instance from a configuration dictionary.

        Args:
            config: Configuration dictionary (e.g. from configs/model.yaml).

        Returns:
            Configured VideoEncoder instance.
        """
        v_cfg = config.get("video_encoder", config)
        return cls(
            pretrained=v_cfg.get("pretrained", True),
            weights=v_cfg.get("weights", "DEFAULT"),
            feature_dim=v_cfg.get("feature_dim", 512),
            freeze=v_cfg.get("freeze_backbone", True),
            apply_transforms=v_cfg.get("apply_transforms", False),
            image_size=tuple(v_cfg.get("image_size", (224, 224))),
        )

