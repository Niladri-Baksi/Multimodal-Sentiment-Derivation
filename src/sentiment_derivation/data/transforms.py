"""Video frame transformation and preprocessing utilities.

Provides reusable transformations for preparing video frames for CNN encoders
(e.g., ResNet-18) according to standard computer vision conventions.

Key functions:
- Video frame resizing and aspect-ratio handling.
- ImageNet channel normalization.
- Batch and temporal dimension preservation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


# Canonical ImageNet normalization constants for pretrained PyTorch vision models
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_frame_transform(
    image_size: Tuple[int, int] = (224, 224),
    is_train: bool = False,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Create a torchvision Compose transform for individual video frames.

    Args:
        image_size: Target height and width (default: (224, 224)).
        is_train: Whether to include data augmentations (e.g. random horizontal flip).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list: List[transforms.Transform] = []

    # 1. Resize frames to standard input size for ResNet
    transform_list.append(transforms.Resize(image_size, antialias=True))

    # 2. Training augmentations (optional, disabled by default for feature extraction)
    if is_train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # 3. Normalize channels using ImageNet statistics
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


class VideoClipTransform(nn.Module):
    """Transform module for video clips preserving temporal dimension (T).

    Applies spatial resizing and normalization across all frames in a video sequence
    while preserving temporal order.

    Expected input tensor shapes:
        - 5D Tensor: (batch_size, num_frames, channels, height, width) [B, T, C, H, W]
        - 4D Tensor: (num_frames, channels, height, width) [T, C, H, W]
        - List of NumPy arrays: list of T frames each with shape (H, W, C)

    Output tensor shape:
        - (B, T, C, image_size[0], image_size[1]) or (T, C, image_size[0], image_size[1])
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        is_train: bool = False,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.is_train = is_train

        # Frame-level transformation pipeline
        self.transform = get_frame_transform(
            image_size=image_size,
            is_train=is_train,
            mean=mean,
            std=std,
        )

    def forward(
        self,
        video: Union[torch.Tensor, List[np.ndarray]],
    ) -> torch.Tensor:
        """Apply video transform to input frames.

        Args:
            video: Input video frames as a 4D/5D Tensor or list of NumPy arrays.

        Returns:
            Normalized float32 tensor of shape (B, T, C, H, W) or (T, C, H, W).
        """
        # Case 1: List of NumPy arrays (T frames, each H x W x C uint8)
        if isinstance(video, list):
            tensor_frames = []
            for frame in video:
                # Convert HWC uint8 -> CHW float32 [0.0, 1.0]
                t_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                tensor_frames.append(t_frame)
            # Stack into (T, C, H, W)
            video_tensor = torch.stack(tensor_frames, dim=0)
            return self._transform_sequence(video_tensor)

        # Case 2: PyTorch Tensor
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or list of numpy arrays, got {type(video)}")

        # Ensure float in [0.0, 1.0] if passed as uint8
        if video.dtype == torch.uint8:
            video = video.float() / 255.0

        if video.ndim == 4:
            # Shape: (T, C, H, W)
            return self._transform_sequence(video)
        elif video.ndim == 5:
            # Shape: (B, T, C, H, W)
            batch_size, num_frames, c, h, w = video.shape
            # Fold batch and time into a single batch of images
            folded = video.view(batch_size * num_frames, c, h, w)
            transformed = self.transform(folded)
            # Unfold back to (B, T, C, target_h, target_w)
            return transformed.view(batch_size, num_frames, c, self.image_size[0], self.image_size[1])
        else:
            raise ValueError(
                f"Expected 4D (T, C, H, W) or 5D (B, T, C, H, W) tensor, got {video.ndim}D tensor"
            )

    def _transform_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Transform a 4D sequence of frames (T, C, H, W)."""
        # sequence shape: (T, C, H, W)
        transformed = self.transform(sequence)
        return transformed

