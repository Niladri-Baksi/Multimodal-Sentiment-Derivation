"""
Audio Encoder and Preprocessing Module for Multimodal Sentiment Derivation System.
Reference: AGENTS.md | Canonical Architecture Stage 2 (Audio Modality)

This module provides:
1. AudioPreprocessor: Standardizes audio from video clips to 16 kHz mono Log-Mel spectrograms.
2. AudioEncoder: Lightweight 2D-CNN mapping Log-Mel spectrograms to [Batch, T=16, D=128].
"""

import os
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class AudioPreprocessor:
    """
    Extracts, standardizes, and converts raw audio waveforms into normalized Log-Mel Spectrograms.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * duration_sec)
        self.n_mels = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def process_file(self, file_path: str) -> torch.Tensor:
        """
        Loads an audio or video file and returns a normalized Log-Mel spectrogram tensor.
        Output shape: [1, n_mels=64, time_steps]
        """
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            # Fallback for silent or unreadable audio tracks
            waveform = torch.zeros(1, self.target_length)
            sr = self.sample_rate

        # 1. Resample to target sample rate (16 kHz)
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 2. Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Standardize duration (pad or center-crop)
        num_samples = waveform.shape[1]
        if num_samples < self.target_length:
            padding = self.target_length - num_samples
            waveform = F.pad(waveform, (0, padding))
        elif num_samples > self.target_length:
            start_idx = (num_samples - self.target_length) // 2
            waveform = waveform[:, start_idx : start_idx + self.target_length]

        # 4. Compute Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6)

        # 5. Instance Normalization (mean 0, std 1)
        mean = log_mel.mean()
        std = log_mel.std() + 1e-6
        norm_log_mel = (log_mel - mean) / std

        return norm_log_mel


class AudioEncoder(nn.Module):
    """
    Lightweight 2D-CNN Audio Encoder mapping Log-Mel spectrograms to time-aligned feature sequences.
    
    Contract (AGENTS.md Section 6):
        Input:  [Batch, 1, n_mels=64, T_mel]
        Output: [Batch, T=16, D_audio=128]
    """
    def __init__(self, target_steps: int = 16, feature_dim: int = 128):
        super(AudioEncoder, self).__init__()
        self.target_steps = target_steps
        self.feature_dim = feature_dim

        # Conv Block 1: 64 mel bands -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Conv Block 2: 32 -> 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # Conv Block 3: Frequency pooled to 1, time pooled to target_steps (T=16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, target_steps)),
        )

    def forward(
        self, x: torch.Tensor, return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input Log-Mel spectrogram tensor of shape [B, 1, 64, T_mel]
            return_dict: If True, returns a structured dictionary matching AGENTS.md contract.
        Returns:
            Tensor of shape [B, 16, 128] or dict {"features": [B, 16, 128]}
        """
        # x: [B, 1, 64, T_mel]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [B, feature_dim=128, 1, target_steps=16]

        # Squeeze frequency dimension: [B, 128, 16]
        x = x.squeeze(2)

        # Permute to canonical [B, T, D] format: [B, 16, 128]
        x = x.permute(0, 2, 1)

        if return_dict:
            return {"features": x}
        return x


if __name__ == "__main__":
    # Self-test when executed directly
    print("Running AudioEncoder self-test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioEncoder(target_steps=16, feature_dim=128).to(device)

    # Synthetic input batch: [Batch=4, Channels=1, Mel_bands=64, Time_frames=401]
    synthetic_audio = torch.randn(4, 1, 64, 401).to(device)
    output = model(synthetic_audio)

    print(f"Device: {device}")
    print(f"Input Shape:  {synthetic_audio.shape}")
    print(f"Output Shape: {output.shape}")
    assert output.shape == torch.Size([4, 16, 128]), "Shape mismatch!"
    print("AudioEncoder self-test PASSED successfully! Output matches [B, 16, 128].")
