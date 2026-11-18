"""
Audio Encoder and Preprocessing Module for Multimodal Sentiment Derivation System.
Reference: AGENTS.md | Canonical Architecture Stage 2 (Audio Modality)

This module provides:
1. AudioPreprocessor: Standardizes audio from video clips to 16 kHz mono Log-Mel spectrograms.
2. AudioEncoder: Lightweight 2D-CNN mapping Log-Mel spectrograms to [Batch, T=16, D=128].
"""

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchaudio
    import torchaudio.transforms as T
except (ImportError, OSError) as exc:  # Keep the CNN usable without audio I/O extras.
    torchaudio = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _TORCHAUDIO_IMPORT_ERROR = exc
else:
    _TORCHAUDIO_IMPORT_ERROR = None


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
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ValueError("duration_sec must be finite and positive.")
        if n_mels <= 0 or n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_mels, n_fft, and hop_length must be positive.")
        if T is None:
            raise ImportError(
                "AudioPreprocessor requires torchaudio for audio loading and mel "
                "spectrogram extraction."
            ) from _TORCHAUDIO_IMPORT_ERROR
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
        if torchaudio is None:
            raise ImportError(
                "AudioPreprocessor requires torchaudio to load audio files."
            ) from _TORCHAUDIO_IMPORT_ERROR
        try:
            waveform, sr = torchaudio.load(file_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Unable to load audio file {file_path!r}: {exc}") from exc

        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
            raise ValueError("Loaded audio waveform must be a [channels, samples] tensor.")
        if waveform.shape[0] <= 0 or waveform.shape[1] <= 0:
            raise ValueError("Loaded audio waveform must have channels and samples.")
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("Loaded audio sample rate must be a positive integer.")
        if not torch.is_floating_point(waveform) or not torch.isfinite(waveform).all():
            raise ValueError("Loaded audio waveform must contain finite floating-point values.")

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
        if (
            not isinstance(mel_spec, torch.Tensor)
            or mel_spec.ndim != 3
            or mel_spec.shape[0] != 1
            or mel_spec.shape[1] != self.n_mels
            or mel_spec.shape[2] <= 0
        ):
            raise ValueError(
                f"Mel transform must return [1, {self.n_mels}, T] with T > 0."
            )
        if not torch.isfinite(mel_spec).all():
            raise ValueError("Mel transform returned NaN or infinite values.")
        log_mel = torch.log(mel_spec + 1e-6)

        # 5. Instance Normalization (mean 0, std 1)
        mean = log_mel.mean()
        std = log_mel.std(unbiased=False) + 1e-6
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
        if target_steps <= 0 or feature_dim <= 0:
            raise ValueError("target_steps and feature_dim must be positive.")
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
        self,
        x: Optional[torch.Tensor],
        return_dict: bool = False,
        *,
        audio_available: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input Log-Mel spectrogram tensor of shape [B, 1, 64, T_mel].
                ``None`` is allowed only with an all-false availability mask.
            return_dict: If True, returns a structured dictionary matching AGENTS.md contract.
            audio_available: Optional boolean tensor of shape [B]. False rows
                are excluded from the CNN so they cannot affect BatchNorm.
                Without a mask, every tensor row is treated as genuine audio.
        Returns:
            Tensor of shape [B, 16, 128] or dict {"features": [B, 16, 128]}
        """
        parameter = next(self.parameters())
        if x is None:
            if audio_available is None:
                raise ValueError("audio_available is required when audio input is None.")
            if not isinstance(audio_available, torch.Tensor) or audio_available.dtype != torch.bool:
                raise TypeError("audio_available must be a boolean torch.Tensor.")
            if audio_available.ndim != 1 or audio_available.numel() == 0:
                raise ValueError("audio_available must be a non-empty tensor shaped [B].")
            if audio_available.device != parameter.device:
                raise ValueError("audio_available must be on the encoder device.")
            if bool(audio_available.any()):
                raise ValueError("audio_available cannot mark rows present when audio input is None.")
            output = parameter.new_zeros(
                (audio_available.shape[0], self.target_steps, self.feature_dim)
            )
            return {"features": output} if return_dict else output

        if not isinstance(x, torch.Tensor):
            raise TypeError("AudioEncoder input must be a torch.Tensor or None.")
        if x.ndim != 4:
            raise ValueError("AudioEncoder expects [B, 1, 64, T_mel] input.")
        if x.shape[0] <= 0:
            raise ValueError("AudioEncoder input batch must not be empty.")
        if x.shape[1] != 1 or x.shape[2] != 64 or x.shape[3] < 4:
            raise ValueError("AudioEncoder expects [B, 1, 64, T_mel] with T_mel >= 4.")
        if not x.is_floating_point():
            raise TypeError("AudioEncoder input must be floating point.")
        if x.device != parameter.device:
            raise ValueError(f"Audio input is on {x.device}; encoder is on {parameter.device}.")
        if not torch.isfinite(x).all():
            raise ValueError("AudioEncoder input contains NaN or infinite values.")
        if x.dtype != parameter.dtype:
            x = x.to(dtype=parameter.dtype)

        if audio_available is None:
            audio_available = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        elif not isinstance(audio_available, torch.Tensor) or audio_available.dtype != torch.bool:
            raise TypeError("audio_available must be a boolean torch.Tensor.")
        elif audio_available.shape != (x.shape[0],):
            raise ValueError(f"audio_available must have shape [{x.shape[0]}].")
        elif audio_available.device != x.device:
            raise ValueError("audio_available must be on the audio input device.")

        if not bool(audio_available.any()):
            output = parameter.new_zeros((x.shape[0], self.target_steps, self.feature_dim))
            return {"features": output} if return_dict else output

        present_indices = audio_available.nonzero(as_tuple=False).squeeze(1)
        x = x.index_select(0, present_indices)

        # x: [B, 1, 64, T_mel]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # [B, feature_dim=128, 1, target_steps=16]

        # Squeeze frequency dimension: [B, 128, 16]
        x = x.squeeze(2)

        # Permute to canonical [B, T, D] format: [B, 16, 128]
        x = x.permute(0, 2, 1)

        # Restore original batch order with explicit zero placeholders for
        # unavailable rows. Their input samples never entered BatchNorm.
        output = parameter.new_zeros(
            (audio_available.shape[0], self.target_steps, self.feature_dim)
        ).index_copy(0, present_indices, x)

        if return_dict:
            return {"features": output}
        return output


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
