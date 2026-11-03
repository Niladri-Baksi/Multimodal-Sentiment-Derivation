# Phase 4 Module Interfaces

## Modality alignment

`ModalityAligner` accepts encoder tensors or encoder output mappings containing
`features`:

- Video: `[B, T_video, 512]` or unbatched `[T_video, 512]`.
- Audio: `[B, T_audio, 128]`.
- Optional text: `[B, T_text, 128]` or `None`.

It projects each available modality to the configured common feature size
(default 128) and linearly resamples the sequence-position axis to the configured
step count (default 16). It returns `AlignedModalities` containing:

- `features`: video, audio, and text tensors, each `[B, 16, 128]` by default.
- `modality_mask`: boolean `[B, 3]` availability mask ordered as
  `(video, audio, text)`.
- `alignment_method`: `sequence_position_linear_resampling`.
- `timestamp_synchronized`: `False`.

The current manifest contains clip metadata (including duration and FPS), but
the encoder feature outputs do not carry common frame, audio-window, or token
timestamps. Audio features come from a fixed-duration center-cropped/padded
waveform, video features come from uniformly sampled frames, and text features
come from pooled transcript tokens. Thus equal sequence lengths do not establish
temporal synchronization. The aligner normalizes ordered sequence positions and
does not claim real-time alignment or invent timestamps.

## Multimodal fusion

`MultimodalFusion` consumes `AlignedModalities`. At each aligned position, a
learned query attends over the available modality features. Attention operates
across modalities and does not mix information across time. The returned mapping
contains `features` shaped `[B, T, D]` (default `[B, 16, 128]`). When requested,
`attention_weights` is `[B, T, 3]` in `(video, audio, text)` order; unavailable
modalities receive zero weight.

Video and audio are required for the AV baseline. Text can be omitted entirely
or marked unavailable per sample with `text_available`. Phase 5 can consume the
fused sequence without further shape conversion.
