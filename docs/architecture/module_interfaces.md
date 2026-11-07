# Phase 4 & Phase 5 Module Interfaces

## Modality alignment (Phase 4)

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

## Multimodal fusion (Phase 4)

`MultimodalFusion` consumes `AlignedModalities`. At each aligned position, a
learned query attends over the available modality features. Attention operates
across modalities and does not mix information across time. The returned mapping
contains `features` shaped `[B, T, D]` (default `[B, 16, 128]`). When requested,
`attention_weights` is `[B, T, 3]` in `(video, audio, text)` order; unavailable
modalities receive zero weight.

Video and audio are required for the AV baseline. Text can be omitted entirely
or marked unavailable per sample with `text_available`. Phase 5 can consume the
fused sequence without further shape conversion.

---

## Affective State & Emotion Derivation (Phase 5 / Canonical Stage 4)

### 1. Parallel Representation & Multi-Label Emotion Classification

`EmotionClassifier` receives the fused multimodal sequence `[B, T, D]` (or pooled
`[B, D]`) and branches into two logically distinct representations per `AGENTS.md`
Constraint 3:

1. **Emotion Branch**: Specialized MLP projection generating `emotion_embeddings`
   `[B, T, emotion_dim]`.
2. **Context Branch**: Dedicated parallel MLP projection generating `context_embeddings`
   `[B, T, context_dim]`.

The Multi-Label Emotion Head projects the Emotion Branch representation to 8 raw logits
and multi-label probabilities $\sigma(\text{logits} / \text{temperature}) \in [0.0, 1.0]$:

- Col 0: Curiosity
- Col 1: Uncertainty
- Col 2: Excitement
- Col 3: Happiness
- Col 4: Surprise
- Col 5: Disgust
- Col 6: Fear
- Col 7: Frustration

### 2. Emotion-wise Confidence and Uncertainty Estimation

Confidence and uncertainty are maintained strictly per-emotion (never collapsed into
a single overall score, satisfying `AGENTS.md` Constraint 4). They have a well-defined
mathematical relationship: $\text{confidence}_k = 1.0 - \text{uncertainty}_k$.

- **Margin Method (`confidence_method="margin"`)**:
  $$\text{confidence}_k = |2 p_k - 1| \in [0.0, 1.0]$$
  $$\text{uncertainty}_k = 1.0 - \text{confidence}_k \in [0.0, 1.0]$$
  When $p_k = 0.5$ (decision boundary), confidence is $0.0$ and uncertainty is $1.0$.
  When $p_k \to 0.0$ or $1.0$, confidence is $1.0$ and uncertainty is $0.0$.

- **Entropy Method (`confidence_method="entropy"`)**:
  $$\text{uncertainty}_k = -\frac{p_k \log_2(p_k) + (1 - p_k) \log_2(1 - p_k)}{1.0} \in [0.0, 1.0]$$
  $$\text{confidence}_k = 1.0 - \text{uncertainty}_k \in [0.0, 1.0]$$

Returns `EmotionOutput` with:
- `logits`: `[B, T, 8]` or `[B, 8]`.
- `probabilities` / `emotion_probabilities`: `[B, T, 8]` or `[B, 8]`.
- `emotion_confidence`: `[B, T, 8]` or `[B, 8]`.
- `emotion_uncertainty`: `[B, T, 8]` or `[B, 8]`.
- `emotion_embeddings`: `[B, T, emotion_dim]` or `[B, emotion_dim]`.
- `context_embeddings`: `[B, T, context_dim]` or `[B, context_dim]`.
- `emotion_names`: Tuple of 8 canonical emotion names.

### 3. Continuous Affective State Estimation

`AffectiveStateEstimator` estimates continuous valence from multimodal/temporal features:

- Predicts continuous valence rating bounded strictly in $[1.0, 7.0]$ (matching EmoReact
  ground truth Likert scale) via $1.0 + (7.0 - 1.0) \cdot \sigma(\text{raw\_valence})$.
- Produces dense latent `affective_state` embeddings shaped `[B, T, affective_dim]`
  (default 64) for subsequent transition representation (Stage 6) and latent sentiment
  mapping (Stage 8).
- Produces `valence_confidence` based on normalized certainty.

Returns `AffectiveStateOutput` with:
- `valence`: `[B, T, 1]` or `[B, 1]`.
- `affective_state`: `[B, T, affective_dim]` or `[B, affective_dim]`.
- `raw_valence`: `[B, T, 1]` or `[B, 1]`.
- `valence_confidence`: `[B, T, 1]` or `[B, 1]`.

### 4. Canonical Unified Pipeline Interface

`MultimodalEmotionPipeline` combines `EmotionClassifier` and `AffectiveStateEstimator`
to fulfill the standardized contract defined in `AGENTS.md` Section 6:

```python
{
    "affective_state": tensor,             # [B, T, affective_dim]
    "valence": tensor,                     # [B, T, 1] in [1.0, 7.0]
    "raw_valence": tensor,                 # [B, T, 1] unbounded
    "valence_confidence": tensor,          # [B, T, 1] in [0.0, 1.0]
    "emotion_probabilities": tensor,       # [B, T, 8] in [0.0, 1.0]
    "emotion_logits": tensor,              # [B, T, 8]
    "emotion_confidence": tensor,          # [B, T, 8] in [0.0, 1.0]
    "emotion_uncertainty": tensor,         # [B, T, 8] in [0.0, 1.0]
    "emotion_embeddings": tensor,          # [B, T, emotion_dim]
    "context_embeddings": tensor,          # [B, T, context_dim]
    "emotion_names": tuple,                # 8 canonical emotion labels
}
```
