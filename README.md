# Multidimensional Emotion-to-Sentiment Derivation and Mapping

A research-driven, multimodal, uncertainty-aware deep learning framework for deriving interpretable sentiment representations from multidimensional emotional dynamics across video, audio, and text.

---

## 1. Project Overview & Research Objectives

Conventional sentiment analysis typically relies on direct black-box classification or naive static mappings from categorical emotions to polarity labels. In contrast, this project investigates how sentiment can be systematically derived through a sequence of affective, temporal, and uncertainty-aware stages:

1. **Affective State**: Continuous valence ratings ($[1.0, 7.0]$ Likert scale).
2. **Multi-Label Emotion Distributions**: Co-occurring discrete emotional expressions across 8 categories.
3. **Emotion-Wise Prediction Confidence**: Independent reliability estimates maintained separately per emotion category.
4. **Predictive Uncertainty & Calibration**: Epistemic variance, aleatoric entropy, and temperature-calibrated probability distributions.
5. **Temporal Dynamics and Emotional Transitions**: Frame-to-frame velocity ($\Delta E_t$), emotional persistence (EMA), and temporal inflection saliency.
6. **Uncertainty-Aware Gating**: Dynamic neural filtering modulating context representations based on predictive certainty.
7. **Latent Sentiment Mapping**: Multi-stream synthesis producing an evolving latent affective representation across time.
8. **Final Sentiment Derivation & Confidence**: 3-class categorical polarity (`Negative`, `Neutral`, `Positive`), continuous sentiment score ($-1.0$ to $1.0$), and sentiment-level confidence.
9. **Explainable AI (XAI)**: Upstream representations preserved for faithful modality, temporal, and emotional attribution.

The system is developed on the **EmoReact (V1.0)** benchmark, capturing natural multimodal emotional reactions of children to diverse stimuli.

---

## 2. Canonical System Architecture

All development strictly adheres to the 10-stage sequential architecture specified in [`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md):

```text
STAGE 1: MULTIMODAL INPUT
│   ├── Video Input (MP4, Uniform 16 Frames)
│   ├── Audio Input (Standardized 16 kHz Mono WAV)
│   └── Text Input (Optional)
▼
STAGE 2: MODALITY-SPECIFIC FEATURE ENCODING
│   ├── Video Encoder (Pretrained ResNet-18 -> [B, T=16, 512])
│   ├── Audio Encoder (Log-Mel Spectrogram 2D-CNN -> [B, T=16, 128])
│   └── Text Encoder (DistilBERT + Adaptive Pooling -> [B, T=16, 128])
▼
STAGE 3: MULTIMODAL FUSION
│   ├── Modality Aligner (Sequence-position linear resampling -> [B, T=16, 128])
│   └── Cross-Modal Attention Fusion (Learned query attention -> [B, T=16, 128])
▼
STAGE 4: TEMPORAL ENCODING
│   └── BiLSTM Temporal Encoder (Bidirectional LSTM + Residual -> [B, T=16, 128])
▼
PARALLEL REPRESENTATION
┌──────────────────────────┴──────────────────────────┐
▼                                                     ▼
EMOTION BRANCH                                        CONTEXT BRANCH
├── Multi-Label Emotion Head ([B, T, 8])              └── Parallel Context
├── Continuous Affective State Head ([B, T, 64])          Embeddings ([B, T, 128])
└── Independent Confidence Estimation ([B, T, 8])
└──────────────────────────┬──────────────────────────┘
                           ▼
STAGE 5: TEMPORAL EMOTION TRANSITION ANALYSIS
│   ├── Delta Emotion Velocity: ΔE_t = E_t - E_{t-1}
│   ├── Emotion Persistence EMA: P_t = α P_{t-1} + (1 - α) E_t
│   └── Temporal Attention Saliency: α_t in [0.0, 1.0]
▼
STAGE 6: TRANSITION REPRESENTATION
│   └── Composite Transition Embedding (Emotion + Affect + Confidence -> [B, T, 128])
▼
STAGE 7: UNCERTAINTY-AWARE GATING
│   └── Dynamic neural gate modulating context via residual prior:
│       h_gated = g * h + (1 - g) * prior,  g in [0.0, 1.0]^D
▼
STAGE 8: LATENT SENTIMENT MAPPING
│   ├── 4-Stream Synthesis (Emotion Dynamics + Gated Context + Transitions + Affect)
│   ├── Latent Affective Sequence: Z in R^{[B, T=16, 128]}
│   └── Temporal Attention Pooling: z_pooled in R^{[B, 128]}
▼
STAGE 9: SENTIMENT DERIVATION AND MAPPING
│   ├── 3-Class Categorical Sentiment: Negative (0), Neutral (1), Positive (2)
│   ├── Continuous Sentiment Polarity Score: s in [-1.0, 1.0]
│   └── Sentiment Prediction Confidence: c in [0.0, 1.0]
▼
STAGE 10: EXPLAINABLE AI (XAI)
    └── Upstream attributions across modalities, emotions, transitions, and gating
```

---

## 3. Work Completed Across Phases

### Phase 1: Dataset Exploration & Preprocessing
- **Empirical Audit**: 1,102 video clips (432 Train, 303 Val, 367 Test) verified with 0 corrupted files.
- **Multi-Label Class Imbalance**: Calculated exact `pos_weight` ratios for `torch.nn.BCEWithLogitsLoss`:
  - *Happiness* ($0.93$), *Curiosity* ($1.92$), *Excitement* ($2.00$), *Uncertainty* ($2.27$), *Surprise* ($2.79$), *Disgust* ($6.20$), *Frustration* ($7.15$), *Fear* ($24.41$).
  - $64.8\%$ of samples feature $\ge 2$ co-occurring active emotions.
- **Preprocessing Module** ([`preprocessing.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/data/preprocessing.py)):
  - Built standardized manifest generator, media validator, frame extractor, and optional 16 kHz mono WAV audio extraction.
  - Generates [`data/processed/processed_manifest.json`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/data/processed/processed_manifest.json) and summary report.
- **Experiment Notebook**: [`notebooks/01_dataset_exploration.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/01_dataset_exploration.ipynb).

### Phase 2 & 3: Modality-Specific Feature Encoders
- **Video Encoder** ([`video_encoder.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/encoders/video_encoder.py)):
  - Pretrained ResNet-18 backbone with pool5 visual feature extraction ($512$-dim per frame).
  - Preserves temporal order via $(B, T, C, H, W) \leftrightarrow (B \cdot T, C, H, W)$ folding/unfolding.
  - Video transforms in [`transforms.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/data/transforms.py).
- **Audio Encoder** ([`audio_encoder.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/encoders/audio_encoder.py)):
  - 64-band Log-Mel spectrogram preprocessing ($16$ kHz, $25$ ms FFT window, $10$ ms hop).
  - 2D-CNN feature extractor mapping spectrograms to $[B, T=16, 128]$.
- **Text Encoder** ([`text_encoder.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/encoders/text_encoder.py)):
  - Pretrained DistilBERT with adaptive pooling to $[B, T=16, 128]$. Transcripts are documented as optional/absent in raw EmoReact V1.0 without fabricating artificial data.
- **Experiment Notebooks**: [`notebooks/02_video_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/02_video_experiments.ipynb), [`notebooks/03_audio_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/03_audio_experiments.ipynb), [`notebooks/04_text_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/04_text_experiments.ipynb).

### Phase 4: Multimodal Alignment & Cross-Modal Attention Fusion
- **Modality Aligner** ([`alignment.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/fusion/alignment.py)):
  - Linear sequence-position resampling mapping heterogeneous modalities (video $512$, audio $128$, text $128$) to common dimension $D=128$ and step count $T=16$.
  - Generates boolean availability mask `modality_mask` ($[B, 3]$) tracking present versus absent modalities.
- **Cross-Modal Attention Fusion** ([`multimodal_fusion.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/fusion/multimodal_fusion.py)):
  - Learned query attending across available modalities at each sequence step independently without temporal leakage.
  - Preserves sequence shape $[B, T=16, 128]$ and exposes cross-modal attention weights for XAI.
- **Unit Tests**: [`tests/test_fusion.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/tests/test_fusion.py) (13 tests passing).

### Phase 5: Affective State & Multi-Label Emotion Derivation
- **Affective State Estimator** ([`affective_state.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/emotion/affective_state.py)):
  - Bounded continuous valence prediction: $\text{valence} = 1.0 + (7.0 - 1.0) \cdot \sigma(\text{raw}) \in [1.0, 7.0]$ matching EmoReact ground truth Likert scale.
  - Generates dense 64-dimensional latent affective state embeddings and normalized certainty scores.
- **Emotion Classifier & Parallel Representation** ([`emotion_classifier.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/emotion/emotion_classifier.py)):
  - **`AGENTS.md` Constraint 3**: Maintains logically separate **Emotion Branch** ($[B, T, 128]$) and **Context Branch** ($[B, T, 128]$).
  - Multi-label sigmoid probabilities across 8 discrete classes.
  - **`AGENTS.md` Constraint 4**: Estimates independent emotion-wise confidence ($c_k = |2p_k - 1| \in [0.0, 1.0]$) and uncertainty ($u_k = 1.0 - c_k$). Never averages confidences into a single scalar.
  - Unified [`MultimodalEmotionPipeline`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/emotion/emotion_classifier.py) satisfying the Section 6 contract.
- **Unit Tests & Notebook**: [`tests/test_emotion.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/tests/test_emotion.py) (15 tests passing) and [`notebooks/05_emotion_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/05_emotion_experiments.ipynb).

### Phase 6: Uncertainty Estimation, Calibration & Gating
- **Temperature Calibrator** ([`calibration.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/uncertainty/calibration.py)):
  - Multi-label temperature scaling (`MultiLabelTemperatureCalibrator`) learning per-emotion temperatures $T_k > 0$ on validation sets via BCE loss minimization.
  - Implements multi-label Expected Calibration Error (ECE across 10 bins) and Brier scores.
- **Certainty Estimator** ([`certainty_estimator.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/uncertainty/certainty_estimator.py)):
  - Margin confidence, Shannon binary entropy, and Monte Carlo (MC) Dropout decomposing epistemic variance from aleatoric entropy.
- **Uncertainty-Aware Gate** ([`uncertainty_gate.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/uncertainty/uncertainty_gate.py)):
  - Modulates contextual representations via dynamic acceptance weights $g_t = \sigma(\text{MLP}([h_t, p_t, u_t, c_{\text{val}, t}])) \in [0.0, 1.0]^D$.
  - Residual prior convex combination: $h_{\text{gated}} = g \odot h + (1 - g) \odot \text{prior}$.
  - Explicit `enabled=False` switch for required ablation experiments. Exposes temporal saliency for XAI.
- **Unit Tests & Notebook**: [`tests/test_uncertainty.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/tests/test_uncertainty.py) (9 tests passing) and [`notebooks/06_uncertainty_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/06_uncertainty_experiments.ipynb).

### Phase 7: Temporal Encoding & Emotion Transition Analysis
- **BiLSTM Temporal Encoder** ([`temporal_encoder.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/temporal/temporal_encoder.py)):
  - 1-layer bidirectional LSTM ($128 \to 128$) with residual skip-connection and LayerNorm stabilization modeling temporal context across $T=16$ steps.
- **Transition Analyzer** ([`transition_analyzer.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/temporal/transition_analyzer.py)):
  - First-order Delta Emotion velocity ($\Delta E_t = E_t - E_{t-1}$ with $\Delta E_0 = \mathbf{0}$).
  - Exponential Moving Average (EMA) Persistence ($P_t = \alpha P_{t-1} + (1 - \alpha) E_t$) separating transient spikes from sustained emotions.
  - Softmax Temporal Attention ($\alpha_t$) weighting salient transition moments across the clip.
  - Disentangled composite transition representation fusing Emotion ($64$), Affect ($32$), and Confidence ($32$) streams into $[B, T, 128]$.
- **Unit Tests & Notebook**: [`tests/test_temporal.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/tests/test_temporal.py) (7 tests passing) and [`notebooks/07_temporal_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/07_temporal_experiments.ipynb).

### Phase 8 & 9: Latent Sentiment Mapping & Sentiment Derivation
- **Latent Sentiment Mapper** ([`latent_mapper.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/sentiment/latent_mapper.py)):
  - Synthesizes 4 distinct streams: Emotion Dynamics ($32$), Gated Context ($128$), Transition Embeddings ($128$), and Affective State ($64$).
  - Produces step-wise latent affective sequence $\mathbf{Z} \in \mathbb{R}^{B \times T \times 128}$.
  - Temporal attention pooling deriving summary latent vector $\mathbf{z}_{\text{pooled}} \in \mathbb{R}^{B \times 128}$ and saliency weights $\alpha_t \in [0.0, 1.0]$.
- **Sentiment Head & Derivation** ([`sentiment_head.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/sentiment/sentiment_head.py)):
  - 3-class categorical sentiment probabilities: $\mathbf{p}_{\text{sentiment}} = \text{Softmax}(\text{logits} / \tau) \in [0.0, 1.0]^3$ over `Negative` (0), `Neutral` (1), and `Positive` (2).
  - Continuous sentiment polarity score: $s \in [-1.0, 1.0]$ via $\tanh$ regression.
  - Sentiment confidence estimation ($c = p_{(1)} - p_{(2)} \in [0.0, 1.0]$).
  - Ground truth derivation utility `valence_to_sentiment_label` with documented neutral margin $\delta = 0.25$ around $4.0$.
  - End-to-end [`SentimentDerivationPipeline`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/src/sentiment_derivation/sentiment/sentiment_head.py) fulfilling the canonical Section 6 dictionary contract.
- **Unit Tests & Notebook**: [`tests/test_sentiment.py`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/tests/test_sentiment.py) (13 tests passing) and [`notebooks/08_sentiment_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/08_sentiment_experiments.ipynb).

---

## 4. Repository Structure

```text
Multimodal-Sentiment-Derivation/
├── README.md                               # Project documentation & architectural summary
├── AGENTS.md                               # Canonical system architecture & development rules
├── CONTRIBUTING.md                         # Contribution & Git guidelines
├── LICENSE                                 # Project license
├── pyproject.toml                          # Project metadata & pytest configuration
├── requirements.txt                        # Project dependencies
│
├── configs/
│   ├── data.yaml                           # Dataset paths & preprocessing parameters
│   └── model.yaml                          # Full 9-stage architecture hyperparameters
│
├── data/
│   ├── README.md                           # Detailed data & split documentation
│   ├── raw/                                # Raw dataset references
│   ├── splits/                             # Partition indices
│   └── processed/                          # Standardized processed outputs
│       ├── processed_manifest.json         # Canonical dataset manifest (1,102 clips)
│       └── preprocessing_summary.json      # Dataset validation summary
│
├── datasets/
│   └── EmoReact_V_1.0/                     # Raw EmoReact benchmark (unmodified)
│       ├── ReadMe.txt                      # Original author documentation
│       ├── Data/                           # Train / Validation / Test MP4 video clips
│       ├── Labels/                         # train_labels, val_labels, test_labels
│       └── Visual_features/                # OpenFace 1.0 extracted feature triplets
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # Phase 1: Dataset audit & statistics
│   ├── 02_video_experiments.ipynb         # Phase 2: Video frames & ResNet-18
│   ├── 03_audio_experiments.ipynb         # Phase 3: Waveforms & Log-Mel spectrograms
│   ├── 04_text_experiments.ipynb          # Phase 3: Text modality audit & DistilBERT
│   ├── 05_emotion_experiments.ipynb       # Phase 5: Parallel branches & multi-label emotions
│   ├── 06_uncertainty_experiments.ipynb   # Phase 6: Calibration, ECE & uncertainty gating
│   ├── 07_temporal_experiments.ipynb      # Phase 7: BiLSTM, Delta Emotion & persistence
│   └── 08_sentiment_experiments.ipynb     # Phase 8/9: Latent mapping & sentiment derivation
│
├── src/
│   └── sentiment_derivation/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py            # Dataset loading, validation & manifest creation
│       │   └── transforms.py               # Video frame transformations
│       ├── encoders/
│       │   ├── video_encoder.py            # Pretrained ResNet-18 visual encoder (512-dim)
│       │   ├── audio_encoder.py            # Log-Mel Spectrogram 2D-CNN audio encoder (128-dim)
│       │   └── text_encoder.py             # DistilBERT text encoder (128-dim)
│       ├── fusion/
│       │   ├── alignment.py                # Sequence-position resampling & modality masking
│       │   └── multimodal_fusion.py        # Cross-modal query attention fusion
│       ├── emotion/
│       │   ├── __init__.py
│       │   ├── affective_state.py          # Continuous valence & affective state estimator
│       │   └── emotion_classifier.py       # Parallel Emotion/Context branches & confidence
│       ├── uncertainty/
│       │   ├── __init__.py
│       │   ├── calibration.py              # Temperature scaling, multi-label ECE & Brier score
│       │   ├── certainty_estimator.py      # Margin, Shannon Entropy & MC-Dropout
│       │   └── uncertainty_gate.py         # Dynamic neural gate with residual prior
│       ├── temporal/
│       │   ├── __init__.py
│       │   ├── temporal_encoder.py         # BiLSTM sequence encoder (128-dim)
│       │   └── transition_analyzer.py      # Delta emotion, persistence EMA & temporal attention
│       └── sentiment/
│           ├── __init__.py
│           ├── latent_mapper.py            # 4-stream latent affective/sentiment synthesis
│           └── sentiment_head.py           # 3-class sentiment head & valence-to-sentiment
│
├── scripts/
│   └── prepare_data.py                     # Preprocessing execution script
│
├── tests/
│   ├── test_data.py                        # 12 unit tests for dataset preprocessing
│   ├── test_fusion.py                      # 13 tests for alignment & cross-modal fusion
│   ├── test_emotion.py                     # 15 tests for affective state & emotion derivation
│   ├── test_uncertainty.py                 # 9 tests for calibration, ECE & gating
│   ├── test_temporal.py                    # 7 tests for BiLSTM & transition analyzer
│   └── test_sentiment.py                   # 13 tests for latent mapper & sentiment derivation
│
├── experiments/
│   └── README.md                           # Experimentation protocols & notebook catalog
│
└── checkpoints/
    └── README.md                           # Model checkpointing & persistence guide
```

---

## 5. Getting Started & Execution

### 1. Environment Setup
Activate the project's virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

Install the documented dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run Data Preprocessing
Generate the canonical dataset manifest and validation statistics:
```powershell
python scripts/prepare_data.py
```
*(To additionally extract standardized 16 kHz mono WAV audio files, append `--extract-audio`)*

### 3. Run Automated Tests
Execute the comprehensive 69-test automated suite across all implemented stages:
```powershell
python -m pytest tests/ -v
```
*(Currently: 67 passed, 2 CUDA tests skipped cleanly when running on CPU environments)*

### 4. Interactive Experimentation
Launch Jupyter Notebook to explore the interactive, self-contained experiment notebooks:
```powershell
jupyter notebook notebooks/
```

---

## 6. Implementation Principles & Scientific Rules

1. **Strict Architectural Integrity**: Stages follow the canonical sequence from Stage 1 through Stage 9 without skipping or shortcuts.
2. **Distinct Emotion and Context Branches**: The Emotion Branch and Context Branch remain logically isolated representations ([`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md) Constraint 3).
3. **Separate Emotion-Wise Confidence**: Prediction confidence and uncertainty are maintained strictly per individual emotion, never collapsed into a single scalar ([`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md) Constraint 4).
4. **Defined Uncertainty Mathematics**: Confidence and uncertainty maintain a rigorous relationship: $\text{confidence}_k = 1.0 - \text{uncertainty}_k$.
5. **No Fabricated Labels**: EmoReact emotion annotations and continuous valence ratings are strictly preserved in their original form. Sentiment labels are derived using explicit, documented, and justified thresholds.
6. **Hardware Portability**: Every single module operates identically on CPU and GPU.
7. **Modular Contracts**: All modules expose typed input/output containers with dataclass and dictionary access matching [`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md) Section 6 contracts.

---

## 7. Next Phase: Stage 10 Explainable AI (XAI)

With Stages 1 through 9 fully implemented and tested, the final architectural stage is **Stage 10: Explainable AI (XAI)**:
- **Modality Attribution**: Quantifying the contribution of visual, acoustic, and textual modalities.
- **Emotion Attribution**: Identifying which discrete emotions drove the derived sentiment.
- **Temporal Saliency**: Explaining the role of emotional transition inflection points using preserved temporal attention weights ($\alpha_t$) and gate weights ($g_t$).
- **Human-Readable Explanations**: Generating interpretable verbal explanations faithful to the actual internal representations without modifying the final predictions.
