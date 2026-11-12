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

The diagram below describes the canonical runtime architecture. The separate 14-phase development sequence (Phases 0–13) is documented in [`AGENTS.md`](./AGENTS.md).

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

## 3. Implemented Components

Video/audio dataset loading, optional transcript input, model integration, multi-task training, checkpoint inference, and held-out evaluation are connected. Text is disabled by default because EmoReact V1.0 does not provide transcripts; enable `text_encoder.enabled` in `configs/model.yaml` when text is present in the manifest. See the [development pipeline readiness matrix](./tests/report/output/development_pipeline_matrix.md) for phase-by-phase status. The tests verify implementation behavior, not dataset performance or scientific validity.

### Phase 2: Dataset Exploration & Preprocessing
- **Dataset profile recorded in the existing project documentation**: 1,102 clips (432 Train, 303 Val, 367 Test). These figures have not been independently revalidated in this checkout because the raw dataset is not included.
- **Previously documented multi-label class imbalance** (not revalidated in this checkout; training computes its positive-class weights from the supplied manifest):
  - *Happiness* ($0.93$), *Curiosity* ($1.92$), *Excitement* ($2.00$), *Uncertainty* ($2.27$), *Surprise* ($2.79$), *Disgust* ($6.20$), *Frustration* ($7.15$), *Fear* ($24.41$).
  - $64.8\%$ of samples feature $\ge 2$ co-occurring active emotions.
- **Preprocessing Module** ([`preprocessing.py`](./src/sentiment_derivation/data/preprocessing.py)):
  - Built standardized manifest generator, media validator, frame extractor, and optional 16 kHz mono WAV audio extraction.
  - Generates [`data/processed/processed_manifest.json`](./data/processed/processed_manifest.json) and summary report.
- **Experiment Notebook**: [`notebooks/01_dataset_exploration.ipynb`](./notebooks/01_dataset_exploration.ipynb).

### Phase 3: Independent Modality Encoders
- **Video Encoder** ([`video_encoder.py`](./src/sentiment_derivation/encoders/video_encoder.py)):
  - Pretrained ResNet-18 backbone with pool5 visual feature extraction ($512$-dim per frame).
  - Preserves temporal order via $(B, T, C, H, W) \leftrightarrow (B \cdot T, C, H, W)$ folding/unfolding.
  - Video transforms in [`transforms.py`](./src/sentiment_derivation/data/transforms.py).
- **Audio Encoder** ([`audio_encoder.py`](./src/sentiment_derivation/encoders/audio_encoder.py)):
  - 64-band Log-Mel spectrogram preprocessing ($16$ kHz, $25$ ms FFT window, $10$ ms hop).
  - 2D-CNN feature extractor mapping spectrograms to $[B, T=16, 128]$.
- **Text Encoder** ([`text_encoder.py`](./src/sentiment_derivation/encoders/text_encoder.py)):
  - Optional pretrained DistilBERT with attention-mask-aware pooling to $[B, T=16, 128]$. Transcript availability is tracked per sample so missing transcripts are excluded from fusion.
- **Experiment Notebooks**: [`notebooks/02_video_experiments.ipynb`](./notebooks/02_video_experiments.ipynb), [`notebooks/03_audio_experiments.ipynb`](./notebooks/03_audio_experiments.ipynb), [`notebooks/04_text_experiments.ipynb`](./notebooks/04_text_experiments.ipynb).

### Phase 4: Multimodal Alignment & Cross-Modal Attention Fusion
- **Modality Aligner** ([`alignment.py`](./src/sentiment_derivation/fusion/alignment.py)):
  - Linear sequence-position resampling mapping heterogeneous modalities (video $512$, audio $128$, text $128$) to common dimension $D=128$ and step count $T=16$.
  - Generates boolean availability mask `modality_mask` ($[B, 3]$) tracking present versus absent modalities.
- **Cross-Modal Attention Fusion** ([`multimodal_fusion.py`](./src/sentiment_derivation/fusion/multimodal_fusion.py)):
  - Learned query attending across available modalities at each sequence step independently without temporal leakage.
  - Preserves sequence shape $[B, T=16, 128]$ and exposes cross-modal attention weights for XAI.
- **Unit Tests**: [`tests/test_fusion.py`](./tests/test_fusion.py) (13 tests passing).

### Phase 5: Affective State & Multi-Label Emotion Derivation
- **Affective State Estimator** ([`affective_state.py`](./src/sentiment_derivation/emotion/affective_state.py)):
  - Bounded continuous valence prediction: $\text{valence} = 1.0 + (7.0 - 1.0) \cdot \sigma(\text{raw}) \in [1.0, 7.0]$ matching EmoReact ground truth Likert scale.
  - Generates dense 64-dimensional latent affective state embeddings and normalized certainty scores.
- **Emotion Classifier & Parallel Representation** ([`emotion_classifier.py`](./src/sentiment_derivation/emotion/emotion_classifier.py)):
  - **`AGENTS.md` Constraint 3**: Maintains logically separate **Emotion Branch** ($[B, T, 128]$) and **Context Branch** ($[B, T, 128]$).
  - Multi-label sigmoid probabilities across 8 discrete classes.
  - **`AGENTS.md` Constraint 4**: Estimates independent emotion-wise confidence ($c_k = |2p_k - 1| \in [0.0, 1.0]$) and uncertainty ($u_k = 1.0 - c_k$). Never averages confidences into a single scalar.
  - Unified [`MultimodalEmotionPipeline`](./src/sentiment_derivation/emotion/emotion_classifier.py) satisfying the Section 6 contract.
- **Unit Tests & Notebook**: [`tests/test_emotion.py`](./tests/test_emotion.py) (15 tests passing) and [`notebooks/05_emotion_experiments.ipynb`](./notebooks/05_emotion_experiments.ipynb).

### Phase 6: Uncertainty Estimation, Calibration & Gating
- **Temperature Calibrator** ([`calibration.py`](./src/sentiment_derivation/uncertainty/calibration.py)):
  - Multi-label temperature scaling (`MultiLabelTemperatureCalibrator`) learning per-emotion temperatures $T_k > 0$ on validation sets via BCE loss minimization.
  - Implements multi-label Expected Calibration Error (ECE across 10 bins) and Brier scores.
- **Certainty Estimator** ([`certainty_estimator.py`](./src/sentiment_derivation/uncertainty/certainty_estimator.py)):
  - Margin confidence, Shannon binary entropy, and Monte Carlo (MC) Dropout decomposing epistemic variance from aleatoric entropy.
- **Uncertainty-Aware Gate** ([`uncertainty_gate.py`](./src/sentiment_derivation/uncertainty/uncertainty_gate.py)):
  - Modulates contextual representations via dynamic acceptance weights $g_t = \sigma(\text{MLP}([h_t, p_t, u_t, c_{\text{val}, t}])) \in [0.0, 1.0]^D$.
  - Residual prior convex combination: $h_{\text{gated}} = g \odot h + (1 - g) \odot \text{prior}$.
  - Explicit `enabled=False` switch for required ablation experiments. Exposes temporal saliency for XAI.
- **Unit Tests & Notebook**: [`tests/test_uncertainty.py`](./tests/test_uncertainty.py) (9 tests passing) and [`notebooks/06_uncertainty_experiments.ipynb`](./notebooks/06_uncertainty_experiments.ipynb).

### Phase 7: Temporal Encoding & Emotion Transition Analysis
- **BiLSTM Temporal Encoder** ([`temporal_encoder.py`](./src/sentiment_derivation/temporal/temporal_encoder.py)):
  - 1-layer bidirectional LSTM ($128 \to 128$) with residual skip-connection and LayerNorm stabilization modeling temporal context across $T=16$ steps.
- **Transition Analyzer** ([`transition_analyzer.py`](./src/sentiment_derivation/temporal/transition_analyzer.py)):
  - First-order Delta Emotion velocity ($\Delta E_t = E_t - E_{t-1}$ with $\Delta E_0 = \mathbf{0}$).
  - Exponential Moving Average (EMA) Persistence ($P_t = \alpha P_{t-1} + (1 - \alpha) E_t$) separating transient spikes from sustained emotions.
  - Softmax Temporal Attention ($\alpha_t$) weighting salient transition moments across the clip.
  - Disentangled composite transition representation fusing Emotion ($64$), Affect ($32$), and Confidence ($32$) streams into $[B, T, 128]$.
- **Unit Tests & Notebook**: [`tests/test_temporal.py`](./tests/test_temporal.py) (7 tests passing) and [`notebooks/07_temporal_experiments.ipynb`](./notebooks/07_temporal_experiments.ipynb).

### Phase 8: Latent Sentiment Mapping & Sentiment Derivation
- **Latent Sentiment Mapper** ([`latent_mapper.py`](./src/sentiment_derivation/sentiment/latent_mapper.py)):
  - Synthesizes 4 distinct streams: Emotion Dynamics ($32$), Gated Context ($128$), Transition Embeddings ($128$), and Affective State ($64$).
  - Produces step-wise latent affective sequence $\mathbf{Z} \in \mathbb{R}^{B \times T \times 128}$.
  - Temporal attention pooling deriving summary latent vector $\mathbf{z}_{\text{pooled}} \in \mathbb{R}^{B \times 128}$ and saliency weights $\alpha_t \in [0.0, 1.0]$.
- **Sentiment Head & Derivation** ([`sentiment_head.py`](./src/sentiment_derivation/sentiment/sentiment_head.py)):
  - 3-class categorical sentiment probabilities: $\mathbf{p}_{\text{sentiment}} = \text{Softmax}(\text{logits} / \tau) \in [0.0, 1.0]^3$ over `Negative` (0), `Neutral` (1), and `Positive` (2).
  - Continuous sentiment polarity score: $s \in [-1.0, 1.0]$ via $\tanh$ regression.
  - Sentiment confidence estimation ($c = p_{(1)} - p_{(2)} \in [0.0, 1.0]$).
  - Ground truth derivation utility `valence_to_sentiment_label` with documented neutral margin $\delta = 0.25$ around $4.0$.
  - End-to-end [`SentimentDerivationPipeline`](./src/sentiment_derivation/sentiment/sentiment_head.py) fulfilling the canonical Section 6 dictionary contract.
- **Unit Tests & Notebook**: [`tests/test_sentiment.py`](./tests/test_sentiment.py) (13 tests passing) and [`notebooks/08_sentiment_experiments.ipynb`](./notebooks/08_sentiment_experiments.ipynb).

---

## 4. Repository Structure

```text
Multimodal-Sentiment-Derivation/
├── README.md                               # Project documentation & architectural summary
├── AGENTS.md                               # Canonical system architecture & development rules
├── CONTRIBUTING.md                         # Contribution & Git guidelines
├── LICENSE                                 # Project license
├── pyproject.toml                          # Empty placeholder
├── requirements.txt                        # Project dependencies
│
├── configs/
│   ├── data.yaml                           # Dataset paths & preprocessing parameters
│   ├── experiment.yaml                     # Empty placeholder
│   ├── model.yaml                          # Model hyperparameters
│   └── training.yaml                       # Empty placeholder
│
├── data/
│   ├── README.md                           # Detailed data & split documentation
│   ├── raw/                                # Local-only data directory; not included
│   ├── splits/                             # Local-only split files; not included
│   └── processed/                          # Standardized processed outputs
│       ├── processed_manifest.json         # Canonical dataset manifest (1,102 clips)
│       └── preprocessing_summary.json      # Dataset validation summary
│
├── datasets/                               # Raw dataset is local-only; not included
│   └── EmoReact_V_1.0/                     # Raw EmoReact benchmark (unmodified)
│       ├── ReadMe.txt                      # Original author documentation
│       ├── Data/                           # Train / Validation / Test MP4 video clips
│       ├── Labels/                         # train_labels, val_labels, test_labels
│       └── Visual_features/                # OpenFace 1.0 extracted feature triplets
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # Phase 2: Dataset audit & statistics
│   ├── 02_video_experiments.ipynb         # Phase 3: Video frames & ResNet-18
│   ├── 03_audio_experiments.ipynb         # Phase 3: Waveforms & Log-Mel spectrograms
│   ├── 04_text_experiments.ipynb          # Phase 3: Text modality audit & DistilBERT
│   ├── 05_emotion_experiments.ipynb       # Phase 5: Parallel branches & multi-label emotions
│   ├── 06_uncertainty_experiments.ipynb   # Phase 6: Calibration, ECE & uncertainty gating
│   ├── 07_temporal_experiments.ipynb      # Phase 7: BiLSTM, Delta Emotion & persistence
│   └── 08_sentiment_experiments.ipynb     # Phase 8: Latent mapping & sentiment derivation
│
├── src/
│   └── sentiment_derivation/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py                   # Empty placeholder
│       │   ├── dataloader.py                # Empty placeholder
│       │   ├── preprocessing.py            # Dataset loading, validation & manifest creation
│       │   └── transforms.py               # Video frame transformations
│       ├── encoders/
│       │   ├── video_encoder.py            # Pretrained ResNet-18 visual encoder (512-dim)
│       │   ├── audio_encoder.py            # Log-Mel Spectrogram 2D-CNN audio encoder (128-dim)
│       │   └── text_encoder.py             # DistilBERT text encoder (128-dim)
│       ├── fusion/
│       │   ├── alignment.py                # Sequence-position resampling & modality masking
│       │   └── multimodal_fusion.py        # Cross-modal query attention fusion
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── runner.py                   # Real-prediction metrics and evaluation figures
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
│       ├── sentiment/
│           ├── __init__.py
│           ├── latent_mapper.py            # 4-stream latent affective/sentiment synthesis
│           └── sentiment_head.py           # 3-class sentiment head & valence-to-sentiment
│       ├── integration/                    # End-to-end model and checkpoint prediction utilities
│       ├── training/                       # Multi-task losses, epoch metrics, and trainer
│       └── utils/                          # Planned utility modules; project-wide helpers remain incomplete
│
├── scripts/
│   ├── prepare_data.py                     # Preprocessing execution script
│   ├── train.py                            # Model training and checkpoint creation
│   ├── evaluate.py                         # Checkpoint inference, metrics, and figures
│   └── predict.py                          # Held-out prediction CSV export
│
├── tests/
│   ├── test_audio_processing.py             # Decoder failures surface instead of silent fallback
│   ├── test_data.py                        # Unit tests for dataset preprocessing
│   ├── test_dataset_runtime.py             # Manifest path validation
│   ├── test_evaluation.py                  # Prediction validation, metrics, and plot generation
│   ├── test_emotion.py                     # Affective state and emotion derivation
│   ├── test_fusion.py                      # Alignment and cross-modal fusion
│   ├── test_integration_pipeline.py        # End-to-end multimodal forward/backward and checkpoint test
│   ├── test_training_pipeline.py           # Training loop, validation, and checkpoint persistence
│   ├── test_temporal.py                    # Temporal encoder and transition analyzer
│   ├── test_sentiment.py                   # Latent mapper and sentiment derivation
│   ├── test_uncertainty.py                 # Calibration, ECE, and gating
│   ├── test_integration.py                 # Module-level integration contract tests
│   └── report/                             # Pipeline readiness and report-generation tools
│
├── experiments/
│   └── README.md                           # Experimentation protocols & notebook catalog
│
└── checkpoints/
    └── README.md                           # Model checkpointing & persistence guide
```

---

## 5. Getting Started & Execution

Create and activate a virtual environment, then install the project dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Prepare the real dataset
Place the EmoReact V1.0 files at the configured paths in `configs/data.yaml`, then create the manifest and extracted audio used by the model:
```bash
python scripts/prepare_data.py --extract-audio
```
The dataset files are not included in this repository. Preprocessing must complete before training; missing or unreadable videos/audio cause an explicit error rather than silent zero-filled input.

### 3. Run tests
```bash
python -m pytest tests/ -q
```

### 4. Train the model
```bash
python scripts/train.py
```
This trains against the manifest's `train` split and selects the best checkpoint using the `val` split. The default ResNet-18 setting downloads pretrained ImageNet weights; add `--no-pretrained` to train without that download.

### 5. Generate held-out predictions
```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --split test
```
This writes sample-level probabilities to `reports/predictions/test_predictions.csv`.

### 6. Run model evaluation and create figures
Evaluation can run inference directly from a checkpoint and compute metrics from the resulting held-out predictions:
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --split test
```
The generated reports include per-emotion and aggregate F1 metrics, confusion matrices, valence regression metrics/plot, and sentiment metrics/plot. Alternatively, pass `--predictions` to evaluate an existing model-exported CSV. Do not use the test split for model selection.

### 7. Interactive Experimentation
Launch the experiment notebooks:
```bash
jupyter notebook notebooks/
```

---

## 6. Implementation Principles & Scientific Rules

1. **Strict Architectural Integrity**: Runtime stages follow the canonical architecture in `AGENTS.md`; development work follows its separate Phase 0–13 sequence.
2. **Distinct Emotion and Context Branches**: The Emotion Branch and Context Branch remain logically isolated representations ([`AGENTS.md`](./AGENTS.md) Constraint 3).
3. **Separate Emotion-Wise Confidence**: Prediction confidence and uncertainty are maintained strictly per individual emotion, never collapsed into a single scalar ([`AGENTS.md`](./AGENTS.md) Constraint 4).
4. **Defined Uncertainty Mathematics**: Confidence and uncertainty maintain a rigorous relationship: $\text{confidence}_k = 1.0 - \text{uncertainty}_k$.
5. **No Fabricated Labels**: EmoReact emotion annotations and continuous valence ratings are strictly preserved in their original form. Sentiment labels are derived using explicit, documented, and justified thresholds.
6. **Hardware Portability**: The end-to-end model and checkpoint workflow are covered by CPU tests; performance and reproducibility on the full dataset remain hardware-dependent.
7. **Modular Contracts**: Implemented modules expose typed input/output containers with dataclass and dictionary access matching `AGENTS.md` Section 6 contracts.

---

## 7. Current validation limits

The code now implements dataset loading, model integration, multi-task training, checkpoint inference, and checkpoint-based evaluation. The tests exercise the real network with synthetic inputs and validate code behavior; they do not represent EmoReact model performance. No dataset or trained checkpoint is bundled, so performance charts and publication claims can only be produced after running the workflow on the real dataset. Explainable-AI functionality remains unimplemented.

The readiness matrix is a source-file audit, not a functional certification. The completed CPU smoke test is not a substitute for full-data training, independent scientific review, or reproducibility testing. Dataset figures above are not produced by model evaluation.
