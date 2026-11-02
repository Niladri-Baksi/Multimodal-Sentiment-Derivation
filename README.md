# Multidimensional Emotion-to-Sentiment Derivation and Mapping

A research-driven, multimodal, uncertainty-aware framework for deriving robust sentiment representations from multidimensional emotional dynamics across video and audio stimuli.

---

## 1. Project Overview & Research Objectives

Conventional sentiment analysis typically relies on direct surface-level classification or simple heuristics mapping categorical emotions directly to sentiment polarities. In contrast, this project models the complete causal and temporal relationship between:

1. **Affective State** (continuous valence dynamics)
2. **Multi-label Emotion Distributions** (co-occurring discrete affective expressions)
3. **Emotion-wise Prediction Confidence**
4. **Predictive Uncertainty Estimation**
5. **Temporal Dynamics and Emotional Transitions**
6. **Latent Sentiment Representation**
7. **Final Sentiment Derivation and its Confidence**
8. **Explainable AI (XAI)**

The system is developed on top of the **EmoReact (V1.0)** benchmark, which captures multimodal reactions of children to diverse stimuli.

---

## 2. Canonical System Architecture

All development strictly follows the 10-stage sequential architecture defined in [`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md):

```text
STAGE 1: MULTIMODAL INPUT
│   ├── Video Input
│   ├── Audio Input
│   └── Text Input (Optional)
▼
STAGE 2: MODALITY-SPECIFIC FEATURE EXTRACTION
│   ├── Video Encoder (ResNet-18)
│   ├── Audio Encoder (Log-Mel CNN / Pretrained baseline)
│   └── Text Encoder (Optional)
▼
STAGE 3: MULTIMODAL FUSION
│   └── Feature Alignment & Cross-Modal Attention
▼
STAGE 4: TEMPORAL ENCODING
│   └── BiLSTM Temporal Encoder
▼
PARALLEL REPRESENTATION
┌───────┴────────┐
▼                ▼
EMOTION BRANCH   CONTEXT BRANCH
│ (Multi-label   │
│  Probabilities │
│  & Confidence) │
└───────┬────────┘
        ▼
STAGE 5: TEMPORAL EMOTION TRANSITION ANALYSIS
▼
STAGE 6: TRANSITION REPRESENTATION
▼
STAGE 7: UNCERTAINTY-AWARE GATING
▼
STAGE 8: LATENT SENTIMENT MAPPING
▼
STAGE 9: SENTIMENT DERIVATION AND MAPPING
▼
STAGE 10: EXPLAINABLE AI (XAI)
```

---

## 3. Work Completed Till Now

### Phase 1: Dataset Exploration & Preprocessing

- **Dataset Audit & Empirical Inspection**:
  - Recursively analyzed `datasets/EmoReact_V_1.0` (1,102 video clips: 432 Train, 303 Validation, 367 Test, split in a person-independent manner).
  - Verified 9-column annotation schema across `Labels/*_labels.text`:
    - Columns 0–7: Discrete multi-label emotions (*Curiosity*, *Uncertainty*, *Excitement*, *Happiness*, *Surprise*, *Disgust*, *Fear*, *Frustration*).
    - Column 8: Continuous *Valence* rating ($[1.0, 7.0]$ Likert scale).
  - Confirmed 0 corrupted or missing video files across all splits.
  - Profiled physical media: 1,034 clips at $1280 \times 720$, 68 clips at $640 \times 360$; 947 at $23.976$ FPS, 155 at $29.97$ FPS; durations range from $2.8$s to $14.3$s.
  - Documented that raw EmoReact V1.0 contains no dialogue/text transcripts; handled gracefully as an optional modality.
  - Identified severe class imbalance and calculated exact positive weight ratios (`pos_weight`) for multi-label `torch.nn.BCEWithLogitsLoss`:
    - *Happiness*: $0.93$ (51.9% positive)
    - *Curiosity*: $1.92$ (34.3% positive)
    - *Excitement*: $2.00$ (33.3% positive)
    - *Uncertainty*: $2.27$ (30.6% positive)
    - *Surprise*: $2.79$ (26.4% positive)
    - *Disgust*: $6.20$ (13.9% positive)
    - *Frustration*: $7.15$ (12.3% positive)
    - *Fear*: $24.41$ (3.9% positive)
  - Multi-label complexity shows $64.8\%$ of clips feature $\ge 2$ active emotions simultaneously.

- **Preprocessing Module (`src/sentiment_derivation/data/preprocessing.py`)**:
  - Implemented modular, beginner-friendly functions with strict type hinting and docstrings.
  - `load_annotations()`: Reads and numerically validates 9-column rows without fabricating artificial labels.
  - `get_split_sample_ids()`: Collects sorted video stems, filtering out OS cache files (e.g., `Thumbs.db`).
  - `match_annotations_with_media()`: Guarantees 1:1 deterministic alignment between annotations and video files.
  - `inspect_media_file()`: Verifies video decodability, duration, frame counts, and resolution via OpenCV.
  - `extract_audio_from_video()`: Standardizes MP4 audio to 16 kHz mono 16-bit uncompressed PCM WAV using FFmpeg.
  - `extract_sample_frames()`: Uniformly samples $T$ RGB frames from video clips.
  - `verify_visual_features()`: Confirms presence of OpenFace 1.0 feature triplets (`.params.txt`, `_au.txt`, `_gaze.txt`).
  - `build_standardized_sample()`: Produces sample dictionaries adhering to architectural contracts.
  - `build_processed_manifest()` & `compute_manifest_statistics()`: End-to-end dataset processing and statistical aggregation with `pos_weight` ratios and video profiling.
  - `save_manifest()` / `load_manifest()`: Manifest JSON serialization and parsing.

- **Configuration & Execution Script**:
  - `configs/data.yaml`: Authoritative configuration for dataset paths, candidate root resolution (Colab/Local), splits, label ordering, and preprocessing flags.
  - `scripts/prepare_data.py`: CLI entry point generating `data/processed/processed_manifest.json` and `preprocessing_summary.json`.

- **Exploration Notebook**:
  - `notebooks/01_dataset_exploration.ipynb`: Executable 21-cell notebook covering Colab/Local environment detection, GPU verification, directory trees, annotation parsing, label distributions with `pos_weight` ratios, emotion-valence correlation heatmap, audio waveform & 64-channel Log-Mel Spectrogram, video metadata profiling, and OpenFace features.

- **Automated Test Suite**:
  - `tests/test_data.py`: 12 comprehensive unit and integration tests covering annotation parsing, schema validation, non-binary error handling, media inspection, manifest construction, and real dataset integrity (100% passing).

- **Documentation**:
  - `data/README.md`: Detailed documentation on the EmoReact dataset, directory structures, schema definitions, and preprocessing execution.

---

### Phase 2: Visual Modality Feature Encoder

- **Video Feature Encoder (`src/sentiment_derivation/encoders/video_encoder.py`)**:
  - Built on top of a pretrained **ResNet-18** visual backbone (`torchvision.models.resnet18`).
  - Classification head (`fc`) replaced with `nn.Identity()` to extract 512-dimensional visual embeddings per frame.
  - **Temporal Sequence Preservation**: Implemented folding $(B, T, C, H, W) \to (B \cdot T, C, H, W)$ for 2D CNN inference, and unfolding $(B \cdot T, 512) \to (B, T, 512)$, strictly preserving frame order ($t_1, \dots, t_T$) for the subsequent Stage 4 BiLSTM temporal encoder.
  - Supports configurable backbone weight freezing (`freeze_encoder()`, `unfreeze_encoder()`) to prevent early overfitting.
  - Full CPU and GPU compatibility.
  - Standardized contract output format (`return_dict=True` returning `{"features": Tensor, "metadata": {...}}`).
  - `VideoEncoder.from_config()` classmethod to construct instances from YAML configuration.

- **Video Frame Transforms (`src/sentiment_derivation/data/transforms.py`)**:
  - `VideoClipTransform`: Handles spatial resizing to $(224 \times 224)$ and ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) across 4D and 5D video tensors while preserving the temporal sequence dimension $T$.

- **Model Configuration (`configs/model.yaml`)**:
  - Specifies visual backbone architecture, weights, feature dimension ($512$), spatial resolution, and freeze settings.

- **Video Modality Experiment Notebook (`notebooks/02_video_experiments.ipynb`)**:
  - Educational notebook detailing frame sampling, sequential timeline visualization, spatial transform pipelines, ResNet-18 layer hierarchy, and temporal order preservation for downstream BiLSTM modeling.

---

## 4. Repository Structure

```text
Multimodal-Sentiment-Derivation/
├── README.md                               # Project documentation (this file)
├── AGENTS.md                               # Authoritative system architecture & rules
├── CONTRIBUTING.md                         # Contribution guidelines
├── LICENSE                                 # Project license
├── pyproject.toml                          # Project configuration
├── requirements.txt                        # Phase 1 & 2 dependencies
│
├── configs/
│   ├── data.yaml                           # Dataset paths and preprocessing configuration
│   └── model.yaml                          # Video encoder & model specifications
│
├── data/
│   ├── README.md                           # Dataset & preprocessing documentation
│   ├── raw/                                # Raw data references
│   ├── splits/                             # Partition indices
│   └── processed/                          # Preprocessed standardized outputs
│       ├── processed_manifest.json         # Canonical dataset manifest
│       └── preprocessing_summary.json      # Dataset validation summary
│
├── datasets/
│   └── EmoReact_V_1.0/                     # Raw EmoReact dataset (unmodified)
│       ├── ReadMe.txt                      # Original author documentation
│       ├── Data/                           # Train / Validation / Test MP4 videos
│       ├── Labels/                         # train_labels, val_labels, test_labels
│       └── Visual_features/                # OpenFace 1.0 extracted features
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb        # Dataset audit and exploration notebook
│   └── 02_video_experiments.ipynb         # Video modality & ResNet-18 experiments
│
├── src/
│   └── sentiment_derivation/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py            # Preprocessing pipeline functions
│       │   └── transforms.py               # Video frame transformations
│       └── encoders/
│           └── video_encoder.py            # Pretrained ResNet-18 Video Encoder
│
├── scripts/
│   └── prepare_data.py                     # Preprocessing execution script
│
└── tests/
    └── test_data.py                        # Unit & integration tests for data pipeline
```

---

## 5. Getting Started & Execution

### 1. Environment Setup
Activate the project's dedicated virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

Install the documented dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Run Data Preprocessing
Generate the standardized dataset manifest and summary report:
```powershell
python scripts/prepare_data.py
```
*(To additionally extract standardized 16 kHz mono WAV audio files, pass `--extract-audio`)*

### 3. Run Automated Tests
Execute the unit and integration test suite:
```powershell
python -m pytest tests/test_data.py -v
```

### 4. Run Exploration Notebooks
Launch Jupyter Notebook to view and run the experiment notebooks:
```powershell
jupyter notebook notebooks/01_dataset_exploration.ipynb
jupyter notebook notebooks/02_video_experiments.ipynb
```

---

## 6. Implementation Principles & Scientific Rules

- **No Artificial Labels**: Emotion labels and affective states are strictly preserved in their original format; no synthetic sentiment labels are fabricated during preprocessing.
- **Temporal Order Preservation**: Visual frame sequences preserve temporal order $T$ throughout the video encoder to provide dynamic sequence inputs to the Stage 4 BiLSTM.
- **CPU & GPU Compatibility**: All modules operate on both CPU and CUDA-enabled devices.
- **Modular Isolation**: Each component adheres strictly to defined interface contracts and type hints without cross-module side effects.
