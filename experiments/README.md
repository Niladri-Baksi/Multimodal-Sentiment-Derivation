# Experiments & Notebooks Documentation

This directory documents the research experiments, evaluation methodology, and interactive Jupyter notebook catalog for the **Multimodal Emotion-to-Sentiment Derivation and Mapping** project.

---

## 1. Experimental Methodology & Protocols

In compliance with [`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md), experiments are structured sequentially to systematically evaluate each stage of the derivation pipeline before end-to-end integration:

1. **Modality Isolation**: Verify individual modality representations (Video, Audio, Text) and their temporal preservation.
2. **Cross-Modal Fusion**: Assess cross-modal attention mechanisms and missing-modality robustness.
3. **Affective & Multi-Label Emotion Derivation**: Verify independent per-emotion confidence estimation and continuous valence bounding.
4. **Uncertainty & Calibration**: Evaluate Expected Calibration Error (ECE), Brier score, and uncertainty-aware gating ablation.
5. **Temporal Dynamics**: Quantify the impact of emotion velocity ($\Delta E_t$), persistence (EMA), and temporal attention.
6. **Latent Sentiment Derivation**: Validate the synthesis of emotional transitions, uncertainty-gated context, and continuous affect into 3-class sentiment polarity.

---

## 2. Interactive Notebook Catalog

The `notebooks/` directory contains 8 standalone, self-contained experiment notebooks covering all architectural phases:

| Notebook | Phase / Stage | Core Topics & Experimental Objectives |
| :--- | :--- | :--- |
| [`01_dataset_exploration.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/01_dataset_exploration.ipynb) | **Phase 1: Dataset** | EmoReact V1.0 dataset audit (1,102 clips), 9-column annotation schema, severe class imbalance analysis, exact `pos_weight` calculation for BCE loss, and OpenFace feature inspection. |
| [`02_video_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/02_video_experiments.ipynb) | **Phase 2: Video** | Uniform 16-frame extraction, spatial resizing & ImageNet normalization, ResNet-18 pool5 feature extraction ($512$-dim), and temporal order preservation for BiLSTM modeling. |
| [`03_audio_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/03_audio_experiments.ipynb) | **Phase 3: Audio** | Audio extraction from MP4 clips, 16 kHz mono standardization, 64-band Log-Mel spectrogram generation, and 2D-CNN feature extraction mapping to $[B, T=16, 128]$. |
| [`04_text_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/04_text_experiments.ipynb) | **Phase 3: Text** | Formal audit documenting absence of raw transcripts in EmoReact V1.0, DistilBERT tokenization, adaptive temporal pooling to $[B, T=16, 128]$, and optional modality masking. |
| [`05_emotion_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/05_emotion_experiments.ipynb) | **Phase 5: Emotion** | Parallel representation (Emotion vs Context branches), 8-emotion multi-label sigmoid probabilities, independent per-emotion confidence ($c_k = |2p_k - 1|$), and continuous valence ($[1.0, 7.0]$). |
| [`06_uncertainty_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/06_uncertainty_experiments.ipynb) | **Phase 6: Uncertainty** | Vector temperature calibration optimizing validation NLL/BCE, multi-label ECE (10 bins), Brier scores, MC-Dropout epistemic uncertainty, and Uncertainty-Aware Gate ablation. |
| [`07_temporal_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/07_temporal_experiments.ipynb) | **Phase 7: Temporal** | 1-layer BiLSTM ($128 \to 128$) sequence modeling, first-order Delta Emotion velocity ($\Delta E_t$), EMA persistence ($P_t$), Softmax temporal attention, and composite transition representations. |
| [`08_sentiment_experiments.ipynb`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/notebooks/08_sentiment_experiments.ipynb) | **Phase 8/9: Sentiment** | 4-stream latent sentiment mapping, temporal attention pooling, 3-class sentiment derivation (`Negative`, `Neutral`, `Positive`), continuous polarity ($-1.0$ to $1.0$), and full 9-stage pipeline validation. |

---

## 3. Evaluation Metrics & Standards

Experiments measure performance across multiple complementary metrics:

### A. Multi-Label Emotion Classification
- **Loss**: Multi-label `torch.nn.BCEWithLogitsLoss` parameterized with positive weight ratios (`pos_weight`) to counter class imbalance.
- **Classification Metrics**: Macro-averaged and Micro-averaged F1-Score, Area Under the ROC Curve (ROC-AUC), and Precision-Recall AUC (PR-AUC) per individual emotion.

### B. Predictive Calibration & Uncertainty
- **Expected Calibration Error (ECE)**: Evaluated across $M=10$ uniform probability bins for each emotion:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score**: Mean squared difference between predicted probabilities and binary ground truths:
  $$\text{BS} = \frac{1}{N \cdot K} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2$$
- **Epistemic Uncertainty**: Variance across stochastic Monte Carlo forward passes with active dropout.

### C. Continuous Affective State & Polarity Regression
- **Mean Squared Error (MSE)** & **Mean Absolute Error (MAE)** on continuous valence ($[1.0, 7.0]$) and continuous polarity ($[-1.0, 1.0]$).
- **Pearson ($r$) & Spearman ($\rho$) Correlation** measuring alignment with human annotator valence dynamics.

### D. Sentiment Classification
- **Accuracy & Macro-F1** across the 3 derived sentiment polarities (`Negative`, `Neutral`, `Positive`).
- **Sentiment Margin Confidence**: $c = p_{(1)} - p_{(2)} \in [0.0, 1.0]$.

---

## 4. Planned Ablation Studies

To rigorously validate each component's contribution per `AGENTS.md` rules:

| Ablation Study | Baseline / Comparison | Hypothesis |
| :--- | :--- | :--- |
| **Gated vs. Un-Gated** | `UncertaintyAwareGate(enabled=False)` vs. `enabled=True` | Uncertainty gating filters out ambiguous/noisy emotional expressions, improving sentiment derivation accuracy. |
| **Temporal vs. Static** | Mean-pooled features vs. BiLSTM + `TransitionAnalyzer` | Modeling frame-to-frame emotion velocity and persistence outperforms static, time-agnostic pooling. |
| **Disentangled Branches** | Single joint head vs. Parallel Emotion & Context branches | Enforcing architectural separation preserves generic contextual cues that would otherwise be suppressed by multi-label emotion optimization. |
| **Modality Contributions** | Video-only vs. Audio-only vs. Audio-Visual (AV) | Multimodal fusion leverages complementary visual facial expressions and vocal prosody. |

---

## 5. Execution Instructions

Launch any notebook in a browser:
```powershell
jupyter notebook notebooks/<notebook_name>.ipynb
```

For Google Colab environments:
1. Open the notebook directly in Colab.
2. The initial environment cell automatically resolves project roots and installs/verifies dependencies.
3. GPU acceleration (`cuda`) is automatically utilized if available; otherwise, execution falls back cleanly to CPU.
