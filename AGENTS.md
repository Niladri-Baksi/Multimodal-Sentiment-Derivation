# AGENTS.md

# Project: Multidimensional Emotion-to-Sentiment Derivation and Mapping

## 1. PROJECT OVERVIEW

This repository implements a multimodal, uncertainty-aware emotion-to-sentiment derivation and mapping system.

The objective is to derive meaningful sentiment representations from multidimensional emotional information obtained from video, audio, and optionally text.

Unlike conventional emotion recognition systems that primarily classify emotions, this project aims to model the relationship between:

1. Affective State
2. Multi-label Emotion Representation
3. Emotion-wise Prediction Confidence
4. Predictive Uncertainty
5. Temporal Dynamics and Emotional Transitions
6. Latent Sentiment Representation
7. Final Sentiment and its Confidence
8. Explainable AI (XAI)

The system should not simply map an emotion label to a sentiment label. It must consider the multidimensional emotional state, reliability of individual predictions, and temporal changes before deriving the final sentiment.

### Core Research Objectives

- Develop a multimodal emotional understanding pipeline.
- Represent emotions using multi-label probability distributions.
- Estimate confidence and uncertainty independently for individual emotions.
- Incorporate temporal information and emotional transitions.
- Develop an uncertainty-aware gating mechanism.
- Learn a latent mapping from multidimensional emotional representations to sentiment.
- Generate interpretable explanations for the final sentiment using XAI.

The project is inspired by existing multimodal emotion recognition research, including EmoReact. Existing emotion recognition approaches should be treated as foundations rather than as the complete contribution of this project.

Do not claim novelty or superiority without experimental evidence.

---

## 2. CANONICAL SYSTEM ARCHITECTURE

All AI coding assistants and contributors MUST follow this architecture.
The following represents the authoritative development pipeline.

````text
STAGE 1: MULTIMODAL INPUT
│
├── Video Input
├── Audio Input
└── Text Input (Optional)
        │
        ▼
STAGE 2: MODALITY-SPECIFIC FEATURE EXTRACTION
        │
        ├── Video Encoder
        ├── Audio Encoder
        └── Text Encoder
                │
                ▼
STAGE 3: MULTIMODAL FUSION
        │
        └── Multimodal Fusion + Attention
                │
                ▼
STAGE 4: TEMPORAL ENCODING
        │
        └── Temporal Encoder (BiLSTM)
                │
                ▼
        PARALLEL REPRESENTATION
                │
        ┌───────┴────────┐
        ▼                ▼
   EMOTION BRANCH   CONTEXT BRANCH
        │                │
        ▼                │
 Multi-label Emotion     │
      Head               │
        │                │
        ▼                │
 Emotion Probabilities   │
        │                │
        ▼                │
 Emotion-wise            │
 Confidence Estimation   │
        │                │
        └────────┬───────┘
                 │
                 ▼
STAGE 5: TEMPORAL EMOTION TRANSITION ANALYSIS
        │
        ├── Emotion Evolution
        ├── Emotion Persistence
        └── Temporal Attention
                 │
                 ▼
STAGE 6: TRANSITION REPRESENTATION
        │
        ├── Emotion Representation
        ├── Affective State Representation
        └── Confidence Representation
                 │
                 ▼
        TEMPORAL / TRANSITION
             REPRESENTATION
                 │
                 ▼
STAGE 7: UNCERTAINTY-AWARE GATING
        │
        └── Evaluate reliability and contribution
            of cumulative emotional states across
            temporal segments.
                 │
                 ▼
STAGE 8: LATENT SENTIMENT MAPPING
        │
        ├── Current Emotional State
        ├── Emotion-wise Confidence
        ├── Temporal Changes
        └── Emotion Persistence
                 │
                 ▼
        LATENT AFFECTIVE REPRESENTATION
        (Evolution across t1, t2, t3 ... tn)
                 │
                 ▼
STAGE 9: SENTIMENT DERIVATION AND MAPPING
        │
        ├── Final Sentiment
        └── Sentiment Confidence
                 │
                 ▼
STAGE 10: EXPLAINABLE AI (XAI)
        │
        ├── Modality-wise Contribution
        ├── Emotion-wise Contribution
        ├── Affective State Explanation
        ├── Temporal Transition Explanation
        ├── Confidence and Uncertainty Explanation
        ├── Uncertainty-Gating Explanation
        └── Latent Sentiment Mapping Explanation
                 │
                 ▼
        FINAL EXPLAINABLE OUTPUT
        │
        ├── Derived Sentiment
        ├── Prediction Confidence
        ├── Key Contributing Emotions
        ├── Important Temporal Changes
        ├── Modality Contributions
        └── Human-Readable Explanation
```

IMPORTANT:

- Temporal analysis must produce transitional embeddings BEFORE final sentiment derivation.
- XAI must be implemented as the final logical stage.
- XAI must explain the complete derivation process, not just the final classification.
- Individual emotion confidence must be maintained separately.
- Do not replace per-emotion confidence with a single overall confidence score.
- The architecture must remain modular and independently testable.

### Architectural Constraints

1. The architecture above is the single source of truth.
2. Do not independently modify the sequence of stages.
3. Emotion and Context branches must remain logically distinct.
4. Emotion-wise confidence must be maintained separately.
5. Temporal transition analysis must precede uncertainty-aware gating.
6. Transition representations must incorporate emotion, affect and confidence.
7. Latent sentiment mapping must incorporate temporal and uncertainty-aware information.
8. XAI must remain the final logical component.
9. XAI must explain the derivation process and must not modify the final sentiment prediction.
10. Any proposed architectural modification must be discussed with the integration lead before implementation.

---

## 3. TECHNOLOGY STACK

Use the following technology preferences unless a documented technical reason requires a change.

### Programming

- Python 3.10 or a compatible version supported by selected dependencies.
- PyTorch as the primary deep learning framework.
- NumPy for numerical operations.
- Pandas for structured data processing.
- Scikit-learn for evaluation and traditional ML utilities.
- Matplotlib and Seaborn for visualization.

### Development Environment

- Google Colab (free GPU environment).
- Git and GitHub for version control.
- Jupyter Notebooks for experimentation.
- Python modules for production-oriented implementation.

### Development Principles

- Prefer lightweight pretrained models.
- Fine-tune selectively rather than training large models from scratch.
- Freeze pretrained encoders initially where appropriate.
- Avoid unnecessarily large models.
- Prioritize reproducibility and efficient GPU utilization.
- Every component must work on CPU when technically feasible.

Do not introduce a new framework or major dependency without discussing its purpose and integration impact.

---

## 4. REPOSITORY STRUCTURE

Maintain the following logical repository organization.

```text

multimodal-sentiment-derivation/
│
├── README.md
├── AGENTS.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── experiment.yaml
│
├── docs/
│   ├── architecture/
│   │   ├── system_architecture.md
│   │   ├── module_interfaces.md
│   │   └── data_flow.md
│   ├── research/
│   │   ├── problem_statement.md
│   │   ├── methodology.md
│   │   └── evaluation_strategy.md
│   └── decisions/
│       └── ADR-001-model-architecture.md
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_video_experiments.ipynb
│   ├── 03_audio_experiments.ipynb
│   ├── 04_text_experiments.ipynb
│   ├── 05_emotion_experiments.ipynb
│   ├── 06_uncertainty_experiments.ipynb
│   ├── 07_temporal_experiments.ipynb
│   └── 08_sentiment_experiments.ipynb
│
├── src/
│   └── sentiment_derivation/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── preprocessing.py
│       │   ├── dataset.py
│       │   ├── dataloader.py
│       │   └── transforms.py
│       │
│       ├── encoders/
│       │   ├── video_encoder.py
│       │   ├── audio_encoder.py
│       │   └── text_encoder.py
│       │
│       ├── fusion/
│       │   ├── alignment.py
│       │   └── multimodal_fusion.py
│       │
│       ├── emotion/
│       │   ├── affective_state.py
│       │   └── emotion_classifier.py
│       │
│       ├── uncertainty/
│       │   ├── certainty_estimator.py
│       │   ├── calibration.py
│       │   └── uncertainty_gate.py
│       │
│       ├── temporal/
│       │   ├── temporal_encoder.py
│       │   └── transition_analyzer.py
│       │
│       ├── sentiment/
│       │   ├── latent_mapper.py
│       │   └── sentiment_head.py
│       │
│       ├── integration/
│       │   ├── model.py
│       │   └── pipeline.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       │
│       └── utils/
│           ├── config.py
│           ├── logger.py
│           └── seed.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_data.py
│   ├── test_encoders.py
│   ├── test_fusion.py
│   ├── test_emotion.py
│   ├── test_uncertainty.py
│   ├── test_temporal.py
│   ├── test_sentiment.py
│   └── test_integration.py
│
├── experiments/
│   ├── README.md
│   └── results/
│
└── checkpoints/
    └── README.md

multimodal-sentiment-derivation/
│
├── README.md
├── AGENTS.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── experiment.yaml
│
├── docs/
│   ├── architecture/
│   │   ├── system_architecture.md
│   │   ├── module_interfaces.md
│   │   └── data_flow.md
│   ├── research/
│   │   ├── problem_statement.md
│   │   ├── methodology.md
│   │   └── evaluation_strategy.md
│   └── decisions/
│       └── ADR-001-model-architecture.md
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_video_experiments.ipynb
│   ├── 03_audio_experiments.ipynb
│   ├── 04_text_experiments.ipynb
│   ├── 05_emotion_experiments.ipynb
│   ├── 06_uncertainty_experiments.ipynb
│   ├── 07_temporal_experiments.ipynb
│   └── 08_sentiment_experiments.ipynb
│
├── src/
│   └── sentiment_derivation/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── preprocessing.py
│       │   ├── dataset.py
│       │   ├── dataloader.py
│       │   └── transforms.py
│       │
│       ├── encoders/
│       │   ├── video_encoder.py
│       │   ├── audio_encoder.py
│       │   └── text_encoder.py
│       │
│       ├── fusion/
│       │   ├── alignment.py
│       │   └── multimodal_fusion.py
│       │
│       ├── emotion/
│       │   ├── affective_state.py
│       │   └── emotion_classifier.py
│       │
│       ├── uncertainty/
│       │   ├── certainty_estimator.py
│       │   ├── calibration.py
│       │   └── uncertainty_gate.py
│       │
│       ├── temporal/
│       │   ├── temporal_encoder.py
│       │   └── transition_analyzer.py
│       │
│       ├── sentiment/
│       │   ├── latent_mapper.py
│       │   └── sentiment_head.py
│       │
│       ├── integration/
│       │   ├── model.py
│       │   └── pipeline.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── metrics.py
│       │
│       └── utils/
│           ├── config.py
│           ├── logger.py
│           └── seed.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_data.py
│   ├── test_encoders.py
│   ├── test_fusion.py
│   ├── test_emotion.py
│   ├── test_uncertainty.py
│   ├── test_temporal.py
│   ├── test_sentiment.py
│   └── test_integration.py
│
├── experiments/
│   ├── README.md
│   └── results/
│
└── checkpoints/
    └── README.md

````

This structure is the intended organization, not a requirement to create empty files immediately.

Do not move, rename, or restructure existing modules without checking their dependencies and informing the team.

---

## 5. MODULE RESPONSIBILITIES

Each module must perform one clearly defined task.

### A. Preprocessing

Responsibilities:

- Load and validate input data.
- Extract video frames.
- Process audio signals.
- Process optional textual inputs.
- Synchronize modalities using timestamps.
- Handle missing or corrupted modalities.

Output:
Cleaned and synchronized modality inputs.

### B. Encoders

Responsibilities:

- Extract visual features.
- Extract audio features.
- Extract textual features when enabled.
- Support pretrained and selectively fine-tuned models.

Suggested initial baselines:

- Visual: ResNet-18.
- Audio: Log-Mel spectrogram CNN or pretrained wav2vec2.
- Text: DistilBERT.

These are starting baselines, not mandatory final choices.

Output:
Modality-specific feature embeddings.

### C. Multimodal Fusion

Responsibilities:

- Combine available modality embeddings.
- Implement feature-level fusion or attention-based fusion.
- Handle missing modalities.
- Maintain consistent embedding dimensions.

Output:
Unified multimodal representation.

### D. Emotional Understanding

Responsibilities:

- Predict affective state.
- Generate multi-label emotion predictions.
- Preserve individual emotion probabilities.
- Estimate emotion-wise prediction reliability.

Output:
Affective state, emotion probability vector and confidence information.

### E. Uncertainty Estimation and Gating

Responsibilities:

- Estimate uncertainty for individual emotion predictions.
- Distinguish confidence from uncertainty.
- Implement an uncertainty-aware gating mechanism.
- Control the contribution of unreliable emotional predictions.

Output:
Uncertainty-aware multidimensional representation.

IMPORTANT:
Confidence and uncertainty must not be treated as interchangeable concepts without a defined mathematical relationship.

### F. Temporal Analysis

Responsibilities:

- Analyze emotional representations across time.
- Detect changes in emotional intensity and composition.
- Capture emotional persistence and transitions.
- Generate transitional embeddings.

Output:
Temporal and transitional representation.

Temporal analysis is not an independent final prediction stage. Its output must contribute to the subsequent latent sentiment mapping.

### G. Sentiment Derivation

Responsibilities:

- Receive the uncertainty-aware representation.
- Incorporate transitional embeddings.
- Generate a latent sentiment representation.
- Derive the final sentiment.
- Estimate sentiment-level confidence.

Output:
Final sentiment, latent representation and sentiment confidence.

The sentiment label space and its relationship to available dataset annotations must be explicitly documented.

Never assume that an emotion label is equivalent to a conventional sentiment label.

### H. Explainable AI (XAI)

XAI is the final logical component of the architecture.

Its responsibility is to explain WHY the system generated a particular sentiment.

It must investigate the contribution of:

1. Visual modality.
2. Audio modality.
3. Text modality, when enabled.
4. Affective state.
5. Individual emotion probabilities.
6. Emotion-wise confidence and uncertainty.
7. Temporal patterns and emotional transitions.
8. Uncertainty-aware gating.
9. Latent sentiment representation.
10. Final sentiment prediction.

Potential approaches include:

- Feature attribution.
- SHAP, where compatible with the model and computational budget.
- Integrated Gradients for differentiable components.
- Attention analysis, where applicable.
- Modality contribution analysis.
- Counterfactual or ablation-based explanations.

Do not assume attention weights alone constitute a faithful explanation.

The XAI module must not modify the sentiment prediction. It receives model outputs and intermediate representations and generates explanations.

Expected XAI output:

- Predicted sentiment.
- Sentiment confidence.
- Important contributing emotions.
- Relevant affective-state information.
- Important temporal transitions.
- Modality-wise contributions.
- Uncertainty-related observations.
- Human-readable explanation.

Explanations must reflect actual model computations. Never generate fabricated explanations or present correlation as causation.

---

## 6. MODULE INTERFACES AND CONTRACTS

All modules must communicate through clearly defined interfaces.

General rules:

- Use Python type hints.
- Use dictionaries or structured dataclasses for module outputs.
- Use consistent tensor dimensions.
- Document tensor shapes and expected data types.
- Avoid hidden dependencies between modules.
- Avoid accessing another module's internal variables directly.
- Never introduce undocumented return values.
- Do not silently change an existing interface.

Example standardized output:

```python
{
    "features": tensor,
    "metadata": {
        "sample_id": str,
        "timestamps": tensor
    }
}
```

Example emotion output:

```python
{
    "affective_state": tensor,
    "emotion_probabilities": tensor,
    "emotion_confidence": tensor,
    "emotion_uncertainty": tensor
}
```

Example temporal output:

```python
{
    "temporal_embedding": tensor,
    "transition_embedding": tensor
}
```

Example sentiment output:

```python
{
    "latent_representation": tensor,
    "sentiment": str,
    "sentiment_confidence": float
}
```

Example XAI output:

```python
{
    "sentiment": str,
    "sentiment_confidence": float,
    "modality_contributions": {},
    "emotion_contributions": {},
    "temporal_explanation": {},
    "uncertainty_explanation": {},
    "human_readable_explanation": str
}
```

These examples establish conceptual contracts. Final schemas, tensor dimensions, class names, and label mappings must be defined in `docs/module_interfaces.md`.

Before implementing a module, inspect the existing interface documentation.

If an interface needs to change:

1. Identify affected modules.
2. Explain the reason for the change.
3. Update the interface documentation.
4. Update dependent implementations and tests.
5. Coordinate the change with the integration lead.

---

## 7. CODING STANDARDS

All contributors and AI coding assistants must follow these rules.

### Python Standards

- Follow PEP 8.
- Use meaningful variable and function names.
- Add type hints to public functions.
- Write docstrings for classes and important functions.
- Avoid unnecessarily complicated abstractions.
- Keep functions focused on a single responsibility.
- Avoid hardcoded paths and hyperparameters.
- Use configuration files for tunable parameters.
- Use logging instead of excessive print statements.
- Handle exceptions explicitly.

### PyTorch Standards

- Use `torch.nn.Module` for model components.
- Keep model initialization separate from training.
- Explicitly manage training and evaluation modes.
- Ensure correct device placement.
- Avoid unnecessary CPU/GPU transfers.
- Use `torch.no_grad()` during inference where appropriate.
- Document all tensor shapes.
- Use reproducible random seeds.
- Avoid loading entire datasets into GPU memory.

### Code Quality

- No unnecessary code duplication.
- No unused imports.
- No unexplained magic numbers.
- No unnecessary dependencies.
- No fabricated model outputs or placeholder predictions in final implementations.
- No silent exception handling.
- No changes to unrelated modules.

---

## 8. DATASET AND EXPERIMENTAL RULES

The project may use EmoReact as its primary research foundation.

All dataset-related decisions must be documented.

Rules:

- Preserve original dataset annotations.
- Do not modify raw data.
- Maintain subject-independent or otherwise justified data splits.
- Prevent train-test leakage.
- Record preprocessing decisions.
- Document emotion label mappings.
- Document any derived sentiment labels and their justification.
- Maintain consistent sample identifiers.
- Respect dataset licensing and usage restrictions.

IMPORTANT:

Emotional labels and sentiment labels are not necessarily interchangeable.

If the dataset does not directly provide the required sentiment ground truth, explicitly document the derivation methodology and its limitations.

Do not silently manufacture ground-truth sentiment labels.

---

## 9. MODEL TRAINING AND RESOURCE MANAGEMENT

The primary training environment is Google Colab with free GPU access.

Therefore:

- Prefer lightweight models.
- Use pretrained weights wherever appropriate.
- Freeze encoders during initial experiments.
- Fine-tune only when justified.
- Use batch sizes appropriate for available GPU memory.
- Use mixed precision when supported and numerically appropriate.
- Save checkpoints regularly.
- Use early stopping where appropriate.
- Record training and validation metrics.
- Avoid repeatedly downloading the same pretrained weights.

Training code must be independent of inference code.

Never commit:

- Large datasets.
- Raw video or audio collections.
- Model checkpoints.
- API keys.
- Credentials.
- Environment-specific temporary files.

Use Git LFS or appropriate external storage only when required and agreed upon.

---

## 10. TESTING AND VALIDATION

Every contributor is responsible for testing their own implementation.

Tests must be created alongside module development rather than postponed until final integration.

### Required Testing Levels

1. Unit Testing
   - Validate individual functions.
   - Check input and output shapes.
   - Test expected and unexpected inputs.

2. Integration Testing
   - Validate communication between modules.
   - Verify interface compatibility.
   - Check tensor dimensions and data types.

3. Pipeline Testing
   - Test the complete multimodal pipeline.
   - Validate the output at every major stage.
   - Test missing-modality scenarios.

4. Model Evaluation
   - Evaluate emotion prediction.
   - Evaluate sentiment derivation.
   - Evaluate confidence calibration.
   - Evaluate temporal consistency.

5. XAI Evaluation
   - Check whether explanations correspond to actual model behavior.
   - Test explanation stability.
   - Compare explanations with modality ablation results where feasible.
   - Evaluate faithfulness and consistency.

Use Pytest for automated testing.

Do not mark a module as completed merely because its code executes without errors.

---

## 11. GIT AND TEAM COLLABORATION RULES

This is a four-member collaborative project.

All four members must actively contribute to implementation.

### Branching Strategy

push only to the my specified branch

Never push directly to `main`.

Use pull requests for integration.

### Commit Convention

Use descriptive commit messages:

```text
feat: add visual feature encoder
feat: implement uncertainty gating
feat: add temporal transition analysis
feat: implement sentiment derivation
feat: add XAI explanation engine

fix: correct tensor dimension mismatch
test: add emotion classifier unit tests
docs: update module interface documentation
refactor: simplify multimodal fusion
```

### Pull Request Requirements

Every pull request must include:

1. Purpose of the change.
2. Files or modules modified.
3. Interface changes, if any.
4. Tests performed.
5. Dependencies introduced.
6. Known limitations.

Do not merge code that breaks an existing module contract.

The integration lead is responsible for architecture consistency and final integration, but each member remains responsible for code quality and testing.

---

## 12. AI CODING ASSISTANT INSTRUCTIONS

These instructions apply to ChatGPT, Cursor, Claude Code, Opencode, Antigravity, GitHub Copilot, and other AI coding assistants working in this repository.

### Before Writing Code

1. Read this AGENTS.md file.
2. Inspect the existing repository structure.
3. Read relevant module interfaces.
4. Identify the module you are working on.
5. Understand its inputs and expected outputs.
6. Inspect existing implementations before creating new files.

### While Writing Code

- Implement only the requested module or task.
- Respect the established architecture.
- Follow existing naming conventions.
- Reuse existing utilities.
- Do not duplicate existing functionality.
- Do not modify unrelated modules.
- Do not change public interfaces without justification.
- Include type hints and documentation.
- Include appropriate unit tests.

### Before Completing a Task

Verify:

- The code executes.
- Inputs and outputs match the documented interface.
- Tensor dimensions are consistent.
- Dependencies are documented.
- Tests pass.
- No unrelated files have been modified.

### AI Assistant Restrictions

NEVER:

- Rewrite the entire project when asked to implement one module.
- Invent undocumented interfaces.
- Change the overall architecture independently.
- Replace an existing model without justification.
- Introduce arbitrary sentiment mappings.
- Assume missing dataset annotations.
- Generate fake training results or evaluation metrics.
- Remove another contributor's implementation without approval.
- Introduce unnecessary dependencies.
- Claim that an implementation is complete without testing it.

If a requirement is unclear, explain the ambiguity and propose a solution before making architectural changes.

If a requested implementation conflicts with this file, identify the conflict instead of silently overriding the project architecture.

---

## 13. EXPERIMENT TRACKING AND REPRODUCIBILITY

Every significant experiment must document:

- Experiment identifier.
- Objective.
- Dataset and split.
- Model architecture.
- Pretrained weights.
- Trainable parameters.
- Hyperparameters.
- Random seed.
- Evaluation metrics.
- Results.
- Observations.
- Limitations.

Keep experimental results separate from production implementations.

Do not overwrite previous experimental results.

Maintain a record of baseline and improved models for comparison.

---

## 14. DOCUMENTATION REQUIREMENTS

Every major module must have documentation explaining:

1. Purpose.
2. Input format.
3. Output format.
4. Model or algorithm used.
5. Dependencies.
6. Tensor dimensions.
7. Training requirements.
8. Testing procedure.
9. Known limitations.

Keep the following documents updated:

- README.md
- docs/architecture.md
- docs/module_interfaces.md
- docs/experiments.md
- docs/research_notes.md

The README must describe the actual implemented system, not merely the planned architecture.

---

## 15. FINAL DEVELOPMENT PRIORITIES

Development should follow this staged order:

1. Phase 0 — Architecture and interface freezing: define architecture, tensor
   dimensions, data contracts, and module responsibilities.
2. Phase 1 — Project foundation: establish environment, dependencies,
   configuration, logging, reproducibility, and repository setup.
3. Phase 2 — Dataset exploration and preprocessing: understand data,
   annotations, modalities, labels, timestamps, and splits.
4. Phase 3 — Independent modality encoders: implement video, audio, and text
   feature extraction.
5. Phase 4 — Multimodal alignment and fusion: align modality representations
   and generate fused representations.
6. Phase 5 — Affective state and emotion derivation: derive affective-state
   representations and multi-label emotion probabilities.
7. Phase 6 — Certainty and uncertainty mechanism: estimate emotion-wise
   confidence, calibrate predictions, and implement uncertainty gating.
8. Phase 7 — Temporal representation and transition analysis: model emotion
   evolution and produce transitional embeddings.
9. Phase 8 — Latent mapping and sentiment derivation: combine the required
   representations to derive sentiment and sentiment confidence.
10. Phase 9 — Complete model integration: connect modules into an end-to-end
    architecture.
11. Phase 10 — Training and optimization: implement losses, training loops,
    checkpoints, and optimization.
12. Phase 11 — Evaluation and experimentation: evaluate modules and the
    complete model, including ablation experiments.
13. Phase 12 — Inference and final pipeline: build prediction scripts,
    reproducible inference, and final documentation.
14. Phase 13 — Explainable AI (XAI) and interpretability: explain actual
    modality, feature, and temporal contributions without changing predictions.

Modules may be developed in parallel only after their interfaces and data
contracts are frozen. The uncertainty gate may be implemented during Phase 6,
but it must be connected only after temporal transition representations are
available. This development order does not change the canonical runtime order:
temporal transition representations must be available before uncertainty-aware
gating and final sentiment mapping. The integration lead must preserve that
runtime order when connecting the modules.

---

## 16. DEFINITION OF DONE

A component is considered complete only when:

- Its documented responsibility is implemented.
- Its input/output contract is satisfied.
- Its unit tests pass.
- Its integration requirements are documented.
- Its dependencies are recorded.
- Its limitations are known.
- Its implementation is committed to the appropriate branch.

The complete project is considered ready for final evaluation when:

1. Video, audio, and optional text inputs are supported.
2. Modality-specific representations are generated.
3. Multimodal fusion works correctly.
4. Affective state and multi-label emotions are predicted.
5. Individual emotion confidence and uncertainty are represented.
6. Temporal transitions are incorporated.
7. Uncertainty-aware gating operates correctly.
8. Latent sentiment mapping produces the final sentiment.
9. Sentiment confidence is estimated.
10. XAI generates meaningful explanations for the derived sentiment.
11. The complete pipeline passes integration testing.
12. Results are reproducible and properly documented.

---

## FINAL PRINCIPLE

Build a modular, research-oriented, computationally efficient and reproducible system.

Prioritize architectural consistency, scientifically defensible methodology, meaningful experimentation, and reliable integration over unnecessary complexity.

Every implementation must contribute to the central research objective:

UNDERSTANDING HOW MULTIDIMENSIONAL EMOTIONAL INFORMATION, ITS UNCERTAINTY, AND ITS TEMPORAL EVOLUTION CAN BE USED TO DERIVE SENTIMENT AND EXPLAIN THAT DERIVATION.
