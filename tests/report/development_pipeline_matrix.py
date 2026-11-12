from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "report" / "output"


@dataclass(frozen=True)
class PipelineStage:
    phase: int
    title: str
    purpose: str
    required_files: tuple[str, ...]


PIPELINE_STAGES = (
    PipelineStage(
        0,
        "Architecture and interface freezing",
        "Define architecture, tensor dimensions, data contracts, and module responsibilities.",
        ("AGENTS.md", "docs/architecture/module_interfaces.md"),
    ),
    PipelineStage(
        1,
        "Project foundation",
        "Set up dependencies, configuration, logging, reproducibility, and repository tooling.",
        (
            "pyproject.toml",
            "requirements.txt",
            "configs/model.yaml",
            "src/sentiment_derivation/utils/config.py",
            "src/sentiment_derivation/utils/logger.py",
            "src/sentiment_derivation/utils/seed.py",
        ),
    ),
    PipelineStage(
        2,
        "Dataset exploration and preprocessing",
        "Understand annotations, modalities, labels, timestamps, splits, and preprocessing.",
        (
            "src/sentiment_derivation/data/dataset.py",
            "src/sentiment_derivation/data/dataloader.py",
            "src/sentiment_derivation/data/preprocessing.py",
            "scripts/prepare_data.py",
            "notebooks/01_dataset_exploration.ipynb",
        ),
    ),
    PipelineStage(
        3,
        "Independent modality encoders",
        "Extract video, audio, and text features independently.",
        (
            "src/sentiment_derivation/encoders/video_encoder.py",
            "src/sentiment_derivation/encoders/audio_encoder.py",
            "src/sentiment_derivation/encoders/text_encoder.py",
        ),
    ),
    PipelineStage(
        4,
        "Multimodal alignment and fusion",
        "Align modality representations and produce fused representations.",
        (
            "src/sentiment_derivation/fusion/alignment.py",
            "src/sentiment_derivation/fusion/multimodal_fusion.py",
        ),
    ),
    PipelineStage(
        5,
        "Affective state and emotion derivation",
        "Derive affective-state representations and multi-label emotion probabilities.",
        (
            "src/sentiment_derivation/emotion/affective_state.py",
            "src/sentiment_derivation/emotion/emotion_classifier.py",
        ),
    ),
    PipelineStage(
        6,
        "Certainty and uncertainty mechanism",
        "Estimate emotion-wise confidence, calibrate predictions, and implement uncertainty gating.",
        (
            "src/sentiment_derivation/uncertainty/certainty_estimator.py",
            "src/sentiment_derivation/uncertainty/calibration.py",
            "src/sentiment_derivation/uncertainty/uncertainty_gate.py",
        ),
    ),
    PipelineStage(
        7,
        "Temporal representation and transition analysis",
        "Model emotional evolution and produce transitional embeddings.",
        (
            "src/sentiment_derivation/temporal/temporal_encoder.py",
            "src/sentiment_derivation/temporal/transition_analyzer.py",
        ),
    ),
    PipelineStage(
        8,
        "Latent mapping and sentiment derivation",
        "Combine required representations to derive sentiment and sentiment confidence.",
        (
            "src/sentiment_derivation/sentiment/latent_mapper.py",
            "src/sentiment_derivation/sentiment/sentiment_head.py",
        ),
    ),
    PipelineStage(
        9,
        "Complete model integration",
        "Connect the modules into the canonical end-to-end architecture.",
        (
            "src/sentiment_derivation/integration/model.py",
            "src/sentiment_derivation/integration/pipeline.py",
        ),
    ),
    PipelineStage(
        10,
        "Training and optimization",
        "Implement losses, training loops, checkpoints, and optimization.",
        (
            "src/sentiment_derivation/training/losses.py",
            "src/sentiment_derivation/training/trainer.py",
            "scripts/train.py",
            "configs/training.yaml",
        ),
    ),
    PipelineStage(
        11,
        "Evaluation and experimentation",
        "Evaluate modules and the complete model, including ablation experiments.",
        (
            "src/sentiment_derivation/training/metrics.py",
            "scripts/evaluate.py",
            "tests/report/model_evaluation_matrix.py",
        ),
    ),
    PipelineStage(
        12,
        "Inference and final pipeline",
        "Build prediction scripts, reproducible inference, and final documentation.",
        ("scripts/predict.py", "README.md"),
    ),
    PipelineStage(
        13,
        "Explainable AI and interpretability",
        "Explain actual modality, feature, and temporal contributions without changing predictions.",
        ("src/sentiment_derivation/explainability/explainer.py",),
    ),
)


def build_pipeline_matrix(root: Path = ROOT) -> list[dict[str, str]]:
    """Assess whether each stage's expected source artifacts are present and non-empty."""
    rows = []
    for stage in PIPELINE_STAGES:
        available = [
            relative_path
            for relative_path in stage.required_files
            if (root / relative_path).is_file()
            and (root / relative_path).stat().st_size > 0
        ]
        missing = [path for path in stage.required_files if path not in available]
        if not missing:
            status = "present"
        elif available:
            status = "partial"
        else:
            status = "not started"

        rows.append(
            {
                "phase": str(stage.phase),
                "title": stage.title,
                "purpose": stage.purpose,
                "status": status,
                "available_files": ", ".join(available) or "None",
                "missing_or_empty_files": ", ".join(missing) or "None",
            }
        )
    return rows


def render_pipeline_report(rows: Sequence[dict[str, str]]) -> str:
    header = (
        "| Phase | Development stage | Purpose | Source-file status | Available evidence | Missing or empty evidence |",
        "|---:|---|---|---|---|---|",
    )
    lines = list(header)
    for row in rows:
        lines.append(
            "| {phase} | {title} | {purpose} | {status} | {available_files} | {missing_or_empty_files} |".format(**row)
        )

    return "\n".join(
        [
            "# Development Pipeline Readiness",
            "",
            "This matrix follows the 14 phases in the agreed development pipeline.",
            "Status measures only whether the listed source files exist and contain data; "
            "it does not certify behavior, interface compatibility, test coverage, or scientific validity.",
            "",
            *lines,
            "",
            "Runtime ordering remains governed by the canonical architecture: temporal "
            "transition representations must be available before uncertainty-aware gating "
            "is connected to final sentiment mapping.",
            "",
        ]
    )


def write_pipeline_reports(root: Path = ROOT, output_dir: Path = OUTPUT_DIR) -> list[dict[str, str]]:
    rows = build_pipeline_matrix(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "development_pipeline_matrix.md").write_text(
        render_pipeline_report(rows), encoding="utf-8"
    )
    with (output_dir / "development_pipeline_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
