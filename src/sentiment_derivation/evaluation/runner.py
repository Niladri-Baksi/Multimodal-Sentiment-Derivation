from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from sentiment_derivation.sentiment.sentiment_head import (
    DEFAULT_SENTIMENT_NAMES,
    valence_to_sentiment_label,
)


EMOTION_PROBABILITY_PREFIX = "emotion_probability_"
SENTIMENT_PROBABILITY_PREFIX = "sentiment_probability_"
PREDICTED_VALENCE_COLUMN = "predicted_valence"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read manifest JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return payload


def _read_predictions(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as file_obj:
            reader = csv.DictReader(file_obj)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError("Prediction CSV must contain a 'sample_id' column")
            rows = list(reader)
            headers = list(reader.fieldnames)
    except OSError as error:
        raise ValueError(f"Unable to read prediction CSV {path}: {error}") from error
    if not rows:
        raise ValueError(f"Prediction CSV contains no predictions: {path}")
    return rows, headers


def _finite_float(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a number; received {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite; received {value!r}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_test_rows(manifest: dict[str, Any], split: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    emotion_names = manifest.get("emotion_names")
    samples_by_split = manifest.get("samples")
    if not isinstance(emotion_names, list) or not emotion_names:
        raise ValueError("Manifest must contain a non-empty 'emotion_names' list")
    if len(set(emotion_names)) != len(emotion_names):
        raise ValueError("Manifest emotion_names must not contain duplicates")
    if not isinstance(samples_by_split, dict) or not isinstance(samples_by_split.get(split), list):
        raise ValueError(f"Manifest must contain a samples list for split {split!r}")

    sample_rows = samples_by_split[split]
    if not sample_rows:
        raise ValueError(f"Manifest split {split!r} contains no samples")

    ids: list[str] = []
    labels: list[list[int]] = []
    valences: list[float] = []
    seen: set[str] = set()
    for row_index, sample in enumerate(sample_rows):
        if not isinstance(sample, dict):
            raise ValueError(f"Manifest sample {row_index} is not an object")
        sample_id = sample.get("sample_id")
        emotion_labels = sample.get("emotion_labels")
        annotations = sample.get("other_original_annotations")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Manifest sample {row_index} has an invalid sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest split {split!r}: {sample_id}")
        if not isinstance(emotion_labels, dict):
            raise ValueError(f"Manifest sample {sample_id} has no emotion_labels object")
        if not isinstance(annotations, dict) or "valence" not in annotations:
            raise ValueError(f"Manifest sample {sample_id} has no original valence annotation")

        sample_labels = []
        for emotion_name in emotion_names:
            value = emotion_labels.get(emotion_name)
            if value not in (0, 1, False, True):
                raise ValueError(
                    f"Invalid label for {emotion_name!r} in sample {sample_id}: {value!r}"
                )
            sample_labels.append(int(value))

        valence = _finite_float(annotations["valence"], f"Valence for {sample_id}")
        valence_range = manifest.get("valence_range", [1.0, 7.0])
        if (
            not isinstance(valence_range, list)
            or len(valence_range) != 2
            or not float(valence_range[0]) <= valence <= float(valence_range[1])
        ):
            raise ValueError(f"Valence for sample {sample_id} is outside the manifest range")

        seen.add(sample_id)
        ids.append(sample_id)
        labels.append(sample_labels)
        valences.append(valence)

    return ids, emotion_names, np.asarray(labels, dtype=np.int64), np.asarray(valences, dtype=float)


def _validate_prediction_rows(
    rows: list[dict[str, str]],
    headers: Sequence[str],
    expected_ids: Sequence[str],
    emotion_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    prediction_ids = [row["sample_id"].strip() for row in rows]
    if any(not sample_id for sample_id in prediction_ids):
        raise ValueError("Prediction CSV contains an empty sample_id")
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Prediction CSV contains duplicate sample_id values")

    expected_set = set(expected_ids)
    predicted_set = set(prediction_ids)
    missing = sorted(expected_set - predicted_set)
    extra = sorted(predicted_set - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} split samples (e.g. {missing[:5]})")
        if extra:
            details.append(f"{len(extra)} unknown samples (e.g. {extra[:5]})")
        raise ValueError("Prediction sample IDs do not match the evaluation split: " + "; ".join(details))

    row_by_id = {row["sample_id"].strip(): row for row in rows}
    ordered_rows = [row_by_id[sample_id] for sample_id in expected_ids]
    emotion_columns = [f"{EMOTION_PROBABILITY_PREFIX}{name}" for name in emotion_names]
    missing_emotion_columns = [name for name in emotion_columns if name not in headers]
    if missing_emotion_columns:
        raise ValueError(
            "Prediction CSV is missing model probability columns: "
            + ", ".join(missing_emotion_columns)
        )

    emotion_probabilities = np.asarray(
        [
            [
                _finite_float(row[column], f"Prediction {column} for {row['sample_id']}")
                for column in emotion_columns
            ]
            for row in ordered_rows
        ],
        dtype=float,
    )
    if np.any((emotion_probabilities < 0.0) | (emotion_probabilities > 1.0)):
        raise ValueError("Emotion probability values must be in [0, 1]")

    sentiment_columns = [
        f"{SENTIMENT_PROBABILITY_PREFIX}{name}" for name in DEFAULT_SENTIMENT_NAMES
    ]
    present_sentiment_columns = [name in headers for name in sentiment_columns]
    if any(present_sentiment_columns) and not all(present_sentiment_columns):
        raise ValueError(
            "Provide all three sentiment probability columns or none: "
            + ", ".join(sentiment_columns)
        )
    sentiment_probabilities = None
    if all(present_sentiment_columns):
        sentiment_probabilities = np.asarray(
            [
                [
                    _finite_float(row[column], f"Prediction {column} for {row['sample_id']}")
                    for column in sentiment_columns
                ]
                for row in ordered_rows
            ],
            dtype=float,
        )
        if np.any((sentiment_probabilities < 0.0) | (sentiment_probabilities > 1.0)):
            raise ValueError("Sentiment probability values must be in [0, 1]")
        if not np.allclose(sentiment_probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Each row of sentiment probabilities must sum to 1")

    predicted_valence = None
    if PREDICTED_VALENCE_COLUMN in headers:
        predicted_valence = np.asarray(
            [
                _finite_float(row[PREDICTED_VALENCE_COLUMN], f"Predicted valence for {row['sample_id']}")
                for row in ordered_rows
            ],
            dtype=float,
        )
        if np.any((predicted_valence < 1.0) | (predicted_valence > 7.0)):
            raise ValueError("Predicted valence values must be in [1, 7]")

    return emotion_probabilities, sentiment_probabilities, predicted_valence


def _plot_emotion_confusion(
    path: Path,
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    emotion_names: Sequence[str],
) -> None:
    columns = 2
    rows = math.ceil(len(emotion_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10, 4.2 * rows), squeeze=False)
    for index, emotion_name in enumerate(emotion_names):
        axis = axes[index // columns][index % columns]
        matrix = confusion_matrix(labels[:, index], predicted_labels[:, index], labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(emotion_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], labels=["Absent", "Present"])
        axis.set_yticks([0, 1], labels=["Absent", "Present"])
        for y in range(2):
            for x in range(2):
                axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for index in range(len(emotion_names), rows * columns):
        axes[index // columns][index % columns].set_visible(False)
    figure.suptitle("Per-emotion confusion matrices (held-out split)", y=1.0)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_f1(path: Path, per_emotion: list[dict[str, Any]], macro_f1: float) -> None:
    names = [row["emotion"] for row in per_emotion]
    values = [row["f1"] for row in per_emotion]
    height = max(4.0, 0.48 * len(names))
    figure, axis = plt.subplots(figsize=(9, height))
    bars = axis.barh(names, values, color="#4472C4")
    axis.axvline(macro_f1, color="#D9534F", linestyle="--", label=f"Macro F1 = {macro_f1:.3f}")
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("F1 score")
    axis.set_title("Per-emotion F1 on held-out split")
    axis.legend(loc="lower right")
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_sentiment_confusion(path: Path, matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(3), labels=DEFAULT_SENTIMENT_NAMES, rotation=20)
    axis.set_yticks(range(3), labels=DEFAULT_SENTIMENT_NAMES)
    axis.set_xlabel("Predicted sentiment")
    axis.set_ylabel("Valence-derived true sentiment")
    axis.set_title("Sentiment confusion matrix (held-out split)")
    for y in range(3):
        for x in range(3):
            axis.text(x, y, str(matrix[y, x]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_valence(path: Path, true_valence: np.ndarray, predicted_valence: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(true_valence, predicted_valence, alpha=0.65, edgecolors="none")
    axis.plot([1, 7], [1, 7], linestyle="--", color="#D9534F", label="Ideal prediction")
    axis.set_xlim(1, 7)
    axis.set_ylim(1, 7)
    axis.set_xlabel("True valence")
    axis.set_ylabel("Predicted valence")
    axis.set_title("Valence regression on held-out split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_predictions(
    manifest_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    threshold: float = 0.5,
    model_name: str = "unspecified",
    checkpoint_path: str | Path | None = None,
    neutral_margin: float = 0.25,
) -> dict[str, Any]:
    """Evaluate CSV predictions against the selected manifest split and save figures."""
    manifest_file = Path(manifest_path)
    predictions_file = Path(predictions_path)
    destination = Path(output_dir)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_file}")
    if not predictions_file.is_file():
        raise FileNotFoundError(f"Model prediction CSV not found: {predictions_file}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if not 0.0 <= neutral_margin < 3.0:
        raise ValueError("neutral_margin must be in [0, 3)")
    checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_file is not None and not checkpoint_file.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_file}")

    manifest = _read_json(manifest_file)
    sample_ids, emotion_names, y_true, true_valence = _manifest_test_rows(manifest, split)
    prediction_rows, headers = _read_predictions(predictions_file)
    emotion_probabilities, sentiment_probabilities, predicted_valence = _validate_prediction_rows(
        prediction_rows, headers, sample_ids, emotion_names
    )
    y_pred = (emotion_probabilities >= threshold).astype(np.int64)

    per_emotion: list[dict[str, Any]] = []
    emotion_confusion: dict[str, dict[str, int]] = {}
    auc_values: list[float] = []
    for index, emotion_name in enumerate(emotion_names):
        true_column = y_true[:, index]
        pred_column = y_pred[:, index]
        matrix = confusion_matrix(true_column, pred_column, labels=[0, 1])
        try:
            auc = float(roc_auc_score(true_column, emotion_probabilities[:, index]))
        except ValueError:
            auc = None
        if auc is not None:
            auc_values.append(auc)
        per_emotion.append(
            {
                "emotion": emotion_name,
                "precision": float(precision_score(true_column, pred_column, zero_division=0)),
                "recall": float(recall_score(true_column, pred_column, zero_division=0)),
                "f1": float(f1_score(true_column, pred_column, zero_division=0)),
                "support": int(true_column.sum()),
                "roc_auc": auc,
            }
        )
        emotion_confusion[emotion_name] = {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics: dict[str, Any] = {
        "model_name": model_name,
        "split": split,
        "sample_count": len(sample_ids),
        "threshold": threshold,
        "emotion_names": emotion_names,
        "emotion_metrics": per_emotion,
        "emotion_confusion_matrices": emotion_confusion,
        "emotion_aggregate": {
            "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": macro_f1,
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        },
        "provenance": {
            "manifest": str(manifest_file.resolve()),
            "manifest_sha256": _sha256(manifest_file),
            "predictions": str(predictions_file.resolve()),
            "predictions_sha256": _sha256(predictions_file),
            "checkpoint": str(checkpoint_file.resolve()) if checkpoint_file else None,
            "checkpoint_sha256": _sha256(checkpoint_file) if checkpoint_file else None,
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    _plot_emotion_confusion(
        destination / "emotion_confusion_matrices.png", y_true, y_pred, emotion_names
    )
    _plot_f1(destination / "emotion_f1_scores.png", per_emotion, macro_f1)

    if sentiment_probabilities is not None:
        y_sentiment, _, _ = valence_to_sentiment_label(
            true_valence, neutral_margin=neutral_margin
        )
        y_sentiment = np.asarray(y_sentiment, dtype=np.int64)
        predicted_sentiment = np.argmax(sentiment_probabilities, axis=1)
        sentiment_matrix = confusion_matrix(
            y_sentiment, predicted_sentiment, labels=[0, 1, 2]
        )
        metrics["sentiment_classification"] = {
            "true_label_source": "continuous ground-truth valence mapped using neutral_margin",
            "neutral_margin": neutral_margin,
            "class_names": list(DEFAULT_SENTIMENT_NAMES),
            "accuracy": float(accuracy_score(y_sentiment, predicted_sentiment)),
            "macro_precision": float(
                precision_score(y_sentiment, predicted_sentiment, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_sentiment, predicted_sentiment, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_sentiment, predicted_sentiment, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_sentiment, predicted_sentiment, average="weighted", zero_division=0)
            ),
            "confusion_matrix": sentiment_matrix.tolist(),
        }
        _plot_sentiment_confusion(
            destination / "sentiment_confusion_matrix.png", sentiment_matrix
        )

    if predicted_valence is not None:
        metrics["valence_regression"] = {
            "mae": float(mean_absolute_error(true_valence, predicted_valence)),
            "rmse": float(np.sqrt(mean_squared_error(true_valence, predicted_valence))),
            "r2": float(r2_score(true_valence, predicted_valence)),
        }
        _plot_valence(
            destination / "valence_regression.png", true_valence, predicted_valence
        )

    _write_metrics_csv(destination / "per_emotion_metrics.csv", per_emotion)
    _write_predictions_csv(
        destination / "evaluated_predictions.csv",
        sample_ids,
        emotion_names,
        y_true,
        emotion_probabilities,
        y_pred,
    )
    (destination / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destination / "evaluation_report.md").write_text(
        _render_markdown_report(metrics), encoding="utf-8"
    )
    return metrics


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions_csv(
    path: Path,
    sample_ids: Sequence[str],
    emotion_names: Sequence[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    fieldnames = ["sample_id"]
    for emotion_name in emotion_names:
        fieldnames.extend(
            [
                f"true_{emotion_name}",
                f"probability_{emotion_name}",
                f"predicted_{emotion_name}",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_id in enumerate(sample_ids):
            row: dict[str, Any] = {"sample_id": sample_id}
            for emotion_index, emotion_name in enumerate(emotion_names):
                row[f"true_{emotion_name}"] = int(y_true[row_index, emotion_index])
                row[f"probability_{emotion_name}"] = float(
                    probabilities[row_index, emotion_index]
                )
                row[f"predicted_{emotion_name}"] = int(
                    predictions[row_index, emotion_index]
                )
            writer.writerow(row)


def _render_markdown_report(metrics: dict[str, Any]) -> str:
    aggregate = metrics["emotion_aggregate"]
    lines = [
        "# Held-out Model Evaluation",
        "",
        f"- Model: `{metrics['model_name']}`",
        f"- Split: `{metrics['split']}`",
        f"- Samples: {metrics['sample_count']}",
        f"- Emotion probability threshold: {metrics['threshold']:.3f}",
        f"- Prediction SHA-256: `{metrics['provenance']['predictions_sha256']}`",
        "",
        "## Multi-label Emotion Results",
        "",
        "| Emotion | Precision | Recall | F1 | Support | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["emotion_metrics"]:
        auc = "N/A" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
        lines.append(
            f"| {row['emotion']} | {row['precision']:.4f} | {row['recall']:.4f} "
            f"| {row['f1']:.4f} | {row['support']} | {auc} |"
        )
    lines.extend(
        [
            "",
            f"- Micro F1: {aggregate['micro_f1']:.4f}",
            f"- Macro F1: {aggregate['macro_f1']:.4f}",
            f"- Weighted F1: {aggregate['weighted_f1']:.4f}",
            f"- Exact-match (subset) accuracy: {aggregate['subset_accuracy']:.4f}",
            f"- Hamming loss: {aggregate['hamming_loss']:.4f}",
            "",
            "## Figures",
            "",
            "- `emotion_confusion_matrices.png`: per-emotion binary confusion matrices.",
            "- `emotion_f1_scores.png`: per-emotion F1 values and macro-F1 reference.",
        ]
    )
    if "sentiment_classification" in metrics:
        sentiment = metrics["sentiment_classification"]
        lines.extend(
            [
                "",
                "## Sentiment Classification",
                "",
                "Ground-truth sentiment classes are derived from original continuous valence annotations "
                f"using a neutral margin of {sentiment['neutral_margin']:.3f}.",
                "",
                f"- Accuracy: {sentiment['accuracy']:.4f}",
                f"- Macro F1: {sentiment['macro_f1']:.4f}",
                "- Figure: `sentiment_confusion_matrix.png`.",
            ]
        )
    if "valence_regression" in metrics:
        regression = metrics["valence_regression"]
        lines.extend(
            [
                "",
                "## Valence Regression",
                "",
                f"- MAE: {regression['mae']:.4f}",
                f"- RMSE: {regression['rmse']:.4f}",
                f"- R²: {regression['r2']:.4f}",
                "- Figure: `valence_regression.png`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Manifest SHA-256: `{metrics['provenance']['manifest_sha256']}`",
            f"- Prediction SHA-256: `{metrics['provenance']['predictions_sha256']}`",
            f"- Checkpoint SHA-256: `{metrics['provenance']['checkpoint_sha256'] or 'not supplied'}`",
            "",
            "Metrics and plots were computed from the supplied model predictions matched by sample ID "
            "to the selected manifest split. File hashes provide traceability only; they do not prove "
            "that the prediction file was produced by the supplied checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)
