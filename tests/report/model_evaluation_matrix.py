from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence


def _validate_classification_inputs(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]
) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not classes:
        raise ValueError("classes must contain at least one label")
    if len(set(classes)) != len(classes):
        raise ValueError("classes must not contain duplicate labels")

    known_classes = set(classes)
    for actual, predicted in zip(y_true, y_pred):
        if actual not in known_classes or predicted not in known_classes:
            raise ValueError(
                f"Unknown class label in predictions: actual={actual}, predicted={predicted}"
            )


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]) -> Dict[int, Dict[int, int]]:
    """Return a confusion matrix keyed by true class and predicted class."""
    _validate_classification_inputs(y_true, y_pred, classes)
    matrix = {cls: {pred_cls: 0 for pred_cls in classes} for cls in classes}

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    return matrix


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]
) -> Dict[str, Dict[str, float | int]]:
    """Compute precision, recall, F1, support per class and macro/weighted averages."""
    _validate_classification_inputs(y_true, y_pred, classes)
    metrics = {}
    unrounded_metrics = {}
    total_support = len(y_true)

    for cls in classes:
        tp = 0
        fp = 0
        fn = 0
        support = 0

        for actual, predicted in zip(y_true, y_pred):
            if actual == cls:
                support += 1
            if actual == cls and predicted == cls:
                tp += 1
            if predicted == cls and actual != cls:
                fp += 1
            if actual == cls and predicted != cls:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        unrounded_metrics[cls] = (precision, recall, f1, support)
        metrics[str(cls)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    macro_precision = round(
        sum(values[0] for values in unrounded_metrics.values()) / len(classes), 4
    )
    macro_recall = round(
        sum(values[1] for values in unrounded_metrics.values()) / len(classes), 4
    )
    macro_f1 = round(
        sum(values[2] for values in unrounded_metrics.values()) / len(classes), 4
    )

    weighted_precision = sum(
        precision * support for precision, _, _, support in unrounded_metrics.values()
    )
    weighted_recall = sum(
        recall * support for _, recall, _, support in unrounded_metrics.values()
    )
    weighted_f1 = sum(
        f1 * support for _, _, f1, support in unrounded_metrics.values()
    )

    weighted_precision = round(weighted_precision / total_support, 4) if total_support else 0.0
    weighted_recall = round(weighted_recall / total_support, 4) if total_support else 0.0
    weighted_f1 = round(weighted_f1 / total_support, 4) if total_support else 0.0

    correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
    accuracy = round(correct / total_support, 4) if total_support else 0.0

    metrics["overall"] = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    return metrics


def markdown_table(rows: Iterable[Dict[str, object]], headers: Sequence[str]) -> str:
    """Return a markdown table for a list of rows."""
    rows = list(rows)
    if not rows:
        return "| No data |\n|---|"

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def export_csv(path: str | Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write rows as CSV if there are data rows; otherwise create an empty file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_json(path: str | Path, payload: Dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model_summary(experiment_name: str, metrics: Dict[str, object], confusion: Dict[int, Dict[int, int]], classes: Sequence[int]) -> Dict[str, object]:
    summary = {
        "experiment_name": experiment_name,
        "classes": [int(cls) for cls in classes],
        "metrics": metrics,
        "confusion_matrix": {str(true_cls): {str(pred_cls): value for pred_cls, value in matrix.items()} for true_cls, matrix in confusion.items()},
    }
    return summary


def model_matrix_report(experiment_name: str, labels: Sequence[int], predictions: Sequence[int], classes: Sequence[int]) -> Dict[str, object]:
    conf = confusion_matrix(labels, predictions, classes)
    metrics = classification_metrics(labels, predictions, classes)
    summary = build_model_summary(experiment_name, metrics, conf, classes)
    return summary
