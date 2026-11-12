from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence


def format_markdown_table(headers: Sequence[str], rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def build_experiment_summary_table(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "_No measured model results were supplied._"

    rows = [
        {
            "Model": name,
            "Accuracy": _format_metric(values, "accuracy"),
            "Macro F1": _format_metric(values, "macro_f1"),
            "Weighted F1": _format_metric(values, "weighted_f1"),
        }
        for name, values in results.items()
    ]
    return format_markdown_table(["Model", "Accuracy", "Macro F1", "Weighted F1"], rows)


def build_ablation_table(results: Dict[str, Dict[str, float]]) -> str:
    if not results:
        return "_No measured ablation results were supplied._"

    rows = [
        {
            "Variant": name,
            "Accuracy": _format_metric(values, "accuracy"),
            "Macro F1": _format_metric(values, "macro_f1"),
            "Gain": _format_metric(values, "gain"),
        }
        for name, values in results.items()
    ]
    return format_markdown_table(["Variant", "Accuracy", "Macro F1", "Gain"], rows)


def _format_metric(metrics: Dict[str, float], name: str) -> str:
    value = metrics.get(name)
    return f"{float(value):.4f}" if value is not None else "N/A"


def build_journal_report(title: str, results: Dict[str, Dict[str, float]], ablation: Dict[str, Dict[str, float]], dataset: str, notes: str) -> str:
    summary_table = build_experiment_summary_table(results)
    ablation_table = build_ablation_table(ablation)

    report = f'''# {title}

## Abstract
This section should summarize the implemented method, dataset, evaluation protocol, and measured findings. Do not report claims until the corresponding experiments have been run.

## Dataset and Experimental Setup
- Dataset: {dataset}
- Evaluation protocol: [Describe the actual split and validation procedure]
- Reporting metrics: accuracy, macro F1, and weighted F1 when measured
- Reproducibility: [Record seeds, preprocessing, software versions, and hardware]

## Main Results
{summary_table}

No performance interpretation is generated automatically. Interpret results only after checking the measured values and evaluation protocol.

## Ablation Study
{ablation_table}

## Discussion
{notes}

## Limitations
[Describe limitations supported by the implementation and experiments.]

## Reproducibility Checklist
- [ ] Fixed train/validation/test split documented
- [ ] Preprocessing and label mapping documented
- [ ] Random seeds and software/hardware recorded
- [ ] Metrics computed from held-out predictions
- [ ] Ablation and error analysis completed
'''
    return report


def write_report(path: str | Path, report_text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
