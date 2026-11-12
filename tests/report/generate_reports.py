from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .development_pipeline_matrix import write_pipeline_reports
    from .journal_paper_report import build_journal_report, write_report
    from .model_evaluation_matrix import export_csv, export_json, model_matrix_report
else:
    from development_pipeline_matrix import write_pipeline_reports
    from journal_paper_report import build_journal_report, write_report
    from model_evaluation_matrix import export_csv, export_json, model_matrix_report


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tests" / "report" / "output"


def generate_evaluation_reports(input_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation input must be a JSON object")

    required_keys = {"classes", "labels", "predictions"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(
            "Evaluation input is missing required fields: "
            + ", ".join(sorted(missing_keys))
        )

    experiment_name = str(payload.get("experiment_name", "Model"))
    model_report = model_matrix_report(
        experiment_name,
        payload["labels"],
        payload["predictions"],
        payload["classes"],
    )
    export_json(OUTPUT_DIR / "model_evaluation_summary.json", model_report)

    overall_metrics = model_report["metrics"]["overall"]
    csv_rows = [{"Model": experiment_name, **overall_metrics}]
    export_csv(OUTPUT_DIR / "model_evaluation_matrix.csv", csv_rows)

    ablation = payload.get("ablation", {})
    if not isinstance(ablation, dict):
        raise ValueError("The optional 'ablation' field must be a JSON object")

    paper_report = build_journal_report(
        title=str(payload.get("title", "Model Evaluation Results")),
        results={experiment_name: overall_metrics},
        ablation=ablation,
        dataset=str(payload.get("dataset", "[Specify the evaluated dataset]")),
        notes=str(payload.get("discussion", "[Add interpretation based on measured results.]")),
    )
    write_report(OUTPUT_DIR / "journal_paper_report.md", paper_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pipeline and model-evaluation reports.")
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        help="JSON file containing real labels, predictions, and class labels.",
    )
    arguments = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = write_pipeline_reports()

    if arguments.evaluation_json is not None:
        generate_evaluation_reports(arguments.evaluation_json)
    else:
        print("No evaluation data supplied; numerical model results were not generated.")

    print(f"Pipeline phases assessed: {len(rows)}")
    print(f"Reports generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
