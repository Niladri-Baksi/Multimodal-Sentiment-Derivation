import json

import pytest

import generate_reports


def test_evaluation_report_is_built_from_supplied_predictions(tmp_path, monkeypatch):
    evaluation_input = tmp_path / "evaluation.json"
    evaluation_input.write_text(
        json.dumps(
            {
                "experiment_name": "Test run",
                "title": "Measured results",
                "dataset": "Test dataset",
                "classes": [0, 1],
                "labels": [0, 0, 1, 1],
                "predictions": [0, 1, 1, 1],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(generate_reports, "OUTPUT_DIR", output_dir)

    generate_reports.generate_evaluation_reports(evaluation_input)

    result = json.loads(
        (output_dir / "model_evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert result["experiment_name"] == "Test run"
    assert result["metrics"]["overall"]["accuracy"] == 0.75
    assert result["metrics"]["overall"]["macro_f1"] == 0.7333
    assert "0.7500" in (output_dir / "journal_paper_report.md").read_text(
        encoding="utf-8"
    )


def test_evaluation_report_rejects_missing_prediction_fields(tmp_path):
    evaluation_input = tmp_path / "incomplete.json"
    evaluation_input.write_text(json.dumps({"labels": [0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        generate_reports.generate_evaluation_reports(evaluation_input)
