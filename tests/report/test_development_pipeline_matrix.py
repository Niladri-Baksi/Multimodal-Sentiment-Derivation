from pathlib import Path

from development_pipeline_matrix import (
    PIPELINE_STAGES,
    build_pipeline_matrix,
    render_pipeline_report,
)


def test_pipeline_matrix_follows_all_fourteen_development_phases():
    assert [stage.phase for stage in PIPELINE_STAGES] == list(range(14))
    assert PIPELINE_STAGES[0].title == "Architecture and interface freezing"
    assert PIPELINE_STAGES[-1].title == "Explainable AI and interpretability"


def test_pipeline_matrix_marks_empty_evidence_as_not_started(tmp_path: Path):
    stage = PIPELINE_STAGES[3]
    for relative_path in stage.required_files:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    rows = build_pipeline_matrix(tmp_path)

    assert rows[3]["status"] == "not started"
    assert rows[3]["available_files"] == "None"


def test_pipeline_matrix_marks_partial_evidence(tmp_path: Path):
    stage = PIPELINE_STAGES[3]
    first_path = tmp_path / stage.required_files[0]
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("encoder implementation", encoding="utf-8")

    rows = build_pipeline_matrix(tmp_path)

    assert rows[3]["status"] == "partial"
    assert rows[3]["available_files"] == stage.required_files[0]


def test_pipeline_report_explains_what_status_does_not_validate():
    report = render_pipeline_report(build_pipeline_matrix())

    assert "does not certify behavior" in report
    assert "temporal transition representations" in report
