import pytest

from model_evaluation_matrix import classification_metrics, confusion_matrix
from journal_paper_report import build_journal_report


def test_confusion_matrix_counts_actual_vs_predicted_labels():
    assert confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], [0, 1]) == {
        0: {0: 1, 1: 1},
        1: {0: 0, 1: 2},
    }


def test_classification_metrics_calculate_class_and_overall_scores():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0, 1])

    assert result["0"] == {
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "support": 2,
    }
    assert result["1"] == {
        "precision": 0.6667,
        "recall": 1.0,
        "f1": 0.8,
        "support": 2,
    }
    assert result["overall"]["accuracy"] == 0.75
    assert result["overall"]["macro_f1"] == 0.7333
    assert result["overall"]["weighted_f1"] == 0.7333


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (confusion_matrix, ([0], [0, 1], [0, 1])),
        (classification_metrics, ([0], [0, 1], [0, 1])),
        (confusion_matrix, ([0], [2], [0, 1])),
        (classification_metrics, ([0], [2], [0, 1])),
        (confusion_matrix, ([0], [0], [])),
        (classification_metrics, ([0], [0], [])),
        (confusion_matrix, ([0], [0], [0, 0])),
        (classification_metrics, ([0], [0], [0, 0])),
    ],
)
def test_classification_helpers_reject_invalid_inputs(function, arguments):
    with pytest.raises(ValueError):
        function(*arguments)


def test_empty_predictions_return_zero_metrics_for_declared_classes():
    result = classification_metrics([], [], [0, 1])

    assert result["0"]["support"] == 0
    assert result["overall"]["accuracy"] == 0.0
    assert result["overall"]["macro_f1"] == 0.0


def test_paper_report_does_not_claim_unmeasured_performance():
    report = build_journal_report(
        title="Draft",
        results={},
        ablation={},
        dataset="Unspecified",
        notes="",
    )

    assert "No measured model results were supplied" in report
    assert "No performance interpretation is generated automatically" in report
    assert "achieves strong predictive performance" not in report
