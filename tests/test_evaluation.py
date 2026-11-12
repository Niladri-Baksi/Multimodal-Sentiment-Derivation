import csv
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentiment_derivation.evaluation import evaluate_predictions


EMOTIONS = ["Curiosity", "Fear"]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.csv"
    samples = [
        ("clip-1", [1, 0], 2.0, [0.9, 0.1]),
        ("clip-2", [0, 0], 4.0, [0.1, 0.1]),
        ("clip-3", [0, 1], 5.2, [0.2, 0.8]),
        ("clip-4", [1, 1], 6.0, [0.6, 0.4]),
    ]
    manifest = {
        "emotion_names": EMOTIONS,
        "valence_range": [1.0, 7.0],
        "samples": {
            "test": [
                {
                    "sample_id": sample_id,
                    "emotion_labels": dict(zip(EMOTIONS, labels)),
                    "other_original_annotations": {"valence": valence},
                }
                for sample_id, labels, valence, _ in samples
            ]
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fields = [
        "sample_id",
        "emotion_probability_Curiosity",
        "emotion_probability_Fear",
        "predicted_valence",
        "sentiment_probability_Negative",
        "sentiment_probability_Neutral",
        "sentiment_probability_Positive",
    ]
    with predictions_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        sentiment_probs = [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.3, 0.6],
        ]
        predicted_valences = [2.0, 4.0, 5.2, 6.0]
        for index, (sample_id, _, _, probs) in enumerate(samples):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "emotion_probability_Curiosity": probs[0],
                    "emotion_probability_Fear": probs[1],
                    "predicted_valence": predicted_valences[index],
                    "sentiment_probability_Negative": sentiment_probs[index][0],
                    "sentiment_probability_Neutral": sentiment_probs[index][1],
                    "sentiment_probability_Positive": sentiment_probs[index][2],
                }
            )
    return manifest_path, predictions_path


def test_evaluation_uses_supplied_predictions_and_writes_real_metrics_and_plots(tmp_path):
    manifest_path, predictions_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "evaluation"

    result = evaluate_predictions(
        manifest_path,
        predictions_path,
        output_dir,
        model_name="fixture-model",
        split="test",
    )

    assert result["sample_count"] == 4
    assert result["model_name"] == "fixture-model"
    assert result["emotion_aggregate"]["macro_f1"] == pytest.approx(0.8333333)
    assert result["emotion_metrics"][0]["f1"] == 1.0
    assert result["emotion_metrics"][1]["f1"] == pytest.approx(2 / 3)
    assert result["sentiment_classification"]["accuracy"] == 1.0
    assert result["valence_regression"]["mae"] == 0.0
    assert result["emotion_confusion_matrices"]["Curiosity"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }
    for filename in (
        "emotion_confusion_matrices.png",
        "emotion_f1_scores.png",
        "sentiment_confusion_matrix.png",
        "valence_regression.png",
        "evaluation_metrics.json",
        "evaluation_report.md",
        "per_emotion_metrics.csv",
        "evaluated_predictions.csv",
    ):
        assert (output_dir / filename).is_file()
        assert (output_dir / filename).stat().st_size > 0


def test_evaluation_rejects_missing_or_extra_prediction_ids(tmp_path):
    manifest_path, predictions_path = _write_inputs(tmp_path)
    predictions_path.write_text(
        predictions_path.read_text(encoding="utf-8").replace("clip-4,", "other-clip,"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample IDs do not match"):
        evaluate_predictions(manifest_path, predictions_path, tmp_path / "evaluation")


def test_evaluation_rejects_invalid_probabilities(tmp_path):
    manifest_path, predictions_path = _write_inputs(tmp_path)
    content = predictions_path.read_text(encoding="utf-8")
    predictions_path.write_text(content.replace("0.9,0.1", "1.5,0.1"), encoding="utf-8")

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        evaluate_predictions(manifest_path, predictions_path, tmp_path / "evaluation")
