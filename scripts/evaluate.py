"""Run a checkpoint on a held-out split and evaluate its real predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_yaml, loader_options, manifest_path, select_device
from sentiment_derivation.data.dataloader import create_dataloader
from sentiment_derivation.evaluation import evaluate_predictions
from sentiment_derivation.integration.pipeline import export_predictions, load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate actual held-out predictions from a trained checkpoint."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Checkpoint to run before evaluation.")
    source.add_argument(
        "--predictions",
        type=Path,
        help="Existing model-exported prediction CSV (must cover the selected split).",
    )
    parser.add_argument("--manifest", type=Path, help="Override the processed manifest path.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "evaluation",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training.yaml",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neutral-margin", type=float, default=0.25)
    parser.add_argument("--model-name", help="Name recorded in report provenance.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = manifest_path(
            load_yaml(args.data_config),
            args.manifest,
        )
        if not manifest.is_file():
            raise FileNotFoundError(f"Processed manifest not found: {manifest}")

        checkpoint_path = args.checkpoint
        if checkpoint_path is not None:
            data_config = load_yaml(args.data_config)
            training_file = load_yaml(args.training_config)
            training = training_file.get("training", {})
            device = select_device(args.device or training.get("device", "auto"))
            model, checkpoint = load_model_checkpoint(checkpoint_path, device)
            options = loader_options(data_config, checkpoint["model_config"], training)
            loader = create_dataloader(
                manifest,
                args.split,
                batch_size=(
                    args.batch_size
                    if args.batch_size is not None
                    else int(training.get("batch_size", 8))
                ),
                shuffle=False,
                **options,
            )
            predictions = export_predictions(
                model,
                loader,
                args.output_dir / f"{args.split}_predictions.csv",
                device=device,
            )
            model_name = args.model_name or checkpoint_path.stem
        else:
            predictions = args.predictions
            model_name = args.model_name or "model-exported-predictions"

        metrics = evaluate_predictions(
            manifest,
            predictions,
            args.output_dir,
            split=args.split,
            threshold=args.threshold,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            neutral_margin=args.neutral_margin,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    aggregate = metrics["emotion_aggregate"]
    print(f"Evaluated {metrics['sample_count']} {metrics['split']} samples.")
    print(f"Emotion macro F1: {aggregate['macro_f1']:.4f}")
    print(f"Emotion micro F1: {aggregate['micro_f1']:.4f}")
    if "valence_regression" in metrics:
        print(f"Valence MAE: {metrics['valence_regression']['mae']:.4f}")
    if "sentiment_classification" in metrics:
        print(
            "Sentiment macro F1: "
            f"{metrics['sentiment_classification']['macro_f1']:.4f}"
        )
    print(f"Reports and figures: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
