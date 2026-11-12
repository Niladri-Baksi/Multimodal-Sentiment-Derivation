"""Export held-out predictions from a trained model checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_yaml, loader_options, manifest_path, select_device
from sentiment_derivation.data.dataloader import create_dataloader
from sentiment_derivation.integration.pipeline import export_predictions, load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output",
        type=Path,
        help="Prediction CSV path (defaults to reports/predictions/<split>_predictions.csv).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_config = load_yaml(args.data_config)
        training_file = load_yaml(args.training_config)
        training = training_file.get("training", {})
        device = select_device(args.device or training.get("device", "auto"))
        manifest = manifest_path(data_config, args.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(f"Processed manifest not found: {manifest}")
        model, checkpoint = load_model_checkpoint(args.checkpoint, device)
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
        output_path = args.output or (
            PROJECT_ROOT
            / "reports"
            / "predictions"
            / f"{args.split}_predictions.csv"
        )
        output_path = export_predictions(
            model,
            loader,
            output_path,
            device=device,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 2

    print(f"Predictions written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
