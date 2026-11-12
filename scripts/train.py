"""Train the video/audio sentiment model from a processed manifest."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_yaml, loader_options, manifest_path, select_device
from sentiment_derivation.data.dataloader import create_dataloader, emotion_positive_weights
from sentiment_derivation.integration.model import MultimodalSentimentModel
from sentiment_derivation.training.losses import MultitaskLoss
from sentiment_derivation.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Override the processed manifest path.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model.yaml",
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
    parser.add_argument("--output-dir", type=Path, help="Override the checkpoint output directory.")
    parser.add_argument("--device", help="Override training.yaml device (auto, cpu, or cuda).")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize ResNet-18 without downloading ImageNet weights.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model_config = load_yaml(args.model_config)
        data_config = load_yaml(args.data_config)
        training_file = load_yaml(args.training_config)
        training = training_file.get("training", {})
        loss_config = training_file.get("loss", {})
        if args.no_pretrained:
            model_config.setdefault("video_encoder", {})["pretrained"] = False

        seed = int(training.get("seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        manifest = manifest_path(data_config, args.manifest)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Processed manifest not found: {manifest}. "
                "Run scripts/prepare_data.py first."
            )
        options = loader_options(data_config, model_config, training)
        batch_size = int(training.get("batch_size", 8))
        train_loader = create_dataloader(
            manifest,
            "train",
            batch_size=batch_size,
            shuffle=True,
            **options,
        )
        validation_loader = create_dataloader(
            manifest,
            "val",
            batch_size=batch_size,
            shuffle=False,
            **options,
        )

        pos_weight = None
        if training.get("use_emotion_pos_weight", True):
            pos_weight = emotion_positive_weights(manifest, split="train")
        loss_fn = MultitaskLoss(
            emotion_weight=float(loss_config.get("emotion_weight", 1.0)),
            valence_weight=float(loss_config.get("valence_weight", 1.0)),
            sentiment_weight=float(loss_config.get("sentiment_weight", 1.0)),
            emotion_pos_weight=pos_weight,
        )
        device = select_device(args.device or training.get("device", "auto"))
        output_dir = args.output_dir
        if output_dir is None:
            configured_output = Path(training_file.get("output_dir", "checkpoints"))
            output_dir = (
                configured_output
                if configured_output.is_absolute()
                else PROJECT_ROOT / configured_output
            )

        model = MultimodalSentimentModel.from_config(model_config)
        trainer = Trainer(
            model,
            loss_fn,
            device=device,
            learning_rate=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            model_config=model_config,
            output_dir=output_dir,
            gradient_clip_norm=float(training.get("gradient_clip_norm", 1.0)),
        )
        trainer.fit(
            train_loader,
            validation_loader,
            epochs=int(training.get("epochs", 30)),
            patience=int(training.get("patience", 5)),
            emotion_threshold=float(training.get("emotion_threshold", 0.5)),
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 2

    print(f"Best checkpoint: {(Path(output_dir) / 'best_model.pt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
