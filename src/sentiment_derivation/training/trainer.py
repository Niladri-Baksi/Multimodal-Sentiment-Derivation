"""Training loop, validation, and atomic checkpoint persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from sentiment_derivation.training.losses import MultitaskLoss
from sentiment_derivation.training.metrics import MetricAccumulator


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MultitaskLoss,
        *,
        device: torch.device,
        learning_rate: float,
        weight_decay: float = 0.0,
        model_config: Mapping[str, Any],
        output_dir: str | Path,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("The model has no trainable parameters.")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        self.device = device
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.model_config = dict(model_config)
        text_encoder = getattr(self.model, "text_encoder", None)
        if text_encoder is not None:
            text_config = dict(self.model_config.get("text_encoder", {}))
            text_config["backbone_config"] = text_encoder.backbone.config.to_dict()
            self.model_config["text_encoder"] = text_config
        self.output_dir = Path(output_dir)
        self.gradient_clip_norm = gradient_clip_norm

    @staticmethod
    def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def run_epoch(
        self,
        loader: DataLoader[dict[str, Any]],
        *,
        training: bool,
        emotion_threshold: float = 0.5,
    ) -> dict[str, float]:
        self.model.train(training)
        accumulator = MetricAccumulator()
        total_loss = 0.0
        samples_seen = 0

        with torch.enable_grad() if training else torch.no_grad():
            for source_batch in loader:
                batch = self._to_device(source_batch, self.device)
                batch_size = batch["frames"].shape[0]
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                model_inputs: dict[str, Any] = {}
                if "input_ids" in batch or "attention_mask" in batch:
                    if "input_ids" not in batch or "attention_mask" not in batch:
                        raise ValueError("Text batches must provide input_ids and attention_mask together.")
                    model_inputs = {
                        "text_input_ids": batch["input_ids"],
                        "text_attention_mask": batch["attention_mask"],
                    }
                    if "text_available" in batch:
                        model_inputs["text_available"] = batch["text_available"]
                outputs = self.model(batch["frames"], batch["audio"], **model_inputs)
                losses = self.loss_fn(outputs, batch)
                if training:
                    losses["loss"].backward()
                    if self.gradient_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_norm,
                        )
                    self.optimizer.step()
                total_loss += float(losses["loss"].detach()) * batch_size
                samples_seen += batch_size
                accumulator.update(outputs, batch)

        if samples_seen == 0:
            raise ValueError("DataLoader produced no samples.")
        metrics = accumulator.compute(threshold=emotion_threshold)
        metrics["loss"] = total_loss / samples_seen
        return metrics

    @staticmethod
    def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(dict(payload), temporary)
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    def fit(
        self,
        train_loader: DataLoader[dict[str, Any]],
        validation_loader: DataLoader[dict[str, Any]],
        *,
        epochs: int,
        patience: int,
        emotion_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        if epochs <= 0 or patience < 0:
            raise ValueError("epochs must be positive and patience cannot be negative.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self.run_epoch(
                train_loader,
                training=True,
                emotion_threshold=emotion_threshold,
            )
            validation_metrics = self.run_epoch(
                validation_loader,
                training=False,
                emotion_threshold=emotion_threshold,
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={validation_metrics['loss']:.4f} "
                f"val_emotion_macro_f1={validation_metrics['emotion_macro_f1']:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_config": self.model_config,
                "model_state_dict": self.model.state_dict(),
                "validation_metrics": validation_metrics,
            }
            self._atomic_torch_save(checkpoint, self.output_dir / "last_model.pt")
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                epochs_without_improvement = 0
                self._atomic_torch_save(checkpoint, self.output_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1

            history_path = self.output_dir / "training_history.json"
            temporary_history = history_path.with_name(history_path.name + ".tmp")
            try:
                temporary_history.write_text(
                    json.dumps(history, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_history, history_path)
            except Exception:
                if temporary_history.exists():
                    temporary_history.unlink()
                raise

            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs without validation improvement.")
                break

        return history
