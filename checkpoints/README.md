# Model Checkpoints Documentation

This directory manages trained model weights, optimization states, and periodic experiment checkpoints for the **Multimodal Emotion-to-Sentiment Derivation and Mapping** system.

---

## 1. Version Control & Git Commit Policy

> [!CAUTION]
> **Do Not Commit Checkpoints to Git**:
> In accordance with [`AGENTS.md`](file:///c:/Niladri/empty/coding/projects/Multimodal-Sentiment-Derivation/AGENTS.md) Section 9 (*Model Training and Resource Management*), **binary weight files (`*.pt`, `*.pth`, `*.bin`, `*.ckpt`) must never be committed to the Git repository**.

This directory is monitored by `.gitignore` to prevent accidental inclusion of large binary weights. For external sharing or persistent archival, use Google Drive mounts (in Colab) or an approved external artifact registry (e.g., Hugging Face Hub or S3).

---

## 2. Checkpoint Storage Structure

When running training scripts or experimental workflows, checkpoints are saved locally under this directory following this organization:

```text
checkpoints/
├── README.md                           # This documentation file
│
├── encoders/
│   ├── video_resnet18_best.pt         # Fine-tuned ResNet-18 visual weights (if unfreezing)
│   └── audio_cnn_best.pt              # Trained Log-Mel 2D-CNN audio encoder
│
├── emotion/
│   ├── emotion_classifier_best.pt     # Best multi-label emotion head checkpoint
│   └── affective_state_best.pt        # Best continuous valence estimator checkpoint
│
├── temporal/
│   ├── temporal_bilstm_best.pt        # Trained BiLSTM sequence encoder
│   └── transition_analyzer_best.pt    # Trained transition analyzer & persistence weights
│
├── uncertainty/
│   └── temperature_calibrator.pt      # Calibrated temperature vector T_k
│
└── pipeline/
    ├── pipeline_epoch_latest.pt       # Periodic training resumption checkpoint
    └── pipeline_best.pt               # End-to-end full derivation pipeline (best Val Loss)
```

---

## 3. Standard Checkpoint Schema

All saved PyTorch checkpoints must include a standardized metadata dictionary to ensure complete experimental reproducibility:

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "metrics": {
        "val_loss": val_loss,
        "val_macro_f1": val_macro_f1,
        "val_macro_ece": val_macro_ece,
        "val_sentiment_acc": val_sentiment_acc,
    },
    "config": config,                  # Exact copy of configs/model.yaml used
    "timestamp": "2026-11-11T20:00:00Z",
    "git_commit": git_hash,
}
torch.save(checkpoint, "checkpoints/pipeline/pipeline_best.pt")
```

---

## 4. Checkpoint Loading Guidelines

To restore a checkpoint for inference or continued training:

```python
import torch
from sentiment_derivation.sentiment import SentimentDerivationPipeline

checkpoint_path = "checkpoints/pipeline/pipeline_best.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Restore model weights
model = SentimentDerivationPipeline.from_config(checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from Epoch {checkpoint['epoch']} with Val Loss {checkpoint['metrics']['val_loss']:.4f}")
```

---

## 5. Google Colab Persistence Protocol

When executing in Google Colab:
1. Mount Google Drive at the beginning of training:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Symlink or copy saved checkpoints from `checkpoints/` to `/content/drive/MyDrive/Multimodal-Sentiment-Derivation/checkpoints/` at the end of each validation epoch to guard against runtime disconnections.
