"""Text encoder tests using mocked tokenizer/backbone; no pretrained weights are fetched."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn

from sentiment_derivation.encoders import text_encoder as text_module


class DummyBackbone(nn.Module):
    """Mock 768-wide token model; pad embeddings remain nonzero to test pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 768)
        self.config = SimpleNamespace(dim=768, vocab_size=100)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask  # Deliberately isolate encoder-side pooling behavior.
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class DummyTokenizer:
    last_text = None
    last_kwargs = None

    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()

    def __call__(self, text, **kwargs):
        type(self).last_text = text
        type(self).last_kwargs = kwargs
        length = kwargs["max_length"]
        ids = torch.zeros(1, length, dtype=torch.long)
        mask = torch.zeros(1, length, dtype=torch.long)
        if text:
            ids[0, :2] = torch.tensor([1, 2])
            mask[0, :2] = 1
        return {"input_ids": ids, "attention_mask": mask}


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        text_module.DistilBertModel,
        "from_pretrained",
        classmethod(lambda cls, _name: DummyBackbone()),
    )
    model = text_module.TextEncoder(target_steps=16, feature_dim=128).eval()
    return model


def test_text_preprocessor_mocked_tokenizer_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocked tokenizer validates adapter shapes, not Hugging Face tokenizer behavior."""
    monkeypatch.setattr(text_module, "DistilBertTokenizer", DummyTokenizer)
    processed = text_module.TextPreprocessor(max_length=12).process_text("sample text")
    assert processed["input_ids"].shape == (12,)
    assert processed["attention_mask"].shape == (12,)
    assert processed["input_ids"].dtype == torch.long
    assert processed["attention_mask"].dtype == torch.long
    assert DummyTokenizer.last_text == "sample text"
    assert DummyTokenizer.last_kwargs["padding"] == "max_length"
    assert DummyTokenizer.last_kwargs["truncation"] is True
    assert DummyTokenizer.last_kwargs["max_length"] == 12


def test_text_preprocessor_rejects_empty_and_non_string_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_module, "DistilBertTokenizer", DummyTokenizer)
    preprocessor = text_module.TextPreprocessor(max_length=12)
    with pytest.raises(ValueError, match="at least one"):
        preprocessor.process_text("   ")
    with pytest.raises(TypeError, match="string"):
        preprocessor.process_text(None)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_text_encoder_batch_shapes_dtype_finite_cpu(encoder, batch_size: int) -> None:
    ids = torch.randint(0, 100, (batch_size, 12))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        features = encoder(ids, mask)
    assert features.shape == (batch_size, 16, 128)
    assert features.dtype == torch.float32
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()


def test_text_encoder_return_dict(encoder) -> None:
    result = encoder(torch.ones(1, 12, dtype=torch.long), torch.ones(1, 12, dtype=torch.long), return_dict=True)
    assert result["features"].shape == (1, 16, 128)


def test_text_pooling_ignores_padding_values(encoder) -> None:
    mask = torch.tensor([[1, 1, 0, 0]])
    ids_a = torch.tensor([[1, 2, 3, 4]])
    ids_b = torch.tensor([[1, 2, 5, 6]])
    with torch.no_grad():
        first = encoder(ids_a, mask)
        second = encoder(ids_b, mask)
    assert torch.allclose(first, second, atol=1e-6)


def test_text_encoder_rejects_empty_sequence(encoder) -> None:
    with pytest.raises(ValueError, match="empty"):
        encoder(torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long))


def test_text_encoder_rejects_mismatched_batch(encoder) -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        encoder(torch.ones(2, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.long))


def test_text_encoder_masks_fully_missing_rows(encoder) -> None:
    ids = torch.tensor([[1, 2, 0], [0, 0, 0]])
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    features = encoder(ids, mask)
    assert features.shape == (2, 16, 128)
    assert torch.count_nonzero(features[1]) == 0


def test_text_encoder_rejects_nonbinary_attention_mask(encoder) -> None:
    with pytest.raises(ValueError, match="binary"):
        encoder(torch.ones(1, 4, dtype=torch.long), torch.tensor([[1, 1, 2, 0]]))


def test_text_encoder_rejects_token_ids_outside_vocabulary(encoder) -> None:
    with pytest.raises(ValueError, match="vocabulary size"):
        encoder(torch.tensor([[100]]), torch.ones(1, 1, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_text_encoder_cuda_when_available(encoder) -> None:
    encoder = encoder.cuda()
    output = encoder(torch.ones(2, 8, dtype=torch.long, device="cuda"), torch.ones(2, 8, dtype=torch.long, device="cuda"))
    assert output.device.type == "cuda"
    assert output.shape == (2, 16, 128)
    assert torch.isfinite(output).all()
