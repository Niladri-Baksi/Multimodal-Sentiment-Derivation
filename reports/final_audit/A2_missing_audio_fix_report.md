# Final Audit A2 — Missing Audio and BatchNorm

## Summary

The mixed-audio batch contract now retains a per-row boolean availability mask from collation through `AudioEncoder` and into the Phase 4 aligner. `AudioEncoder` runs its CNN only on available rows, so missing rows cannot affect BatchNorm training statistics. For all-audio-missing batches, the encoder returns zero feature placeholders without running the CNN. The availability mask remains authoritative; zero values are not used to infer missingness.

## Original defect and root cause

The Phase 2 collator represents missing audio rows in a mixed batch with zero-filled spectrograms and returns `modality_mask` ordered as `(video, audio, text)`. The Phase 3 `AudioEncoder.forward` previously passed the entire batch through convolution and BatchNorm without receiving that mask. Consequently, missing rows contributed to the batch statistics during training, and their resulting features could be mistaken for genuine silence.

The audit also reproduced a collator row-order defect: the mask comprehension used a loop index left over from an earlier loop, so every batch row reused the last sample's audio flag. That could corrupt the mask before it reached the encoder. The comprehension now enumerates the samples explicitly.

## Files changed for A2

- `src/sentiment_derivation/data/dataloader.py`
- `src/sentiment_derivation/encoders/audio_encoder.py`
- `tests/test_missing_audio_batchnorm.py` (new)
- `reports/final_audit/A2_missing_audio_fix_report.md` (this report)

No alignment implementation change was needed. The existing Phase 4 `ModalityAligner` already accepts `audio_available`, zeros masked aligned rows, and rejects a true availability mask when audio features are absent.

## Exact interface behavior

### Collator

`collate_manifest_samples` checks that every sample has a boolean audio availability value and that it agrees with whether its audio tensor is present. A mismatch raises `ValueError`. Present audio must be a finite floating tensor shaped `[1, M, T]` with positive dimensions; present items in a batch must agree on mel-band count. Mixed batches still use zero-filled rows for tensor stacking, but set `modality_mask[:, 1]` from the matching sample's actual availability. All-missing batches return `audio=None` and an all-false audio column. The mask order remains `(video, audio, text)`.

### Audio encoder

The backward-compatible `AudioEncoder.forward(x, return_dict=False, *, audio_available=None)` contract is:

- `x`: `[B, 1, 64, T_mel]` floating tensor, with `T_mel >= 4`, or `None` for an all-audio-missing batch.
- `audio_available`: optional boolean `[B]` tensor on the same device. If omitted with a tensor input, rows are treated as present as before. Callers handling collated batches must pass `batch["modality_mask"][:, 1]` explicitly. If `x is None`, the mask is required and must be all false.
- Output: `[B, target_steps, feature_dim]` (default `[B, 16, 128]`), or `{"features": tensor}` when `return_dict=True`.
- Masked rows are excluded before the first convolution/BatchNorm and restored as zero placeholders in their original batch positions. An all-false mask bypasses the CNN and does not update BatchNorm state.
- Wrong mask type, shape, device, or a true mask with `x=None` raises a clear `TypeError` or `ValueError`.

Only available audio rows enter BatchNorm during training. Gradients flow through available rows; unavailable input rows receive no encoder gradient. The caller must pass the same audio availability column to the aligner so output placeholders remain marked absent.

### Alignment compatibility

The unchanged Phase 4 `ModalityAligner.forward(video_features, audio_features, ..., audio_available=...)` accepts the encoder's batched feature output and the same boolean mask. The regression test verifies that all-missing audio features remain zero and the aligner's audio mask stays false. This is a synthetic contract check, not evidence of a production orchestrator calling both modules.

## Regression coverage

`tests/test_missing_audio_batchnorm.py` adds nine collected cases covering:

1. Mixed availability: BatchNorm running means, variances, and tracked-batch counters match a reference encoder run on available rows only.
2. Mixed availability: available-row features match that reference, missing-row outputs are zero, and backward propagation produces nonzero gradients only for available audio.
3. All-present mask behavior matches the legacy unmasked path and BatchNorm state.
4. All-missing collation returns `None`, the encoder bypasses BatchNorm and returns zeros, and alignment retains the absent mask.
5. Encoder rejects inconsistent mask type, shape, and true availability with absent audio.
6. Collator rejects both directions of disagreement between audio tensor and availability flag.
7. Collator preserves row ordering and the `(video, audio, text)` mask order for mixed batches.

## Test execution results

Commands were run with `PYTHONPATH=src` and pytest's cache provider disabled. The initial pre-fix focused attempt exposed the collator row-index defect (1 test failure); other fixture-based tests in that attempt also errored because the sandbox could not access the default Windows temporary directory. The defect was corrected, then the final reruns used a temporary directory inside the repository, which was removed after each run.

| Run | Result |
|---|---|
| Initial pre-fix attempt: new tests plus `test_audio_encoder.py`, `test_dataset.py`, `test_alignment.py` | 27 passed, 1 failed (mixed mask row-order check exposed the collator defect described above), 1 skipped, 10 errors from pytest fixture setup denied by Windows temp-directory permissions. This was a diagnostic run, not the final result. |
| New A2 tests: `tests/test_missing_audio_batchnorm.py -v` | 9 passed, 0 failed, 0 skipped; 1 unrelated AnyIO deprecation warning. |
| Focused A2/data/encoder/alignment suite: new tests plus `test_audio_encoder.py`, `test_dataset.py`, `test_alignment.py` | 38 passed, 0 failed, 1 skipped (CUDA unavailable); 1 AnyIO deprecation warning. |
| Selected Phase 2–6 regression suite: A2 tests plus config, logger, seed, data, preprocessing edge cases, dataset, transforms, video/audio/text encoders, encoder integration, alignment, fusion, Phase 4 integration, emotion, uncertainty, and Phase 6 integration tests | 179 passed, 0 failed, 11 skipped; 2 warnings (AnyIO pytest deprecation and PyTorch LBFGS tensor-to-scalar warning). |

No model training, external data, pretrained checkpoint download, or real media pipeline execution was performed.

## Remaining limitations

- Evidence is from synthetic/unit tests. Real dataset decoding, audio extraction, CUDA execution, and production inference/training remain unverified.
- No top-level production orchestrator is part of this change. Any caller that consumes a mixed collated audio batch must pass the same `modality_mask[:, 1]` into both `AudioEncoder` and `ModalityAligner`. Omitting the mask on a tensor input retains legacy semantics and treats every row as present; availability cannot safely be inferred from zero values because silence may be valid audio.
- If a batch contains no audio, callers may pass `None` plus its all-false mask to `AudioEncoder`, then pass the returned zero feature tensor and same mask to the aligner. Alternatively, the aligner accepts `None` directly with an all-false/default absent mask.
- Short sequences are processed with their current collator zero padding. This fix excludes whole missing samples, but does not make BatchNorm ignore temporal padding within an available spectrogram.

## Readiness

The identified A2 BatchNorm contamination path is addressed and synthetically regression-tested. The Phase 2→3→4 audio availability contract is statically compatible when callers forward the mask. Production integration remains unverified until a real pipeline forwards that mask consistently and is exercised on representative data.
