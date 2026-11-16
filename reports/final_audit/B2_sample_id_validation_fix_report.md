# Final Audit B2 — Stable Sample-ID Validation

## Original defect and root cause

`ManifestDataset._validate_split_leakage()` previously skipped records that were not mappings or whose `sample_id` was falsey, and converted remaining IDs to strings. Thus missing, null, empty, zero, or false IDs escaped duplicate/leakage checks; truthy non-string IDs were silently coerced. `ManifestDataset.__getitem__()` then used the dataset index for a missing ID and stringified the result. This could hide malformed manifest data and return an unstable identity.

## Files changed

- `src/sentiment_derivation/data/dataset.py`
- `tests/test_dataset.py`
- `reports/final_audit/B2_sample_id_validation_fix_report.md` (this report)

No changes were made to preprocessing, `tests/test_data.py`, annotation-to-video matching, configuration, or Phases 5–6 architecture/formulas.

## Implementation

`ManifestDataset._validate_split_leakage()` now validates each split list and each record before the requested split is selected. Every record must be a mapping with a `sample_id` that is a Python string and contains at least one non-whitespace character. Invalid IDs raise `ValueError` with the split and record index. IDs are never stripped, normalized, converted, or replaced. Duplicate checks continue to use the exact string value and still reject duplicates within a split and across splits.

`ManifestDataset.__getitem__()` now reads `sample["sample_id"]` directly. It does not generate an index fallback or coerce the ID. Valid IDs containing surrounding whitespace are preserved exactly, as required. Constructor diagnostics and the invalid-video skip list also use the already-validated ID directly.

## Regression tests

Added to `tests/test_dataset.py`:

- Parameterized rejection for missing, `None`, empty, whitespace-only, integer, and boolean IDs, asserting split/index context.
- Rejection of a malformed ID in an unselected validation split when constructing the training Dataset.
- Exact preservation of valid string IDs through `__getitem__()` and collation, including IDs with surrounding whitespace.

Existing tests continue to cover duplicate IDs within a split and across splits (`test_split_leakage_duplicate_and_empty_batch_are_rejected`) and acceptance of a preprocessing-generated filename-stem ID (`test_manifest_producer_to_dataset_sample_contract`). No `tests/test_data.py` change was needed.

## Validation results

The requested test runs completed with `PYTHONPATH=src` and pytest cache disabled. The focused run used the requested test selection and required temporary fixture access; the combined run used a repository-local pytest temporary directory, removed after execution.

| Run | Result |
|---|---|
| `python -m pytest -p no:cacheprovider tests/test_dataset.py tests/test_data.py -v` | **26 passed, 1 skipped, 0 failed.** The skip is the real EmoReact integrity test because the raw dataset is not present. One installed AnyIO pytest-plugin deprecation warning. |
| Selected Phase 2–6 suite, including `tests/test_missing_audio_batchnorm.py` and the same config/logger/seed, preprocessing, dataset, transform, encoder, alignment, fusion, Phase 4 integration, emotion, uncertainty, and Phase 6 integration selection used for the prior A2 validation | **187 passed, 11 skipped, 0 failed.** Two warnings: installed AnyIO pytest-plugin deprecation and PyTorch LBFGS tensor-to-scalar warning. |

## Checked-in manifest verification

Re-read `data/processed/processed_manifest.json` after the implementation. It contains 1,102 records; all 1,102 IDs are nonempty strings, and no duplicate IDs were found within or across splits. The checked-in manifest therefore requires no migration for this ID-validation change. This check does not validate whether the IDs are correctly associated with annotation rows; the separate A1 annotation-order question remains unresolved.

## Remaining limitations

- The tests use synthetic fixtures. Other external or user-supplied manifests with invalid IDs will now fail at Dataset construction and must be corrected at their source.
- ID validity does not prove that an ID accurately identifies the media or annotation. It also does not resolve the positional annotation-to-video matching risk described in A1.
- The real EmoReact dataset is unavailable in this checkout, so real-data identity and leakage properties remain unverified.
