# A1 Annotation-to-Video Association Validation

## Scope and conclusion

The original EmoReact dataset is unavailable in this environment, and annotation rows contain no sample identifiers. The repository therefore cannot establish which annotation row belongs to a specific video. **A1 association correctness remains unverified.** The positional matching algorithm is unchanged. Provenance flows from generated manifests through manifest loading, Dataset items, and collated batches.

A missing provenance field is the legacy case and is interpreted as `method: unknown`, `verification_status: unverified`, and `evidence: null`. An explicitly present empty or partial provenance mapping is invalid. A caller declaration is recorded as `caller_declared_verified` with an evidence reference; this software cannot independently verify it.

## Source findings and contract

- `src/sentiment_derivation/data/preprocessing.py:248-323`, `load_annotations()`, parses annotation labels and valence without generating annotation sample IDs.
- `src/sentiment_derivation/data/preprocessing.py:325-356`, `get_split_sample_ids()`, sorts video filename stems. Sorting is deterministic but does not prove correspondence to annotation row order.
- `src/sentiment_derivation/data/preprocessing.py:358-424`, `match_annotations_with_media()`, retains positional `zip()` pairing and attaches the normalized policy. The default is unverified.
- `src/sentiment_derivation/data/preprocessing.py:68-245` validates caller policy declarations and separately validates recorded provenance. Caller input status `verified` is translated to `caller_declared_verified`; recorded raw status `verified` is rejected.
- `src/sentiment_derivation/data/preprocessing.py:727-895`, `build_processed_manifest()`, records policy at manifest and sample level. The standard CLI calls the builder without a policy (`scripts/prepare_data.py:224-238`), so it remains unverified and emits the builder warning.
- `save_manifest()` and `load_manifest()` use `_normalize_manifest_association_provenance()` to enforce the same structure and provenance contract (`preprocessing.py:132-188`, `:1064-1131`). The `samples` root must be a mapping, every split name must be a nonblank string, every split value must be a list, and every record must be a mapping. Every present manifest-level or per-sample provenance object is validated; raw `verified`, empty/partial mappings, unsupported fields, and invalid evidence are rejected. Missing provenance alone is the legacy case: the normalized manifest policy becomes unknown/unverified, and a sample with no explicit policy inherits that policy. Explicit per-sample provenance retains precedence over the manifest policy.
- `save_manifest()` fully normalizes and JSON-serializes the input before creating the output directory. It writes to a temporary sibling file and atomically replaces the destination, so invalid structures/provenance and serialization errors cannot leave a partial manifest (`preprocessing.py:1064-1103`).
- `src/sentiment_derivation/data/dataset.py`, `ManifestDataset`, uses the same recorded-provenance validator for in-memory manifests, checks records across splits, inherits the manifest policy for samples without a provenance key, and returns provenance/status/evidence on each item (`dataset.py:71-145`, `:302-385`).
- `src/sentiment_derivation/data/dataloader.py`, `collate_manifest_samples()`, validates and preserves per-row provenance, status, and evidence in input sample order (`dataloader.py:135-169`). It adds metadata only; tensor construction and modality mask order are unchanged.

Recorded provenance must contain exactly `method`, `verification_status`, and `evidence`. Empty mappings, partial mappings, unsupported fields/methods/statuses, blank or non-string evidence, and caller-declared verification without evidence are rejected. An absent provenance key is distinct from an empty object and remains backward-compatible as legacy unknown/unverified. Only `unverified` and `caller_declared_verified` are valid recorded statuses. Independent verification cannot be asserted by this software.

The repository and reviewed documentation do not provide authoritative mapping evidence. Synthetic tests cannot establish the actual EmoReact annotation ordering.

## Regression coverage

- `tests/test_data.py` covers save/load preservation of valid top-level and per-sample provenance, legacy manifests acquiring unknown/unverified provenance, rejection of empty and partial mappings, raw `verified` status at nested sample level, invalid evidence, and non-list split rejection by both load and save. The save tests also verify invalid data does not create the destination directory or output file.
- `tests/test_dataset.py` covers legacy default behavior, manifest-level inheritance when the sample field is absent, both precedence directions (manifest unverified/sample caller-declared verified and manifest caller-declared verified/sample unverified), invalid evidence, empty sample mappings, rejection of raw independent `verified`, and mixed batch ordering.
- The mixed-provenance collation test checks that video remains `[B,T,C,H,W]`, audio remains absent when all rows lack audio, and the modality mask remains ordered `(video, audio, text)` with unchanged values. Existing collation tests continue to check audio/text tensor shapes.

All test records are synthetic/placeholders. They validate provenance handling only, not real EmoReact association.

## Test results

Environment: isolated Windows Python 3.12.10 environment at `C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv`, CPU PyTorch 2.11.0. `PYTHONPATH` pointed at the current worktree `src/`. No packages, data, or weights were downloaded or installed.

Focused command:

```powershell
& 'C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv\Scripts\python.exe' -m pytest -p no:cacheprovider -q -ra tests/test_data.py tests/test_dataset.py
```

Final result after the boundary-hardening changes: **56 passed, 1 skipped, 0 failed**. The skipped test checks real EmoReact integrity; the dataset is absent. The new tests confirm malformed split containers and nested invalid provenance are rejected, while valid legacy omission and valid provenance round-tripping remain supported.

Selected Phase 0-6 regression command:

```powershell
& 'C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv\Scripts\python.exe' -m pytest -p no:cacheprovider -q -ra tests/test_config.py tests/test_logger.py tests/test_seed.py tests/test_data.py tests/test_preprocessing_edge_cases.py tests/test_dataset.py tests/test_transforms.py tests/test_video_encoder.py tests/test_audio_encoder.py tests/test_text_encoder.py tests/test_encoder_integration.py tests/test_alignment.py tests/test_fusion.py tests/test_phase4_integration.py tests/test_emotion.py tests/test_uncertainty.py tests/test_phase6_integration.py tests/test_missing_audio_batchnorm.py
```

Final result after the boundary-hardening changes: **223 passed, 7 skipped, 0 failed**. Six tests were skipped because CUDA is unavailable; one was the real EmoReact integrity test. The suite uses synthetic/local inputs and does not verify annotation ordering.

## Remaining limitations and evidence needed

- The actual EmoReact row-to-video association remains unverified. Establishing it requires authoritative mapping or dataset-order documentation for the exact dataset version and validation against its split files.
- Caller evidence is checked for presence and type only; this software does not authenticate its authority, scope, or correctness.
- Training does not automatically reject unverified or legacy samples. Callers must inspect each batch row's status if verified labels are required.
- `load_manifest()` is the provenance validation boundary for file-based manifests. Code that bypasses it and directly parses JSON also bypasses that boundary; `ManifestDataset` still validates in-memory mappings before use. `save_manifest()` requires a `samples` mapping and list-valued splits; metadata-only objects without the normal manifest sample structure are rejected.
- Real-data decoding, CUDA execution, and pretrained model loading are outside these synthetic provenance tests.
- The existing processed manifest was not regenerated or modified.
