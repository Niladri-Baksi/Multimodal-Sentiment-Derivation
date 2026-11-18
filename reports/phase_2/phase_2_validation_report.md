# Phase 2 — Dataset, Preprocessing, Bug Fixing and Validation

## Scope and evidence

This report covers the Phase 2 pipeline in the current `testing-validation_Dey` checkout. Source was inspected in `src/sentiment_derivation/data/{preprocessing.py,dataset.py,dataloader.py,transforms.py,__init__.py}`, `scripts/prepare_data.py`, `configs/data.yaml`, the requested Phase 0 interface reports, Phase 1 config/logger/seed utilities, the existing preprocessing tests, and the new Dataset/transform tests. The Phase 3 encoder signatures were inspected for batch compatibility. No real EmoReact dataset was available; no real dataset run, model training, checkpoint download, or dependency installation was performed.

The initial unprivileged pytest attempt could not access Windows temporary directories and produced fixture permission failures. The same suite was rerun with filesystem access required for its temporary fixtures and completed successfully. The successful run is the reported result below.

## Phase 2 handover status

**A1 is COMPLETE for implementation and synthetic validation and accepted for handover.** Its provenance safeguard is frozen. The actual EmoReact annotation-to-video mapping remains **UNVERIFIED** because the original dataset and authoritative ordering evidence are unavailable. The latest A1 focused tests completed with **56 passed, 1 skipped, 0 failed**; the selected Phase 0–6 regression suite completed with **223 passed, 7 skipped, 0 failed**. The real-dataset integrity test was skipped because the dataset is absent; six CUDA-specific tests were skipped because CUDA is unavailable. These results do not verify real annotation ordering.

## Architecture and current data flow

`scripts/prepare_data.py` loads the data YAML through the Phase 1 config utility, applies the configured seed, resolves configured dataset paths with supported fallback candidates, and invokes the preprocessing builder. Preprocessing reads annotation rows, enumerates video file stems, associates rows and sorted stems positionally, inspects media, optionally extracts frames/audio, builds split-indexed samples and statistics, and saves a JSON manifest. `ManifestDataset` reads one split and decodes/transforms video frames, optionally processes audio/transcript inputs, validates labels, and preserves identity and metadata. `create_dataloader` selects seeded shuffled training sampling or ordered validation/test sampling and uses the project collator.

The complete workflow has been exercised with synthetic fixtures from manifest production through Dataset retrieval. A separate real dataset preparation run remains unverified.

## Manifest and sample contracts

The manifest produced by `build_processed_manifest()` has these top-level fields:

| Field | Contract |
|---|---|
| `manifest_version` | String, currently `"1.0"` |
| `dataset_name`, `dataset_root` | Dataset identifier and root path strings |
| `emotion_names` | Ordered list of the eight emotion labels |
| `valence_range` | `[1.0, 7.0]` |
| `statistics` | Per-split counts, label/valence summaries and modality summaries |
| `corrupted_or_missing` | List of media/processing issues found during construction |
| `samples` | Mapping of split name to list of sample records |

Each sample preserves `sample_id`, `split`, `video_path`, optional `video_frame_paths`, `audio_path`, optional `text`, `emotion_labels` (name-to-binary-value mapping), `other_original_annotations.valence`, `modality_availability`, `visual_feature_paths`, and `metadata`. Annotation matching is positional after sorting video stems; the expected correspondence between that ordering and source annotation rows is assumed by the code and cannot be confirmed without the dataset.

`ManifestDataset` emits one mapping per sample: string `sample_id` and `split`; video float tensor `[T,C,H,W]`; optional audio tensor `[1,64,Tmel]` after the injected/selected audio preprocessor; optional raw transcript and token features; eight-element float emotion-label tensor; scalar float valence tensor; per-modality booleans; paths and metadata. Missing/invalid required video is either rejected with a sample-specific error or filtered according to the configured invalid-sample policy. Audio and text are optional.

The collator returns video `[B,T,C,H,W]`, emotion labels `[B,8]`, valence `[B]`, boolean `modality_mask [B,3]` in `(video,audio,text)` order, ID/split/path/metadata lists, optional audio `[B,1,64,Tmel]` padded across time, and optional `input_ids`/`attention_mask [B,L]` padded across text length. Missing audio rows are represented by zeros only when an audio tensor exists for another row; the mask marks those rows absent. Missing text rows are padded only when token features are provided; otherwise raw transcripts remain in the text list and token tensors are `None`.

## Dataset and DataLoader implementation

- `ManifestDataset` accepts a loaded mapping or manifest path and a selected split. It validates the manifest/split structure, duplicate IDs and cross-split ID leakage, annotation presence and values, video availability, and finite valence in `[1,7]`.
- Video is decoded from stored frame paths when usable, otherwise sampled from the video path, transformed to a fixed number of frames, and checked to return `[T,3,H,W]`.
- Audio and text processors are optional/injectable. Their outputs are retained for encoder input; file paths are not substituted for feature tensors.
- `collate_manifest_samples` rejects empty batches and incompatible required tensors, pads variable audio/text lengths, retains IDs and labels, and returns the explicit availability mask.
- `create_dataloader` supports batch size, worker count, drop-last, training shuffle, and a supplied seed. Validation/test preserve manifest order and reject a shuffle request. Worker Python/NumPy seeds derive from the PyTorch worker seed.
- `data/__init__.py` lazily exposes the Dataset/DataLoader API so importing preprocessing helpers does not eagerly import those components.

## Bugs found and fixes

Confirmed regression cases are listed in `bug_fix_log.md`.

1. **Invalid valence accepted:** the parser now rejects non-finite values and values outside `[1,7]`, with line context. Five parameterized edge cases (NaN, positive/negative infinity, 0, and 8) pass.
2. **Media path reconstruction lost filename case/extension:** manifest construction now keeps the actual matched media path and suffix. A regression test confirms uppercase `.MP4` survives.
3. **Unreadable video could be reported available:** manifest construction reflects the media inspection validity in `modality_availability.video`; corrupt media is recorded invalid. The corrupt-file regression test confirms the invalid result (using an isolated mocked OpenCV capture because OpenCV is unavailable in this environment).
4. **Duplicate identifiers and split leakage could make labels ambiguous:** matching rejects repeated IDs and Dataset validation rejects duplicates/leakage across manifest splits. Focused tests cover duplicate and leakage inputs.
5. **Extracted audio validity/cache handling was weak:** extraction validates nonempty WAV output and expected sample-rate/channel metadata, removes unusable partial output, and retries invalid cached files. The P2-05 handover section below records synthetic regression evidence and fixes for the defects reproduced by those tests; real FFmpeg/media extraction remains unexecuted.
6. **Optional media/dependency handling was inconsistent:** OpenCV is optional at import time; absent OpenCV yields an explicit unavailable/invalid inspection result and extraction raises a clear import error. Audio/text processing is optional in Dataset and availability is represented in the batch mask. Runtime tests with actual OpenCV/torchaudio media decoding were not possible.
7. **Frame transform input assumptions were unchecked:** transform code validates empty/invalid RGB frames, ensures contiguous arrays/float32, handles non-contiguous tensors using reshape, and rejects invalid channel or empty sequence/batch input. The transform test module was skipped because torchvision is unavailable.

Other implementation details and source locations are recorded in the repository modules; no production files outside the Phase 2 data modules and `scripts/prepare_data.py` were changed for this task. Phase 1 utility files were not modified in this task.

## Configuration, paths, labels and split risks

`configs/data.yaml` supplies dataset split annotation/media locations, expected split sizes, ordered emotion labels, valence bounds, modality options, extraction settings and manifest/report output locations. `prepare_data.py` now uses the central config loader, configures logging inside `main`, consumes the configured seed, derives paths from the configuration, and checks configured expected split counts. Its configured split-size values are validation expectations, not a substitute for checking the source dataset.

The annotation-to-media relationship still uses sorted video stems zipped with annotation rows, after a count check. The positional algorithm is unchanged. Its provenance is explicitly unknown/unverified by default, and the accepted A1 implementation validates and propagates provenance without claiming that the ordering is correct. A same-count ordering mismatch remains undetectable without authoritative sample IDs or documentation. Split leakage detection checks repeated sample IDs across manifest splits; subject-level leakage cannot be assessed because subject identifiers are not present in the verified sample contract.

Dataset label validation currently requires all eight configured emotion keys with binary values and `other_original_annotations.valence` finite in `[1,7]`. It preserves those original labels and valence; it does not derive sentiment labels.

## Phase 3 batch compatibility

| Connection | Static contract finding | Runtime evidence |
|---|---|---|
| Dataset/collator → `VideoEncoder` | **Static interface compatible:** batch video is `[B,T,3,H,W]`, float32, matching the encoder's batched frame contract. | Synthetic test confirms the emitted shape; encoder forward was not run. |
| Audio batch → `AudioEncoder` | **Static interface compatible when audio preprocessing is configured:** collator produces `[B,1,64,Tmel]`; missing rows are explicitly zero-filled and marked absent when other rows have audio. | Synthetic collation tested, but actual `AudioEncoder` and audio preprocessing were not run. |
| Text batch → `TextEncoder` | **Conditionally compatible:** tokenizer features produce padded `input_ids` and `attention_mask [B,L]`. Without an injected tokenizer/processor the Dataset retains raw text and emits `None` token tensors, which cannot be passed directly to `TextEncoder`. | Synthetic optional-text collation tested; text encoder forward was not run. |
| Batch modality mask → `ModalityAligner` | The current Phase 4 aligner accepts `audio_available`; Phase 2's per-row mask can be forwarded while preserving the fixed `(video,audio,text)` ordering. This closes the earlier static audio-availability mismatch. Full production model wiring remains unverified. | Phase 4 synthetic integration/regression tests; no real-data end-to-end execution. |

These are batch-level/static findings, not production integration verification. `VideoEncoder`, `AudioEncoder`, `TextEncoder`, and `ModalityAligner` were not modified.

## Tests and validation results

Successful command:

```text
python -m pytest tests/test_config.py tests/test_logger.py tests/test_seed.py tests/test_data.py tests/test_preprocessing_edge_cases.py tests/test_dataset.py tests/test_transforms.py -v
```

Result: **54 passed, 3 skipped, 0 failed, 0 errors, 1 warning**; pytest collected 56 test items, with one additional module-level skip during collection. The three skips were the torchvision-dependent transform module, unavailable-CUDA seed case, and real EmoReact integrity case. The warning is an installed AnyIO pytest plugin deprecation warning. No package was installed.

The narrower Phase 2 command also completed: **24 passed, 2 skipped, 0 failed, 0 errors, 1 warning**. It skipped real dataset integrity and the torchvision-dependent module. The initial sandbox-restricted attempt was invalidated by `PermissionError` while creating/cleaning Windows temporary directories; it is not counted as a code test failure. The rerun above had the needed fixture access and passed.

Synthetic producer-to-Dataset integration verified expected sample count, labels, IDs, video tensor shape/dtype, and split handling. Synthetic collator tests covered optional audio/text and variable audio lengths. No accuracy or real-data result is claimed.

## P2-05 audio extraction and WAV cache validation

The extraction path was inspected and tested with a mocked FFmpeg subprocess and temporary WAV fixtures. The tests reproduced three defects: an empty WAV raised `EOFError` that `_wav_is_usable()` did not classify as invalid; a subprocess exception could leave partial output; and an invalid cached WAV was not removed before the builder retried extraction. The implementation now handles `EOFError`, cleans partial output after subprocess exceptions, and removes invalid cached WAV files before invoking extraction. A valid cache is reused without running FFmpeg.

Focused command, run in the isolated Python 3.12 environment:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path; & 'C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv\Scripts\python.exe' -m pytest -p no:cacheprovider -q -ra tests/test_audio_extraction.py
```

Final result: **8 passed, 0 skipped, 0 failed**. The first focused run had **3 failures** that reproduced the defects above; after the targeted fixes, the focused rerun passed.

The selected Phase 0–6 regression suite was rerun with `tests/test_audio_extraction.py` added to the established test selection:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path; & 'C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv\Scripts\python.exe' -m pytest -p no:cacheprovider -q -ra tests/test_config.py tests/test_logger.py tests/test_seed.py tests/test_data.py tests/test_audio_extraction.py tests/test_preprocessing_edge_cases.py tests/test_dataset.py tests/test_transforms.py tests/test_video_encoder.py tests/test_audio_encoder.py tests/test_text_encoder.py tests/test_encoder_integration.py tests/test_alignment.py tests/test_fusion.py tests/test_phase4_integration.py tests/test_emotion.py tests/test_uncertainty.py tests/test_phase6_integration.py tests/test_missing_audio_batchnorm.py
```

Result: **231 passed, 7 skipped, 0 failed**. Six skips were CUDA-dependent tests; one was the real EmoReact dataset integrity check because the dataset is absent.

These tests use synthetic WAV data and mocked subprocess results. They do not establish that FFmpeg is installed, validate its actual command-line behavior, or verify extraction/decoding on real EmoReact media. No dataset or audio files were downloaded or modified.

## Remaining limitations and readiness

- No EmoReact files were available, so annotation ordering, actual split counts, real media health, and dataset-level subject leakage remain unverified. A1 implementation and synthetic validation are complete; real mapping verification remains a data/evidence prerequisite.
- In the original Phase 2 validation environment, OpenCV, torchvision, and torchaudio were unavailable. Actual video decode/transform and FFmpeg audio extraction remain unverified; the new P2-05 tests mock subprocess execution and use generated WAV fixtures. FFmpeg availability was not established by those tests.
- Text batches are encoder-ready only when a text processor is supplied. Audio batches are encoder-ready only when a working audio preprocessor is supplied.
- The aligner now supports an audio-availability mask, but the complete production model/pipeline does not yet establish end-to-end forwarding of all Phase 2 masks on real data.
- The DataLoader supports fixed-size transformed video clips; arbitrary variable frame counts are normalized by Dataset sampling/padding rather than retained as ragged video tensors.
- Empty manifest splits are representable at Dataset construction, but creating a DataLoader for a zero-length dataset is rejected. Empty minibatches are rejected by the collator.

**Phase 2 readiness: ready for synthetic pipeline testing and for real-data validation once the dataset/media dependencies are available; not yet verified for production dataset preparation.** The manifest-to-Dataset-to-DataLoader path and A1 provenance safeguards are implemented and synthetic-tested. Actual annotation-to-video identity, real media extraction, and complete production encoder/pipeline integration remain unverified. The aligner now accepts the audio availability mask, but full production mask forwarding has not been demonstrated.
