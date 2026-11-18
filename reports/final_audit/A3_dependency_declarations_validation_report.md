# A3 Dependency Declaration Validation - Phases 0-6

## Scope and outcome

Completed isolated dependency installation and validation against a temporary snapshot of the current source tree. The snapshot excluded the repository's pre-existing untracked `src/sentiment_derivation.egg-info`; the source snapshot matched all 32 current Python source files by SHA-256 before validation. No source algorithms, tests, dependency declarations, or the existing egg-info were changed during this validation.

**Overall status: partially accepted.** The clean dependency installation, editable packaging, imports, `pip check`, and synthetic Phase 0-6 tests passed. The FFmpeg availability check was executed but failed because `ffmpeg` is not installed or not on `PATH`; real dataset and CUDA-specific tests were skipped. Pretrained model loading/downloads were not attempted.

## Validation environment

- OS: Windows (`platform win32`), 64-bit Windows wheels (`win_amd64`).
- Python: 3.12.10, including the isolated venv interpreter.
- Pip frontend: 26.1.1, invoked as `python -m pip --python <venv-python>` because `venv` was created with `--without-pip`. This targeted all package operations at the venv; nothing was installed globally.
- Venv and project snapshot: `C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv` and `...\project`.
- The isolated snapshot contained `pyproject.toml`, `requirements.txt`, `src/`, and `tests/`. The copied stale egg-info directory was removed only from the temporary snapshot before installation. The live repository directory was not modified.
- CPU PyTorch was selected because this validation targeted CPU tests and did not establish CUDA hardware/runtime requirements. PyTorch's published Windows CPU matrix lists the matching `torch 2.11.0`, `torchvision 0.26.0`, and `torchaudio 2.11.0` set. See [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) and [Get Started Locally](https://pytorch.org/get-started/locally/).

## Commands executed and results

### Isolated environment and packaging

```powershell
python -m venv --without-pip C:\Users\subha\AppData\Local\Temp\msd-a3-final-90402df63ae342b09da50b4141c422b3\venv
```

**Passed.** The venv interpreter reported `Python 3.12.10`. The venv's bundled pip was not used; the installed pip frontend targeted the venv explicitly.

The project snapshot was copied from the current working tree. Its Python source files were compared with the working tree:

```text
SOURCE_SNAPSHOT_MATCHED_PY_FILES=32
```

The Python test files were also hash-compared before recording the run:

```text
TEST_SNAPSHOT_MATCHED_PY_FILES=22
```

The pre-install snapshot contained no `src/sentiment_derivation.egg-info`. After the editable build, setuptools generated fresh `src/multimodal_sentiment_derivation.egg-info` metadata in that temporary snapshot.

TOML and metadata assertions were run with Python 3.12 `tomllib`. They verified the setuptools backend, project name/version, Python minimum, all five optional groups, and package discovery `where = ["src"]`:

```text
TOML_METADATA_OK ['audio', 'dev', 'research', 'text', 'video']
```

**Passed.**

The first unpinned CPU index resolution command was:

```powershell
python -m pip --python <venv-python> install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

It completed, but independently resolved Torch `2.15.0+cpu`, torchvision `0.29.2+cpu`, and torchaudio `2.11.1+cpu`. To validate a documented matching trio rather than rely on separately resolved versions, those venv packages were replaced with:

```powershell
python -m pip --python <venv-python> install --no-deps --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

**Passed.** Installed versions reported: `torch-2.11.0+cpu`, `torchvision-0.26.0+cpu`, `torchaudio-2.11.0+cpu`. `--no-deps` was used for this trio replacement because the first install had already placed Torch's shared dependencies in the venv.

All declared modality and development extras were then installed from the isolated current-source snapshot:

```powershell
python -m pip --python <venv-python> install -e ".[video,audio,text,dev]"
```

**Passed.** Pip built an editable wheel for `multimodal-sentiment-derivation 0.1.0` and installed the declared NumPy, PyYAML, video, audio, text, and pytest dependencies. No research extra was required for this test selection.

The editable installation points at the snapshot, not generated or stale source metadata:

```text
Name: multimodal-sentiment-derivation
Version: 0.1.0
Editable project location: ...\msd-a3-final-90402df63ae342b09da50b4141c422b3\project
SOURCE_PACKAGE ...\project\src\sentiment_derivation\__init__.py
DIRECT_URL {"dir_info": {"editable": true}, "url": "file:///.../project"}
```

This independently validates editable packaging from the current Python source snapshot while preserving the live untracked `src/sentiment_derivation.egg-info` (`git ls-files` returned no tracked files for that path).

### Core and optional imports

Commands imported the installed package, core data/utilities and dependency modules, then each modality's package dependencies and implementation modules. Results:

| Probe | Result |
|---|---|
| `sentiment_derivation`, `.data`, `.utils.config`, `.utils.logger`, `.utils.seed`, NumPy, YAML, Torch | **Passed.** Package version `0.1.0`; NumPy `2.5.4`; PyYAML `6.0.3`; Torch `2.11.0+cpu`. The package resolved to the temporary snapshot source path. |
| OpenCV, torchvision, `data.transforms`, `encoders.video_encoder` | **Passed.** OpenCV `5.0.0`; torchvision `0.26.0+cpu`. |
| torchaudio, `encoders.audio_encoder` | **Passed.** torchaudio `2.11.0+cpu`. |
| Transformers, `encoders.text_encoder` | **Passed.** Transformers `5.23.0`. |

The text import emitted a PyTorch informational warning: `Redirects are currently not supported in Windows or MacOs.` The import completed successfully.

### Dependency consistency

```powershell
python -m pip --python <venv-python> check
```

**Passed:** `No broken requirements found.`

### Phase 0-6 regression tests

The following selected suite was run from the isolated snapshot. It includes the configuration, logging, seed, Phase 2 data/preprocessing/dataset/transforms, Phase 3 encoders and integration, Phase 4 alignment/fusion/integration, Phase 5 emotion, Phase 6 uncertainty/integration, and missing-audio BatchNorm regression tests:

```powershell
<venv-python> -m pytest -p no:cacheprovider -q -ra `
  tests/test_config.py tests/test_logger.py tests/test_seed.py `
  tests/test_data.py tests/test_preprocessing_edge_cases.py tests/test_dataset.py `
  tests/test_transforms.py tests/test_video_encoder.py tests/test_audio_encoder.py `
  tests/test_text_encoder.py tests/test_encoder_integration.py tests/test_alignment.py `
  tests/test_fusion.py tests/test_phase4_integration.py tests/test_emotion.py `
  tests/test_uncertainty.py tests/test_phase6_integration.py `
  tests/test_missing_audio_batchnorm.py
```

Final execution result:

```text
193 passed, 7 skipped in 6.40s
```

Failed: **0**. Skipped:

- CUDA unavailable: seed reproducibility, video encoder, audio encoder, text encoder, fusion, and emotion GPU tests (6 skips).
- EmoReact dataset absent at its default location: real dataset integrity test (1 skip).

The first sandboxed execution of the same selection collected 200 tests but failed fixture setup because the sandbox denied writes to pytest's Windows temporary directories (7 failed, 34 errors, 161 passed, 7 skipped). The identical selection then passed when run with permission for pytest's temporary fixtures. These initial errors were environmental; the successful run above is the final test result. No model weights, datasets, or pretrained checkpoints were downloaded.

### FFmpeg external prerequisite

```powershell
ffmpeg -version
```

**Not available in this environment.** PowerShell returned exit code 1: `ffmpeg : The term 'ffmpeg' is not recognized as the name of a cmdlet, function, script file, or operable program.` FFmpeg is an external executable prerequisite for media audio extraction; it is not a Python dependency. Audio module imports and synthetic tests passing do not verify real FFmpeg extraction.

## Validation summary

| Check | Result |
|---|---|
| Python 3.12 isolated venv | Passed |
| TOML/build metadata/package discovery assertions | Passed |
| Dependency-resolved install of `.[video,audio,text,dev]` | Passed |
| Matched CPU Torch family | Passed: Torch 2.11.0, torchvision 0.26.0, torchaudio 2.11.0 |
| Editable install independent of stale working-tree egg-info | Passed from SHA-256-verified temporary source snapshot |
| Core imports | Passed |
| Video/audio/text imports | Passed |
| `pip check` | Passed; no broken requirements |
| Phase 0-6 synthetic regression selection | 193 passed, 7 skipped, 0 failed |
| FFmpeg availability | Failed; executable not found on `PATH` |
| CUDA runtime tests | Skipped; CUDA unavailable |
| Real-data test | Skipped; EmoReact dataset unavailable |
| Pretrained model/checkpoint download or load | Not run by design |

## Remaining limitations and acceptance

- Install FFmpeg and make it available on `PATH` before validating audio extraction on actual media.
- CUDA-specific execution, real dataset integrity, real video/audio decoding, and pretrained-model loading remain unverified.
- The clean environment demonstrates that the declared dependency groups resolve and that all selected synthetic Phase 0-6 tests pass. It does not establish production-data readiness.
- **A3 is not marked fully accepted** because the requested FFmpeg availability check did not pass, and hardware/data-dependent coverage was skipped.
