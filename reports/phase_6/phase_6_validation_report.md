# Phase 6 — Certainty and Uncertainty Validation

## Scope and evidence

This audit covers the Phase 6 uncertainty estimator, calibration utilities, uncertainty-aware gate, and synthetic Phase 5-to-6 connection. Source contracts were inspected in `src/sentiment_derivation/uncertainty/`, Phase 4 fusion and Phase 5 emotion/affect output code, `configs/model.yaml`, `configs/experiment.yaml`, `docs/architecture/module_interfaces.md`, and the relevant Phase 2–6 tests. No model weights or dataset were downloaded; no training or real-data evaluation was performed.

Labels in this report distinguish source inspection from execution. Tests used CPU synthetic inputs except where noted. A first combined run under the default sandbox could not write/clean Windows temporary files; it showed temp permission failures. The same suite was rerun with temporary-directory access and completed successfully. The permission-limited run is not counted as the final regression result.

## Uncertainty estimator

`UncertaintyEstimator` accepts floating probability tensors shaped `[K]`, `[B,K]`, or `[B,T,K]`, where `K=num_emotions` (default 8). Inputs must be non-empty, finite, and within `[0,1]`. Output `UncertaintyOutput` preserves the input shape for per-emotion confidence and uncertainty, along with the selected method and optional MC proxy fields. The implementation preserves device and floating dtype; entropy calculations use float32 working precision for half/bfloat16 inputs.

The supported estimators are:

- **Margin:** `c = clamp(2 * abs(p - 0.5), 0, 1)` and `u = 1 - c`. This is the same confidence complement used by the Phase 5 margin baseline.
- **Binary entropy:** `H(p) = -p log2(p) - (1-p) log2(1-p)`, with probabilities clamped for numerical stability; `u = H(p)` and `c = 1-u`. This is recomputed from probabilities and does not consume Phase 5's `emotion_uncertainty` field. Except at the midpoint and extremes, entropy and margin scores differ.
- **MC-dropout summary:** the estimator requires a `model_fn` returning one valid probability tensor per stochastic pass. It computes the normalized variance proxy `min(Var_S(p)/0.25, 1)` and the mean binary-entropy proxy over passes. The reported total is their equal-weight average, and confidence is its complement. These are heuristic proxies, not an exact or empirically validated epistemic/aleatoric decomposition. Stochastic behavior/dropout activation is the callback's responsibility.

The former silent fallback from MC-dropout mode to deterministic entropy when no callback was provided is corrected: it now raises a clear error. Invalid ranks, sizes, values, and callback outputs are rejected. Extreme probabilities and degenerate settings receive explicit numerical/configuration validation.

## Calibration implementation and metrics

`MultiLabelTemperatureCalibrator` uses a separate positive temperature per emotion: `T_k = exp(log_T_k)` and `p_cal,k = sigmoid(z_k/T_k)`. `fit()` optimizes `BCEWithLogitsLoss(z/T, y)` with LBFGS on the caller-provided calibration set. For independent binary labels, this is mean Bernoulli negative log-likelihood. The API calls its fitting data a validation set, but source inspection cannot establish that callers use a held-out calibration split or keep the test split separate. No production pipeline calls `fit()` or applies the calibrator.

The implemented metrics are:

- **Binary ECE:** within equal-width probability bins, `sum_b (n_b/N) * abs(mean(y_b)-mean(p_b))`. Empty bins contribute zero. Multi-label ECE computes this independently per emotion and reports the macro mean across emotions.
- **Brier score:** for each emotion, `mean_n((p_n-y_n)^2)`, then macro-averaged over labels for multi-label inputs.
- **NLL:** there is no public standalone NLL metric. BCE-with-logits is used internally as the temperature-fitting objective.

The metrics treat outputs as independent binary labels rather than mutually exclusive softmax classes. Input ranks/shapes, finite values, probability range, binary labels, non-empty data, bin count, and emotion-name consistency are validated. Synthetic known-value cases test the formulas. This verifies implementation arithmetic, not calibration quality. Standard PyTorch `state_dict()` round-trip covers learned temperature parameters; emotion names/configuration are not encoded in the parameter state itself, and no full checkpoint lifecycle exists here.

## Uncertainty-aware gate

The gate accepts feature vectors/sequences and corresponding emotion probability and uncertainty tensors. For the documented batched temporal contract:

| Input | Shape |
|---|---|
| Context/features `h` | `[B,T,D]`, normally `D=128` |
| Emotion probabilities `p` | `[B,T,K]`, normally `K=8` |
| Emotion uncertainty `u` | `[B,T,K]` |
| Optional valence confidence | `[B,T,1]`; omitted input defaults to `0.5` |

It validates ranks, dimensions, non-empty values, finite floating inputs, `[0,1]` ranges for probabilities/uncertainty/valence confidence, device consistency, then aligns floating inputs to the module parameter dtype. A learned MLP consumes concatenated `[h,p,u,valence_confidence]` and applies sigmoid to obtain elementwise gate weights `g` in `[0,1]`. With a prior, output features are `g*h + (1-g)*prior`; without a prior, output features are `g*h`. Output gate weights have the feature shape, while temporal saliency is `mean_D(g)` with shape `[B,T,1]`. Squeezed vector inputs have correspondingly squeezed outputs. The disabled path returns the validated input unchanged and unit gates.

The saliency value is a descriptive mean gate activation, not normalized temporal attention or a faithful causal explanation. The MLP has no monotonic constraint: larger uncertainty is not guaranteed to reduce `g`, and higher confidence is not guaranteed to increase it. Sigmoid gates can saturate near zero/one; with no prior, a near-zero gate suppresses feature magnitude. These are design/training properties, not grounds to claim a calibrated reliability mechanism. Tests check bounds, exact modulation, finiteness, and gradients.

## Phase 5-to-6 compatibility

Phase 5's classifier output supplies `[B,T,8]` probabilities, confidence, uncertainty and `[B,T,128]` context embeddings. Its affect branch supplies valence confidence `[B,T,1]`. These contracts are statically compatible with the estimator and gate, respectively. The estimator accepts probabilities and independently recomputes its chosen score; it does not use Phase 5 confidence/uncertainty fields unless the caller chooses to pass the Phase 5 uncertainty directly to the gate. The gate itself does not complement or otherwise transform uncertainty. The synthetic integration test explicitly selects the margin estimator and checks that its scores match Phase 5's margin confidence complement. It exercises fusion → emotion/affect → estimator → gate and verifies shape preservation and gradients.

This is a **synthetic component integration**, not production integration. There is no end-to-end production orchestrator in the Phase 6 source path that connects the estimator and gate, chooses between recomputed and Phase 5 uncertainty, or forwards the actual Phase 5 outputs. No real data, pretrained model, CUDA path, or learned calibration artifact was exercised.

## Configuration and downstream compatibility

`configs/model.yaml` configures `uncertainty_gate` but has no `uncertainty` section for estimator/calibrator options. Their `from_config()` methods accept a mapping and fall back to defaults when the section is absent, so passing the current full model configuration does not make those choices explicit. `configs/experiment.yaml` is empty. The gate preserves `[B,T,128]`, which is statically compatible with Phase 7 `TemporalEncoder(input_dim=128)`. No production call from the gate into Phase 7 was found or runtime-tested.

## Bug fixes and validation

The Phase 6 implementation now rejects malformed/empty/non-finite/out-of-range estimator inputs, validates MC callback results, requires a callback for MC mode, and validates metric/calibrator inputs and learned temperatures. The gate validates auxiliary shapes/ranges/devices and feature tensors, handles dtype alignment, and checks finite outputs. These corrections are detailed in `reports/phase_6/bug_fix_log.md`.

Test execution on Python 3.12.10, pytest 9.2.0, Windows, CPU:

| Suite | Result |
|---|---|
| Focused Phase 6: `tests/test_uncertainty.py tests/test_phase6_integration.py` | 24 passed, 0 failed, 0 skipped; 2 warnings; 2.13 seconds |
| Combined related Phase 2–6 regression plus sentiment downstream compatibility: 15 test files | 153 passed, 0 failed, 10 skipped; 2 warnings; 7.98 seconds |

Skips were dependency/hardware/data dependent (including unavailable OpenCV/torchvision/CUDA or absent real dataset); they are not counted as passes. Warnings were an installed AnyIO pytest deprecation and PyTorch LBFGS's tensor-to-scalar warning in the temperature optimization test. Tests used synthetic tensors/mocks; no production dataset or pretrained weights were loaded. The focused integration test confirmed gradient propagation through fusion, Phase 5 classifier/affect, and the Phase 6 gate; unit tests also confirmed gradients through estimator/gate/calibrator paths where differentiable.

## Remaining limitations and readiness

- Predictive uncertainty and calibration quality remain unverified without held-out labeled data and a defined evaluation protocol. Margin and entropy quantify probability ambiguity; they do not establish correctness.
- Valence confidence remains Phase 5's distance-from-midpoint proxy, not calibrated prediction reliability. The gate accepts it as an input scalar and does not correct that semantic limitation.
- The MC variance/entropy combination is heuristic, and callback-controlled dropout behavior has not been exercised with a real model.
- Calibration fitting, inference-time application, test-set isolation, and full checkpoint metadata are not wired into a production pipeline.
- No monotonic reliability behavior is guaranteed for the learned gate; saliency is not an explanation.
- `compute_reliability_curve` is not implemented as a public function despite its former mention in source documentation; ECE and Brier are the available metrics.
- Phase 5/6 and Phase 6/7 compatibility is shape-level and synthetic/static only. Production integration remains unverified.

**Readiness:** Phase 6 components are source-hardened and their synthetic CPU contracts, known metric examples, and gradient paths are validated. Phase 6 is ready for controlled integration work, but it is not yet evidence of calibrated predictive uncertainty or production-ready end-to-end behavior.

Phase 13 — XAI is planned future work and is outside the scope of this audit.
