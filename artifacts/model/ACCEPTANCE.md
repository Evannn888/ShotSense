# ShotSense Production Acceptance Report (2026-10-02)

The local parameter-recommendation MVP is implemented. It supports unedited DNGs with valid camera WB and returns absolute legacy Camera Raw PV2003 values. Acceptable practical error remains uncalibrated.

## Data and training

- 5000 inputs: 4946 successful, 3 invalid labels, 51 unsupported missing-camera-WB inputs. Original data retained.
- Fixed photo-ID split: 3957 training, 495 validation, 494 test; seed 42. Burst/near-duplicate grouping unavailable.
- Frozen EfficientNet-B0, trainable MLP, best checkpoint selected by validation macro MAE; backbone weights/BN buffers unchanged.

| Model | Validation normalized MAE | Test normalized MAE |
|---|---:|---:|
| constant | 0.081530 | 0.082541 |
| physical_linear | 0.068191 | 0.068102 |
| semantic | 0.074476 | 0.076934 |
| dual | 0.066111 | 0.067113 |

## Final per-parameter test

| Parameter | Unit | Dual MAE | Constant MAE | Status |
|---|---|---:|---:|---|
| Exposure | EV | 0.281 | 0.409 | Beats constant on validation; recommended |
| Contrast | slider | 5.989 | 5.759 | Experimental |
| Saturation | slider | 2.958 | 2.656 | Experimental |
| Temperature | K | 592.006 | 773.889 | Experimental |
| Tint | slider | 7.454 | 8.174 | Experimental |
| HighlightRecovery | slider | 7.428 | 10.147 | Beats constant on validation; recommended |

Dual test macro error was 18.7% below constant. Highlight recovery was slightly worse than physical-linear test MAE 7.298; dual is not best for every field.

## White balance and strata

Camera/Catalog-tag/training-p1/p99-tail counts and MAE/RMSE are in `evaluation.json`. Catalog tags overlap and are incomplete, not full scene ground truth. Temperature/Tint lack matching online absolute baselines and remain experimental.

- Validation explicit As-Shot coverage: 341/495 (68.9%). Matched baseline/model Temperature MAE 703.2/555.0 K, Tint MAE 7.49/7.83.
- Test explicit As-Shot coverage: 340/494 (68.8%). Matched baseline/model Temperature MAE 707.7/600.4 K, Tint MAE 7.03/7.76.

## Deployment

- FP32 ONNX, dynamic batch, dual inputs, in-graph standardization; 18.30 MiB.
- Maximum PyTorch/ORT normalized difference 3.2e-07 (tolerance 1e-4).
- macOS-14.8.4-arm64-arm-64bit, CPUExecutionProvider, 4 threads, batch=1, 10 warmups/100 timed runs: p50 18.66 ms/p95 19.39 ms. No quantization.
- Full RAW→parameters includes decoding/features; local-job timing includes process startup and is separate from forward time.
- Independent inference passed with torch/torchvision/network disabled. Offline/online same-DNG features/JPEG matched exactly.

## Product scope

Local upload/sample/baseline development, recommended/experimental separation, and JSON download. Limits: 128 MB/40 megapixels, 60-second bounded worker, temporary cleanup, explicit failure feedback. Replacing input clears old results. Faithful Lightroom high-resolution rendering, modern Highlights mapping, and practical error calibration remain incomplete. No XMP or accumulated Delta.

## Initial application acceptance

22 checks passed. Real browser sample/DNG upload/recommendation/downloaded JSON passed; replacing input cleared old results and corrupt DNG produced an error without stale values. Saved `browser_acceptance.json` and `example_parameters.json`. The initial screenshot and exact source ZIP remain historical evidence, retained locally/earlier Git history with hashes in `artifacts/SNAPSHOT_PROVENANCE.md`.

## Initial approximate preview addendum

Added baseline/approximate-after images and 224×224 PNG download, simulating only validated exposure gain and a soft recovery proxy. No Lightroom equivalence/clipped-highlight reconstruction; experimental values excluded. Recompute from baseline without accumulation. Two targeted numerical/UI checks passed. Saved user browser denial blocked newer real-browser acceptance. Initial screenshot/source snapshot describe the first recommendations page.

## Linear RAW preview v2 addendum

Implemented independent aspect-preserving maximum-edge-1600 RAW preview per `PREVIEW_IMPROVEMENT_PLAN.md`. Protection/strength/applied values/clipping diagnostics and PNG/JSON align. 25 checks passed. Three real-photo offline reviews and numerical results are in `artifacts/preview_v2/acceptance.json`. Replaces 224-JPEG previews without changing recommendations. Custom approximate development; latest browser acceptance, Lightroom equivalence, and regional semantic editing incomplete.

## Preview v3 (2026-10-02)

26 checks passed. Continuous shoulder reduces premature midtone compression; recovery knee 0.8 reduces normal-highlight darkening. Page/JSON diagnose developed display-source ceilings, not sensor overexposure. Sixteen v2/v3 comparisons (12 regressions plus 4 unseen test photos) completed offline review, with zero newly full-value-channel pixels at both strengths. See `artifacts/preview_v3`. Model metrics unchanged; latest browser/regional semantics/faithful highlight reconstruction incomplete.

## English localization

Current interface, documentation, development milestones, and review observations have been translated to English. Numerical contracts/model artifacts remain unchanged. Original immutable Chinese-interface evidence is retained separately and described by an English provenance record rather than silently rewritten.

All 26 tests passed after localization, including English sample/strength/protection/ceiling-warning/download assertions. ONNX, checkpoint, preprocessing, and renderer identities were checked against the previous Git revision and remain identical.
