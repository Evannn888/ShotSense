# Shadow amplification repair diagnostic

Date: 2026-10-02. Decision: preserve this failure regression and continue a separately designed noise/color-aware curve experiment; **do not deploy this recipe**.

Used the same 16 already inspected development pairs and three frozen gamma checkpoints. No retraining, new sample selection or official `eval15` access. Verified historical model/training-source/manifest/split/checkpoint/source-member/JPEG hashes and exact gamma replay; the preceding experiment and production source/model remain unchanged.

The guard replaces channel powers with `min(x**gamma, 8*x)` in encoded sRGB. It gives a linear dark branch, monotonic continuous join, black/white endpoints and float slope at most eight for the retained gamma bounds. A single input code maps to at most eight codes rather than approximately 111 at gamma 0.15. The separate fixed 3×3 median-input baseline applies the same original-source predicted gamma after smoothing; it is not a learned denoiser and changes textures.

## Results across 16 pairs and three frozen checkpoints

| Recipe | Mean per-image PSNR | Mean RGB brightness | Dark median-residual proxy (8-bit codes) |
|---|---:|---:|---:|
| Original power curve | 18.29 dB | 0.4598 | 5.43 |
| Bounded power curve | 16.67 dB | 0.3283 | 5.52 |
| Fixed median + bounded curve | 16.72 dB | 0.3288 | 1.41 |

The proxy is mean absolute residual from a per-channel 3×3 median inside originally dark nonzero source pixels whose maximum RGB code is ≤16. It includes actual texture and edges, not just sensor/JPEG noise. Median+guard lowers this proxy by 74.37% relative to guard alone, but this is **not a calibrated 74% denoising/quality improvement**. Guard alone does not reduce the mean proxy; it controls extreme amplification rather than smoothing all local variation. Average new full-channel fractions are approximately 0.00870% for original/guard and 0.00896% for median+guard. No method's clipping score establishes restored details.

Reviewed the same first eight development IDs and a fixed nearest-neighbor enlarged room-view crop. The guard removes the conspicuous bright/magenta/yellow near-black speckles in room views 102/104, utensil view 140 and plant view 50, but their dark areas remain substantially darker than references. Residual noise and color cast persist. The median baseline visibly reduces isolated speckling further; it also smooths small structures and cannot be assumed to preserve fine detail. Blue ceiling 142 still has insufficient visibility/color match. Corridor 576 still has a green cast. Shower 221 changes little because its pixels mostly avoid the capped dark branch.

This is a genuine visibility/noise tradeoff, not a reason to optimize paired PSNR blindly or to increase gain until the reference is matched. The frozen network was trained for the unguarded curve, so the guarded recipe is not a calibrated newly trained candidate. Three-seed means here describe the same small inspected development set, not independent final quality or human preference.

## Checks and reproduction

49 tests passed in 20.67 s, with the existing optional Matplotlib warning. New checks cover monotonicity, exact identity/zero strength, endpoints, strength interpolation, continuous join, numerical float slope bound, one-code amplification and invalid strength/gamma. Frozen predictions replay, original per-image MSE matches historical metrics, bounded rendering repeats exactly, and source/historical-parent hashes verify. The diagnostic loop took 3.23 s, excluding later tests/review; it is not a deployment latency measurement.

Run `venv/bin/python -m scripts.test_shadow_guard` with the recorded local dataset/checkpoints. The script refuses to overwrite a completed report. Protocol, frozen-source hashes and every per-photo/seed metric are in `PROTOCOL.md` and `report.json`. External photos/crops remain local and are excluded from GitHub; reports/code are tracked.

## Next action

Use the bounded/linear shadow behavior in the **training** renderer rather than adding it only after a power-curve model is trained. Compare a shared brightness curve that preserves channel ratios against the current independent channel curves, and add an explicit noise/detail control alongside reference matching. Keep any fixed edge-preserving denoiser a separately measured baseline; a smoother appearance is not recovered texture. Freeze a small protocol, audit new scene-exclusive development data and evaluate both full images and noise/detail/color crops. Previously inspected images remain regression examples. Keep official test pixels untouched until a fixed final evaluation. Preserve the current website model while this remains experimental.
