# Small grouped JPEG curve experiment: numeric gain, visual rejection

Date: 2026-10-02. Decision: **retain as a research prototype; do not deploy or promote**. The predefined numerical expansion screens passed, but visible chroma noise and color artifacts failed visual acceptance. This experiment trains a separate low-light image task, not compatible RAW adjustment parameters.

## Data and grouping

Selected 48 paired images from the author's LOL `our485` partition: the prior 24-pair pilot plus 24 added filenames drawn with seed 20261004 before image/quality inspection. Verified the original archive hash and all 96 source PNG hashes. Preserved source PNGs and quality-95 JPEG derivatives; official `eval15` pixels were not extracted or scored. Absent ICC is still assumed sRGB, and native source/target dimensions match; subpixel/color calibration is not established.

Reused the existing input visual-hash helper, screening both low and reference images. Reference pixels informed scene grouping only, never model features. Reviewed all 48 low/reference thumbnails and three candidate scene pairs before training; merged all three and recorded additional conservative toy/color-chart, refrigerator, hedge and swimming-venue groups. Capture/location metadata was not available, so wider venue groups are precautionary and overall grouping remains incomplete.

Groups containing previously inspected pilot IDs were forced into training. A fixed group shuffle assigned 32 pairs to training and 16 to development validation across 15 validation groups. No related group crosses partitions, and none of the prior pilot IDs enters validation. Reference thumbnails were seen during grouping; this is small development validation, not an untouched external final test. The two room-view validation IDs 102/104 share a group and are not independent observations.

## Model and controlled training

The 739-parameter candidate maps 19 source-only encoded-sRGB statistics through 19→32→3 layers to three bounded channel gamma values. Train-only feature statistics are checkpoint buffers. Curves operate in display sRGB, not RAW-linear exposure space. A zero-initialized output layer starts as exact identity. Strength blends original and adjusted pixels; channel curves are monotonic and preserve black/white endpoints in float. They have no spatial processing or denoising, and can amplify quantization noise/change hue.

Fixed CPU/one-thread AdamW 0.003, weight decay 0.0001, batch 16, maximum 100 epochs/patience 15. Trained on maximum-edge-96 proportional resized pairs; selected by small validation MSE including identity epoch 0. Native-resolution evaluation uses only source JPEG pixels/features. Three seeds share the same reviewed split and predefined settings. The constant baseline learns three gamma values from the same training pixels with the same loss/optimizer/checkpoint rule; no target-informed per-image curve fit enters inference.

| Run | Selected epoch / epochs run | Native mean validation PSNR | Native mean validation MSE |
|---|---:|---:|---:|
| Identity | — | 8.12 dB | 0.173210 |
| Existing experimental JPEG/v3 | — | 11.13 dB | 0.097057 |
| Fixed +1 EV/v3 | — | 9.04 dB | 0.145552 |
| Learned constant gamma | 100 / 100 | 15.79 dB | 0.043340 |
| Conditional gamma, seed 42 | 32 / 47 | 18.11 dB | 0.022969 |
| Conditional gamma, seed 43 | 29 / 44 | 18.39 dB | 0.022630 |
| Conditional gamma, seed 44 | 28 / 43 | 18.36 dB | 0.022973 |

The three-seed mean is 18.29 dB: +7.16 dB over existing experimental-v3, +2.50 dB over the constant baseline. All three seed means beat both; individual seeds improve MSE over experimental-v3 on 14–15/16 pairs and over the constant baseline on 12–13/16. Per-image PSNR averages are not PSNR of pooled MSE. No statistically independent final-quality claim follows from this small development set.

The constant baseline reaches the maximum 100 epochs, so its optimization plateau is not established. The experiment preserves the fixed matched budget rather than silently extending this run. The candidate's small-resolution selection loss differs from native-resolution evaluation; both are recorded.

All numerical screens pass: predefined ≥0.5 dB gain over both baselines, ≥2 seeds better, and new full-channel fractions ≤0.5%. Candidate mean new full-channel fraction is approximately 0.00870% in each seed. This does not measure noise, color fidelity, fine texture or aesthetic preference.

## Visual review: failed

Reviewed the first eight validation IDs, fixed by sorted ID before prediction, in five-way native-derived sheets: original, existing-v3, learned constant, seed-42 candidate and normal-light reference. Corridor/shower examples gain visibility and some approach the reference more closely. Room views 102/104, utensils 140, blue ceiling 142 and plant/cabinet view 50 exhibit clearly amplified colored speckles; some become too bright or acquire color casts. Seed-42 validation gamma values range from approximately 0.15 to 0.419. The room-view 102 estimate nearly reaches the 0.15 lower bound in all channels.

The curve itself explains a concrete failure mechanism: `x**gamma` with gamma below one has a large slope near black. At gamma 0.15, a single encoded level 1/255 maps to approximately 0.436 (111/255), whereas zero stays zero. Small JPEG/sensor variations can therefore become conspicuous colored spots. This is a mechanism consistent with the observed noise, not evidence of recovered dark detail. Per-channel curves also alter channel relationships. Lower paired MSE is insufficient for image acceptance.

## Verification and provenance

- Full existing suite plus curve invariant test: **48 passed in 21.51 s**, one existing optional Matplotlib warning.
- Monotonicity, endpoint, exact identity/zero strength, intermediate strength, invalid gamma/strength and finite exponent gradients passed.
- All selected-checkpoint validation gamma predictions replay exactly within 1e-6; actual native Torch/NumPy float maximum difference was zero in this environment. This is fixed-device calculation/checkpoint replay, not exact end-to-end retraining replay.
- Verified manifest/review/split/source/checkpoint hashes, all source members and JPEG derivatives, group exclusivity, old pilot training-only placement and no `eval15` extraction.
- The four-run training/native evaluation/report loop took 7.63 s on CPU, excluding preparation and later verification/tests. No deployment latency/memory gate is established by that loop time.
- Production RAW/JPEG model, existing renderer and app source remain unchanged. Checkpoints and external photo sheets stay local; tracked files contain code, protocol, source/manifest/split/config/history/metrics and this decision.

Reproduce preparation with `venv/bin/python -m scripts.train_jpeg_curve prepare`, review `group_review.json`, then run the `train` stage. Completed manifest/report directories are protected from overwrite; preserve or version them before replay. The local author archive is required. See `config.json`, `report.json` and `verification.json` for exact hashes/results.

## Next bounded hypothesis

Before expanding training or wiring this prototype into the site, test a monotonic curve with bounded near-black slope/linear dark segment and explicit noise/color controls. Compare against both the constant curve and this failed candidate using a separately fixed protocol. A simple global denoiser may be an additional baseline, but could remove real detail and needs separate review; it is not an assumed fix. These inspected validation images are now development examples. Preserve the official test and obtain fresh scene-exclusive validation for subsequent quality claims. Do not automatically brighten every JPEG or replace the current release with this result.
