# Noise-aware shared-curve pilot: improved development metrics, visual gate failed

Date: 2026-10-02. Decision: retain prototype/evidence; **do not enable it on the website or promote a new production model**. All predefined numerical expansion screens pass, but visible colored grain/casts and inadequate extreme-dark visibility fail visual acceptance. The implementation/training task is complete; image-quality acceptance remains incomplete.

## New development data and fixed protocol

Added 32 previously unextracted `our485` training pairs using seed 20261006, alongside all 48 earlier inspected pairs. Reviewed all 80 low/reference thumbnails and ten hash candidate pairs before training. Carried prior groups forward; merged repeated captures and precautionary related fridge/bedroom/bathroom/AC/toy/bookshelf/gym/pool views. Scene/venue metadata is not available, so grouping remains conservative/incomplete and some extensions may overgroup.

Every prior ID and any related group is training-only. Frozen seed-20261007 split: 64 training pairs and 16 new development-validation pairs in 13 groups. Related bed and gym captures stay together. No `eval15` pixels were extracted/scored. Reference thumbnails were seen for grouping; this is new development data, not an untouched external final test. Original archive, all 160 source PNG hashes and JPEG derivative hashes are verified. RGB 8-bit/dimension compatibility was checked; absent ICC is assumed sRGB, and calibrated color/subpixel alignment is not established.

The user approved this bounded trial. See `PROTOCOL.md` for settings fixed before training. During implementation, before scoring, epoch-zero semantics were clarified: candidate/constant start with a neutral **filtered** pipeline; exact unchanged output is the separate identity baseline/strength-zero path, not filtered neutral output at strength one. Curve gain bounds refer to filtered pixels; spatial filtering can mix original neighbors.

## Candidate and controls

The 772-parameter MLP uses the same source-only 19D statistics and training-only normalization. One shared maximum-channel brightness curve is capped at gain eight, followed by endpoint-preserving per-channel color correction limited to ±0.15. Shared brightness avoids the preceding independent-power channel jumps, while retaining bounded color flexibility. Fixed native bilateral filtering (d=5, sigmaColor=8, sigmaSpace=2) precedes the curve. It is fixed smoothing, not learned denoising or recovered detail.

Training uses five fixed native 96×96 crops per source image, rather than downsizing before training. Candidate/constant loss combines RGB MSE, encoded-luminance gradient, chroma, target-flat/original-dark local-variation and bounded-color regularizers. All crops/features stay with their source group; no reference enters inference. Matched independent-gamma control uses RGB MSE and raw crops, the same split/feature normalization/optimizer/budget. Constant shared curve uses the candidate renderer/prefilter/composite objective. Comparisons evaluate complete recipes, not causal isolation of each component.

CPU/one Torch thread, AdamW 0.003/weight decay 0.0001/batch16, maximum60 epochs/patience10. Selection uses the predefined validation objective on native crops; full-image metrics are calculated separately.

| Run | Selected epoch / epochs run | Native mean PSNR | Native RGB MSE | Dark median-residual proxy | Target chroma MSE |
|---|---:|---:|---:|---:|---:|
| Identity | — | 7.23 dB | 0.220525 | 0.853 codes | 0.002067 |
| Existing experimental JPEG/v3 | — | 10.15 dB | 0.133523 | 2.526 codes | 0.001199 |
| Fixed +1 EV/v3 | — | 8.04 dB | 0.192650 | 1.387 codes | 0.001677 |
| Matched independent-gamma control | 3 / 13 | 16.30 dB | 0.029586 | 6.259 codes | 0.005771 |
| Learned shared constant | 60 / 60 | 14.48 dB | 0.072111 | 2.801 codes | 0.002541 |
| Candidate seed42 | 57 / 60 | 15.04 dB | 0.068302 | 2.713 codes | 0.002516 |
| Candidate seed43 | 4 / 14 | 14.90 dB | 0.069391 | 2.601 codes | 0.002324 |
| Candidate seed44 | 59 / 60 | 15.06 dB | 0.068129 | 2.723 codes | 0.002545 |

Three-seed mean PSNR is 15.00 dB: +4.85 dB over existing-v3, +0.52 dB over the learned constant and −1.30 dB versus the noisy matched gamma control. Each candidate seed improves RGB MSE over existing-v3 on 15/16 pairs. The candidate mean dark median-residual proxy is 2.679 codes, 57.19% below matched gamma; chroma MSE is 0.002462, 57.34% below gamma; luminance-gradient L1 is 0.018967, 34.14% below gamma. Mean new full-channel fraction is approximately 0.000313%.

The proxy includes texture/edges and is not a calibrated noise measure. Neither a 57% denoising claim nor broad aesthetic acceptance follows. Compared specifically with existing-v3, candidate dark residual is slightly higher and chroma MSE is approximately twice as high; these scores must not be misrepresented as improvements over every baseline. Existing-v3 is darker, affecting these comparisons. Per-image PSNR averages differ from pooled-MSE PSNR. Validation pairs are not statistically independent scenes, and a new set prevents direct comparison with the preceding 18.29 dB score.

All fixed numerical screens pass: gain over v3, closeness to constant, three seed means better than v3, dark residual ≤70% of matched gamma, chroma error ≤110% of matched gamma and new full-channel fraction ≤0.5%. Constant and two candidate runs reach the budget limit; optimization plateaus are not established. No settings/thresholds were changed after scores.

## Visual acceptance failed

Reviewed the first eight sorted new validation IDs (113,27,28,35,546,559,58,583), full images and fixed native center crops enlarged with nearest-neighbor interpolation. Also reviewed prior ID102 as a training regression only, excluded from validation statistics.

Shared/filtered curves reduce the very bright magenta/yellow spikes of the matched independent-gamma control. Book/sofa/TV scenes gain useful visibility over existing-v3. Book pages, wall/AC and upholstery crops retain visible colored grain and can lose subtle tonal/fine detail. Kitchen utensils 113 and bedroom views 27/28 remain much darker than normal-light references, with green/brown casts. Washroom 583 remains visibly cyan/green and noisy. The limited color correction does not establish correct white balance. Old regression102 loses the severe colorful spikes, but this is a trained-on example and is not new acceptance evidence.

The shared curve/gain cap and fixed prefilter reduce a known failure mechanism, but do not solve spatial noise restoration or extreme-dark color/visibility. The simple constant recipe is close to the conditioned model, so adding a larger global statistics head is not currently supported by strong evidence. No public benchmark result or population-quality claim is made.

## Verification and reproduction

- Full suite: **50 passed in 22.93 s**, one existing optional Matplotlib warning. No new dependencies were installed.
- Shared neutral brightness monotonicity, channel endpoints, curve gain, strength-zero bypass, neutral no-filter identity, intermediate strength, bounded color, finite composite-loss gradients and NumPy/Torch encoding checks passed.
- Verified all160 source hashes/JPEG derivatives, complete/exclusive groups, oldIDs training-only, at least eight validation groups, recomputed train-only feature statistics, source/protocol/manifest/review/split/checkpoint hashes and no eval15 extraction.
- All selected checkpoint parameters replay with zero error; actual native NumPy/Torch encoded outputs agree exactly (maximum/fraction differences zero) for every model and scored/regression image. This is checkpoint/calculation replay, not exact complete retraining replay.
- Training/native scoring/report loop:226.54s. Individual completed loops gamma3.36s,constant67.62s,candidate42/43/44 66.32/15.70/68.22s. Preparation and later tests/verification/review are excluded; these are not deployment latency or memory gates.
- Existing production model/renderer/app/source contracts remain unchanged. External images and experimental checkpoints remain local; English protocols/code/configuration/groups/manifests/results/decision are tracked.

Reproduce with `venv/bin/python -m scripts.train_noise_candidate prepare`, review `group_review.json`, then the `train` stage. Preserve/version completed directories because the commands refuse overwrite. `venv/bin/python -m scripts.verify_noise_candidate` checks retained artifacts. Local author-linked archive and source files are required; exact history/metrics are in `report.json`, with verification in `verification.json`.

## Remaining work

Do not increase the cap or change loss thresholds merely to make this inspected validation pass. The next justified research direction is a small spatial noise/color restoration baseline, compared with the fixed filter/constant/shared curve using a new fixed budget and fresh scene-exclusive development data. Audit native pair alignment and low-light color/export assumptions before fitting spatial targets. Keep detailed/noisy/cast/dark cases in separate visual acceptance, and preserve previous sets for regressions. Any deployment still requires independent unseen-image acceptance, export/runtime/latency and UI checks. Official test pixels remain untouched until a frozen final protocol.
