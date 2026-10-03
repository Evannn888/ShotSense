# Small spatial residual restoration pilot

Fixed before new extraction/training, 2026-10-02. The user approved trying a lightweight local noise/color model. This experiment compares a small residual CNN with a frozen bounded shared-curve pipeline; no production promotion or final test follows from a small development score.

## Data, grouping and alignment

Use the existing verified LOL archive. Retain all80 previously inspected `our485` pairs for training/regression. Add40 unseen training filenames using seed20261008, independent of pixels/quality. Carry prior scene groups forward and review all120 low/reference contact thumbnails plus hash candidates before splitting. OldIDs and their related groups are training-only. Assign approximately16 new validation pairs by eligible-group shuffle seed20261009, at least8 validation groups. All expert/encoding/crop versions stay grouped. No eval15 pixels are extracted/scored. Reference thumbnails are only for pairing/grouping, never inference input.

Before split/training, screen alignment on native grayscale edge maps: resize proportionally to maximum edge320, apply Sobel gradient magnitude, normalize, and phase-correlate with a Hann window. Record response/native displacement for every pair. Quarantine clear shifts where response≥0.2 and maximum absolute native displacement>2 pixels. Mark response<0.2 as uncertain rather than claiming accurate registration. Do not warp or overwrite source/reference data. Inspect fixed first-six uncertain examples and flagged displacement pairs, and preserve every exclusion in the manifest. This screen is incomplete under noise/illumination changes; validate it on a synthetic known translation. If fewer than8 validation groups remain, stop before training. Rejected/uncertain alignment is a real limitation for spatial supervision.

## Architecture and budget

Freeze the prior noise-aware seed42 model/prefilter as a source-only base pipeline. Pick that seed before new scoring. The parent was trained on the previous64-photo training split; it is a fixed existing pipeline, not a retrained same-budget baseline. Preserve its exact feature statistics/configuration/checkpoint/source hashes.

Concatenate original encoded-sRGB input and base result (6 channels). CNN: 3×3 conv6→12, ReLU, conv12→12, ReLU, conv12→3; zero initialize last convolution. Predict `clip(base+0.5*tanh(residual),0,1)`. Initial output is exactly the base pipeline. This allows local color/noise residuals but does not establish true texture recovery. Strength zero returns original RGB exactly; intermediate strength blends original and complete output. CNN padding is zero; check native image edges. Approximately2295 trainable parameters, no BatchNorm/dropout/pretrained image generator.

Five fixed native96px crops per eligible training image, source/base computed at native resolution first. Exclude a3px crop border from loss to avoid artificial crop-padding supervision. Use the prior composite RGB/gradient/chroma/flat-variation loss with zero color-coefficient penalty (no trainable curve coefficients here). AdamW lr0.001, weightdecay0.0001, batch16, max30epochs/patience6. Select by same validation-crop loss, including parent pipeline epoch0. No loss/settings grid. Seed42 first; confirm43/44 only if seed42 numerical screen passes. Save initial/final frozen-parent state checks, histories/native predictions and source/data/config/checkpoint hashes.

DefaultCPU. If local MPS is available, first require CPU/MPS smoke forward≤1e-5, gradient≤1e-5 and finite optimizer updates on the actual training crops; then explicitly record MPS training. Evaluate selected checkpoints/metrics on CPU. No cross-device trajectory equivalence claim.

## Fixed expansion screen

On new development validation, candidate mean per-image native PSNR≥ frozen parent +0.5dB, original-dark median-residual proxy≤ parent (texture/edge caveat), target chroma MSE≤ parent, target luminance-gradient error≤110% parent, and new full-channel fraction≤0.5%. Confirm at least2/3 seeds improve PSNR if allthree run. Metrics describe recipes, not an isolated causal effect or independent scene-level significance. Include identity and existing-v3 native baselines; never compare absolute PSNR across different development sets.

Review first8 sorted validation IDs fixed before predictions at native full/crop/edge resolution; check colored spikes, casts, extreme-dark visibility, edge seams and fine text/texture loss. Also inspect old102 and27 as training regression examples, excluded from validation metrics. Visual failure blocks integration regardless of numeric screens. Official final evaluation, unseen personal photos, export/runtime/latency and UI gates remain incomplete; respect existing browser restrictions.
