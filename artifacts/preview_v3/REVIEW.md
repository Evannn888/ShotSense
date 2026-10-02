# Preview v3 Development Record (2026-10-02)

Problem: v2's global rational exposure mapping compressed midtones early. Its recovery knee at 0.6 also darkened ordinary highlights, flattening some images and turning clipped clouds gray.

Change: for exposure gain g>1, set shoulder start k=0.6/g. Below k, retain g*m; above it, use 0.6+g*(m-k)/(1+b*(m-k)), with b=(g-1)/(0.4*(1-k)). Value and first derivative are continuous at the join; endpoint is 1; mapping is monotonic and preserves RGB ratios. Recovery proxy knee moves to 0.8. This is a custom global curve, not faithful legacy Lightroom highlight recovery.

Page/JSON include linear display-source at-least-one-channel and three-channel-white ceiling fractions with explicit semantics. These are measured after RAW development, resizing, and display-gamut clipping; they do not detect sensor overexposure or prove texture recovery. Show a notice when channel-ceiling pixels reach at least 0.1%.

Acceptance: 12 previous regression cases plus 4 unseen test photos, excluding earlier candidates and selected before inference with seed 20261002. All 16 had zero new full-channel pixels at 100% and 75%. All baseline/v2/v3 sheets were reviewed: portrait/snow/bark midtones lift more fully; clipped clouds look less gray but missing texture is not restored; backlit subjects remain constrained by global adjustment. No obvious new halos at display size. Pixel-level noise, latest real-browser, and Lightroom checks were not performed. Four new test photos are a small-sample check, not population quality evidence.

Retain `preview_v2_reference.py`, existing `artifacts/preview_variety` reports, comparison JPGs, four overviews, and parameter/hash `report.json`. Individual PNGs and original ZIPs remain local with provenance documented separately. Model/development inputs/features/predictions remain unchanged.

Reproduce: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -m scripts.compare_preview_v3`.
