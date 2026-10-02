# Preview Improvement Plan (2026-10-02)

## Goal and scope

Improve washed-out skies, reduced tonal separation, and low-resolution previews. Model recommendations, applied preview values, and downloads must agree. Retain the verified ONNX model, 224×224 model inputs, 132D features, labels, and formal splits. Apply only validated Exposure/HighlightRecovery; other parameters remain experimental. Rendering remains a custom approximation, not Lightroom/PV2003 equivalence or recovery of clipped RAW channels.

## Causes and constraints

1. The previous preview decoded an encoded 8-bit JPEG; clipped information was already absent. Its 224×224 shape also distorted aspect ratio.
2. After multiplying exposure by 2^EV, the old highlight curve could exceed 1; RGB clipping then created flat white areas.
3. Stronger compression must not change model recommendations. Record the custom curve and actual applied parameters explicitly.
4. Saved browser permissions prohibit access to localhost. Do not bypass them. Use numerical checks, offline real-DNG rendering, and Streamlit integration tests; latest real-browser acceptance remains incomplete.

## Implementation sequence

### 1. Independent preview source

- [x] Reuse accepted dual-output RAW development in independent `preview.py`, preserving camera WB/orientation/geometry.
- [x] Convert linear 16-bit ProPhoto to linear sRGB using matching LibRaw matrices; document display-gamut clipping.
- [x] Preserve aspect ratio, maximum edge 1600, without upscaling. Model inputs remain separate 224×224 JPEGs.
- [x] Generate the source in the bounded worker and return NPZ with `allow_pickle=False`; clean temporary DNG/NPZ files. Keep source in session so strength changes do not reread RAW.
- [x] Ask old sessions to regenerate instead of silently using old JPEGs or mismatched results.

### 2. Monotonic highlight protection (v2)

- [x] Exposure requests gain g=2^EV. Protection defaults on. Use maximum RGB channel m and scale all channels together to preserve their ratios.
- [x] For g>1, v2 uses t=g*m/(1+(g-1)*m), monotonic on 0≤m≤1 without a hard full-value plateau; dark-pixel gain approaches g.
- [x] For g≤1, t=g*m. The explicitly labeled HighlightRecovery proxy monotonically compresses t>0.6 without affecting lower values.
- [x] Allow protection-off direct-exposure comparison and report newly full-value-channel pixels; do not present it as the default quality result.
- [x] Recompute from the same linear source. Zero strength exactly returns baseline; repeated actions do not accumulate.

### 3. Page and export

- [x] Display aspect-preserving before/approximate-after images, actual dimensions, and approximation limits.
- [x] Strength 0–100%, default 100%. Effective EV=recommended EV×strength and recovery=recommended value×strength; keep recommendations separate.
- [x] Show applied values/protection state and clarify that the proxy is not modern Highlights mapping.
- [x] PNG reflects current state; JSON includes source/dimensions/renderer/applied values/strength/protection/clipping metadata.
- [x] Do not add JPG/PNG parameter prediction, training dependencies, or a rendering service.

### 4. Numerical and workflow acceptance

- [x] Test zero-strength identity, positive/negative exposure, monotonic protection/RGB ratios, unchanged dark pixels under recovery, finite boundaries, and invalid-source/parameter rejection.
- [x] Check dimensions/aspect ratio/repeated rendering/same-source strength changes/PNG–JSON correspondence.
- [x] Verify unchanged model JPEG/X/prediction for the same DNG, passing deployment hashes, and Torch-free inference.
- [x] App tests cover sample/two images/strength/protection/two downloads; run the necessary full suite.

### 5. Real-photo acceptance and delivery

- [x] Fix three inputs: desert sample, low-brightness sample, and high-brightness sample selected from cached input brightness, not test quality tuning. Use the existing fixed model.
- [x] Compare baseline/old 224-JPEG/direct-exposure/protected renderings; measure clipping/dimensions/foreground brightness/timing.
- [x] Review sky detail/color cast/halos/geometry. Subjective improvement is not model accuracy or Lightroom equivalence.
- [x] Save comparisons/acceptance JSON and update README/master plan. Investigate before delivery if protection fails to reduce direct-exposure clipping.

## Completion criteria

Default preview uses a higher-resolution aspect-preserving linear source. Numerical/app checks pass, recommendations/applied state are auditable, multiple real inputs have reviewable comparisons, and model contracts remain unchanged. Faithful Lightroom and latest real-browser gates remain incomplete.

## v2 results

25 checks passed, including Torch/network-disabled RAW preview. Model/training-cache contracts stayed unchanged. Three fixed samples produced four-column comparisons and offline visual reviews. New full-channel fractions for desert/dark samples fell from 1.3167%/1.7179% to zero. The bright negative-exposure sample had zero in both modes. Sizes: desert 1600×1060, dark 1060×1600, bright 1600×1065. Reports/comparisons are in `artifacts/preview_v2`; original individual PNGs remain local. Saved permissions prevented real-browser acceptance.

## Future regional semantic adjustments (deferred)

The dual-branch model already combines full-image semantics and physical statistics. Separate sky/ground/face adjustments require defined masks/local parameters/soft boundaries and validation of regional labels/losses/color consistency. This expands the six-global-field contract; global recommendations cannot be assumed optimal for each region. This work only fixes previews and does not use these three photos to select/retrain models.

## Round 2: Normal highlights and ceiling diagnostics (2026-10-02)

The user approved global-curve optimization and overexposure diagnostics first. Preserve v2 source and 12 existing comparisons; keep model/RAW input/regional-semantic contracts unchanged.

- [x] Replace global compression with a continuous shoulder. Below output 0.6 retain linear gain; above it use a monotonic rational shoulder with endpoint 1 and continuous first derivative. Avoid early compression of normal shadows/midtones.
- [x] Move the recovery-proxy knee from 0.6 to 0.8 to reduce normal-highlight darkening; do not claim legacy Lightroom equivalence.
- [x] Report linear-preview channel/three-channel-white ceiling fractions in page/JSON. These are measured after display-gamut conversion and cannot distinguish sensor overexposure/development/display clipping.
- [x] Test monotonicity/shoulder continuity/RGB ratios/zero-strength identity/export correspondence; compare v2/v3 on 12 fixed photos.
- [x] Treat the four previously viewed test photos as regressions, not independent acceptance. Add unseen fixed-random test photos and do not retune based on their results.
- [x] Adopt the new renderer after numerical/offline review; retain browser/faithful-development limitations and update development records.

Round 2: 26 checks passed; all 16 fixed DNGs had zero newly full-value-channel pixels at both strengths and completed offline review. Adopted `linear-raw-shoulder-v3`. Original/new comparisons and provenance are in `artifacts/preview_v3`. Diagnostics only describe the developed display source; faithful highlight reconstruction remains incomplete.
