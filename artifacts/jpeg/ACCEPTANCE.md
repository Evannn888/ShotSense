# Experimental JPG/JPEG Input Acceptance

Date: 2026-10-02. Scope: input/worker/UI workflow only. JPEG prediction accuracy has not been validated.

The English page accepts `.dng`, `.jpg`, and `.jpeg`. The existing DNG path retains its verified RAW preprocessing and model. JPEG runs through `src.jpeg_inference` in the same bounded worker arrangement: 128 MB, 40 megapixels, 60-second timeout, one local image job at a time.

JPEG decoding verifies actual JPEG content, applies EXIF orientation, converts valid embedded ICC profiles to sRGB, and assumes sRGB when no profile is present. CMYK without a profile and malformed profiles fail explicitly. Physical features are estimated from linearized display sRGB using the existing Lab statistics; this does not reconstruct RAW scene values or undo camera/software edits. Semantic JPEG encoding and normalization use the selected bundle's explicit geometry contract.

Every JPEG output is experimental. `recommended_absolute` is empty; model outputs for Exposure, Contrast, Saturation, and HighlightRecovery appear under `experimental_absolute`. Temperature/Tint are unavailable. Legacy outputs must not be interpreted as validated RAW settings or adjustment deltas. JPEG clipping cannot be reversed.

The default preview preserves the decoded input. **Apply experimental JPEG estimates to preview** is off by default; enabling it simulates exposure/highlight proxies only. PNG and JSON downloads match the selected preview settings. JSON identifies the rendered-JPEG input status, processing version/source hash, model version, orientation/color handling, unavailable fields, and limitations.

Verified checks cover both extensions, EXIF/ICC handling, exact worker/backend agreement, Torch/network-disabled inference, malformed/mislabeled/profile rejection, English page warnings, default identity preview, optional estimates, and downloads. A separate synthetic orientation/ICC fixture is recorded in `acceptance.json`; it is workflow evidence, not an aesthetic or accuracy benchmark.

The complete suite passed 40 checks at this input-support/candidate-pilot milestone. These tests establish workflow correctness, not JPEG model accuracy.

Real-browser localhost acceptance remains unavailable under saved browser permissions. Backend and Streamlit AppTest checks passed without bypassing that restriction. The production model checkpoint/ONNX and core RAW/color/JPEG-decoder modules remain unchanged. The inference router gained explicit candidate-geometry support, and the page gained JPEG upload controls; no candidate model was promoted.

Future accuracy acceptance needs a separately defined rendered-image task/dataset and evaluation protocol. The RAW model's historical accuracy numbers do not apply to JPEG uploads.

Follow-up verification: four actual training-photo semantic JPEGs passed exact decoded-baseline and zero-strength identity checks, with no new full-channel pixels in the opt-in preview. See `real_photo_workflow.json` and `real-photo-workflow-review.jpg`. These 224×224 inputs are workflow smoke fixtures, not full-resolution or JPEG prediction accuracy acceptance.

Preview usability fix: an unchanged JPEG result is explicitly labeled and explained; strength/highlight protection controls are disabled when no adjustments are enabled. Optional manual exposure/highlight controls override estimates, record `adjustment_mode=manual` and effective values in preview JSON, and reset when a new input is processed. A Streamlit AppTest regression verifies inactive status and manual +1 EV produces a changed preview without altering model recommendations.
