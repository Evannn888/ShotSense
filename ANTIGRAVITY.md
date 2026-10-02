# Antigravity Sync Protocol

This file defines the strict autonomous behavior rules for the agent working on the ShotSense project.

## Core Rules
1. **Auto-Initialization**: At the beginning of every new conversation, the agent MUST automatically read the `SHOTSENSE_MASTER_PLAN.md` file located in the root directory to understand the project context, architecture, and current progress.
2. **Auto-Tracking**: Update `SHOTSENSE_MASTER_PLAN.md` after each task, marking only work actually completed and verified with `[x]`. A plan edit or prototype does not complete implementation or acceptance gates.
3. **Progress Logging**: Following any task completion or major milestone, the agent MUST automatically append a new log entry to the `Progress Log` section at the bottom of `SHOTSENSE_MASTER_PLAN.md`, detailing the date and a brief summary of the completed work.
4. **V3.1 Ground-Truth Protocol**: Follow the current contracts and gates in `SHOTSENSE_MASTER_PLAN.md`. Training labels must come from the official Catalog, never from a regression proxy over rendered images; the model itself learns supervised parameter regression. Use rawpy for deterministic DNG development and colour-science for verified RGB -> XYZ D50 -> Lab conversions. Document necessary architecture amendments in the plan before implementing them.

## Preprocessing & Data Rules
5. **Dual-Output RAW Pipeline**: Use one rawpy.imread() context and two explicit postprocess() calls: linear ProPhoto 16-bit with gamma=(1,1) for 132D Lab features, and sRGB 8-bit with gamma=(2.4,12.92) for the 224x224 JPEG cache. Keep WB, geometry and exposure options consistent. Never apply ProPhoto CCTF decoding to linear output. Verify the LibRaw matrix and D50 handling; the old PSNR observation does not prove unavoidable conversion loss. Training and inference share the preprocessing contract.
6. **Normalization Spec Adherence**: Follow the master plan mapping: absolute targets, fixed field order, Contrast [-50,100], Temperature [2000,50000] and the other table ranges. Reject or quarantine invalid labels, never silently clip them. Mapping changes require updating the plan, ParameterNormalizer when implemented, cache/model versions and export metadata. A normalized sign is not an adjustment direction; Delta requires a same-version baseline. HighlightRecovery is not modern Highlights.
7. **Known Issue Awareness**: Before modifying data pipeline code, the agent MUST review the `Known Issues & Decisions` section in `SHOTSENSE_MASTER_PLAN.md` to avoid re-introducing fixed issues (e.g., KI-001 IncrementalTemperature bug).
8. **Label Integrity and Leakage**: Open the Catalog read-only. Require one nonempty record per source image/expert, match complete keys and apply only verified omitted-field defaults. Preserve source data and extraction snapshots while fixing parsing bugs. Never use expert-edited pixels or expert WB as inputs. Fit feature statistics only on training data; persist the image-level split and processing version.

## Environment Notes
- **Python**: System Python 3.9.8 was observed on 2026-10-02. An isolated project venv and requirements.lock.txt have been verified with pip check; do not use the incompatible global OpenCV environment. Reuse torchvision EfficientNet-B0 without adding timm.
- **CPU**: 8 cores. Pilot was verified with 1/2/6 workers and identical output; full preprocessing uses 6 workers (3.053 images/s, 2.75 GB peak RSS on the 20-image benchmark). Keep one worker for small Pilot runs. Expose --workers, limit nested library threads and support macOS spawn.
- **colour-science API**: Use RGB_to_XYZ(img, colourspace=RGB_cs, illuminant=..., apply_cctf_decoding=False) with a verified linear-input colourspace. API equivalence does not verify gamma or whitepoint correctness.
- **Deployment**: Verify FP32 ONNX and preprocessing parity before optional quantization. Measure model-only and RAW-to-result latency separately. Label CSS/Canvas previews approximate unless renderer compatibility is verified.
