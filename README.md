# ShotSense

A local photo-adjustment prototype that automatically processes uploads, with protected adaptive JPEG color/shadow adjustment and conservative RawTherapee DNG tone adjustment at original resolution. Optional manual controls and legacy parameter research remain in Advanced options. The separate RAW recommender combines EfficientNet-B0 semantic features with 132-dimensional linear Lab statistics to predict the arithmetic mean of five FiveK experts' parameters.

Recommendations are absolute values for legacy Camera Raw **PV2003**. `HighlightRecovery` is not modern `Highlights`. Only exposure and highlight recovery passed the recommendation gate; contrast, saturation, temperature, and tint remain experimental.

## Current progress · 2026-10-06

- **Practical local workflow:** automatic JPG/JPEG/DNG processing, original-resolution PNG saving, matching JSON and optional adjustments are implemented. The page follows **Upload → Compare → Save**, with research modes in the sidebar and details collapsed. See the [interface verification](artifacts/experiments/ui_cleanup_v1/REPORT.zh-CN.md).
- **Verified delivery:** 91 project tests pass. Actual in-app-browser upload and PNG download of one 678×452 JPEG produced the exact current worker/render bytes and pixels. Original-resolution 20MP export and separate approximately50MP JPEG compatibility checks are recorded; these checks do not establish photographic quality on every input.
- **Research beta:** scene-aware policy and bounded white balance are connected behind a default-off option. The [current WB v2 review](artifacts/experiments/white_balance_v2/REPORT.zh-CN.md) covers114diagnostic images:18applyWB and96retain the no-WB result. Warm-backlight preservation improves, but an indoor people cast remains unresolved and pooled expert-reference distance slightly worsens.
- **Next gate:** independent photo/user-preference review, mixed-light/color-intent and reliable skin appearance need validation before promoting scene/WB to the default. No sharpening is applied.
- **GitHub snapshot scope:** application/tests/preparation scripts, engine profiles, reports and summary metadata are distributed. Personal/dataset photos, external model weights, downloaded upstream source mirrors and duplicate historical source archives stay local; referenced native exports and logs remain local evidence. Prepare the pinned LUT assets with `venv/bin/python scripts/prepare_adaptive_lut.py` and the native engine as described below before using their respective workflows on a new Mac.

## Run locally

From the project root:

```sh
venv/bin/python -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501/, upload an unedited DNG or JPG/JPEG, compare the result, then click **Save PNG to Downloads**. Alternative modes and **Use project sample** are in the sidebar's **Advanced options**. Inputs are limited to 128 MB, JPEG to 64 megapixels and DNG to 40 megapixels, with a 60-second processing-worker timeout. The app processes one image job at a time locally. PNG rendering/export takes additional time.

The [high-resolution compatibility check](artifacts/experiments/photo_engine_highres_v6/ACCEPTANCE.md) records actual 50.33 MP/49.63 MP JPEG workers and native PNG exports without resizing: approximately 36/30 seconds for the worker calls plus 7/5 seconds for PNG rendering on this Mac. JPEG manual-preview resizing also bounds float32 roundoff before strict validation (`rendered-jpeg-srgb-v2`). Restart the local service after source changes so worker and renderer imports use the same current version; refresh and upload again after a restart.

**Default: automatic adjustment.** Upload a JPG/JPEG or DNG; processing begins without choosing a recipe, changing sliders or clicking Process image. JPEG uses the protected adaptive-color result and automatically suggested shadow lift; DNG uses the existing conservative native automatic tone recipe. Compare **Original** and **Adjusted** (or **Original retained**), then click **Save PNG to Downloads** to save the original-resolution result. Nothing is saved automatically. **Optional adjustments** and **Inspect photo details** are collapsed and are not required to use the tool. Automatic choices are provisional and cannot infer photographic intent; this is a beta, not completed independent personal-camera/preference acceptance.

Each new upload is processed once, identified by filename, suffix and byte hash. Saving, preparing downloads, inspecting details and optional controls reuse its result. Replacing/removing an upload clears previous pixels, settings and export state. A failed file shows an error and does not retry on every page rerun; use **Advanced options → Process image** to retry. Legacy sessions start with the new automatic workflow widget. No backend formulas/weights, format/size limits or worker deadlines changed. See the [automatic-flow check](artifacts/experiments/automatic_photo_ui_v1/ACCEPTANCE.md).

The [actual automatic-default validation](artifacts/experiments/automatic_photo_validation_v1/REPORT.zh-CN.md) processes17existing photographic regressions through the current default: all pass source/native PNG/JSON consistency. Landscape JPEG results are provisionally useful, while extreme-low-light results remain dark/noisy and DNG is conservative tone adjustment. This is limited photographic evidence, not universal direct-to-use or independent camera/preference acceptance.

**Optional manual exposure.** Open **Advanced options**, select **Manual exposure**, upload JPG/JPEG, then click **Process image**. The preview starts unchanged at0EV. Use **Manual exposure (EV)** and highlight protection to adjust brightness and compare before/after. Click **Save PNG to Downloads** to save the current preview directly on this Mac. The optional browser route uses **Prepare PNG download**, then **Download approximate preview PNG**; JSON is in the separate metadata expander. The shared linear RGB tone gain does not add model-generated color offsets; it is a manual preview, not automatic learned enhancement or a denoising solution. The maximum preview/export edge is1600px; automatic, Natural adjustment and Natural color use original-resolution exports.

**Natural adjustment (experimental):** prepare the pinned local engine once on macOS12.3+:

```sh
venv/bin/python scripts/prepare_photo_engine.py
```

This downloads and verifies the official RawTherapee5.11 universal archive, installs the separate unmodified app under `/Applications/RawTherapee.app`, and retains its CLI under `data/tools/`. A different existing RawTherapee installation is never overwritten. The adapter currently targets this Mac setup; Linux/Windows are not implemented. No Python dependencies are added, and processing after setup runs offline with isolated settings/cache and temporary source copies.

Select **Natural adjustment (experimental)** under **Advanced options → Photo workflow**, choose a recipe, upload a DNG or JPG/JPEG, and process. **Automatic conservative choice** applies +0.35EV with engine highlight compression only to moderately dim images, otherwise preserves the baseline; it also rejects candidates that introduce more than0.1% new full-value-channel pixels. These provisional scene statistics cannot determine artistic intent. **Gentle lift** makes that recipe explicit; **Gentle lift + chroma denoising** adds mild chroma filtering that may soften detail. **Lift shadows** uses native tone-equalizer bands0/40/20/0/0/0, regularization2 and zero global EV for a moderate dark-foreground adjustment that retains deeper blacks. It may reveal noise and affect highlights; reduce strength when needed. Automatic selection does not enable denoising or shadow lift. Full-strength recipes use their own PP3 controls, independently of Adobe model predictions.

Compare before/after and adjust strength from0–100%. Click **Save PNG to Downloads** to write the current PNG directly to this Mac's Downloads folder. The page reports the absolute path after the write succeeds, or an error if it fails. This requires an explicit click, with no preparation or automatic save. An identical existing file is reused; conflicting files and symlinks are refused. Filenames identify the recipe, strength, dimensions and a short output hash. See the [local delivery check](artifacts/experiments/photo_engine_delivery_v4/ACCEPTANCE.md).

After processing in Natural adjustment, open **Color and local contrast**. **Warmth** moves from cool to warm; **Tint** from green to magenta; **Vibrance** adjusts less saturated colors with native skin protection; **Local contrast** adds a restrained amount of contrast around lightness transitions. All four start at zero. Click **Apply fine tuning** to apply the current controls at full resolution. **Reset fine tuning** restores the current recipe and clears these controls. Each application starts from the same recipe result, so repeated clicks do not accumulate edits. Overall **Adjustment strength** blends the recipe plus applied tuning with the original decoded JPEG or neutral RAW baseline; zero strength still restores that original baseline.

Fine tuning supports both JPEG and DNG in this workflow as relative edits to the already rendered sRGB result. Warmth is not the camera's original Kelvin and cannot undo JPEG white-balance processing. DNG camera-WB development and the legacy parameter model are preserved. Local contrast operates on neighborhoods throughout the image; it is not a foreground/sky selection or recovered sharpness, and can emphasize noise or bright edges. The page reports applied settings and warns when tuning adds more than0.1% pixels with a full-value channel. These explicit edits are not automatic color correction or an accepted universal recipe. See the [fine-tuning protocol](artifacts/experiments/photo_engine_finetune_v7/PROTOCOL.md).

Full-resolution PNG, matching JSON and native detail crops all use the applied fine-tuned result and overall strength. PNG filenames add `finetuned`; JSON records the tuning version/settings/input-output pixel hashes/native profile and timing separately from the original recipe. Applying or resetting invalidates prepared downloads. A failed application preserves the last successful pixels and reports the error. High-resolution fine tuning runs in the same serialized,60-second bounded worker; export adds rendering time.

For an optional browser download, open **Other download options**, click **Prepare PNG download**, then **Download full-resolution PNG**. Preparing keeps the panel expanded and the PNG button visible. Matching **adjustment JSON** is in the separate **Adjustment metadata (JSON)** expander; it is a settings file, not an image. Changing the result/settings invalidates the prepared pair. Each PNG and JSON has a separate file URL, MIME type and widget identity. Browser downloads do not trigger another processing rerun. The2026-10-06 [actual browser verification](artifacts/experiments/ui_cleanup_v1/browser_download_verification.json) confirms one current default JPEG upload/download has correct PNG bytes, dimensions and pixels; local saving and the other modes/JSON/cache contracts are covered separately by functional checks.

The image-proto payload and PNG download use the same original-resolution bytes through a PNG data URL. The overview uses content width: large images fit the column and small images are not enlarged. JPEGs below1MP show their dimensions and detail limits in **Inspect photo details**. Earlier default Streamlit image calls silently converted RGB PNGs to JPEG and resized wide images to1460px, so the earlier displayed/downloaded-byte claim was not established. **Inspect detail at original size** shows adjustable native320px crops from the actual blended result, with coordinates and no upscaling/sharpening. The [export/display correction](artifacts/experiments/photo_engine_export_v3/ACCEPTANCE.md) records the reproduction and actual file-manager checks. The current browser PNG download proof is limited to the JPEG case recorded above; it does not retroactively verify earlier downloads.

JSON records dimensions, strength, recipe/input/profile/engine identities and output hash. Export is8-bit sRGB; a JPEG is decoded after EXIF orientation and ICC conversion, and zero strength restores those exact decoded pixels. For DNG, zero strength restores neutral engine development with camera white balance, not original sensor data or the old RAW preview. RAW inputs still require valid camera WB. Outputs contain no invented Lightroom parameter recommendations. Original files are preserved; EXIF/GPS and other input metadata are not copied into PNG exports. PNG preserves available pixels; it cannot reconstruct source blur or missing detail.

When the result says **Unchanged**, the page shows the actual processed recipe and distinguishes zero strength from automatic preservation, including the candidate's clipping percentage when applicable. **Apply gentle lift** explicitly reprocesses the same input, selects Gentle lift and restores100% strength; the input hash must still match. It does not relax the automatic guard or promise a large visual change. See the [unchanged-result UX correction](artifacts/experiments/photo_engine_v1/ui_feedback_v1/ACCEPTANCE.md).

If gentle lift leaves the foreground too dark, click **Apply shadow lift** beneath the comparison, or select **Lift shadows** and process again. This reuses the same worker/hash check and resets strength to100%; zero strength restores the baseline. The historical [stronger shadow record](artifacts/experiments/photo_engine_shadow_v2/ACCEPTANCE.md) used screenshot proxies. Actual user JPEG review later showed excessive lifted blacks and amplified compression; the [native quality correction](artifacts/experiments/photo_engine_quality_v5/DECISION.md) replaces that strong recipe with a gentler one after viewing the three actual small JPEGs and portrait/synthetic controls. Runtime/renderer version is `rawtherapee-recipes-v5`; gentle/chroma/automatic thresholds remain unchanged. An older cached shadow result is marked and offers **Reprocess with gentler shadows** while its source is still available. The new recipe brightens less and does not reconstruct compressed/missing detail. Fresh user preference acceptance remains pending.

The [engine pilot acceptance](artifacts/experiments/photo_engine_v1/ACCEPTANCE.md) records 15 inspected public/synthetic diagnostics, native detail review, actual timing and 59 passing tests. No severe new color/detail failure was observed in that finite review, but extremely dark photos remain dark, source clipping persists, and independent personal-photo preference acceptance is pending. Automatic is now the default; manual and research workflows remain in Advanced options. PNG/TIFF/HEIC and other camera RAW extensions are not accepted by this application, even if the underlying engine can decode them. The 128 MB, JPEG 64 MP/DNG 40 MP and 60-second worker limits are guards, not a guarantee that every file up to those limits completes within the timeout. JPEG acceptance and native PNG output/rendering use the same pixel bound; no input is silently downscaled to pass it.

**AI enhancement withdrawn (2026-10-04):** a real user photo showed severe over-brightening, color artifacts and a red/purple eye. HVI-CIDNet is no longer available in the website workflow. Previously cached AI results show only their preserved input preview; processing again uses the chosen manual/legacy workflow. Original uploaded files are never overwritten. Historical sources/checkpoints and diagnostic records remain available for research, but passing functional tests did not establish acceptable photo quality. See [failure and withdrawal record](artifacts/jpeg_restoration/ACCEPTANCE.md).

The **Legacy parameter estimates** workflow retains the earlier JPEG estimates and manual exposure controls. **JPEG support is experimental.** The model was trained on DNGs; an already processed JPEG has a different input distribution. All JPEG estimates appear under experimental outputs, Temperature/Tint are unavailable, and `recommended_absolute` is empty. EXIF orientation is applied; embedded ICC profiles are converted to sRGB, with sRGB assumed when no profile is present. CMYK JPEGs require a valid ICC profile. Physical features come from linearized display sRGB, which does not undo camera/software processing. Clipped detail cannot be recovered. The baseline JPEG preview appears immediately after processing; **Apply experimental JPEG estimates to preview** is off by default. The page explicitly marks an unchanged preview and disables strength until adjustments are enabled. **Adjust JPEG preview manually** provides exposure and highlight compression controls; manual values replace model estimates for the preview. PNG/JSON downloads record the selected mode and applied settings. See [JPEG acceptance](artifacts/jpeg/ACCEPTANCE.md).

For DNG inputs, the page shows baseline RAW development and an approximate adjusted preview side by side, preserving the original aspect ratio with a maximum edge of 1600 pixels. Highlight protection is enabled by default. Adjustment strength ranges from 0–100%; applied values and clipping diagnostics appear on the page and in JSON. PNG downloads reflect the current settings. Only validated exposure and highlight compression are simulated. This custom preview is not equivalent to Lightroom and cannot guarantee recovery of clipped detail.

## Environment and reproduction

Verified on macOS arm64 with Python 3.9.8. Set up the development and training environment:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.lock.txt
venv/bin/python -m pip check
venv/bin/python -m pytest -q
```

The full test suite requires the locally retained dataset and caches. Original data live in `data/raw/dngs/` and `data/raw/fivek_dataset/raw_photos/fivek.lrcat`. The Catalog is read-only, invalid labels are quarantined, and previous labels are preserved as immutable snapshots. Training requires pretrained weights at `artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`; preparing those weights initially requires network access. ONNX inference runs offline.

```sh
venv/bin/python -m src.extract_labels
venv/bin/python -m src.preprocess --limit 20 --workers 1
venv/bin/python -m src.train --pilot --metadata data/processed/pilot/metadata.npz --output-dir artifacts/experiments/pilot-new --epochs 30
venv/bin/python -m scripts.prepare_data --workers 6
```

The included production bundle was trained/exported under the historical protocol. New training runs now require a **new output directory**, select checkpoints on validation P2, and produce validation results only. They do not grant validated recommendation status or replace the included bundle. A changed training seed does not change the split seed.

After reviewing photo groups and completing candidate preprocessing acceptance, use the local group map and fixed split:

```sh
venv/bin/python -m src.train --output-dir artifacts/experiments/model_vnext/control-seed42 --seed 42 --split-seed 42 --groups data/processed/model_vnext/groups.json --split-path data/processed/model_vnext/splits.json
```

Replace the run name and model seed for subsequent experiments; reuse the same group map and split. Run directories cannot be overwritten. Once architecture/settings/checkpoint selection are frozen, explicitly perform final evaluation:

```sh
venv/bin/python -m src.train --final-evaluate --output-dir artifacts/experiments/model_vnext/control-seed42
```

This creates an exclusive test-access record and refuses repeat evaluation of the same run. Training uses the training-mean constant baseline; the historical trainer used a training median. Final results do not automatically authorize promotion: three-seed comparisons, uncertainty, visual review, and deployment gates remain required. All large experimental caches/checkpoints stay local.

Caches are identified by processing-code, dependency, label, and configuration hashes. Repeated runs resume and rebuild corrupt files. Configuration changes require a new output directory or explicit `--rebuild`. Pilot and full data, splits, and models are separate. The full photo-ID split uses seed 42 and an 80/10/10 ratio. Feature standardization uses only training data. Burst/near-duplicate grouping is unavailable, so this split does not establish that all similar-scene leakage has been excluded.

## Command-line inference

```sh
venv/bin/python -m src.inference path/to/input.dng --output result.json --preview input.jpg
```

The [isolated pretrained LUT pilot](artifacts/experiments/adaptive_lut_v1/ACCEPTANCE.md) runs the pinned paired-sRGB Image-Adaptive-3DLUT on CPU with no new dependencies. Its ten-photo/native/synthetic comparison found stronger color, crushed shadows and gray-ramp reversals, so the unprotected author result is **not enabled in the website or accepted for automatic adjustment**. A separate protected color wrapper is available as an explicit experiment below. Both actual50MP JPEG workers/full-size sRGB exports pass functional checks; those checks do not establish naturalness. To reproduce the separate JPEG-only inference:

```sh
venv/bin/python scripts/prepare_adaptive_lut.py
venv/bin/python -m src.adaptive_lut path/to/input.jpg --output lut.json --preview-source lut.npz
```

The preparation downloads verified pinned weights and author/license sources into local `data/external_models/`; inference afterwards is offline. The NPZ stores full original/adjusted RGB arrays for the shared PNG renderer, and JSON records model/table/pixel identities and unclamped diagnostics. The existing website/default/current recipes remain unchanged by this trial. DNG diagnostic comparisons used previously frozen neutral developments; the new CLI does not accept DNG.

For the withdrawn HVI research backend only (not recommended for photo editing), prepare pinned weights with `venv/bin/python scripts/prepare_jpeg_restoration.py`, then:

```sh
venv/bin/python -m src.jpeg_restoration path/to/input.jpeg --output enhancement.json --preview-source restoration.npz
```

The application renders/downloads the PNG from this cached source at the chosen strength. To reproduce retained local diagnostic examples, run `venv/bin/python scripts/verify_jpeg_restoration.py`.

For experimental JPEG estimates:

```sh
venv/bin/python -m src.jpeg_inference path/to/input.jpeg --output result.json --preview-source preview.npz
```

JSON separates `recommended_absolute` and `experimental_absolute`, with units, legacy process semantics, model/data versions, and timing. No Delta is returned without a baseline using the same parameter version. ONNX inference does not import torch/torchvision; it still requires NumPy, ONNX Runtime, rawpy, colour-science, OpenCV, and Pillow.

## Acceptance evidence

- `data/intermediate/label_audit.json`: field sources, evidence for omitted defaults, invalid IDs, 43 repaired fields, process versions, and label hashes.
- `data/intermediate/color_audit.json`: linear/D50 configuration and pixel comparisons between shared and separate RAW contexts.
- `data/intermediate/preprocessing_benchmark.json`: throughput and peak memory for 1/2/6 workers on the same pilot.
- `data/processed/manifest.json` and `splits.json`: accounting, versions, and mutually exclusive splits. Of 5000 inputs, 4946 succeeded, 3 had invalid labels, and 51 lacked valid camera WB.
- `data/processed/input_support_audit.json`: per-file RAW-header support audit and original failure report. Unknown processing errors still fail; automatic WB is not silently enabled.
- [Model evaluation](artifacts/model/evaluation.json): constant/physical-linear/semantic/dual comparisons, final test metrics, expert disagreement, and camera/Catalog-tag/tail/As-Shot WB strata.
- [Deployment report](artifacts/model/deployment_report.json): PyTorch/ORT parity and CPU forward p50/p95, reported separately from RAW end-to-end time.

Beating a baseline on a fixed validation set is evidence of research effectiveness; acceptable error for practical use remains uncalibrated. Six parameters cannot fully reproduce expert development. Modern Lightroom mapping, faithful high-resolution development, additional RAW formats, and edited JPEG inputs are outside the verified scope. Final test macro-average error was 18.7% below the constant baseline; exposure MAE was 0.281 EV and highlight recovery MAE was 7.428. FP32 CPU forward p95 was 19.39 ms. All 26 checks passed after English localization. See [acceptance](artifacts/model/ACCEPTANCE.md) and the [master plan and development log](SHOTSENSE_MASTER_PLAN.md).

## Model improvement roadmap

The [detailed model improvement plan](MODEL_IMPROVEMENT_PLAN.md) defines grouped evaluation, validation-only experiment selection, matched baselines, aspect-preserving semantic inputs, a bounded head search, and conditional partial backbone fine-tuning. Phase A is implemented: a [production baseline manifest](artifacts/experiments/baseline_manifest.json), [reviewed grouping summary](artifacts/experiments/group_audit_summary.json), independent split/training seeds, protected run directories, and explicit final evaluation. Screening 4946 inputs proposed 11 pairs; review merged 8 and rejected 3, yielding 4938 groups. Two merged pairs crossed historical partitions. Grouping remains incomplete; a FiveK resplit is not a fresh external benchmark.

Phase B established a [reproduced grouped validation reference](artifacts/experiments/model_vnext/PHASE_B_ACCEPTANCE.md): the seed-42 dual control scored 0.277752 EV Exposure MAE and 7.108073 HighlightRecovery MAE, with P2 32.9% below the training-mean constant and 5.1% below physical linear. These are single-seed validation comparisons, separate from the production model's historical test results. Neither the new test set nor production promotion was used. See the [experiment ledger](artifacts/experiments/model_vnext/experiment_ledger.csv) and [summary](artifacts/experiments/model_vnext/phase_b_summary.json). Phase C [geometry comparison](artifacts/experiments/model_vnext/PHASE_C_ACCEPTANCE.md) is complete: letterbox dual P2 was 0.052948, 0.09% higher error than stretch. The single-seed bootstrap interval crosses zero; retain stretch geometry and skip candidate seed confirmation. Phase D [bounded head search](artifacts/experiments/model_vnext/PHASE_D_ACCEPTANCE.md) is also complete: the original configuration remained best, and its three-seed P2 mean/sample SD was 0.052239/0.000626. No alternative was promoted. Validation-only [failure analysis and partial-tuning smoke](artifacts/experiments/model_vnext/PHASE_E_SMOKE_ACCEPTANCE.md) now verify the experimental training boundaries and CPU cost. The [full three-seed comparison](artifacts/experiments/model_vnext/PHASE_E_ACCEPTANCE.md) is now complete: mean P2 improved only 0.245%, its bootstrap interval crossed zero, and only one seed improved. Retain the frozen control; no new model was promoted. White-balance/style/regional work remains pending.

## Preview development

The [preview improvement plan](PREVIEW_IMPROVEMENT_PLAN.md) records each stage. Three-photo v2 results are retained in [the v2 report](artifacts/preview_v2/acceptance.json). Current reproduction scripts call the current renderer; historical results identify their original source hashes.

The current renderer is `linear-raw-shoulder-v3`: normal midtones retain linear exposure gain, highlights use a continuous shoulder, and baseline channel-ceiling diagnostics appear in the page and JSON. These diagnostics cannot establish sensor overexposure or texture recovery. See the [16-photo v2/v3 review](artifacts/preview_v3/REVIEW.md). Reproduce with `venv/bin/python -m scripts.compare_preview_v3` after preparing local data. Browser acceptance of the latest preview remains incomplete because saved browser permissions block localhost access.

![Baseline, v2 and v3 comparisons](artifacts/preview_v3/overview-3.jpg)

## Repository contents

The repository includes code, pinned dependencies, the master plan/development log, the production ONNX model and training checkpoint, acceptance JSON, comparison JPGs, and an English snapshot/provenance record. Original FiveK DNG/Catalog files, extracted labels, preprocessing/semantic/pretrained-backbone caches, the virtual environment, and duplicate PNG previews remain local.

The included production model can process your own supported DNG without retraining. **Use project sample** requires a DNG under local `data/raw/dngs/`. Full data tests, training, and comparison reproduction require separately prepared data and caches. `data/` audit paths and individual PNGs in reports refer to locally retained evidence.

Original historical ZIP snapshots and the initial Chinese-interface screenshot are preserved locally and in earlier Git history, rather than altered and presented as original evidence. Their hashes and purpose are recorded in [snapshot provenance](artifacts/SNAPSHOT_PROVENANCE.md). Current documentation and application text use English; prior commits remain intact.

The [optimization research review](OPTIMIZATION_RESEARCH.md) compares primary literature with the negative geometry/head/fine-tuning results and prioritizes a separate JPEG enhancement task, bounded RAW objective alignment and stronger regularized baselines. Proposed gains remain unverified.

The [dataset shortlist](DATASET_RECOMMENDATIONS.md) prioritizes full FiveK resources, MSEC exposure variants, PPR10K portraits and Rendered WB, with targeted LCDP/LOL/SICE diagnostics. The approved [100-source paired-target pilot](artifacts/experiments/fivek_paired_targets_v1/REPORT.zh-CN.md) acquired original Expert C TIFFs and prepared99provisionally aligned source-default-crop development pairs (79train/20validation), retaining one orientation mismatch in quarantine. This is data preparation, not model training, native subpixel alignment or independent photo-quality acceptance.

The [current automatic effect review](artifacts/experiments/fivek_current_effect_v1/REPORT.zh-CN.md) runs unchanged JPEG processing on those99development views and compares twelve preselected inputs with actual outputs and Expert C references. It finds useful shadow visibility in some examples, but raised night blacks, remaining casts and weak subject separation; no model training or new quality acceptance is implied.

An optional **Use scene-aware adjustment (experimental)** checkbox in Advanced options connects the locally prepared MobileCLIP2-S0 image ONNX/fixed text vectors to the real JPEG worker. It still processes uploads automatically and restricts night/sunset/uncertain changes; missing or invalid recognition assets fall back to the unchanged existing route without downloading. The [109-input development comparison](artifacts/experiments/scene_policy_connection_v1/REPORT.zh-CN.md) finds better night black levels but false restrictions on dark-background flowers/birds and slightly worse pooled Expert C proximity. The option stays off by default; independent preference validation and default promotion remain pending. Recognition assets currently reuse the verified local research bundle at `data/user_photo_diagnostics/scene-analysis-research-v1/`; they are not distributed with the repository. CLI: `python -m src.lut_natural photo.jpg --scene-aware --output result.json --preview-source pixels.npz`.

The current [v2 night-gate follow-up](artifacts/experiments/scene_policy_connection_v2/REPORT.zh-CN.md) requires night ranked first in both existing views before suppressing lift. Known lily/owl/sunset inputs regain mild lift while moonlight/galaxy outputs stay exact;109classifier scores remain unchanged and only9outputs differ from v1. Pooled Expert C proximity improves slightly versus v1 but remains worse than the original automatic route, with candle/flash-scene counterexamples. The checkbox stays default-off; its policy version invalidates old beta upload caches once.

The [bounded WB pilot](artifacts/experiments/white_balance_v1/REPORT.zh-CN.md) adds optional source-only Deep White-Balance Editing AWB to this checkbox, with neutral-evidence/mood/luminance/headroom checks and smaller changes for a skin-color proxy (not face segmentation). On109reused development inputs13are corrected and96remain exact scene-v2 outputs; some indoor casts improve, but pooled reference distance worsens slightly, false night skips and warm-backlight changes remain. This is experimental, default-off and not independent quality acceptance. Prepare its pinned local research assets with `python scripts/prepare_white_balance.py` (CC BY-NC-SA4.0 author bundle); no photo-time downloads. CLI with WB: `python -m src.lut_natural photo.jpg --scene-aware --white-balance --output result.json --preview-source pixels.npz`. Only `--scene-aware` retains v2. Current JPEG/DNG defaults are unchanged; missing WB assets preserve v2, missing recognition preserves the original route.


The [small JPEG curve experiment](artifacts/experiments/jpeg_curve/ACCEPTANCE.md) records three-seed development gains and a visual rejection caused by amplified dark-region chroma noise. The prototype is separate from the production model and is not enabled in the app.

The [bounded-shadow regression](artifacts/experiments/jpeg_shadow_guard/ACCEPTANCE.md) tests a linear dark branch/gain cap and a separate fixed median-filter baseline. It records the visibility/noise tradeoff and remaining color/detail problems without promoting the recipe.

The [noise-aware native-crop training pilot](artifacts/experiments/jpeg_noise_v1/ACCEPTANCE.md) uses a new grouped development set, shared bounded brightness, fixed filtering and noise/detail/color losses. Numerical screens pass, but remaining grain, casts and dark visibility fail visual acceptance; the website still uses the existing release.

The [small spatial residual pilot](artifacts/experiments/jpeg_spatial_v1/ACCEPTANCE.md) uses104 training pairs and16 new grouped development pairs over a frozen shared-curve base. Seed42 improves PSNR by3.48dB but increases dark-region variation by15.95% and retains visible grain/casts, so conditional seeds and website integration were rejected.

The [separate denoising/illumination pilot](artifacts/experiments/jpeg_illumination_v1/ACCEPTANCE.md) reduces grain but fails fidelity/color and visual gates on14 new development pairs; strong filtering erases fine detail. After HVI's real-photo failure, the [updated existing-solutions research](EXISTING_SOLUTIONS_RESEARCH.md) prioritizes a controlled mature engine, real user preference review and full-resolution exports. The first frozen adaptive-LUT comparison now runs locally but fails the natural-photo gate; further LUT or specialized restoration candidates remain conditional future work.

### Protected natural color (experimental)

Open **Advanced options** and select **Natural color (experimental)**, upload **JPG/JPEG**, then click **Process image**. It uses the existing pinned Image-Adaptive-3DLUT predictor and basis tables to produce a per-photo color suggestion. The new wrapper limits the color change, automatically selects inner color strength (at most35%), protects dark/bright/neutral regions, and uses a separate mild monotone brightness curve. It does not apply the author model’s full tone transform that crushed shadows in prior tests. **Adjustment strength** blends this protected result with the original;100%means all of the protected result, and0%restores the exact decoded original. Automatic strength is a provisional scene-statistics rule, not an understanding of photographic intent or a named style selector.

**Independent shadow control:** Natural color now suggests **Shadow lift** from the photo’s brightness distribution, then lets you adjust it from0–100%. It starts at the automatic suggestion (at most75%). The lift concentrates on darker tones, preserves deep blacks and tapers off in brighter tones. It uses common RGB gain to keep the fixed base’s color ratios to rounding; it cannot distinguish dark trees from equally dark clouds or identify photographic intent. Reduce it to preserve a silhouette or moody sky.

**Shadow lift0%** removes only the extra shadow lift, retaining the fixed natural-color result and its mild base brightness. **Adjustment strength0%** restores the exact decoded original; that slider blends color and shadow adjustments together. Shadow changes always start from the same cached natural base and do not rerun the LUT model or accumulate edits. PNG/JSON/native crops/local save reflect the current shadow and overall strength. JSON records the fixed base/model under `natural_color`, the applied brightness and automatic suggestion under `shadow_adjustment`, and the final blend/PNG under `export`. Older cached results must be processed again. See the [shadow-control check](artifacts/experiments/adaptive_lut_shadows_v3/ACCEPTANCE.md).

Full-resolution PNG, matching JSON, **Inspect detail at original size**, and **Save PNG to Downloads** use the same result. The existing **Color and local contrast** controls belong to Natural adjustment; they are not additional controls in this new mode. DNG/WebP are not supported in the new uploader; the project DNG sample is disabled for this mode. Inference stays local and requires the verified model prepared with `scripts/prepare_adaptive_lut.py`; it does not download weights during photo processing. Existing formats/recipes and optional manual/research workflows remain available.

The [protected-color check](artifacts/experiments/adaptive_lut_natural_v2/ACCEPTANCE.md) records12previously inspected scenes, native/synthetic controls and actual50MPworkers. No new all-black/full-channel pixels appeared in these checks; gray ramps retain neutrality/order. This remains an opt-in experiment: existing source blur/noise/clipping remain, atmospheric silhouettes may stay dark, and portrait/general-camera/user-preference acceptance is incomplete. It does not denoise, sharpen, reconstruct texture, detect semantic masks or supply landscape/portrait/film presets.

Standalone JPEG-only reproduction:

```bash
venv/bin/python -m src.lut_natural path/to/input.jpg --output natural-color.json --preview-source natural-color.npz
```
