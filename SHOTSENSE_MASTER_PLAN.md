# ShotSense V3.1: AI Photo Adjustment Recommendations — Revised Implementation Plan

## Current status and plan review (2026-10-02)

- [x] Reviewed all existing source, 5000 DNGs, expert JSON, read-only Catalog queries, and official documentation for key libraries; revised data contracts, task order, and acceptance gates.
- **Implementation includes training, ONNX, and the local application**: the isolated environment, label audit, color pipeline, cache, Dataset, model, and 20-photo pilot passed acceptance. Full support auditing/preprocessing completed with N=4946. The production model, FP32 ONNX, Torch-free inference, and sample/upload/download workflow passed acceptance.
- **Label repairs**: selected one nonempty record per photo/expert from 50000 joined Catalog rows. Repaired 43 fields, quarantined 3 photos without absolute WB, and retained 4997 candidates. Named history records prove omitted zero defaults for Saturation/HighlightRecovery. All expert records were confirmed as PV2003.
- **Reproducible environment**: isolated venv, locked dependencies, locally cached official EfficientNet-B0 weights, and successful `pip check`. Source and model hashes identify versions.
- **Validation**: 26 checks passed after English localization. Maximum production PyTorch/ORT normalized difference was 3.21e-7; offline/online features and JPEGs for the same DNG were identical. Final test macro-average error was 18.7% below the constant baseline. Exposure/highlight recovery are validated recommendations; the other four fields are experimental.
- **Delivery**: local recommendations MVP at http://127.0.0.1:8501/ with JSON export. Contrast/Saturation/Temperature/Tint remain experimental. Faithful Lightroom rendering, modern parameter mapping, and practical error calibration remain incomplete. See `artifacts/model/ACCEPTANCE.md`.
- **JPEG input extension**: JPG/JPEG uploads use a separate experimental rendered-image path with EXIF orientation and ICC-to-sRGB handling. No JPEG outputs are validated; Temperature/Tint are unavailable and automatic preview estimates are opt-in. This does not expand the original RAW model's accuracy acceptance. See `artifacts/jpeg/ACCEPTANCE.md`.

### Next model research cycle

- [x] Created and reviewed [the detailed model improvement plan](MODEL_IMPROVEMENT_PLAN.md), including evaluation leakage controls, grouped splits, matched baselines, geometry experiments, bounded optimization, conditional fine-tuning, and promotion gates.
- [x] Implemented validation-only experiments, independent split/training seeds, protected run directories, explicit recorded final evaluation, and conservative reviewed photo grouping before new full-data training.
- [ ] Run the staged experiments and three-seed confirmations using validation only.
- [ ] Freeze a candidate, perform recorded final evaluation, and promote only after deployment acceptance.

The production model and its historical results remain the current release. Proposed model experiments and thresholds are not completed acceptance evidence.

Phase A evidence is in `artifacts/experiments/baseline_manifest.json` and `group_audit_summary.json`. All 4946 inputs were screened; 11 candidate pairs were visually reviewed, 8 merged and 3 rejected, producing 4938 groups. Two merged pairs crossed historical partitions. The new group-exclusive split remains 3957/495/494 with seed 42 and lives separately under local `data/processed/model_vnext/`. Its train-only physical normalization preview was recorded. Grouping is conservative/incomplete, and previously inspected FiveK data cannot establish fresh external acceptance. New experiments select checkpoints on validation P2 and use a training-mean constant baseline; the historical trainer used a training median. Phase B grouped controls have since been trained and reproduced on validation only; no new model has been promoted.

### Review findings and priorities

| Priority | Issue | Resolution |
|---|---|---|
| P0 | Absolute values, Delta, and normalized signs were conflated | Fix absolute targets; require a per-photo baseline for Delta (KI-004) |
| P0 | Contrast/Temperature ranges did not cover valid labels | Expand ranges; never silently clip valid labels (KI-003) |
| P0 | Default rawpy BT.709 encoding was decoded as ProPhoto CCTF | Use explicit linear physical output, verify D50 matrices, lock sRGB encoding (KI-002) |
| P0 | Empty settings overwrote records; partial key matches and generic zero defaults | Repair extraction, deduplication, key boundaries, and default provenance (KI-001) |
| P0 | Legacy highlight recovery was described as modern Highlights | Retain legacy fields; validate cross-version conversion separately (KI-005) |
| P1 | Camera WB is already applied, yet targets are absolute Kelvin/Tint | Add a WB feasibility gate, context, and fallback delivery rules (KI-006) |
| P1 | Physical feature scales differ; splits/statistics were not persisted | Persist photo-level splits and train-only standardization |
| P1 | Freezing gradients was treated as fully freezing the backbone | Keep backbone in eval mode and verify BatchNorm buffers |
| P1 | Resume checked only JPEGs | Validate image/features/config per ID; write atomically |
| P1 | Dynamic INT8 assumed for CNN; entire RAW pipeline promised under 30 ms | Start with FP32; measure before quantization; separate forward/end-to-end timing |
| P1 | CSS/Canvas treated as faithful Lightroom development | Deliver recommendations/export first; validate preview separately (KI-007) |
| P2 | Dependencies differed from the local machine; timm proposed unnecessarily | Reuse torchvision and verify dependencies in an isolated environment |

### Execution order (20-photo pilot completed)

**Dependencies/labels → color correctness → serial pilot → recovery/failure tests → Dataset contract → small parallel comparison → full preprocessing → training baselines.**

1. Repair extraction and produce an audit. Preserve the previous JSON snapshot and original Catalog; confirm field defaults and parameter versions.
2. Correct physical-output gamma, linear resizing, and D50 conversion; migrate `RGB_to_XYZ` and verify API equivalence separately from color correctness.
3. Implement one RAW read/two developments, 224×224 JPEGs, per-photo feature caches, `--limit`, `--workers`, and failure lists.
4. Run a serial pilot on the first 20 sorted candidate IDs. Separately test the 3 known invalid IDs and one simulated failure for filtering/retry; do not count them in the 20 candidates.
5. Implement normalization, Dataset, and small data tests. Repeat the same pilot to verify duplicate-free resume, corruption repair, and version mismatch rejection.
6. Compare serial/parallel values, throughput, and peak memory on identical samples; then process all data and fix the formal split.

## Core Architecture

Recommend six official Catalog parameters from DNGs without expert adjustments. Retain V3.1 supervised dual-branch regression. Labels are expert edit records, not unique aesthetic truth or complete Lightroom recipes.

- **Semantic branch**: torchvision `EfficientNet-B0`, fixed `IMAGENET1K_V1` weights, classifier removed, global pooling to **1280D**. Freeze gradients and keep the backbone in `eval()` while training only the MLP. Prepare/version weights in advance; inference does not download them.
- **Physical branch**: verified linear ProPhoto → XYZ D50 → Lab, producing **132D** descriptive statistics. These do not directly measure sensor exposure or scene illumination.
  - 96D: normalized global 32-bin histograms for each of L/a/b.
  - 18D: L/a/b mean/std in upper/middle/lower spatial thirds.
  - 18D: L/a/b mean/std for L≤33.3, 33.3<L≤66.7, and L>66.7. Occupancy fractions are absent; describe this as luminance-region statistics, not a complete Zone System.
- **Fusion head**: concatenate train-standardized physical features and semantic features: **1280+132=1412D → 512 → 128 → 6**, ReLU hidden layers and Tanh output. If KI-006 requires WB context, explicitly revise the input contract/dimensions rather than inserting it into existing 132D fields.
- **Targets**: arithmetic mean of experts A–E for each parameter in original units, then normalize. Store original-unit means in `Y`, expert values, and `Y_std` for disagreement auditing. Parameter means need not render the mean expert image or any individual expert style.
- **Field order**: `Exposure, Contrast, Saturation, Temperature, Tint, HighlightRecovery`, identical in the model, NPZ, export, and API. UI labels: exposure, contrast, saturation, temperature, tint, and legacy highlight recovery.
- **Initial inputs**: DNGs compatible with the training-development pipeline. Edited JPEG/PNG, additional RAW formats, and modern Lightroom mapping require separate validation.

### Normalization Specification

These are this model's **legacy-process encoding ranges**, not universal ranges for all Lightroom versions. They cover existing valid expert labels. Statistics were recomputed from 4997 repaired valid labels and agree with this table's three-decimal display; exact values are in `label_audit.json`.

| Parameter | Original-unit range | Normalized range | Center mapped to 0 | Expert-mean p1–p99 | Expert-mean min–max |
|---|---|---|---|---|---|
| Exposure | `[-4,4]` EV | `[-1,1]` | 0 EV | -0.344–2.254 | -1.518–3.274 |
| Contrast | `[-50,100]` | `[-1,1]` | 25 | 0–34.008 | -6.8–68 |
| Saturation | `[-100,100]` | `[-1,1]` | 0 | -4.8–16.408 | -15.6–30.8 |
| Temperature | `[2000,50000]` K | `[-1,1]` | 26000 K | 2379.938–8202.301 | 2000–35558.054 |
| Tint | `[-150,150]` | `[-1,1]` | 0 | -21.6–37.808 | -96.4–97 |
| HighlightRecovery | `[0,100]` | `[-1,1]` | 50 | 3.8–68.208 | 0–98.4 |

`norm = 2 * (raw - min) / (max - min) - 1`; inverse: `raw = (norm + 1) / 2 * (max - min) + min`.

- Validate individual expert values and aggregated Y. Missing, NaN/Inf, and out-of-range values must fail/be quarantined with reasons, not clipped and presented as valid truth. Preserve valid high temperatures.
- Valid labels include **20 photos with mean Contrast<0**, **13 with mean Temperature>10000 K**, and **101 individual Temperature values>10000 K**. Previous ranges could not represent them.
- `norm=0` is a range center, not “no adjustment.” Expanding the temperature range changes loss scale; report Kelvin/mired errors separately and consider nonlinear encoding only if validation justifies it.
- Return `recommended_absolute`. Only compute `delta = recommended_absolute - current_absolute` when the caller supplies a same-version baseline. Without one, return no Delta/direction accuracy. Camera WB multipliers are not Lightroom Kelvin/Tint.
- `ParameterNormalizer` has NumPy/Tensor round-trip verification. Future changes must update this specification, code, cache/model versions, and export metadata.

## Data, Color, and Reproducibility Contracts

### Catalog labels

1. Open SQLite read-only and verify expert A–E collection names/IDs. Associate `(source-file ID, expert)`, discard empty settings, and require exactly one valid record. Zero/multiple nonempty records must fail, not arbitrarily overwrite.
2. Match complete keys. Temperature/Tint must not match Incremental or other prefixed fields. Define Custom-field priority when `WhiteBalance=Custom`; otherwise use valid absolute fields.
3. **Only confirmed omitted defaults**: 17390 missing Saturation and 8597 missing HighlightRecovery values in nonempty expert records. Confirm serialization semantics before filling zero and record provenance. No generic missing→zero rule. Temperature/Tint are each missing in 15 records from 3 known invalid photos; relative values cannot replace absolute WB.
4. Audit ProcessVersion, CameraProfile, WhiteBalance, source-record IDs, and default rules. Only **283 of 25000 records explicitly contain `ProcessVersion="5.0"`**; 24717 omit it. Do not infer universal PV2010 without verifying omission semantics.
5. Never modify the original Catalog. Preserve old JSON extraction snapshots, output repaired labels/differences, and fix parser bugs rather than keeping them for provenance.
6. Do not use expert TIFFs, expert WB, or `(default) Input with ExpertC WhiteBalance minus1.5` settings as model inputs. Labels always come from the official Catalog.

### RAW and color

- One `rawpy.imread()` context and two `postprocess()` calls with consistent WB, demosaicing, orientation, cropping, and exposure baseline. Explicitly lock `use_camera_wb=True`, `use_auto_wb=False`, `no_auto_bright=True`, `bright=1.0`, `half_size=False`, highlight mode, and all other output-affecting options. Record rawpy/LibRaw versions.
- **Physical output**: `output_color=ProPhoto, output_bps=16, gamma=(1,1)`. Normalize linear RGB, area-resize to 224×224, then convert XYZ D50→Lab without ProPhoto CCTF decoding. Verify LibRaw/colour-science matrix compatibility rather than assuming D50 from an enum name.
- **Semantic output**: `output_color=sRGB, output_bps=8, gamma=(2.4,12.92)`, resize to 224×224, JPEG quality=95, explicit RGB/BGR handling. Training/inference share JPEG encode/decode, resizing, and ImageNet mean/std; do not mix JPEG and uncompressed inputs.
- rawpy defaults to gamma `(2.222,4.5)`, not automatically ROMM encoding for ProPhoto. The previous ProPhoto CCTF decode of default output was unverified. [rawpy parameters](https://letmaik.github.io/rawpy/api/rawpy.Params.html), [LibRaw parameters](https://www.libraw.org/docs/API-datastruct-eng.html).
- LibRaw 0.22.1 calls this output `ProPhoto D65`; its automatic ICC derives D50 coefficients. Source-derived versus standard ProPhoto matrix maximum coefficient difference was approximately `1.22e-4`. Explicitly document matrices/whitepoint/tolerance and avoid duplicate chromatic adaptation. [Color constants](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/tables/colorconst.cpp), [conversion implementation](https://github.com/LibRaw/LibRaw/blob/0.22.1/src/postprocessing/postprocessing_utils_dcrdefs.cpp).
- Use `RGB_to_XYZ(..., colourspace=..., illuminant=..., apply_cctf_decoding=False)`. Old/new API difference was zero on a small numerical check; that establishes migration equivalence, not old gamma correctness.
- Retain direct sRGB output via the second development. Historical PSNR 23.5 dB lacks a unified encoding/whitepoint baseline and cannot establish unavoidable colour-science conversion loss.
- Initial model inputs resize the whole frame to 224×224, with aspect-ratio distortion documented. This differs from pretrained resize-256/center-crop. Compare aspect-preserving preprocessing only if needed and update training/inference together. [torchvision weights/transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
- Wide-gamut Lab a/b may exceed histogram ranges. Merge overflow into endpoint bins and record fractions, never discard it silently. Means/std use unclipped Lab. Empty luminance regions return zeros; spatial thirds must be nonempty.
- `HighlightMode.Clip` and low-resolution statistics cannot retain all recoverable RAW highlights. Validate highlight parameters against baselines/ablations; never promise recovery of clipped JPEG detail.

### Caches, splits, and standardization

- `metadata.npz`: `X float32 (N,132)`, `Y float32 (N,6)`, `Y_std float32 (N,6)`, `ids Unicode (N,)`. X/Y retain original features/units; Dataset normalizes labels. Load with `allow_pickle=False`.
- IDs are lowercase full filenames; map to original-case paths and reject canonicalization collisions. Sort tasks and aggregate by ID independently of worker completion order.
- Per-ID feature cache/JPEG, parent-process NPZ/report aggregation, temporary files validated before atomic replacement. Resume requires valid image/features/label-source/version; a JPEG alone is insufficient.
- Configuration records label hashes, field/feature order, development/resizing/JPEG options, code/dependency versions. Mismatch explicitly requires rebuild; do not mix versions or introduce a database/complex queue.
- `--limit N` selects the first N sorted candidates, including cached ones. Repeated pilots select the same IDs. Version-compatible pilot caches can be reused for full processing; temporary splits cannot overwrite formal splits.
- Success (including resume), filtered, and failed inputs are mutually exclusive and sum to all inputs. No successful photos means no trainable NPZ. Unresolved failures produce nonzero exit status and a report.
- Fix valid IDs, then split by **original photo**, seed 42, 80/10/10. Persist `splits.json`/data version; A–E cannot cross sets. Group identifiable bursts/near-duplicates when possible, allow approximate ratios, and document unavailable grouping.
- Fit X statistics only on training data. Use scale=1 for zero variance. Save means/scales as model buffers and export them into ONNX; Dataset does not standardize twice. Fixed label ranges cannot be fitted on validation/test data.
- Initial training does not independently augment color, exposure, WB, or spatial statistics. Future augmentation must update both branches and affected labels together.

## Directory Architecture

Implemented structure; `preview.py` is independent and does not change trained inputs:

```text
ShotSense/
├── data/                              # Local only
│   ├── raw/dngs/                      # Original DNGs, unchanged
│   ├── raw/fivek_dataset/raw_photos/   # Read-only Catalog
│   ├── intermediate/                 # Versioned labels and audits
│   └── processed/                    # Images, features, metadata, splits
├── src/
│   ├── color_pipeline.py              # Shared RAW/color pipeline
│   ├── extract_labels.py              # Complete-key parsing and audits
│   ├── preprocess.py                  # Shared features and resumable cache
│   ├── dataset.py                     # Dataset, normalization, fixed splits
│   ├── model.py                       # Frozen backbone and MLP
│   ├── train.py                       # Baselines, training, checkpoints
│   ├── export_onnx.py                 # FP32 export and parity checks
│   ├── inference.py                   # Torch-free online inference
│   └── preview.py                     # Independent approximate rendering
├── artifacts/                         # Models, evaluation, development evidence
├── app/streamlit_app.py                # Local application
├── scripts/                           # Data preparation and preview comparisons
├── tests/                             # Data, model, inference, and UI checks
├── ANTIGRAVITY.md
├── SHOTSENSE_MASTER_PLAN.md
└── requirements.txt
```

## Phases and Milestones

### Phase 1: Reliable labels, color correctness, and resumable data processing

- [x] Original 5000 DNGs and Catalog available; IDs aligned.
- [x] GATE 3 prototype: 5000×5×6 numeric records extracted; formal semantic audit completed (`label_audit.json`).
- [x] GATE 1/2 prototype: rawpy/colour-science/132D implementation present; numerical color/RAW-context parity passed (`color_audit.json`).
- [x] **P1-A Environment**: correct Torch/OpenCV requirements, reuse torchvision, remove unused dependencies, retain one OpenCV package. Verify conflicts, two RAW developments, and model imports in an isolated versioned environment.
- [x] **P1-B Labels**: deduplicate and parse complete keys, confirm omitted defaults/ProcessVersion, audit differences/quarantines/reasons, recompute normalization statistics.
- [x] **P1-C Color**: explicit linear physical/sRGB output; black/white/gray/color checks for conversion/gamma with neutral a/b near zero. Compare shared two-development context against separate contexts for at least one DNG; record matrices/tolerances/versions. Separate API parity from color acceptance.
- [x] **P1-D Serial pilot**: per-photo caches, atomic writes, filtering, ID alignment, `--limit 20 --workers 1`. Handle errors throughout development/conversion/features/writes.
- [x] **P1-E Data tests**: normalization limits/round trips, valid negative Contrast/high temperatures, missing-value rejection, Tensor/batch shapes, finite values, JPEG size/RGB order; idempotent resume, corruption repair, version mismatch rejection, unique IDs.
- [x] **P1-F Parallel pilot**: macOS spawn, module-level worker, `__main__` guard, `--workers`. Compare 1/2/6 workers with controlled OpenCV/OpenMP threads; choose defaults from throughput/peak memory, not core count alone.
- [x] **P1-G Full data/split**: account for all 5000 inputs, resolve failures, verify complete caches, fix filtering/version, record final N, persist mutually exclusive formal splits.

**Phase 2 entry**: defaults/version semantics confirmed, color numerical checks and pilot/recovery tests passed, full failures resolved, formal splits fixed. Generated files/correct shapes alone are not acceptance.

### Phase 2: Baselines, dual-branch training, and effectiveness

- [x] **Training baselines**: per-field training median, simple physical-feature linear regression with intercept (prefer NumPy), semantic/dual comparisons on identical splits. Store expert disagreement without calling it calibrated confidence.
- [x] **WB gate (KI-006)**: camera/scene Temperature/Tint strata versus constants and available As-Shot baselines. Add online-available context/update input contract if needed, or deliver only non-WB fields beating baselines. Currently only Exposure/HighlightRecovery are validated; four fields remain experimental.
- [x] **Model**: 1412D MLP, train-only X buffer standardization; verify `(B,6)`, finite values, Tanh range, unchanged frozen weights/BatchNorm buffers.
- [x] **Training checks**: fixed Python/NumPy/Torch seeds/device; small-batch overfit check for gradients/pairing before full training. Overfitting is not generalization evidence.
- [x] **Training loop**: equal mean of per-field Smooth L1/Huber on normalized targets, AdamW, MLP only initially. Record validation-selected learning rate, batch, epoch cap, and patience.
- [x] **Model selection**: best validation six-field macro normalized MAE. Save field order, mappings, X statistics, development/data/split/weights versions, and seed. One final test after selection; further tuning after failure requires a new final-evaluation arrangement, not repeated test peeking.
- [x] **Evaluation**: per-field original-unit MAE/RMSE, mired temperature MAE, tails/camera/scene errors and counts. Add Delta three-way direction metrics only with reliable matching baselines; report zero tolerance/class balance/coverage.

**Phase 3 entry**: validation macro error beats the training median; each claimed field beats its own constant baseline, with simple-regression comparison reported. On failure inspect labels/scales/WB context/information loss before enlarging the network. Practical absolute-error calibration remains pending; call this a research prototype.

### Phase 3: FP32 ONNX, parity, and measured deployment

- [x] FP32 dual inputs: `image float32 (B,3,224,224)` and `physical_raw float32 (B,132)` → normalized absolute values in fixed order. Include X standardization in the graph; record opset/input names/dynamic batch/provider.
- [x] ONNX checker and PyTorch/ORT batch/boundary parity. Initial maximum normalized FP32 tolerance `1e-4`; investigate violations before revising tolerance. Check denormalization and single/batch results.
- [x] Shared development/features/JPEG codec/mapping; identical DNG offline/online X/image/prediction comparison. Inference acceptance disables network and torch/torchvision; RAW processing still needs rawpy/colour-science/NumPy/image libraries.
- [x] FP32 benchmark: hardware/provider/threads/batch=1, 10 warmups, at least 100 timed runs, p50/p95/size. **Under 30 ms is a model-forward target**, not a RAW end-to-end promise. Measure read/two developments/features/JPEG/end-to-end separately without substituting cached timing.
- [x] **Quantize only if needed**: production FP32 18.30 MiB, CPU 4 threads p50=18.66 ms/p95=19.39 ms meets current needs; no INT8. If measured needs change, prefer training-calibrated static QDQ INT8 for CNN; evaluate dynamic INT8 for MLP only. Require provider support, ≤2% relative per-field MAE degradation (numerical tolerance if baseline error=0), and measured size/latency benefits or target compliance; otherwise retain FP32. [ORT quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

**Phase 4 entry**: framework/preprocessing parity, complete model/config bundle, reproducible environment, measured quality/timing. INT8 and under 30 ms are not unconditional promises.

### Phase 4: Local recommendations MVP and independent preview acceptance

- [x] Streamlit invokes local inference: DNG upload→recommendations→baseline display→structured download. Add FastAPI only for a real remote multi-client need.
- [x] Validate file type/size/decoded pixels, bounded RAW worker, temporary cleanup, clear loading/failure states. Configure limits from memory/timing evidence, not extension checks alone.
- [x] Fields/units/version semantics/model/data versions/validation state; absolute suggestions without current parameters. Applying values sets them rather than repeatedly adding Delta.
- [x] JSON first. XMP/Lightroom import only after validating fields/legacy mapping; never rename HighlightRecovery to Highlights2012/modern Highlights.
- [x] **Approximate preview**: independent linear 16-bit RAW → aspect-preserving maximum-edge-1600 preview, default highlight protection, strength/applied values, PNG/JSON alignment; numerical/application/three-DNG offline checks passed. Model inputs unchanged. See `PREVIEW_IMPROVEMENT_PLAN.md` and subsequent v3 records.
- [ ] **Faithful preview gate**: custom curves are not Lightroom/PV2003 equivalent. Validate a full renderer, matching configuration, and RAW highlight reconstruction separately. Saved browser permissions block latest real-browser acceptance.
- [ ] Representative faithful single/combined adjustments and idempotent application. Experts also edit Brightness/Shadows/FillLight/ToneCurve/CameraProfile and other unmodeled fields; complete expert TIFF pixel differences cannot alone validate six-field predictions.

**MVP complete**: Phases 1–3 passed, upload/recommend/download/error handling work, scope/legacy semantics are clear. Faithful high-resolution development and modern Lightroom mapping remain incomplete.

## Known Issues & Decisions

### KI-001: Label boundaries, empty settings, and omitted defaults

- Invalid IDs: `a3131-ke_.dng`, `a3214-ke_-8375.dng`, `a3741-ke_-8337.dng`; retain actual path casing. Cause: missing absolute WB, not PV2003 without evidence.
- Reverse the earlier filter-only decision: fix parsing/record selection, retain old snapshots, then quarantine photos still lacking absolute WB. Temperature<1000 K is a known-error guard, not complete quality validation.
- Complete JSON keys do not prove correct defaults/versions. Repair/audit completed: 43 changed fields, 3 quarantined photos, previous snapshots retained.

### KI-002: Keep dual development; redo color acceptance

- Keep one read/two developments, explicit linear physical output/sRGB encoding, and separate API/color acceptance.
- PSNR 23.5 dB does not establish unavoidable conversion loss. Earlier “strict pipeline complete” GATE 1/2 claims were downgraded to prototype completion.

### KI-003: Cover valid labels in normalization

- Replace Contrast `[0,100]` and Temperature `[2000,10000]` with current table ranges. Do not clip labels to p1–p99; report tails separately.
- Temperature center 26000 K is an encoding consequence, not neutral WB. Decide on mired/nonlinear encoding using validation errors.

### KI-004: Absolute parameters and Delta

- Targets are original-unit arithmetic expert means. Delta requires a same-version caller baseline; no baseline means no direction accuracy.
- Tanh sign/range center cannot establish adjustment direction.

### KI-005: Legacy recovery versus modern Highlights

- Labels are HighlightRecovery. Process controls/algorithms differ; renaming/range changes are insufficient. [Adobe process versions](https://helpx.adobe.com/ie/lightroom-classic/desktop/process-and-develop-photos/develop-module-options.html), [legacy differences](https://helpx.adobe.com/sk/archive/lightroom/lightroom-5-troubleshooting.pdf).
- Verify Catalog process semantics first; validate modern compatibility and faithful previews independently.

### KI-006: WB context and target leakage

- Inference: applied camera WB removes part of the original color cast. Existing pixel features omit WB baselines and cannot guarantee recovery of absolute Kelvin/Tint; verify capabilities by strata.
- Catalog `InputAsShotZeroed` collection `943690` has 5000 unique nonempty records: **3395 explicitly contain Temperature/Tint, 1605 omit them**. Use it to investigate baselines, not to assume universal coverage or fill missing values using expert WB.
- Phase 2 must evaluate the WB gate. Added context must have matching training/online definitions and update caches/model inputs. Earlier fallback was to deliver validated non-WB fields and keep WB experimental; actual validation approved only Exposure/HighlightRecovery. Never improve metrics using unavailable expert inputs.

### KI-008: Unsupported missing-camera-WB inputs

- Full development found 51 DNGs with camera WB `[0,1,0,0]` from older cameras. Implicit LibRaw AWB violates the training contract. Reject consistently online/offline; do not add unvalidated fallback/expert WB.
- Separate support audit only classifies this confirmed violation as unsupported. Recheck per-file LibRaw headers, retain IDs/camera/file metadata/original failure report/selection version. Unknown decode/numerical/I/O errors must still fail.
- Exclude 51 from 4997 candidates → N=4946. With 3 invalid-label photos, 54 inputs are quarantined. Retain existing successful cache contracts/hashes; original data/color algorithms remain unchanged. Create the formal split after support is fixed.

### KI-007: Deployment and rendering boundaries

- Separate model and RAW→result timing; quantize based on measured benefit. Unverified CSS/Canvas rendering is approximate.
- Six fields are not a complete expert recipe; clipping/downsampling limit recovery.

### Environment and dependency checkpoints

- Verified isolated Python 3.9.8, NumPy 1.26.4, rawpy 0.27.0/LibRaw 0.22.1, colour-science 0.4.4, opencv-python-headless 4.11.0.86, Torch 2.8.0/torchvision 0.23.0, onnx 1.19.1/onnxruntime 1.19.2, Streamlit 1.50.0. Full environment locked in `requirements.lock.txt`; `pip check` passed.
- Same 20-photo pilot: 1/2/6 workers achieved 1.072/1.813/3.053 photos/s and process-tree peak RSS 0.82/1.33/2.75 GB, with identical outputs. One worker for pilots, explicit six for full processing, nested threads limited.
- Before Git initialization, source/label/config hashes and snapshots identified versions. Git was initialized and uploaded on 2026-10-02; do not invent historical commit IDs.

## Progress Log

Historical entries preserve what was recorded at the time. Earlier strict-color completion claims, normalization ranges, filter-only parsing decisions, and PSNR explanations were superseded by the contracts above.


- **2026-07-21 — Project initialization**: Created the master architecture/tasks and the EfficientNet-B0 1280D + physical 132D fusion (1412D). Initialized Antigravity synchronization/logging rules.
- **2026-07-25 — V3.1 Ground-Truth Parameter Supervision**: Replaced incomplete Kaggle JPGs with the official approximately 50 GB MIT-Adobe FiveK dataset. Parsed `fivek.lrcat` labels (GATE 3), prototyped rawpy/colour-science color processing (GATE 1/2), and implemented 132D features. Later review corrected the original color-completion claim.
- **2026-09-12 — Phase 1 audit/plan revision**: Audited code/data/environment. Found IncrementalTemperature partial matching in 3 PV2003 photos (0.06%; then planned preprocessing filtering), deprecated `RGB_to_XYZ` API with verified migration parity, one-context/two-development support, and multiprocessing compatibility with observed 1.9× speedup on 8 cores. Planned six-field normalization from 25000 labels, Dataset, and data tests. Later review corrected filter-only handling and conversion-loss interpretations.
- **2026-10-02 — Resume status review**: Confirmed 5000 DNGs/25000 expert records/3 known invalid photos. JPEG/metadata, Dataset/model/training/application were not implemented yet. Added 20-photo pilot→tests→full preprocessing order and acceptance, clarified absolute/Delta and dependency discrepancies. Only plan/read-only work completed.
- **2026-10-02 — Plan corrections**: Verified code/full labels/read-only Catalog, 25000 empty settings, omitted defaults, 20 negative Contrast means, 13 Temperature means>10000 K, 15 missing-absolute-WB records, and incomplete As-Shot baselines. Verified new/old `RGB_to_XYZ` parity and default gamma/ProPhoto decode mismatch. Revised absolute/legacy semantics and data/training/export/app gates; implementation still pending.
- **2026-10-02 — Implementation/pilot acceptance**: Built isolated environment/locks, top-level Catalog parsing/default evidence, linear LibRaw→D50 Lab, resumable caches, Dataset/splits, frozen-backbone training, FP32 ONNX, Torch-free inference, and local page. 18 checks passed; serial/parallel pilot outputs matched. Full six-worker preprocessing started; final training/strata/deployment/UI acceptance pending.
- **2026-10-02 — Full data acceptance**: Accounted for 5000 inputs: 4946 successful, 3 invalid labels, 51 unsupported missing camera WB (28 DCS460D, 17 PowerShot S70, 6 EOS D30). Rechecked LibRaw headers and retained original failure reports. Color algorithms/AWB unchanged; unknown errors remain blocking. Fixed 3957/495/494 splits and started full training.
- **2026-10-02 — Production model/local MVP acceptance**: Frozen backbone remained unchanged. Dual validation/test macro normalized MAE 0.066111/0.067113 versus constant 0.081530/0.082541. Exposure test MAE 0.281 EV, recovery 7.428; only those two passed the gate. Recorded camera/Catalog-tag/tail/3395-available-As-Shot evaluation. FP32 ONNX 18.30 MiB, parity 3.21e-7, CPU p95 19.39 ms, no quantization. 22 tests and sample/upload/JSON download passed. Sample local job approximately 2.00 s, RAW→result approximately 0.69 s. Faithful rendering/modern mapping/practical calibration incomplete.
- **2026-10-02 — Browser workflow**: Real DNG upload→recommendation→downloaded JSON verified fields/model version. Replacing input cleared old results; corrupt DNG failed without stale parameters. Restored sample/local app. Saved acceptance, example JSON, screenshot, and exact source snapshot in `artifacts/model`.
- **2026-10-02 — Initial approximate preview**: Added baseline/approximate side-by-side display and 224×224 PNG download for validated exposure/recovery proxies only. Recomputed from baseline without accumulation. Gray-ramp checks covered identity/exposure direction/monotonic highlights/unchanged dark pixels/invalid inputs/repeatability. Two targeted numerical/UI checks passed. Faithful development gate incomplete; saved browser denial prevented latest browser acceptance.
- **2026-10-02 — Preview v2 plan/implementation**: Independent linear RAW source, aspect preservation/max edge 1600, monotonic highlight protection, 0–100% strength, applied values/clipping diagnostics, PNG/JSON alignment. 25 checks passed and Torch/network-disabled RAW preview passed. New full-channel fractions for desert/dark samples fell from 1.3167%/1.7179% to zero; foreground brightened. Three offline samples showed no obvious halos and improved geometry/detail. Model/features/data unchanged. Existing full-image semantics documented; local semantic adjustments deferred for separate validation. Browser/Lightroom gates incomplete.
- **2026-10-02 — Refresh diagnosis**: Old server was running with `runOnSave=false`. Restarted to clear process caches, enabled `runOnSave=true`/`fileWatcherType=poll`, and added `linear-raw-protected-v2` caption. Verified localhost listener/import/config. Refresh alone does not process RAW; regenerate/upload again. Browser restrictions were not bypassed.
- **2026-10-02 — Multi-scene preview check**: Tested 12 DNGs: 8 training-scene checks plus 4 held-out test photos, covering portraits/dark interiors/night/backlight/snow/sunset/textiles/lake. Saved baseline/100%/75% comparisons and JSON. All zero-strength PNGs exactly matched baselines; neither strength introduced full-channel pixels. Offline review found improved portrait/interior/night visibility, limited backlit-subject lifting, and flat gray clipped clouds. Did not interpret clipping reduction as texture recovery. Recorded `artifacts/preview_variety/REVIEW.md`; no model/renderer change or population-accuracy claim.
- **2026-10-02 — Preview v3 shoulder/ceiling diagnostics**: Planned first, preserved v2 source/12 comparisons, then implemented `linear-raw-shoulder-v3`. Linear midtones, continuous monotonic shoulder, recovery proxy knee 0.8. Page/JSON diagnose developed display-source channel/white ceilings, not sensor overexposure/recovery. 26 checks passed. Twelve regressions plus four unseen random test photos had zero new full-channel pixels at 100%/75%. Improved portrait/snow/bark midtones and reduced gray cloud blocks; backlight/missing texture limitations remain. Saved code/images/reports/scripts in `artifacts/preview_v3`; model/preprocessing unchanged. Restarted app; latest real-browser acceptance not performed.
- **2026-10-02 — GitHub preparation**: User supplied `git@github.com:Evannn888/ShotSense.git` and authorized upload. Verified empty target and working SSH; initialized Git, excluding raw data/environment/secrets/training caches/duplicate PNGs. Included model/source/acceptance/development/comparison JPGs/source snapshots; documented clone/data requirements.
- **2026-10-02 — Initial GitHub upload**: Pushed `3d0b7fd6899d19d6809a5b1e2fd2e1d270ccaedd` to `main`, verified matching remote/local SHA. 98 files, approximately 46.43 MiB. Original data/environment stayed local. Staged formatting/size/credential-pattern checks passed; runtime unchanged, prior 26 passing checks retained. Upload record committed as `4c79d10ae4dda98a0a4a4352466dc7bc33ce2a68`.
- **2026-10-02 — English localization in progress**: User requested English throughout the app and GitHub project. Translate current UI/docs/development records/review JSON; preserve numerical contracts and all milestone history. Keep original Chinese screenshots/immutable ZIPs locally with hashes in an English provenance record instead of falsifying historical evidence. Update the GitHub About description and retain prior Git history.

- **2026-10-02 — English localization verified**: Translated all current tracked UI/documentation/development milestones/review JSON to English; app sample, strength, warnings, and downloads passed in the full 26-test suite. No Chinese text remains in tracked text files. Verified unchanged ONNX/checkpoint/preprocessing/renderer bytes. Saved original ZIP/screenshot hashes and retained originals locally/earlier Git history; created a separate English source snapshot. GitHub About description saved in English and verified in the signed-in page. Local app will be restarted before delivery; latest real-browser localhost acceptance remains blocked.

- **2026-10-02 — English GitHub publication verified**: Localization commit `82d4b14194b2adbdd59ea0aa0b16dfb89d3c9717` was pushed to `main`. The signed-in GitHub page visibly shows the English README and English About description. Saved `artifacts/github-english-verification.jpg` as browser evidence. The local English app was restarted and its localhost listener verified; full 26-test integration coverage passed, without bypassing the saved localhost browser restriction. Historical milestones remain translated in the current plan and originals remain retrievable from prior commits.

- **2026-10-02 — Detailed model improvement planning:** Reviewed current training/test access, group-split support, coupled split/training seeds, cached semantic features, and preprocessing/deployment contracts. Created `MODEL_IMPROVEMENT_PLAN.md` with validation-only experiments, grouped matched baselines retrained from ImageNet, geometry-first comparisons, bounded head search, conditional partial fine-tuning, three-seed confirmation, uncertainty, proposed promotion gates, and later WB/style tracks. Linked the roadmap from README. Documentation only; no training, split/cache mutation, or model replacement.

- **2026-10-02 — Model research Phase A acceptance:** Saved a compact production baseline manifest at revision `6d4fa17ce71747f09cd3facc003c1322f6dccd11`. Added validation-only experiments, P2 checkpoint selection, separate split/training seeds, exposed group/split paths, non-overwriting run directories, failure/config records, validation predictions, frozen-artifact checks, and explicit one-time final evaluation. Synthetic checks prove no test metrics during training, stable grouped splits across seeds, rejected run overwrites/artifact tampering, and logged final access. Screened 4946 unedited previews/source metadata, reviewed 11 candidates, merged 8/rejected 3 into 4938 groups; 2 accepted pairs crossed historical partitions. Saved a separate 3957/495/494 split and train-only normalization preview, with camera coverage and incomplete-grouping limitations. Full suite: 29 passed; original production RAW/ONNX/offline inference still works, and model/preprocessing/original cache/split hashes are unchanged. No new full-data training, production replacement, or fresh external acceptance claim.

- **2026-10-02 — Model research Phase B acceptance:** Added developed-Lab-lightness strata with training-only tertile thresholds and reports for all four baselines. Trained grouped seed-42 controls from official ImageNet initialization with fixed split seed 42, P2 checkpoint selection, and frozen backbone/BatchNorm. Validation dual Exposure/HighlightRecovery MAE 0.277752 EV/7.108073, P2 0.052900 versus training-mean constant 0.078813 and physical linear 0.055715 (32.9%/5.1% lower). Contrast/Saturation still fail to improve on constant. Replayed both heads with identical training histories and zero prediction difference. Full run 288.44 s, peak RSS 1379155968 bytes; 30 tests passed. Recorded experiment ledger/configuration/hashes/full metrics/reproduction/acceptance. No test metrics, production replacement, or fresh external acceptance; candidate geometry and seeds 43/44 remain pending.

- **2026-10-02 — Experimental JPEG support and Phase C input pilot:** User requested JPG/JPEG support while the geometry experiment was active. Added a separate bounded, Torch-free rendered-JPEG worker with content/pixel/size checks, EXIF orientation, ICC-to-sRGB conversion, explicit sRGB assumption, and unavailable Temperature/Tint. All JPEG estimates are experimental; the English page uses opt-in exposure/highlight preview and matching PNG/JSON downloads. Retained DNG model/core preprocessing bytes and added explicit versioned candidate geometry routing. Synthetic/backend/AppTest checks passed, with 39 full-suite checks at this milestone and additional mixed-cache/router checks passing afterward. Independently verified 20 letterboxed RAW inputs: physical features exactly match originals, cached and online helper inputs agree, and mixed caches/unknown versions fail. Full 4946-photo candidate cache rebuilding is in progress; no geometry-model result or promotion yet. Browser permission was not bypassed.

- **2026-10-02 — JPEG workflow verification:** Full suite passed 40 checks; JPG/JPEG worker paths, EXIF/ICC handling, unavailable WB fields, experimental-only outputs, default identity/opt-in preview, and PNG/JSON downloads passed. Saved `artifacts/jpeg/ACCEPTANCE.md` and synthetic fixture JSON; documented English UI/CLI usage. Local Streamlit listener and automatic file watching remain active. Production checkpoint/ONNX, core RAW preprocessing/decoder, and Phase B frozen artifacts are preserved. Full letterbox cache rebuilding continues; no candidate model result or real-browser acceptance is claimed.

- **2026-10-02 — Phase C completed, stretch retained:** Rebuilt all 4946 letterbox inputs with zero failures and exactly unchanged physical vectors; retained Phase B split memberships/settings. Seed-42 dual validation P2 0.052948 versus stretch 0.052900 (0.09% higher error); Exposure MAE 0.282241 EV and HighlightRecovery MAE 7.061645. Paired 10000-resample validation-group bootstrap relative improvement interval −2.65% to +2.42% crosses zero; no population regression claim. Semantic-only improvement did not satisfy the dual gate, so retained stretch and did not trigger three-seed confirmation. Exact head replay passed with zero prediction difference. Saved acceptance, full metrics/configuration/hash manifests, preprocessing resources, uncertainty summary and ledger. Moved the builder's Dataset import to the parent entry point to avoid Torch imports in preprocessing workers; no new speed claim. Final full suite: 40 passed, one existing optional Matplotlib warning. Verified production/core RAW processing, original data and both frozen runs remain intact; no final test access or production replacement. JPG/JPEG workflow remains experimental, WB unavailable, real-browser/JPEG-accuracy acceptance incomplete.

- **2026-10-02 — Verification and Phase D bounded head search:** User requested tests/results inspection and proceeding if verified. Rechecked all production/core RAW/original data hashes, Phase B/C frozen artifacts and replay/source integrity; pip check passed and the existing 40 tests passed. Reviewed four actual training-photo semantic JPEGs: decoded baseline and zero-strength PNGs exactly match input RGB, no new full-channel pixels under experimental preview, and no obvious thumbnail halos; backlit subject remains dark. These are workflow checks, not JPEG accuracy or full-resolution acceptance; real-browser restrictions remain respected. Documented the Phase D protocol before implementation, reused the native trainer and verified embeddings, reused exact H1 seed-42 control, and completed H2/H3/H4 seed-42 screens plus H1 seeds 43/44. H1 remains best (P2 0.052900 versus 0.053309/0.053130/0.053063); H1 three-seed mean/sample SD 0.052239/0.000626. Winner/control are identical runs, so zero improvement/bootstrap interval reflects identity rather than independent equivalence evidence. No alternative passes the improvement gate. All five new runs exactly reproduced dual/semantic histories and predictions; backbone/BatchNorm and frozen hashes preserved. Full final suite: 41 passed, one existing optional Matplotlib warning. Saved complete metrics, configuration/split/replay/hash reports, summary, ledger, JPEG comparison and acceptance. Retained control; no final test access, export, production replacement or automatic backbone fine-tuning. Phase E requires separate failure-analysis justification.

- **2026-10-02 — JPEG unchanged-preview usability correction:** User screenshot showed nearly identical JPEG before/after despite 100% strength. Code review found strength remained active while default opt-out produced zero applied parameters, and the result caption implied adjustment. Now explicitly explain inactive JPEG adjustment, disable inactive strength/protection, and label byte-identical previews unchanged. Added optional manual JPEG Exposure/HighlightRecovery controls that override experimental estimates, export the selected adjustment mode/effective values, and reset with new input. AppTest verifies inactive status and manual +1 EV changes rendering without changing model recommendations. Full suite: 42 passed, one existing optional Matplotlib warning. Local Streamlit listener and automatic polling reload verified; real-browser acceptance remains blocked and JPEG model accuracy remains unvalidated. Production model and renderer unchanged; no automatic tuning or promotion.
