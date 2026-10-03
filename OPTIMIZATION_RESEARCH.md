# ShotSense optimization research and next experiments

Research date: 2026-10-02. Scope: existing ShotSense evidence plus selected primary papers and author implementations, including 2024–2026 work. This is a targeted engineering review, not an exhaustive literature survey or proof that any proposed method will improve ShotSense. Runtime/model/data remain unchanged.

## Recommendation

Prioritize a separately defined rendered-JPEG enhancement task, because the user's visible goal is a better-looking adjusted image. Retain RAW Catalog-parameter prediction as a separate capability. Before investing in a larger network, run one bounded RAW loss-alignment experiment and strengthen its regularized baselines. For the JPEG path, begin with a small controllable global curve model, then an adaptive LUT only if the curve model's failures justify it. Regional models come after evidence shows that global transformations are insufficient.

## What our own evidence establishes

| Observation | Evidence | Interpretation |
|---|---|---|
| Letterbox did not improve dual validation P2 | [Phase C](artifacts/experiments/model_vnext/PHASE_C_ACCEPTANCE.md) | Geometry preservation is useful for display but has no demonstrated accuracy benefit in this controlled model experiment. |
| Original head settings remained best in the fixed grid | [Phase D](artifacts/experiments/model_vnext/PHASE_D_ACCEPTANCE.md) | More of the same learning-rate/decay search has weak justification. |
| Last-stage tuning improved three-seed mean P2 only 0.245%; interval −0.398% to +0.912% | [Phase E](artifacts/experiments/model_vnext/PHASE_E_ACCEPTANCE.md) | A larger/unfrozen network is not the next supported investment. This does not prove that every other architecture would fail. |
| Control training loss fell from 0.02664 at best epoch 3 to 0.001705 at epoch 23, while validation P2 rose from 0.05290 to 0.05512 | [Control history](artifacts/experiments/model_vnext/control-seed42/evaluation.json) | Consistent with generalization/overfitting limits; additional training alone is not supported. Loss and P2 are different objectives, so this is not a causal diagnosis. |
| Dark validation Exposure bias is only −0.037 EV | [Failure analysis](artifacts/experiments/model_vnext/failure-analysis/summary.json) | Do not uniformly brighten every dark scene to compensate for an unproven model bias. |
| Training optimizes equal six-field SmoothL1, while selection/release prioritize two fields | [Trainer](src/train.py) | Loss alignment is a concrete testable hypothesis. It is not an established explanation of current error. |
| JPEG uses a RAW-trained model after display-sRGB linearization and returns legacy absolute outputs | [JPEG implementation](src/jpeg_inference.py) | JPEG processing is already baked in; these outputs are not learned JPEG correction deltas. Inverse transfer encoding does not recover the original RAW or editing baseline. |
| Current preview only implements global exposure/highlight proxies | [Renderer](src/preview.py) | Accurate six-field prediction would still not reproduce complete expert development. Global changes cannot independently brighten a subject and leave an equally dark background untouched. |

The current 47 passing tests establish workflow, invariants and computation parity. They do not establish that JPEG estimates are aesthetically correct.

## Relevant research and practical fit

| Method | What the primary source supports | ShotSense implication and limitation |
|---|---|---|
| [FiveK original study](https://people.csail.mit.edu/vladb/photoadjust/) | Photographic adjustment varies by user; supervised examples and personalization were studied. | Averaging five experts is one task, not a unique preferred style. Explicit style conditioning is a future task change requiring matched baselines. |
| [Neural Color Operators, ECCV 2022](https://arxiv.org/abs/2207.08080), [author implementation](https://github.com/amberwangyili/neurop) | Learns sequential color operators with scalar strengths; offers controllable rendered output. | Strong conceptual match for an editable JPEG workflow. Its strengths are learned operator controls, not Adobe slider identities. Inspect color/input contracts and export support before adoption. |
| [Image-adaptive 3D LUT](https://arxiv.org/abs/2009.14468), [author implementation](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | A small image-conditioned predictor combines basis LUTs for color/tone enhancement. The repository supplies lower-resolution paired data and uses custom interpolation code. | A compact full-resolution color transform is plausible. A global LUT applies the same mapping to identical colors everywhere; it does not provide arbitrary independent subject/background editing. Avoid assuming its old custom extension fits our environment. |
| [HDRNet](https://arxiv.org/abs/1707.02880), [author implementation](https://github.com/google/hdrnet) | Predicts local affine transforms on a bilateral grid from a low-resolution input, then applies them to the full image. Its reference stack uses older TensorFlow/custom operators. | A later regional-enhancement direction. Modern runtime/export compatibility and boundary artifacts need separate engineering; the published speed does not predict Mac/ONNX speed. |
| [Zero-DCE](https://arxiv.org/abs/2001.06826), [Zero-DCE++](https://arxiv.org/abs/2103.00860) | Learns image-dependent low-light curves using non-reference losses; paired reference targets are not required by this training formulation. | Useful optional low-light baseline when paired targets are unavailable. Training still requires data and designed losses; it is not a universal aesthetic model or a clipped-detail reconstruction guarantee. |
| [ICELUT, 2024](https://arxiv.org/abs/2403.19238) | Converts learned pointwise/global components into lookup tables for efficient deployment. | A later efficiency option after a JPEG quality baseline exists. Author hardware timings are not measured ShotSense latency. |
| [Exposure-slot, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jung_Exposure-slot_Exposure-centric_Representations_Learning_with_Slot-in-Slot_Attention_for_Region-aware_Exposure_CVPR_2025_paper.html) | Learns region-aware exposure representations with hierarchical slot attention. | Supports considering regional exposure representations if simple global models fail; it does not justify immediately importing a more complex attention model. |
| [PPR10K, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Liang_PPR10K_A_Large-Scale_Portrait_Photo_Retouching_Dataset_With_Human-Region_Mask_CVPR_2021_paper.html) | Portrait data include human-region masks, multiple expert edits and photo groups, emphasizing human regions and consistent group tone. | Relevant only if portrait retouching becomes a declared product focus. Human masks do not cover pets or general object subjects. Group-exclusive evaluation remains necessary. |
| [Illuminant-adaptive camera LUTs, July 2026 preprint](https://arxiv.org/abs/2607.11681) | Studies illuminant-aware camera color correction into device-independent color coordinates. | Relevant to a later RAW color/WB context investigation, not a demonstrated solution for already-rendered JPEG aesthetics. Preprint results require independent verification. |
| [Acceptable tonal-adjustment ranges](https://projects.csail.mit.edu/acceptable-adj/) | Studies multiple acceptable brightness/contrast renditions using human judgments. | Supports evaluating user preference/acceptability in addition to one reference image. This is inspiration for evaluation, not a calibrated confidence score for our current model. |

These method families are alternatives, not a stack to combine indiscriminately. No paper's PSNR, benchmark speed or reported improvements transfer to our preprocessing/splits without reproduction.

## Priority 1: define a real JPEG enhancement task

### Contract and data

- Input: an already rendered, correctly oriented, color-managed sRGB image. Preserve the current bounded decoder and original image. Any later color-space convention must be explicit and versioned.
- Output: a separate transformed image and its own operator/curve parameters, with an exact zero-strength identity path. Do not label JPEG outputs as absolute Lightroom Exposure/Temperature/Tint or RAW recommendations.
- Supervision: choose one fixed expert/style before the experiment, and use actual compatible input/target renderings. Other expert styles are a later comparison. Do not average five target renderings and assume the result matches a user preference.
- Local inventory found source RAWs and Catalog data, but no standalone paired expert TIFF/JPEG files under `data/raw`. Expert parameter records alone are insufficient to train faithful expert-image enhancement. Obtain compatible reference exports/paired data before claiming supervised JPEG quality.
- Record source IDs, rendering settings, color profiles, bit depth and target style. Keep all variants and expert edits of one source in the same group. Retain disclosed FiveK validation as development data and reserve new personal-camera photos before inspection for external acceptance.
- Public pretrained FiveK models may have trained on our validation/test photo IDs. Until provenance is checked, use them only as quarantined visual prototypes, never as evidence of fresh held-out accuracy.
- Synthetic exposure/gamma/JPEG variants can help robustness or implementation tests, but synthetic defects and our own v3-rendered targets are not independent expert-quality labels. Do not promote a synthetic-only result as real-photo generalization.

### Minimal baseline and progression

1. Establish unchanged-input, current opt-in preview, and a documented deterministic tone-curve baseline on the same evaluation images.
2. Test a small image-conditioned global monotonic curve predictor with bounded controls, exact identity and an independent strength blend. A curve can protect highlights while lifting shadows; it still cannot distinguish two regions with identical input tone solely through the curve.
3. If paired reference data are available, train against the chosen rendered target and compare reference errors plus blind visual preferences. If paired data remain unavailable, assess Zero-DCE/Zero-DCE++ only as an explicitly limited low-light prototype.
4. Try an adaptive 3D LUT only when errors show color/tone capacity beyond the curve model. Check interpolation layout, RGB order, gamut bounds, identity behavior, skin/neutral color shifts and CPU/MPS/ONNX parity.
5. Add an edge-aware spatial branch or subject mask only after fixed review cases demonstrate a global transform failure. Evaluate halos/noise, not just average brightness. Define pets/general subjects separately from human portraits.

### Proposed acceptance protocol, to freeze before implementation

Select a fixed development review set of about 40 photos spanning indoor dark subjects, backlight, normal daylight, night scenes and bright/highlight scenes. These are minimum coverage targets, not currently collected samples. Record unavailable categories. Reserve a separate untouched external set; do not reuse the user's hamster screenshot as independent acceptance after repeatedly optimizing against it.

Report paired PSNR/SSIM and color difference only on aligned, color-compatible reference pairs. Keep those scores separate from RAW parameter MAE/P2. Add shuffled before/after preference judgments (including ties), severe artifact counts, subject visibility and high-resolution crops for noise/halos. Human evaluations need real reviewers; the assistant's own viewing is not a human study. Do not claim statistical preference evidence from a few hand-picked images.

Suggested prototype gates: exact zero-strength identity; finite/range-safe output; no reproducible severe new artifact on the fixed review; documented preference protocol and uncertainty; full-resolution and decoder-to-export consistency; measured CPU and MPS end-to-end latency and memory. Define quality/latency thresholds for this new task before tuning rather than borrowing the RAW model-only 30 ms gate. Insufficient preference/data evidence keeps the feature experimental.

## Priority 2: one bounded RAW objective-alignment experiment

Keep the same original Catalog mean targets, grouped split, stretch input, frozen backbone, normalization, 512→128→6 head, seed 42, optimizer and checkpoint rule. Change only the auxiliary-field weighting in the loss:

`L = mean(SmoothL1[Exposure, HighlightRecovery]) + 0.25 × mean(SmoothL1[Contrast, Saturation, Temperature, Tint])`

Rescale the existing control objective as `L_control = mean(primary) + 2 × mean(auxiliary)` so both objectives have the same primary coefficient. Reproduce that rescaled control on the same seeds instead of treating the original equal-six-field averaged loss as an identical-gradient control; a global loss multiplier can affect optimizer behavior. Freeze this comparison before training. Do not tune a large set of weights after seeing results.

This tests whether auxiliary targets distract from the two delivered fields. It preserves the RAW task and field meanings, but any gain is unknown. Confirm a passing candidate on seeds 42/43/44 with the existing 5%/2%/group-bootstrap gates. Report every auxiliary field and retain experimental status; improving P2 must not imply that all six are accurate. If it fails, stop rather than repeatedly searching loss weights.

Also add a bounded ridge-regression baseline on fused semantic/physical features, with train-only scaling and a small predeclared regularization grid, to test whether the large nonlinear head offers sufficient benefit. This is a stronger baseline than physical-only OLS and can use existing verified embeddings. Treat any learned transformations/scalers as train-only artifacts.

## Priority 3: features, style and usable uncertainty

- Existing global L histograms already encode much brightness occupancy. Merely appending histogram-derived occupancy does not supply new scene information; call it an inductive-bias experiment if attempted. Channel-ceiling statistics, spatial shadow occupancy, RAW WB/camera context and online-available metadata are candidates only after a specific failure analysis supports them.
- Do not equate developed/gamut-clipped channel ceilings with sensor clipping, ISO with a measured noise map, or camera WB multipliers with Lightroom Kelvin/Tint.
- If preferences differ, define expert-specific/style-conditioned targets and matched per-style baselines. Five expert edits per photo are not five independent photos. Personal preferences need held-out personal data.
- Treat expert disagreement and three-model prediction spread as diagnostics, not calibrated per-image confidence. For useful confidence/abstention, reserve a separate calibration partition and evaluate actual coverage on untouched data before attaching numerical confidence labels.

## What to postpone

Full-backbone unfreezing, larger transformers/diffusion, unbounded hyperparameter searches, segmentation for every image and quantization do not address the demonstrated JPEG task mismatch. Quantization may improve deployment cost but does not establish better image quality. Changing to a newer backbone is a later controlled comparison if task-aligned baselines plateau.

## Concrete next deliverables

| Order | Deliverable | Evidence required to continue |
|---|---|---|
| 1 | JPEG input/target/style and rendering manifest; frozen evaluation protocol | Compatible reference pairs or an explicitly limited no-reference baseline; source-group exclusivity and external-set separation |
| 2 | Independent controllable curve prototype with identity/no-change behavior | Numerical invariants, real-image/crop review, matching exports and measured deployment path |
| 3 | Small JPEG curve-model comparison; conditional adaptive LUT | Task-matched baselines, paired/visual metrics and uncertainties, no reproducible severe failures |
| Parallel low-cost track | RAW loss-alignment and fused ridge comparisons | Same target/split and matched controls; three-seed 5% improvement plus field protection |
| Later | Style/context/region adaptation | Failure evidence identifies the specific missing capability |

Estimated effort is one protocol/data-audit session, one curve-prototype session and one bounded model-comparison session once usable target data exist. These are planning units, not guaranteed dates. Missing compatible target images is a real prerequisite for supervised enhancement; it should not be concealed by labeling proxy renderings as expert truth.
