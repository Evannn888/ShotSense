# ShotSense Model Improvement Plan

Date: 2026-10-02. Status: Phases A and B accepted; grouped seed-42 validation reference trained and reproduced. Candidate geometry experiments and release promotion remain pending.

This plan improves parameter prediction. Preview rendering has its own acceptance process in `PREVIEW_IMPROVEMENT_PLAN.md`. A brighter preview does not establish a more accurate model.

## 1. Objective and current evidence

The first objective is to improve Exposure and legacy HighlightRecovery without losing reproducibility or local inference support. Contrast, Saturation, Temperature, and Tint remain experimental until they pass their own gates. Practical error tolerances still require user evaluation.

Current historical benchmark:

| Item | Recorded result or contract |
|---|---|
| Supported photos | 4946 of 5000; 3 invalid-label and 51 unsupported-WB inputs excluded |
| Split | 3957 train / 495 validation / 494 test; photo-ID split, seed 42 |
| Labels | Six absolute PV2003 fields; arithmetic mean of experts A–E |
| Model | Frozen ImageNet EfficientNet-B0 1280D + train-standardized Lab 132D; MLP 512 → 128 → 6 |
| Validation / test macro normalized MAE | 0.066111 / 0.067113 |
| Test Exposure MAE | 0.281 EV; constant baseline 0.409 EV |
| Test HighlightRecovery MAE | 7.428; constant baseline 10.147; physical linear baseline 7.298 |
| CPU FP32 forward p95 | 19.39 ms, four threads, batch size one |
| Deployment parity | Maximum normalized PyTorch/ONNX difference approximately 3.21e-7 |

These numbers describe the existing split, not the future grouped benchmark. The physical baseline already beats the dual model on test HighlightRecovery, so a larger network is not automatically the best choice.

## 2. Fixed decisions for the first wave

1. Keep the six-field order, label extraction, arithmetic-mean target, and normalization ranges unchanged.
2. Use only unedited RAW development and metadata available at inference. Never supply expert WB, expert-rendered images, or Catalog-only edit settings as inputs.
3. Keep the current production bundle and its preprocessing usable. Candidate cache directories, checkpoints, and contracts must be separate.
4. Change one major factor at a time: evaluation protocol → semantic geometry → head optimization → partial backbone fine-tuning.
5. Keep the physical 132D feature calculation unchanged during the geometry comparison. Keep the preview renderer fixed during all first-wave model comparisons.
6. Use the existing dependencies and EfficientNet-B0. No additional model service, new backbone family, segmentation system, or quantization in the first wave.
7. Do not begin a training run until the evaluation protocol and that run's input preprocessing checks pass. The existing stretch-input pipeline is already verified for Phase B controls; Phase C candidate geometry must pass its own checks before candidate training.

## 3. Phase A — Make experiments reliable

### A1. Freeze the current baseline

- [x] Record the Git revision, checkpoint/ONNX hashes, dependency versions, source hashes, labels, supported IDs, original split, and original metrics in a baseline manifest.
- [x] Verify the original bundle still loads and processes a supported DNG after experimental support is added.
- [x] Retain the original source snapshots and acceptance evidence; do not rewrite historical results.

### A2. Audit related photos and establish a grouped split

- [x] Generate duplicate/burst candidates using source metadata where available and inexpensive image similarity from unedited previews. Use existing OpenCV/standard-library tools.
- [x] Review candidate matches before using model errors or expert edits. Similar-looking landscapes alone do not prove that photos belong to the same burst.
- [x] Persist a complete photo-ID → group-ID map, the matching rules, reviewed examples, and unresolved cases. Connected related-photo groups must remain within one partition.
- [x] Create an approximately 80/10/10 group-exclusive split with fixed split seed 42. Report actual photo/group counts and camera coverage; exact ratios are secondary to group isolation.
- [x] If reliable grouping is incomplete, report that limitation rather than claiming all scene leakage has been removed.
- [x] Fit physical feature means/scales on the new training partition only. Verify no overlap by photo ID or group.

The existing split helper supports groups; the training entry point now exposes `--groups`, `--split-path`, and `--split-seed`. Screening used 64-bit difference/perceptual hashes (limits 4/8) and matching camera/capture times within three seconds, followed by visual review. All 11 candidates were reviewed: 8 merges, 3 rejections, 4938 groups. Camera/time metadata was unambiguous for 2348 of 4946 photos. The local candidate map/split and normalization preview are separate from production. Source-camera coverage remains incomplete; timestamps without serial information may be unreliable. See `artifacts/experiments/group_audit_summary.json`.

**Important:** a new split of FiveK is not a genuinely unseen external benchmark. Previous work has already inspected parts of FiveK. Retrain all grouped-benchmark baselines from ImageNet initialization; do not reuse the current ShotSense checkpoint, which may have trained on newly held-out photos. Separately collected personal RAW files and final settings can later supply an external evaluation set.

### A3. Separate validation from final testing

- [x] Make experimental training validation-only by default. The historical trainer computed test metrics and Catalog test diagnostics on every run; that behavior is removed from the experiment path.
- [x] Provide a separate, explicit final-evaluation action that loads frozen candidates. Record each final test access.
- [x] Separate the fixed split seed from the model initialization/data-order seed. Changing training seeds must never change the split or invalidate it accidentally.
- [x] Save training configuration, split/group hashes, preprocessing version, seed, checkpoint-selection rule, validation predictions, original-unit errors, timing, and failure reasons for every run.
- [x] Reject incompatible caches and checkpoints. Use a new directory rather than silently rebuilding or overwriting production files.

**Exit gate:** reproducible group-exclusive partitions; train-only statistics; validation-only experiments; original production bundle still usable. No model-quality claim at this stage.

## 4. Phase B — Establish matched baselines

- [x] Retrain a training-mean constant baseline, physical linear baseline, frozen semantic baseline, and current dual architecture on the new grouped split.
- [x] Use identical labels, split, validation IDs, normalization, and evaluation code across the comparisons.
- [x] Use the predefined validation P2 score for checkpoint selection in both trainable controls and candidates; do not compare checkpoints selected using different objectives as a single-factor experiment.
- [x] Run the dual control with training seed 42 initially and reproduce both trainable heads from ImageNet initialization.
- [ ] Repeat the matched control with seeds 43 and 44 when a candidate reaches the confirmation stage.
- [x] Report Exposure/HighlightRecovery separately, all six original-unit MAEs, the historical six-field macro metric, and camera/brightness/error-tail strata.
- [x] Define strata from input information or training-derived thresholds. Record small sample counts; do not claim subgroup improvement from a few examples.

Phase B evidence: `artifacts/experiments/model_vnext/PHASE_B_ACCEPTANCE.md`, `phase_b_summary.json`, `experiment_ledger.csv`, and the full validation report under `control-seed42/`. Dual Exposure/HighlightRecovery MAE is 0.277752 EV / 7.108073; P2 is 0.052900 versus constant 0.078813 and physical linear 0.055715. This is a single-seed validation reference, not a promoted model. Head replay produced exactly identical training histories and validation predictions; no final test access occurred.

**Exit gate:** a reproducible reference on the new split. The existing 18.7% improvement is historical context and must not be used as the denominator for new-split improvements.

## 5. Phase C — Preserve geometry in the semantic input

The current semantic cache stretches the whole image to 224×224. Portraits and wide scenes therefore have distorted geometry. The high-resolution preview already preserves aspect ratio, but the model input does not.

- [ ] Add a candidate semantic pipeline that fits the complete frame into 224×224 while preserving aspect ratio, with centered padding and a fixed documented RGB padding color based on ImageNet mean values.
- [ ] Apply this before the current square JPEG is created. Padding an already stretched JPEG cannot restore the original geometry.
- [ ] Keep orientation, RAW WB/development, JPEG quality, JPEG decoding, and ImageNet normalization identical apart from geometry.
- [ ] Keep the physical 132D branch on its existing calculation for this controlled experiment.
- [ ] Version the candidate pipeline and cache. Ensure training and online inference use exactly the same resize, padding, JPEG, and normalization behavior.
- [ ] Check wide, square, and portrait fixtures; subject retention; supported RAW decoding; deterministic offline/online input parity; and explicit rejection of mixed cache versions.
- [ ] Rebuild candidate semantic caches from the original DNG development, then train the same head with seed 42 and the same training settings as the matched control.

**Exit gate:** if geometry preservation improves the predefined validation score without unacceptable field regressions, confirm it across three seeds. If it does not, record the result and retain the control geometry. Appearance alone is insufficient.

## 6. Phase D — Optimize the head with a bounded search

Use the geometry selected in Phase C. Keep the 512 → 128 head, six targets, SmoothL1 loss, and frozen backbone fixed initially.

| Screening run | Learning rate | Weight decay | Model seed |
|---|---:|---:|---:|
| H1 | 0.001 | 0.01 | 42 |
| H2 | 0.001 | 0.001 | 42 |
| H3 | 0.0003 | 0.01 | 42 |
| H4 | 0.0003 | 0.001 | 42 |

- [ ] Reuse an identical existing run where its complete configuration matches a screening row.
- [ ] Keep batch size 128, maximum 120 epochs, and early-stopping patience 20 for this comparison. Define the validation checkpoint score before running the grid.
- [ ] Screen the four configurations using validation only. Confirm the winner with seeds 42, 43, and 44 against the matched control using the same three seeds.
- [ ] Report the mean and spread across seeds. Do not select a deployment seed because it happened to score best on the test set.
- [ ] Stop this search after the predefined grid. Head width, dropout, loss weighting, and target encoding are later experiments only if recorded evidence identifies a specific problem.

**Exit gate:** a repeatable improvement over the matched control. If results are inconclusive, retain the simpler control and proceed only if the failure analysis supports fine-tuning.

## 7. Phase E — Test partial backbone fine-tuning

Run this only after the preceding comparisons. Cached semantic embeddings support head training, but cannot support backbone fine-tuning.

- [ ] Add an image-minibatch training path that uses the same versioned semantic input and physical features.
- [ ] Start from a head checkpoint trained only on the new training partition. Unfreeze the last MBConv stage; record the exact module/parameter names after inspecting the installed torchvision model.
- [ ] Keep earlier stages and BatchNorm running statistics frozen. Update the existing `train()` behavior deliberately, since it currently forces the complete backbone into evaluation mode and freezes its parameters.
- [ ] Verify that intended backbone parameters receive gradients and change, while frozen parameters and BatchNorm buffers do not.
- [ ] Initial experiment settings: backbone learning rate 1e-5, head learning rate 1e-4, weight decay 0.01, batch size 16, maximum 30 epochs, early-stopping patience 5. These are starting settings, not promised optima.
- [ ] Run a short smoke test and measure throughput/memory before the full experiment. Use CPU by default; use local acceleration only after compatibility and numerical checks.
- [ ] If the seed-42 pilot improves validation, confirm with seeds 43 and 44. If it fails the gate, stop; do not expand to full-network training automatically.

**Exit gate:** a robust gain that justifies training cost, with deployment input parity and latency still acceptable. Retain the frozen model if the gain is marginal or uncertain.

## 8. Metrics, selection, and promotion gates

Use a primary score focused on the two currently validated recommendations:

`P2 = 0.5 * (MAE_Exposure / 8 + MAE_HighlightRecovery / 100)`

The denominators are the fixed legacy encoding spans. This score gives the two fields equal range-normalized weight; it does not establish equal perceptual or practical importance. Report original units alongside it. The historical six-field normalized metric remains a secondary diagnostic, so the wide Temperature range cannot conceal regressions in the delivered fields.

Proposed research promotion rules, fixed before experiments:

| Gate | Requirement |
|---|---|
| Repeatability | Three fixed training seeds; improvement in at least two of three paired comparisons |
| Primary improvement | At least 5% lower mean validation P2 than the matched control |
| Delivered-field protection | Neither Exposure nor HighlightRecovery mean MAE worsens by more than 2% |
| Uncertainty | Report paired bootstrap intervals over photo groups; a 95% interval crossing zero means the gain is inconclusive |
| Baseline relevance | Evaluate each delivered field against the constant and physical linear baselines; investigate any failure to beat a simpler baseline |
| Experimental fields | Report all four; do not enable them solely because aggregate P2 improves |
| Export parity | Maximum normalized PyTorch/ONNX difference ≤1e-4 on varied fixtures |
| Inference performance | Proposed CPU model-forward p95 ≤30 ms under the recorded four-thread, batch-one protocol; benchmark RAW end-to-end separately |
| Usability | No new reproducible severe failure in the fixed visual review; no unsupported-input fallback presented as validated |

The 5%, 2%, and 30 ms thresholds are initial research decisions, not measured user tolerances. Freeze them before training; do not relax them after seeing a candidate's results. If no candidate passes, retain the current release and document what failed.

For uncertainty, compare candidate and control errors on the same validation groups. Report per-seed effects and group bootstrap intervals separately; neither photos within a burst nor repeated seeds are independent new photos.

## 9. Visual review and final evaluation

- [ ] Select a fixed validation-only review set before looking at candidate outputs: target 30–50 photos covering available portraits, interiors, night, sky, snow, backlight, and unusual cameras. Report missing categories rather than inventing coverage.
- [ ] Compare the same RAW baseline with candidate/control recommendations through the unchanged v3 renderer, at 100% and 75% strength.
- [ ] Record excessive brightening, flat highlights, blocked shadows, visible noise, and color problems. Preserve image IDs, versions, settings, and review notes.
- [ ] Keep parameter accuracy and preview preference separate. Pixel PSNR against an expert image is not a valid standalone score for a six-parameter recipe with a different renderer.
- [ ] Freeze architecture, preprocessing, hyperparameters, checkpoint rule, and release candidate before final testing. Preselect seed 42 for the deployable checkpoint; the other seeds establish stability.
- [ ] Evaluate the frozen candidate and matched controls in one recorded final-test event. Publish all per-field metrics and uncertainty; require the candidate to maintain the improvement and delivered-field protection gates on this split.
- [ ] If final testing fails, reject the candidate. Do not retune against the same test set and call a second result independent acceptance. A later research cycle must disclose prior test exposure and seek fresh external evidence.

Existing preview review photos include test photos from the historical split. They can serve as historical regressions, but do not establish fresh model generalization.

## 10. Deployment and development records

- [ ] Export the accepted candidate to FP32 ONNX with all required standardization inside the graph and explicit input/version metadata.
- [ ] Test decoder → semantic input/features → PyTorch → ONNX → JSON consistency, including supported and rejected RAW inputs.
- [ ] Verify dynamic batches, offline operation, and no torch/torchvision import in the inference path.
- [ ] Check the English app, recommendation status, downloads, stale-result clearing, and model/preprocessing version display. Browser acceptance must be reported accurately if browser permissions remain unavailable.
- [ ] Keep the previous bundle as a rollback option. Promote only the accepted bundle and matching preprocessing contract.
- [ ] Update `SHOTSENSE_MASTER_PLAN.md`, acceptance evidence, README results, and the experiment ledger. Mark implementation complete only after its checks pass.

Experiment layout; Phase A/B directories now exist, while candidate/final-evaluation outputs remain pending:

```text
data/processed/model_vnext/             # Candidate inputs, metadata, group mapping, split
artifacts/experiments/model_vnext/
  experiment_ledger.csv                # Every run, including failed/negative runs
  <run_id>/
    config.json                        # Source/data/split/preprocessing hashes and seeds
    validation_metrics.json
    validation_predictions.npz
    training_history.json
    best.pt
  final_evaluation/                    # Created only after candidate freeze
  ACCEPTANCE.md
```

Keep RAW data, large caches, and experimental checkpoints local. Commit English plans, compact experiment summaries, version manifests, and accepted evidence. Do not duplicate the approximately 47 GB source dataset. Candidate geometry may require additional JPEG/embedding storage; estimate that from the pilot before rebuilding all photos.

## 11. Second wave — Conditional improvements

### White-balance context

Proceed if WB errors and camera strata show useful headroom. Add only RAW metadata reproducibly available online, such as camera WB multipliers and supported camera identifiers. Use a documented unknown-camera representation and train-only categorical vocabulary. Version the additional input dimensions explicitly; do not hide context inside the existing 132 fields.

Compare context versus no context on identical splits. Evaluate Temperature in Kelvin and optionally mired, Tint in native units, and unseen-camera behavior. The 3395 Catalog As-Shot records are diagnostic evidence, not a universally available online input. Camera multipliers are not automatically Lightroom Kelvin/Tint. Keep WB experimental unless its separate feasibility and accuracy gates pass.

### Expert-specific styles

If users want a particular editing style, compare explicit A–E style conditioning with the existing mean-target task. Keep every edit of an original photo in the same group/partition. Five experts provide multiple labels per photo, not 25000 independent photos. Evaluate against matched expert-specific baselines; mean-target and expert-target MAEs are different tasks and cannot be compared as one improvement percentage.

### Better physical features

Consider brightness-band occupancy, developed-highlight fractions, or shadow statistics only after errors identify a missing signal. Define what each feature measures and its RAW/development limitations. Version feature order/dimensions and fit normalization only on training data. Test additions individually before combining them.

### Personal data and regional editing

Collect unedited personal RAWs, compatible final settings, camera/source provenance, and explicit editing intent. Reserve a genuinely untouched subset before any model inspection. Use it first to measure external generalization; fine-tuning comes later with a separate training subset.

Sky/face/background masks and independent regional adjustments require a separate product/model plan. Global semantic features can inform global parameters, but cannot apply independent local exposure controls. Do not add segmentation until the intended regional behavior and acceptance criteria are defined.

## 12. Execution order, effort, and stopping rules

| Stage | Deliverable | Effort estimate | Dependency / stop rule |
|---|---|---|---|
| A | Frozen baseline, reviewed grouping, validation-only training protocol | 1–2 development sessions, plus grouping review | Stop on leakage, incompatible contracts, or unexplained excluded photos |
| B | Matched grouped baselines | One setup session plus measured training time | Stop on irreproducible metrics or failed baseline reproduction |
| C | Versioned aspect-preserving semantic experiment | One implementation/verification session plus rebuild/training time | Keep original geometry if gain is inconclusive |
| D | Four-configuration screening and three-seed confirmation | Training time estimated from the first run | Stop after the predefined grid |
| E | Partial fine-tuning pilot and conditional confirmation | One training-path session; compute time estimated from smoke test | Stop if pilot has no supported gain or local cost is excessive |
| Final | Frozen evaluation, ONNX acceptance, English release records | One acceptance session plus final evaluation | Reject candidate on failed gates; retain prior release |

A session is a rough development unit, not a promised completion date. Log elapsed time, peak memory, cache size, and throughput in the first pilot, then estimate the remaining work from measured values. Avoid unbounded searches. Do not start a long fine-tuning run before its measured runtime fits the available local compute window.

## 13. Risk checklist before each experiment

- [ ] The experiment has a specific hypothesis and one main changed factor.
- [ ] Split/group identity is independent of training seed.
- [ ] The new baseline has not trained on this split's validation/test photos.
- [ ] Labels and expert edits have not leaked into inputs or preprocessing choices.
- [ ] Target changes are treated as a new task with matched baselines.
- [ ] Production files and preprocessing hashes remain usable.
- [ ] Fine-tuning reads images rather than detached cached embeddings.
- [ ] Frozen gradients, BatchNorm state, and train-only standardization are verified.
- [ ] No naive exposure/WB/color augmentation is applied with unchanged targets.
- [ ] The test set stays outside checkpoint selection and experiment screening.
- [ ] Negative results and limitations are retained in the development record.

## Revision record

- **2026-10-02:** Created the detailed improvement plan after reviewing the current trainer, split support, input geometry, evaluation, and deployment contracts. Defined phased experiments and proposed promotion gates. No training, dataset split change, model replacement, or runtime change performed.
- **2026-10-02 — Phase A:** Implemented validation-only training and P2 checkpoint selection, independent seeds, immutable run directories, failure records, validation prediction exports, frozen artifacts, and explicit one-time final evaluation. Saved the production baseline and conservatively reviewed all 11 grouping candidates. Created a separate grouped split and train-only normalization preview. Synthetic protocol checks and existing production RAW/inference checks passed; no new full-data training or model promotion.
- **2026-10-02 — Phase B:** Clarified that each run requires its own input-preprocessing acceptance; the existing stretch pipeline is verified for controls. Added train-derived Lab-lightness strata and per-stratum reports for all four baselines. Trained seed-42 grouped controls from official ImageNet initialization and replayed the two heads with identical histories/predictions. Recorded full validation metrics, observed runtime/memory, hashes, and ledger. Thirty tests passed; test metrics, candidate geometry, three-seed confirmation, and production replacement remain pending.
