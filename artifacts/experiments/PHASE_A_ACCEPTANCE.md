# Model Research Phase A Acceptance

Date: 2026-10-02. Scope: evaluation infrastructure and conservative photo grouping. No new full-data model has been trained or promoted.

## Verified changes

- Production reference: `baseline_manifest.json` records the source revision, file hashes, model contract, original split identity/counts, and historical metrics. Original snapshots remain intact.
- Training is validation-only. Head checkpoints select on Exposure/HighlightRecovery P2. Test predictions/metrics and test Catalog diagnostics are absent from experimental training.
- Model seeds and split seeds are independent. `--groups` and `--split-path` reuse the existing validated group-exclusive splitting helper.
- Every experiment requires a new output directory. Status/config/failure records, validation predictions, control parameters, and frozen artifact hashes are retained.
- `--final-evaluate` checks frozen data/configuration/source/input identities and records test access before prediction. Exclusive creation prevents repeated/concurrent final evaluation of the same run. Failures retain the access record. This protects the supported workflow; it cannot prevent manual test access or deliberate edits outside it.
- Experimental checkpoints have no validated recommendation fields. Final evaluation records metrics without automatically granting release promotion.

## Grouping evidence

| Item | Result |
|---|---:|
| Unedited input photos screened | 4946 |
| Unambiguous camera/capture metadata | 2348 |
| Visually reviewed candidate pairs | 11 |
| Accepted / rejected pairs | 8 / 3 |
| Resulting photo groups | 4938 |
| Accepted pairs crossing historical partitions | 2 |
| New train / validation / test photos | 3957 / 495 / 494 |
| Split seed | 42 |

Candidates used difference/perceptual hash distances ≤4/≤8 together, or matching camera/capture metadata within three seconds. Review used unedited images, source naming/time, and scene relationships, without model errors or expert edits. Three timestamp matches were rejected because scenes differed. One session pair with different people was conservatively grouped based on matching source sequence/time and shared setting.

The full local photo-group map, reviewed pairs/contact sheet, split, and train-only physical normalization preview live under `data/processed/model_vnext/`. Compact decisions, hashes, and per-partition camera counts are in `group_audit_summary.json`. Screening is conservative and incomplete. Metadata conflicts/absence and missing serials limit burst detection. These groups do not establish that all scene leakage is excluded. Previously inspected FiveK data are not a fresh external benchmark.

## Validation

The complete suite passed **29 tests**. New synthetic checks verify validation-only predictions/diagnostics, stable partitions across model seeds, whole-group isolation, recorded failures, P2 units, overwrite rejection, frozen-artifact tamper rejection, and one-time final access. The grouping check verifies that timestamp candidates still require review rather than automatic merging.

Existing production checks verify real RAW offline/online parity, ONNX batches, and Torch/network-disabled inference. Production checkpoint/ONNX/bundle, original preprocessing/decoder/inference source, original metadata/split, and existing audit hashes remain unchanged. The only warning is the existing optional Matplotlib availability notice from colour-science.

The historical constant baseline used a training median; new experiments use the plan's training mean. Historical and new-split results must be presented separately.

## Remaining work

Matched grouped baselines, candidate geometry acceptance, hyperparameter screening, three-seed confirmation, partial fine-tuning, uncertainty estimates, final candidate comparisons, deployment promotion, and practical error calibration remain pending. Latest real-browser localhost acceptance remains unavailable under saved browser permissions; no bypass was attempted.
