# Phase C geometry acceptance

Status: complete validation screen; retain the stretch control. Production remains unchanged.

## Controlled input change

`letterbox-imagenet-mean-v1` preserves the complete frame at 224×224 with centered RGB [124,116,104] padding. Only semantic geometry changes. All 4946 supported RAW inputs succeeded with physical vectors exactly equal to the original cache. Targets, groups, split memberships, head architecture, initialization, training settings, and model/split seed 42 match Phase B. Wide/portrait/square fixtures, subject retention, JPEG decoding, cache rejection, and offline/online helper parity passed. This verifies input routing, not exported candidate ONNX parity; no candidate was exported or promoted.

Preprocessing used six workers and took 1173.51 seconds. Candidate training, including input validation and fresh embeddings, took 288.91 seconds with peak process RSS 1693204480 bytes. These observations are machine-specific, not latency guarantees or aggregate preprocessing process-tree memory. After the measured preprocessing run, the builder's Dataset import was moved inside the parent entry point to avoid loading Torch in spawned workers; pixel processing is unchanged and no improved runtime is claimed.

## Validation decision

| Model | Stretch P2 | Letterbox P2 |
|---|---:|---:|
| Training-mean constant | 0.078813 | 0.078813 |
| Physical linear | 0.055715 | 0.055715 |
| Semantic only | 0.066409 | 0.064861 |
| Dual | 0.052900 | 0.052948 |

The primary dual score has 0.09% higher error. Exposure MAE rises from 0.277752 to 0.282241 EV; HighlightRecovery MAE falls from 7.108073 to 7.061645. Semantic-only improvement does not satisfy the dual-model gate. A paired bootstrap over 495 validation photo groups, 10000 resamples with seed 20261002, gives a relative dual improvement interval of −2.65% to +2.42%. This interval crosses zero; one seed cannot establish a population-level improvement or regression.

Retain stretch geometry. Three-seed confirmation was not triggered because the candidate did not improve the predefined primary score. Both heads replayed from official ImageNet initialization with identical histories and zero validation prediction difference. No final test access occurred; no production bundle was replaced. Grouping remains incomplete and FiveK data were previously inspected, so this is not fresh external evaluation.

## Evidence and reproduction

- `letterbox_preprocessing_acceptance.json`: counts, cache/source hashes, geometry and timing.
- `phase_c_summary.json`: full matched aggregate metrics, paired bootstrap, resources, hashes and decision.
- `letterbox-seed42/`: configuration, split membership, full validation strata, frozen artifact hashes, completion status and exact replay report.
- `experiment_ledger.csv`: both control and candidate runs.

```bash
venv/bin/python -m scripts.preprocess_letterbox --help
venv/bin/python -m scripts.train_geometry --candidate-dir data/processed/model_vnext/letterbox --output-dir artifacts/experiments/model_vnext/NEW-RUN
venv/bin/python -m scripts.verify_baseline_reproduction --run-dir artifacts/experiments/model_vnext/NEW-RUN --metadata data/processed/model_vnext/letterbox/metadata.npz
```

Use a new run directory; large datasets, caches, embeddings and checkpoints remain local. JPEG support has separate [workflow acceptance](../../jpeg/ACCEPTANCE.md), and remains experimental without JPEG accuracy validation. Real-browser acceptance remains incomplete because the saved localhost browser permission blocks access.
