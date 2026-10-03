# Phase D head-search acceptance

Date: 2026-10-02. Status: completed validation-only grid and matched three-seed reference. Decision: retain control.

## Verification before training

Forty existing tests passed, `pip check` found no broken requirements, and production/core preprocessing/original data hashes matched the preserved baseline. Phase B/C frozen hashes, trainer sources and exact replay records were intact. JPEG workflow tests cover real workers and Streamlit AppTest; real-browser localhost acceptance remains blocked by saved permissions. Four existing training-photo JPEGs additionally passed exact decoded-baseline/zero-strength identity checks and had no newly introduced full-channel pixels under the experimental preview. Visual inspection found global brightening without obvious halos at this thumbnail size; the backlit subject remains dark and this is not JPEG accuracy acceptance. See `artifacts/jpeg/real_photo_workflow.json` and comparison image.

## Fixed comparison

All runs use accepted stretch input, the unchanged frozen EfficientNet-B0 and 512→128 head, equal six-field SmoothL1 loss, batch 128, maximum 120 epochs, patience 20, validation P2 checkpoint selection, split seed 42, and identical group-exclusive memberships. Only learning rate and weight decay vary in the four predefined seed-42 rows. Reused H1 and copied its embeddings into new run directories; the native cache fingerprint validator and numerical feature hashes verify reuse. No source data, model or native trainer changed.

| Row | Learning rate | Weight decay | Dual validation P2 | Exposure MAE (EV) | Recovery MAE |
|---|---:|---:|---:|---:|---:|
| H1 | 0.001 | 0.01 | 0.052900 | 0.277752 | 7.108073 |
| H2 | 0.001 | 0.001 | 0.053309 | 0.278850 | 7.176205 |
| H3 | 0.0003 | 0.01 | 0.053130 | 0.287913 | 7.027143 |
| H4 | 0.0003 | 0.001 | 0.053063 | 0.286923 | 7.025995 |

H1 remained best. Its paired confirmation uses the same control configuration at seeds 42,43,44: P2 0.052900,0.051656,0.052162; mean 0.052239, sample SD 0.000626. The winning candidate is literally the control, so paired differences and bootstrap interval are zero by identity, not independent statistical evidence of equivalence. Alternatives lost the screening comparison; none passes the 5% improvement gate. No deployment seed was selected.

All five newly trained runs replayed from official ImageNet initialization with identical dual/semantic histories and zero validation prediction difference. Native-score versus original-unit bootstrap calculations agree; frozen backbone/BatchNorm and run artifacts remain intact. Full per-field/stratified metrics and hashes are retained. Grouping remains incomplete; previously inspected FiveK data are not fresh external evaluation. Validation selection bias is not corrected by the bootstrap.

## Evidence and next decision

`phase_d_summary.json`, `experiment_ledger.csv`, and each `head-*-seed*/` JSON report record settings, splits, complete metrics, source hashes, completion status and replay. Large checkpoints/caches stay local. Runtime fields left blank in the ledger were not measured; no speed claim is made.

```bash
venv/bin/python -m scripts.train_head_grid
venv/bin/python -m pytest -q
```

The grid reuses only complete, exactly matched runs and refuses to overwrite its summary. Archive the previous summary intentionally before a separate reproduction; never overwrite accepted run artifacts. Final test data were not evaluated and production remains unchanged. Stop the bounded grid. Partial backbone fine-tuning requires a separate failure-analysis justification and recorded plan before implementation; this result alone does not justify increasing training complexity.
