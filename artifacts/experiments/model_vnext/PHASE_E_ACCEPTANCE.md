# Phase E full last-stage tuning acceptance

Date: 2026-10-02. Status: completed validation-only seed-42 pilot and conditional seeds 43/44. Decision: retain frozen control. No new model is promoted.

## Fixed protocol and verified computation

Initialized each seed from its matched Phase B/D H1 checkpoint trained only on the new training partition. Kept the original 3957/495/494 group-exclusive split and accepted stretch input. Unfroze only `backbone.features.7` plus the existing head; earlier stages, final projection and every BatchNorm running buffer remain frozen. Batch size 16, stage/head learning rates 1e-5/1e-4, weight decay 0.01, equal six-field SmoothL1 beta 0.1, maximum 30 epochs and patience 5 were fixed before training. Validation P2 chooses checkpoints, with the unchanged parent eligible at epoch zero.

Only the unchanged prefix before stage 7 is cached. All 4452 training/validation image bytes, frozen prefix weights, decoder/data/source/torch identities and cache bytes are versioned and verified. Test-image activations were not computed. Stage 7/final projection/head remain in the differentiable graph. Tests prove fresh/reused cache paths preserve RNG, reject changed cache artifacts, and match the full image path before/after updates. Every training batch checks finite gradients; every epoch checks frozen state/BatchNorm unchanged.

The first cache build was explicitly interrupted before training to fix cache-related RNG consumption. A corrected CPU run completed three epochs, each approximately 101 seconds; its progress, configuration and exact source snapshot are preserved. After CPU/MPS compatibility checks passed, complete runs restarted from their original parents on MPS. Forward/gradient compatibility maximum differences were 2.68e-7/3.73e-8; post-update image/cache difference 2.98e-7. CPU remains the CLI default and MPS requires explicit selection and compatibility evidence. MPS training and CPU trajectories are not claimed to be identical.

## Full validation results and decision

Selected-checkpoint predictions for all 495 validation photos were recomputed on CPU. Across runs, CPU/device maximum normalized difference is at most 3.58e-7; full-image/cached-tail before/after differences are at most 5.36e-7, well within the predefined 1e-5 checks.

| Seed | Control P2 | Candidate P2 | Best epoch | Epochs run | Candidate Exposure MAE (EV) | Candidate recovery MAE |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.052900 | 0.052516 | 1 | 6 | 0.275221 | 7.062851 |
| 43 | 0.051656 | 0.051656 | 0 | 5 | 0.273965 | 6.906572 |
| 44 | 0.052162 | 0.052162 | 0 | 5 | 0.279062 | 6.944151 |

Control mean/sample SD P2: 0.052239/0.000626. Candidate mean/sample SD: 0.052111/0.000432. Mean improvement is 0.245%; Exposure/Recovery mean MAE improves 0.305%/0.216%. Only one seed improves; seeds 43/44 select exactly unchanged parent parameters. Tiny numerical prediction differences from different batch computations are not gains.

Paired photo-group bootstrap of per-photo losses averaged over three fixed seeds, 10000 draws with seed 20261002, gives relative improvement interval −0.398% to +0.912%. It crosses zero. The two-of-three repeatability, 5% mean improvement and uncertainty gates fail; delivered-field protection passes. Retain frozen control and stop this fine-tuning experiment. Do not expand to full-network tuning or select a lucky deployment seed.

Prefix preparation took 240.74 seconds once. Complete cached MPS runs report approximately 42.29/30.53/30.67 seconds; epoch times mostly 5.2–5.4 seconds after initial warmup. Reported elapsed time includes setup/cache verification/training/prediction parity, but excludes subsequent diagnostics/serialization. CPU and MPS timings are observations of these paths, including gradient-check instrumentation, rather than an isolated hardware speedup benchmark. CPU peak RSS excludes Metal allocation accounting; no combined GPU memory claim is made.

## Evidence and limits

- `phase_e_summary.json`: paired metrics, original control comparison, bootstrap, gates, hashes and negative result.
- `phase_e_training_summary.json`: full pilot/confirmation training summaries.
- `finetune-seed42/`, `finetune-seed43/`, `finetune-seed44/`: configuration, split, history/progress, complete stratified metrics, selected checkpoint hashes and status.
- `frozen_prefix_acceptance.json`, `mps_compatibility.json`: cache and accelerator checks.
- `finetune-seed42-interrupted-cache-build/`, `finetune-seed42-interrupted-cpu/`: preserved unsuccessful/interrupted attempts; the CPU source snapshot matches its recorded hash.
- `experiment_ledger.csv`: complete and interrupted runs; large checkpoint/cache files remain local.

```bash
venv/bin/python -m scripts.train_finetuning --device mps
venv/bin/python -m pytest -q
```

Run directories are protected and refuse overwrites. Preserve existing evidence before a separately named reproduction. The compatibility entry point also refuses to overwrite its accepted output. Exact full-training replay was not performed; seed comparisons, image/device parity and RNG/cache regression checks were performed. Validation-selected uncertainty does not correct selection bias. Previously inspected FiveK and incomplete grouping remain limitations. No new test evaluation, JPEG accuracy acceptance, export/latency acceptance or production replacement occurred. Real-browser localhost restrictions remain respected.
