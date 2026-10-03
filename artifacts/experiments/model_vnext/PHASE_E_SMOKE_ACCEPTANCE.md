# Validation failure analysis and partial-tuning smoke

Date: 2026-10-02. Status: analysis and implementation/cost smoke passed. Full Phase E quality evaluation remains pending.

## Findings and interpretation

Analyzed all 495 validation photos using unchanged, hash-verified seed-42 predictions and original-unit targets. Brightness thresholds use training inputs only. Residual CSV contains every validation ID and no new test IDs; weighted stratum scores match the existing aggregate metric.

| Stratum | Count | Dual P2 | Semantic-only P2 | Dual Exposure bias (EV) |
|---|---:|---:|---:|---:|
| Dark | 170 | 0.053141 | 0.059437 | −0.036996 |
| Middle | 147 | 0.048647 | 0.058582 | −0.011738 |
| Bright | 178 | 0.056181 | 0.079532 | +0.001939 |

The worst 20 photos contribute 12.21% of dual error; dual beats physical linear on 52.73% of individual photos. Exposure absolute error and expert Exposure disagreement have correlation 0.214, which does not establish a cause or irreducible error. The worst-input contact sheet includes night/backlit interiors, highlights, portraits and objects; large residuals are mixed across Exposure/Recovery and directions. It does not establish that all dark scenes should be brightened or that semantic adaptation will fix them. No blanket dark-image compensation was added. Post-selection descriptive findings justify a limited domain-adaptation implementation experiment, not a promised accuracy benefit.

## Implemented boundary and measured cost

A separate `FineTuneCandidate` retains the original architecture/forward path and unfreezes only installed torchvision EfficientNet-B0 `backbone.features.7` (last MBConv stage) plus the existing head. All earlier stages and final projection stay frozen. The overridden training mode enables this stage while forcing every backbone BatchNorm to evaluation mode; running buffers stay unchanged. Production source/model and native trainer are unchanged.

The CPU smoke initializes from grouped H1 seed 42, whose head used only the new training partition. It uses the first 128 fixed training IDs, eight batches of 16, stage/head learning rates 1e-5/1e-4, weight decay 0.01 and equal-parameter SmoothL1 beta 0.1. One fixed 16-photo validation batch is diagnostic only. Every intended trainable parameter receives finite gradients, stage/head weights change, frozen parameters receive no gradients, and all frozen state/BatchNorm buffers remain exactly unchanged. No checkpoint is promoted or saved as a quality candidate.

Measured training time: 11.528 seconds for 128 photos (11.10 photos/s); peak macOS process RSS: 1167163392 bytes. Linear extrapolation estimates 356.38 seconds of training per full 3957-photo epoch, excluding validation/checkpoint overhead. A maximum 30-epoch run would imply roughly three hours of training by this extrapolation; this is not a measured full-run time. The smoke subset metrics must not be compared with the full validation baseline as gain evidence.

## Evidence and reproduction

- `failure-analysis/summary.json`: stratum errors/bias, descriptive comparisons and limitations.
- `failure-analysis/validation_residuals.csv`: all 495 validation residuals, expert disagreement and developed lightness.
- `failure-analysis/worst-validation-inputs.jpg`: baseline model inputs only; captions show prediction/target.
- `finetune-smoke-seed42/config.json`, `smoke_report.json`, `run_status.json`: exact selected IDs, sources, parameter boundaries, loss/cost and completion.
- Tests protect the actual gradient/BatchNorm boundary, validation-only analysis and exact manual JPEG PNG/JSON exports at 100% and 50% strength.

```bash
venv/bin/python -m scripts.analyze_validation_failures
venv/bin/python -m scripts.smoke_finetuning
venv/bin/python -m pytest -q
```

These entry points refuse to overwrite existing evidence directories. Preserve the accepted outputs before an intentional separate reproduction. Remaining work: full image-minibatch training/early stopping and full validation-quality pilot, conditional three-seed confirmation, export/inference checks and final acceptance. No new test evaluation, JPEG accuracy acceptance or production replacement occurred. Real-browser localhost restrictions remain respected.
