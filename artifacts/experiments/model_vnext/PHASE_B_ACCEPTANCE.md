# Phase B: Matched Grouped Validation Baselines

Date: 2026-10-02. Status: accepted as a reproducible validation reference. Not accepted for production promotion.

## Experiment contract

The 4946 supported FiveK inputs use the reviewed group map and fixed split seed 42: 3957 training, 495 validation, and 494 test photos. Group screening remains incomplete. This resplit is not fresh external evidence because FiveK has already been inspected.

All four baselines use identical labels, validation IDs, parameter ranges, and metrics. Physical standardization is fitted only on training photos. The constant uses the training mean; physical linear uses least squares. Semantic and dual heads start from official ImageNet EfficientNet-B0 initialization, with no existing ShotSense checkpoint loaded. The backbone stays frozen, including BatchNorm buffers. Semantic inputs retain the original 224×224 stretch geometry for this control.

Settings: model seed 42, CPU/four threads, 512 → 128 → 6 head, batch size 128, AdamW learning rate 0.001/weight decay 0.01, equal-field SmoothL1 beta 0.1, maximum 120 epochs/patience 20. Checkpoints select on validation P2. Dual stopped after 23 epochs; semantic after 22.

## Validation results

All values below are validation metrics, not test results. Slider fields retain legacy PV2003 units.

| Baseline | Exposure EV | Contrast | Saturation | Temperature K | Tint | HighlightRecovery | P2 | Six-field normalized MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Training mean | 0.408677 | 5.628 | 2.740 | 786.1 | 8.329 | 10.654 | 0.078813 | 0.084329 |
| Physical linear | 0.297620 | 5.797 | 2.656 | 620.4 | 8.466 | 7.423 | 0.055715 | 0.068167 |
| Frozen semantic | 0.357444 | 5.689 | 2.667 | 709.9 | 7.625 | 8.814 | 0.066409 | 0.074761 |
| Frozen dual | 0.277752 | 5.858 | 2.796 | 617.4 | 7.922 | 7.108 | 0.052900 | 0.066034 |

`P2 = 0.5 * (Exposure MAE / 8 + HighlightRecovery MAE / 100)`.

Dual P2 is 32.9% below the constant and 5.1% below physical linear on this seed/split. Both delivered fields improve relative to those baselines. Contrast and Saturation remain worse than the constant; the aggregate score does not justify enabling them. Absolute WB remains experimental without online context. The historical production test improvement of 18.7% is a different metric/split/baseline and is not directly comparable.

## Diagnostic strata

Approximate developed Lab lightness uses L-histogram bin centers. Training-input tertiles define thresholds 31.263660 and 39.985044; these do not measure sensor exposure. Camera, Catalog tag, and label-tail diagnostics retain counts; tags are overlapping/incomplete and small groups must be interpreted cautiously. All four baselines are reported in every stratum.

| Validation brightness group | Count | Constant P2 | Physical P2 | Semantic P2 | Dual P2 |
|---|---:|---:|---:|---:|---:|
| Dark | 170 | 0.07379 | 0.05560 | 0.05944 | 0.05314 |
| Middle | 147 | 0.07309 | 0.05105 | 0.05858 | 0.04865 |
| Bright | 178 | 0.08833 | 0.05967 | 0.07953 | 0.05618 |

The full camera/tag/tail/WB diagnostic report is in `control-seed42/evaluation.json`. Catalog WB is used only for diagnostics, never as an input. Strata do not establish a generalization claim from one seed.

## Reproduction and observed resources

`scripts.verify_baseline_reproduction` reinitialized both heads from ImageNet with seed 42, reused only version-checked frozen embeddings, and repeated training. Training histories were identical and the maximum normalized validation prediction difference was **0.0** for both heads. No ShotSense checkpoint was used as replay initialization.

The full first run, including fresh embedding extraction, took 288.44 seconds. macOS `/usr/bin/time -l` reported peak RSS 1379155968 bytes (approximately 1.28 GiB). Head training took 2.84 seconds for dual and 2.34 seconds for semantic. These are observed local training measurements, not an inference benchmark or a promised future runtime.

Thirty tests passed, including train-only brightness thresholds, all-baseline strata, the validation-only protocol, real RAW/ONNX parity, and offline inference. Production model/preprocessing/original cache/split hashes remain unchanged. No `final_test_access.json` or `final_evaluation.json` exists for this full-data run; test metrics were not computed.

## Records and next gate

Configuration, frozen artifact hashes, original-unit metrics, training histories, run status, and reproduction evidence are retained under `control-seed42/`. Large checkpoints/embeddings/prediction arrays and raw logs remain local. Compact evidence is committed with `phase_b_summary.json` and `experiment_ledger.csv`.

Next: validate a separate aspect-preserving semantic pipeline, then run a matched geometry comparison. Seeds 43/44 are deferred to candidate confirmation as planned. Uncertainty, final test comparison, deployment checks, and all release promotion gates remain pending. The current application continues to use the previous production bundle.
