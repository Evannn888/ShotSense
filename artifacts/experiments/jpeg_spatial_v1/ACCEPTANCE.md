# Spatial residual pilot: better fidelity, noise and visual gates failed

Date: 2026-10-02. Decision: preserve this isolated research prototype and results; **do not integrate it into the website**. Seed42 improves paired development fidelity and chroma error over the frozen shared-curve base, but worsens the predefined dark-region variation proxy and leaves visible grain, washed-out shadows and inaccurate color. Seeds43/44 were correctly skipped under the fixed conditional budget. This is not a three-seed or independent final-test result.

## Data and fixed experiment

The verified author-linked LOL archive supplied120 `our485` pairs:80 previously inspected pairs forced into training/regression plus40 new filenames chosen with seed20261008 before pixel review. Reviewed all120 low/reference thumbnails,20 hash candidates and prior/conservative scene groups. New toy/chart, bedroom, desk, foosball and swimming-pool views remain grouped with related old images. The fixed seed20261009 split contains104 training pairs and16 new validation pairs in13 scene groups. All original/JPEG variants share a partition; no old IDs enter validation. Grouping is visual/conservative and may miss shared venues.

The exposure/noise-dependent edge-phase alignment screen records118 passing screens,2 uncertain pairs (142,648), and zero clear-shift quarantines. Both uncertain pairs were visually reviewed and are training-only. A synthetic4-pixel translation check passes. This is an alignment screen, not precise registration or a guarantee that every spatial target is accurate. Sources were preserved without warping. Official `eval15` pixels remain unextracted and unscored.

The parent is the previous noise-aware seed42 checkpoint, selected before new scores and frozen with its original64-training-image feature statistics and native bilateral prefilter. It is an existing fixed baseline, not a same-budget model retrained on104 pairs. The2295-parameter CNN takes source/base RGB as6 channels and predicts a bounded additive residual with three3×3 convolutions,12 hidden channels and zero-initialized final weights. It begins as the base exactly. Source-only features/base enter inference; references only supervise training and evaluation.

Five native96px crops per image, with3px border excluded from the inherited composite RGB/gradient/chroma/flat-variation loss. AdamW lr0.001, weight decay0.0001, batch16, maximum30epochs/patience6. Seed42 selected epoch16 and stopped after22 epochs. No settings/gates changed after scores.

The actual-crop CPU/MPS three-step compatibility smoke failed: maximum forward difference5.96e-5 and gradient difference0.00695 versus1e-5 thresholds. Training therefore used CPU with one Torch thread. The test does not establish the cause of device disagreement or general MPS unsuitability. All final native metrics use CPU.

## Native development scores

| Pipeline | Mean per-image PSNR | RGB MSE | Dark median-residual proxy, codes | Chroma MSE | Luminance-gradient L1 |
|---|---:|---:|---:|---:|---:|
| Identity | 7.73 dB | 0.177776 | 0.842 | 0.002099 | 0.016663 |
| Existing experimental JPEG/v3 | 10.96 dB | 0.094519 | 2.822 | 0.001183 | 0.018934 |
| Frozen shared-curve base | 15.69 dB | 0.039529 | 2.597 | 0.002948 | 0.018767 |
| Spatial seed42 | 19.16 dB | 0.015716 | 3.011 | 0.001447 | 0.019067 |

Against the same-set frozen base, PSNR rises3.48dB and chroma error falls50.93%, but dark variation increases15.95%. Gradient error rises1.60%, within the predefined110% bound. Mean new full-channel fraction is0.0156%, below0.5%. Four of five numerical screens pass; the dark-variation screen fails. The residual proxy includes texture and edges, not calibrated sensor noise. Mean PSNR is not pooled-MSE PSNR;16 pairs in13 groups are not16 independent scenes. Do not compare this score directly with previous experiments on different sets, or claim every metric beats existing-v3.

## Visual decision

Reviewed first8 sorted validation IDs132,145,154,160,162,172,238,594 as full-image comparisons, nearest-neighbor-enlarged native center crops and top-left native edge crops. Also reviewed102/27 as old training regressions, excluded from validation scores.

The CNN improves visibility and partly reduces the yellow/green dominance of the frozen base. However, jar/cupboard scenes154/160/162, projector172 and hall238 retain strong colored/mottled grain and gray raised shadows. Jar132 becomes excessively pale; clothes145 retain a pink cast; shower594 remains dark/cyan. Fine cup markings, projector text and wall detail are not faithfully restored. Bedroom27 becomes more visible but remains grainy, green/gray and far below the normal-light reference. The edge review did not establish clean native borders or a comprehensive seam guarantee: noise/casts themselves already fail acceptance.

A plausible interpretation is that this small7-pixel receptive-field additive network mainly learns brightness/color offsets while leaving much of the amplified noise. This is a hypothesis, not an ablation or proof that larger networks will solve the problem. Higher PSNR does not justify deployment.

## Verification and records

- Full suite: **51 passed in22.85s**, one existing optional Matplotlib warning, no added dependencies.
- Verified all240 originalPNG and120 derivativeJPEG hashes, source/protocol/manifest/review/split/parent/configuration/checkpoint provenance, complete scene-exclusive partitions, old IDs training-only and untouched production ONNX.
- Selected-checkpoint native metric replay is exact (maximum difference0); strength-zero output exactly reproduces source bytes. Parent state is unchanged. Initial parent identity, finite gradients, intermediate strength and synthetic alignment checks pass. This is checkpoint/calculation replay, not complete training reproduction.
- Training/native-scoring/report loop107.49s; seed42 loop104.45s. Preparation/tests/review/later verification are excluded. These timings do not certify production latency, memory, export or UI acceptance.
- Histories/per-image scores: `report.json`; data/groups/split: `manifest.json`, `group_review.json`, `split.json`; frozen identities/settings: `config.json`; replay: `verification.json`; accelerator screen: `device_compatibility.json`.
- English code/protocols/manifests/results are tracked. Dataset photographs, contact sheets and experimental checkpoints remain local. App and production renderer/model are unchanged.

Reproduce using `venv/bin/python -m scripts.train_spatial_candidate prepare`, manual group review, then `train`. Completed preparation/results refuse overwrite: preserve/version the directory before another trial. `venv/bin/python -m scripts.verify_spatial_candidate` verifies retained local artifacts. The verified local archive/source/checkpoint files are required.

## Next bounded direction

Do not relax the failed gate or fit more seeds to this inspected set. A further experiment should separate denoising from illumination adjustment, include a matched frozen-denoiser control, and test whether a modest wider spatial context plus explicit shadow/black-level and detail constraints helps. Fix its budget and fresh scene-exclusive development data before training; retain this set for regressions. The evidence does not yet warrant a large generative model or a semantic head expansion. Final unseen-image quality, export/runtime and UI gates still precede any deployment.
