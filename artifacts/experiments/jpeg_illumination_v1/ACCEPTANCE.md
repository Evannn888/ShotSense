# Separate denoising and illumination: fidelity/color and visual gates failed

Date: 2026-10-02. Decision: retain research evidence; **do not integrate this recipe into the website**. The fixed denoiser visibly reduces colored grain, but removes fine detail. Spatial illumination supplies little fidelity gain and increases color error. Both failed numerical gates and the fixed visual review block expansion; seeds43/44 were skipped.

## Fixed data and implementation

Verified author-linked LOL archive, previous120 inspected pairs plus48 new `our485` filenames selected with seed20261010 before review. Reviewed all168 low/reference thumbnails and45 hash candidates, carrying prior scene groups and conservatively merging additional repeated toy/chart, closet, bedroom, bookshelf, retail and dish scenes. Only14 pairs in12 groups remain eligible as new validation; all were used under the approximate16-pair/minimum8-group rule.154 pairs train. Every old ID is training-only. No extra acquisition or split/quality threshold change was used to reach a target count. Grouping may still miss shared venues.

Prior edge-phase alignment screen:166 pass,2 uncertain old training-only pairs142/648, zero clear shifts. Sources preserved without warping; no eval15 pixels extracted or scored. This screen is not calibrated registration. PNG/JPEG display encoding is assumed sRGB without profiles, not a recovered radiometric pipeline.

The prior noise-aware seed42 shared curve/bilateral pipeline and original64-training-image statistics remain frozen. Native controls: identity, existing-v3, frozen shared base, fixed nonlocal-means denoised base, and a trained one-scalar global illumination control. OpenCV colored nonlocal means uses h7, hColor10, template7/search21 after explicit RGB/BGR conversion. Filtering occurs once at full native size before cropping; no filter grid.

The881-parameter illumination CNN receives only denoised RGB downsampled by four, three3x3 convolutions3->8->8->1, ReLU, zero final weights. Its bilinearly expanded scalar field gives gain in[0.25,4], applying the shared endpoint-preserving shoulder1-(1-base)^gain. Neutral gain reproduces the denoised base; exact zero/one base endpoints remain fixed. Strength zero returns original source bytes. No free additive RGB offset or reference pixels enter inference. Wider local context is still not a semantic scene model or learned denoiser.

Five native96px crops,16px loss border, inherited RGB/gradient/chroma/flat-variation objective plus0.1 dark-reference absolute error. CPU/one Torch and OpenCV thread, AdamW lr0.001/weight decay0.0001/batch16/max30epochs/patience6. Validation-crop loss selects checkpoints including neutral epoch0. Global seed42 selected3/9epochs; spatial seed42 selected4/10epochs. Original planned parameter count corrected from889 to881 before fitting without changing architecture.

## Native new-development results

| Recipe | Mean per-image PSNR | RGB MSE | Dark median-residual proxy, codes | Chroma MSE | Luma-gradient L1 |
|---|---:|---:|---:|---:|---:|
| Identity | 8.17 dB | 0.181371 | 1.820 | 0.002478 | 0.020356 |
| Existing experimental JPEG/v3 | 12.82 dB | 0.092227 | 4.030 | 0.001374 | 0.021490 |
| Frozen shared base | 16.94 dB | 0.040394 | 6.739 | 0.001602 | 0.022988 |
| Fixed denoised base | 16.88 dB | 0.040845 | 1.869 | 0.001493 | 0.021040 |
| Trained global illumination | 17.06 dB | 0.040042 | 2.064 | 0.001799 | 0.021499 |
| Spatial illumination | 17.22 dB | 0.035395 | 2.444 | 0.002848 | 0.022549 |

Spatial gain over the frozen base is0.284dB, below the fixed0.5dB gate; gain over global is only0.160dB. Dark variation drops63.73% and gradient error drops1.91% against the base, but chroma error increases77.74%, failing the no-worse-color gate. New full-channel fraction0.00896% passes0.5%. Four of six gates pass. The proxy mixes noise, texture and edges: smoothing detail lowers it too. It is not a63.73% calibrated noise-removal claim. Fixed filtering alone already lowers the proxy72.26%, with slightly lower mean PSNR than the base.

This set differs from previous pilots; their absolute PSNR values cannot rank experiments fairly.14 pairs in12 groups are not14 independent scenes; no confidence/generalization/public-benchmark claim is made. The parent was not retrained on154 pairs; recipe comparisons do not isolate every causal contribution.

## Visual review failed

Fixed first-eight validation IDs18,254,33,43,44,529,565,6 were reviewed as full/native-center/native-top-left-edge comparisons. Old102 and27 are training regressions only. Fixed filtering substantially smooths grain in chair, wall, bookcase and bedroom crops. However, building254 remains grossly overbright from the frozen base and loses window detail; foliage/roof565 become blurred blobs; cabinet6 loses wood grain and book text. Shared illumination cannot repair those removed details.

Spatial illumination introduces stronger yellow/green tones in chair18, tea table33 and old bedroom27, while shadows can retain dark patchiness. Cabinet/ornament scenes retain inaccurate saturation and fine-detail loss. Bedroom27 is still far darker than the normal-light reference. The checks do not establish clean seams or faithful textures. Endpoint preservation avoids an unconstrained additive gray offset but is insufficient for overall image quality.

## Verification and handoff

Full suite52passed22.82s, one existing optional Matplotlib warning, no dependencies added. Verified336 originalPNG/168JPEG hashes, complete/exclusive partitions, old IDs training-only, prior archive/manifest, parent/configuration/source/checkpoint identities and unchanged production ONNX. Native metrics replay exactly for both checkpoints and fixed controls; initial denoised identity and strength-zero identity are byte-exact.881 spatial/1 global parameters verified. Source/black-white endpoints, finite gradients and RGB/BGR channel-order checks passed. This is checkpoint/calculation replay, not full training reproduction.

Training/filtering/native-scoring loop95.58s; global9.51s/spatial13.48s loops. Preparation/tests/verification/review are excluded; no deployment latency or memory acceptance follows. English protocols/manifests/review/split/configuration/histories/metrics/verification and source are retained. External images/contact sheets/checkpoints stay local. App/model/renderer and official final-test data remain unchanged.

Reproduce with `venv/bin/python -m scripts.train_illumination_candidate prepare`, manual group review, then `train`; preserve/version completed directories because preparation/completed reports refuse overwrite. `venv/bin/python -m scripts.verify_illumination_candidate` verifies retained artifacts. Local archive/parent weights/sources are required.

The user subsequently requested research before more implementation. Stop further recipe training and evaluate the evidence/availability of established pretrained restoration methods first; see `EXISTING_SOLUTIONS_RESEARCH.md`. Do not ease gates or repeat fitting on this inspected set.
