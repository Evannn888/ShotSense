# LOL small-batch pilot results

Date: 2026-10-02. Scope: data suitability and current JPEG/curve behavior on 24 deterministic **training** pairs. This experiment did not train or release a new enhancement model.

## Acquisition and validation

The FiveK paired-release author listings returned 404 for both Google Drive and OneDrive. The LOL author-linked archive was accessible: 347,426,968 bytes, SHA-256 `47d85314b7927470cd48c97e5c1d6c896a56d6217c8e508d0736aff2d91aadcc`. Download took 51.25 s; original archive and extracted data stay in ignored local `data/external/lol_pilot/`.

The archive contains macOS resource-fork entries. An initial preflight correctly rejected an unexpected training-member count before any pair extraction/scoring; matching was corrected to the exact `our485/low/` and `our485/high/` paths, excluding `__MACOSX`. The final script verified 485 matching training filenames and selected 24 with seed 20261003 before inspecting pixels. Only these training pairs were extracted. Official `eval15` pixels were not extracted or scored. Archive directory names were inspected to diagnose packaging, not test content.

All 24 pairs decode as dimension-compatible RGB 8-bit PNGs, with no embedded ICC profile. This pilot assumes sRGB; absence of ICC does not prove the underlying camera/export color calibration. Original PNG hashes and image sizes are recorded. Quality-95, no-chroma-subsampling JPEG derivatives were generated for the real JPEG inference path. Mean per-image JPEG-versus-PNG PSNR was 43.43 dB. This derivative is not an original camera JPEG.

Zero-strength results exactly match decoded JPEG RGB; repeat rendering is deterministic. Model-only/source hashes identify the unchanged production model and current preview implementation. Pair dimensions are checked, but subpixel registration was not calibrated. Representative sheets use the first eight selected IDs, fixed independently of scores.

## Quantitative development observations

Numbers are averages of per-image scores, not a held-out benchmark or aesthetic preference study. PSNR measures closeness to the provided normal-light target in encoded display sRGB.

| Method | Mean PSNR (dB) | Mean MSE (RGB normalized 0–1) | Mean RGB brightness | Pairs with lower MSE than identity |
|---|---:|---:|---:|---:|
| Decoded JPEG identity | 8.63 | 0.178428 | 0.0775 | Baseline |
| Existing RAW-trained experimental JPEG estimates + v3 renderer | 12.47 | 0.087264 | 0.2032 | 24/24 |
| Fixed manual +1 EV, protected v3 | 9.83 | 0.149800 | 0.1183 | 24/24 |
| Reference-informed fixed curve grid | 16.62 | 0.046941 | 0.2971 | 24/24 |
| Provided normal-light references | — | — | 0.4524 | — |

The grid searches 11 exposure values from −2 to +4 EV and recovery values 0/50/100, with highlight protection enabled. It uses each target to choose parameters and is **not a prediction model**. This is only the best result on the selected finite grid; continuous experimental predictions can outperform that grid on individual photos. It does not bound all possible curves or justify a deployable 16.62 dB claim.

22/24 grid choices hit +4 EV, and two choose +1 EV. Existing experimental exposure estimates range from 0.829 to 3.142 EV. This is evidence of insufficient brightening within the tested model/renderer/range on these particular dark pairs; it is not proof that removing the exposure limit alone would solve the task. Color cast and noisy dark detail remain. Experimental adjusted images introduced no new full-channel pixels. Fixed +1 EV and grid averages introduced small nonzero full-channel fractions (0.00000521 and 0.00003611 respectively); protection is not an unconditional zero-clipping guarantee after 8-bit encoding.

## Visual review and decision

Offline overview review found visibly improved visibility from experimental estimates, but extreme-dark kitchen, shelf, bed and ceiling examples remain much darker than their references even with reference-informed grid selection. The window pair approaches the reference more closely after stronger exposure. Mildly dark toy/color-chart examples are already handled reasonably by existing adjustments. Color casts and noise are visible in strongly lifted examples; this renderer has no denoiser or learned color correction.

IDs 328 and 330 look like closely related captures of the same toy/color-chart setup. This is a specific warning to audit source-scene groups before any new training/validation split. The pilot counts **pairs**, not 24 verified independent scenes. No pet-specific acceptance follows from these household examples.

Proceed with a separately versioned low-light curve-training protocol after grouping and color/pair audits. Include identity/current adjustment baselines, a wider-purpose monotonic curve that can lift shadows independently of highlights, and controls for noise/color. Maintain general retouching and RAW parameter prediction as separate tasks. Start with a small training subset and scene-exclusive validation; keep official test pixels untouched until a fixed final evaluation. Do not replace the production model based on this diagnostic.

## Reproduction and limits

`venv/bin/python -m scripts.test_paired_jpeg_pilot` uses the recorded local archive, refuses to overwrite a completed report, verifies archive provenance and writes the manifest/metrics/comparison sheets. Preserve or move the completed output as a version before reproducing. Source/member hashes are in `report.json`; acquisition metadata is in `acquisition.json`. Comparison JPGs stay local until image redistribution terms are established; links do not bundle dataset photographs in GitHub.

Eight existing JPEG/preview tests passed in 3.65 s with the existing optional Matplotlib warning. The pilot's actual 24-pair identity/repeatability/source checks passed in 32.34 s. No production source/model changes, supervised training, browser acceptance or new generalization claim occurred.
