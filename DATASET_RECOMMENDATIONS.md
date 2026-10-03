# Dataset recommendations for ShotSense

Research date: 2026-10-02. This is a primary-source shortlist and proposed acquisition/evaluation protocol. No new dataset has been downloaded, audited or used for training in this task. Published download listings do not prove that every archive is currently retrievable or correctly paired.

## Recommended order

Start with paired rendered FiveK inputs/targets for general JPEG enhancement, then a separate LOL low-light experiment. Initially reserve SICE for external exposure stress testing. Consider PPR10K when portrait priority or a larger parameter-supervision task is needed. DPED is a later phone-camera domain experiment.

| Dataset and primary source | Available supervision | Best fit | Important task boundary |
|---|---|---|---|
| [FiveK 480p release from the adaptive-LUT authors](https://github.com/HuiZeng/Image-Adaptive-3DLUT), derived from [original FiveK](https://people.csail.mit.edu/vladb/photoadjust/) | Resized 8-bit sRGB / 16-bit XYZ inputs and 8-bit sRGB targets are listed. | First general tone/color enhancement prototype with modest image sizes. Select the sRGB branch for the rendered-image task. | These are the same underlying FiveK photos already used by ShotSense, not a fresh independent dataset. Audit expert identity, export recipe, file pairing and source IDs before use. |
| [LOL / RetinexNet author project](https://daooshee.github.io/BMVC2018website/) | Captured low/normal-light image pairs; synthetic pairs are offered separately. | Learn visibility improvement for dark JPEG-like rendered inputs; evaluate amplified noise and color shifts. | Low-light restoration is a different target from aesthetic expert retouching. The author implementation uses 485 training pairs; preserve its held-out evaluation and create development validation only from training groups. |
| [SICE author repository](https://github.com/csjcai/SICE) | 589 multi-exposure sequences, 4,413 source images. References were selected subjectively from fusion/HDR algorithm results. Download listings split sequences into Part 1 (360) and Part 2 (229). | Under/overexposure and high-contrast external stress tests, followed by a separately defined exposure-training task if justified. | References are selected fused results, not CameraRaw slider labels. Keep all exposures of one scene together. Use one input image per inference; do not give a single-image model the other exposures. |
| [PPR10K author repository](https://github.com/csjliang/PPR10K) | 11,161 RAW portraits in 1,681 groups, three expert targets, human-region masks, source XMP and three expert-target XMP sets. The listed 360p image package is 91 GB and the entire folder is 406 GB. | Portrait retouching, human-region objectives, group consistency; possible future RAW parameter supervision after an XMP compatibility audit. | Human masks do not establish pet-subject performance. All augmentations/experts from a group must share a split. Dataset agreement limits images and derived data to non-commercial research; the separate Apache code license does not remove that condition. |
| [DPED author project](https://aiff22.github.io/) | Synchronously captured three-phone/Canon 70D images; aligned 100x100 training patches are listed as 6.2 GB, original photos as 54 GB and sample photos as 125 MB. | Later phone-camera quality/domain adaptation study. | Phone-to-DSLR mapping combines camera differences, texture and tone. Original full images are not perfectly aligned. Group patches/devices by source scene and do not treat patch count as independent scene count. |

The [DRBN author repository](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) also lists expanded real/synthetic low-light collections. If evaluating an LOL-v2 release, first verify its exact archive identity and real/synthetic partition against the accompanying paper and manifest; do not mix benchmark variants or quote their results as interchangeable.

## Why these data do not directly replace the current labels

ShotSense's current RAW task predicts six absolute legacy PV2003 Catalog parameters. Most shortlisted targets are images, not compatible absolute parameter records. Training a JPEG enhancer means a separate rendered-image loss/output/evaluation contract; an image target does not by itself identify unique Exposure, Contrast, white balance or HighlightRecovery values.

PPR10K is an important exception to an image-only description: the authors explicitly provide source and expert CameraRaw XMP adjustments. This makes a parameter-data expansion worth investigating. It does **not** establish PV2003 compatibility. Before importing labels, inspect ProcessVersion, exact field names, defaults, absolute versus incremental WB, local edits/masks, source processing and target rendering. Modern Highlights cannot be substituted for legacy HighlightRecovery. The RAW formats also extend beyond the currently accepted DNG path.

The previous local inventory found no standalone expert TIFF/JPEG target package in `data/raw`. Existing DNGs and the Catalog support current parameter supervision; embedded previews and ShotSense's approximate renderer are not calibrated expert-image ground truth. Paired rendered FiveK exports would fill a different supervision gap.

## Bounded first experiment

1. Acquire only the rendered sRGB input/target branch of the FiveK 480p release first. Record archive hashes, source URL, dataset-specific terms, expert/style identity, bit depth, ICC/transfer encoding, dimensions and export provenance. Do not infer that a repository code license covers photographs.
2. Build an immutable manifest with source photo/group IDs, input/target paths and hashes. Map IDs to the existing conservative group split; all crops, alternate exposures, expert versions and encodings of a photo stay together. Explicitly quarantine unmatched IDs rather than silently dropping or reassigning them. A separate rendered-image task may audit the full 5,000-photo release, but cannot silently alter the RAW task's supported 4,946-photo cohort.
3. Inspect a small fixed set of pairs for alignment, orientation, color encoding and target style before training. Export identity and current manual-curve baselines alongside targets. Fix one expert style initially rather than averaging differently edited image pixels.
4. Train the proposed small controllable curve model only on eligible training groups. Use task-matched validation, identity/no-change checks, paired PSNR/SSIM where images are genuinely aligned, color checks, visual crops and measured latency. Preserve the current RAW model and its separate acceptance criteria.
5. Run LOL as a distinct low-light experiment with matched baselines. Report dark visibility, noise amplification, clipping and color separately from general retouching scores. Do not merge target styles indiscriminately.
6. Freeze the model/protocol before evaluating reserved SICE scenes and a new personal-photo set covering pets, indoor darkness, backlight, skies and skin tones. Previously reviewed hamster/desert examples remain development examples. If external failures drive changes, that inspected set becomes development data and a fresh final set is needed.

No public dataset alone establishes the user's preferred look. A small, separately held-out collection of real photos and preferred edits is still useful for practical acceptance. Paired capture/retouching data support learning; repeatedly choosing improvements on the final test set does not support an independent accuracy claim.

## Status

- [x] Reviewed primary author pages/repositories and documented task fit, supervision, grouping and the explicit PPR10K data agreement.
- [ ] Verify archive access, contents, per-release terms and pair/export provenance.
- [ ] Acquire and audit a bounded paired sRGB release.
- [ ] Implement and validate a separate JPEG enhancement experiment.

Documentation-only change. Previous 47-test runtime acceptance is retained, not newly rerun. No new quality result, trained candidate or production promotion is claimed.
