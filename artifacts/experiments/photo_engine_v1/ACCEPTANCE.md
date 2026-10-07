# Conservative photo-engine pilot — 2026-10-04

Decision: deliver an **opt-in experimental** original-resolution workflow. Manual JPEG exposure remains the default. The finite offline review found no conspicuous new cast, eye-color corruption or severe detail destruction; it does not establish preferred results on personal photos or superiority over the old renderer. The practical-tool photographic acceptance gate is still open.

## What is delivered

Pinned RawTherapee5.11, compatible with the actual macOS14.8.4 arm64 host, is installed as a separate unmodified application. The official archive, CLI and application-tree identities are retained in [the manifest](../../photo_engine/manifest.json); the preparation script refuses replacement of a different installation. No new Python dependencies or model training were required. Processing is offline with isolated settings/cache, temporary inputs, no source-sidecar loading and versioned PP3 configuration.

The website's **Natural adjustment (experimental)** accepts DNG/JPG/JPEG, offers automatic conservative choice, gentle lift, or gentle lift with optional chroma denoising. Moderate dimness gets a fixed0.35EV lift with engine highlight compression; bright/extremely dark inputs and candidates introducing over0.1% new full-value-channel pixels are preserved by auto selection. Thresholds are engineering choices, not confidence or learned aesthetic judgment. Denoising is never inferred automatically.

Strength0–100 blends the full-resolution8-bit sRGB output with its baseline. JPEG zero strength restores the exact EXIF-oriented, ICC-managed decoded pixels, rather than original compressed file bytes. DNG zero restores neutral engine development with valid camera WB, rather than sensor pixels or the old RAW pipeline. The displayed result and PNG download share bytes; matching JSON records strength, dimensions, source/engine/profile hashes and output PNG hash. EXIF/GPS/input metadata are not copied to exports. RAW model predictions and legacy Adobe labels do not configure this engine.

## Fixed diagnostics and review

The [protocol](PROTOCOL.md) was recorded before scoring. The successful [frozen manifest](diagnostics-v2/manifest.json) has six previously inspected LOL JPEGs, six previously inspected FiveK RAW regression inputs and three synthetic controls. All15 sources and full exported hashes are retained; [summary](summary.json) and [recipe report](diagnostics-v2/report.json) record the outcomes. Source images and generated comparison media stay local. This is not a fresh camera/generalization set, not a human preference experiment, and uses no LOL eval15 or new model-test metrics.

Auto preserved8/15 inputs and gently adjusted7/15. Five of six severely dim LOL scenes stayed unchanged; the remaining scene got a modest lift. Five of six RAW scenes got a lift; a scene with extensive baseline white clipping stayed unchanged. Synthetic daylight and extreme-dark controls stayed unchanged. Native center crops retain visible source noise/blur; chroma filtering can modestly smooth texture. The sunset gets brighter even though its intent might favor preserving dusk. Neutral RAW already clips some highlights, and no recovery claim is made.

Viewed all three full-frame and all three native-center sheets, plus an additional native portrait-eye/glasses panel. The eye panel retained source detail and pre-existing lens fringing without the prior generated red/purple-eye failure. This is one public portrait, not a reproduction of the user's earlier source-photo failure; only screenshots of that failure are available. Pet fur, mixed lighting, additional cameras and fresh personal photos remain unaccepted.

Local review artifacts:

- [Full-frame comparison 1](diagnostics-v2/overview-1.jpg), [2](diagnostics-v2/overview-2.jpg), [3](diagnostics-v2/overview-3.jpg).
- [Native detail comparison 1](diagnostics-v2/native-crops-1.jpg), [2](diagnostics-v2/native-crops-2.jpg), [3](diagnostics-v2/native-crops-3.jpg).
- [Portrait eye detail](diagnostics-v2/portrait-details.jpg), [crop/source record](diagnostics-v2/portrait-details.json).

The overview's “Current +0.35EV” column is the old custom protected tone function applied to the engine-neutral baseline. It is **not** the current model recommendation or existing RAW development. Its1600px native crops originally had a different scale and were subsequently omitted; they were excluded from detail assessment. The [review correction](diagnostics-v2/review-amendment.json) records this limitation. No renderer-quality ranking is derived from these sheets.

The first comparator stopped after a float32 comparison-buffer bound error, with a case-sensitive historical-ID exclusion mistake. Preserve its [failure record](diagnostics/failure.json), manifest, source snapshot and partial outputs. The corrected run clips the comparison-only float buffer and excludes historical IDs case-insensitively before selecting the first six already-reviewed regressions; no outcome-based split selection occurred. All initial inputs were also previously inspected.

## Verification and measurements

Real engine tests cover deterministic JPEG replay; offline worker execution; orientation/ICC; immutable input; odd/tiny dimensions; neutral grayscale and dark color patches; exact zero strength; full-resolution PNG size/hash; invalid strength; missing/checksum-mismatched CLI; nonzero exit and native timeout; clipping fallback; and separate DNG development. Streamlit AppTest covers actual worker routing, full-resolution JPEG and RAW download bytes/JSON at different strengths, stale-cache clearing on mode/recipe changes, and preservation of the existing workflows. The parent60-second timeout sends termination to the worker process group so its native engine is stopped too; this path is verified with a controlled process-boundary test.

Full regression: **59 passed**, no skips/failures, 38.99s. Dependency check passed with no broken requirements. Results and final implementation identities are recorded in [verification](verification.json). One existing optional Matplotlib warning remains. Actual browser visual interaction is unverified because saved browser permissions block localhost; no bypass was used. Server health on port8501 returned `ok`.

Subsequent user feedback confirmed an unchanged result. The [UX correction](ui_feedback_v1/ACCEPTANCE.md) adds explicit reasons, processed-recipe status and one-step gentle reprocessing; its full suite has60 passing tests. The original59-test source identities are preserved in that directory. Engine/PP3/auto thresholds and the photographic acceptance status are unchanged.

The [timing record](benchmark.json) measures three sequential isolated decode-to-PNG exports, including Python startup/import, decoding, selected auto recipe, both PNG encodes and temporary PNG/JSON writes. It excludes web rendering and NPZ serialization. The12MP JPEG is a uniform synthetic control and cannot predict textured-photo cost. `/usr/bin/time -l` reports maximum resident set size; it is not a sampled sum of Python and native-engine process-tree RSS. Loaded attempts are retained separately. Single observations are not latency percentiles or40MP performance acceptance.

| Input | Full PNG dimensions | Measured wall time | Reported max RSS |
|---|---|---|---|
| Public moderately dim JPEG | 600×400 | 1.27s | 182MiB |
| Uniform synthetic JPEG | 4000×3000 | 5.32s | 405MiB |
| Public portrait DNG | 2848×4282 | 11.19s | 617MiB |

The comparison report's auto times exclude PNG export and worker startup:600×400 JPEGs0.031–0.214s, these six6–12.7MP DNGs3.98–8.44s. Do not compare those directly with complete export/website times. App limits remain128MB/40MP/60s, and do not promise completion for all inputs at the limits.

## Remaining practical acceptance

Human original/baseline/candidate preference review on separately reserved personal-camera photos remains required before changing the default or claiming natural/stable automatic adjustment. Keep the earlier proposal of30 development plus20 reserved photos as a planning target, not completed acquisition. Include real eyes/skin/fur/text/foliage, mixed light, backlight, noisy darkness, ordinary daylight and intentional night mood. Freeze criteria before tuning. Expand format support only with decoder/color/orientation/export fixtures for that format. Do not automatically start darktable, adaptive-LUT or neural training merely because this gentle pilot does not restore extreme low light.

Reproduce in a new output directory after preparing the local engine and source data:

```sh
venv/bin/python scripts/prepare_photo_engine.py
venv/bin/python scripts/compare_photo_engine.py --output-dir artifacts/experiments/photo_engine_v1/new-diagnostics
venv/bin/python -m pytest -q
```

Full-resolution CLI processing is available with `venv/bin/python -m src.photo_engine input.jpg --recipe auto --output result.json --preview-source source.npz`; `render_photo` produces the PNG from those cached pixels at the selected strength.
