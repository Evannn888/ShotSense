# Automatic novice workflow — 2026-10-05

The default page now processes a newly uploaded photo automatically and shows original/result comparison plus explicit original-resolution PNG save. Mode selection and retry are in collapsed **Advanced options**; optional shadow/overall adjustments and native detail inspection are also collapsed. The user does not have to choose a LUT, recipe or parameter, or click Process image. The JPEG path reuses protected Natural color and its source-dependent shadow suggestion; DNG reuses the existing conservative native automatic recipe.

This implements the user's clarification that the product is for people who cannot edit photos. The [frozen protocol](PROTOCOL.md), [pre-edit file identities](baseline.json) and source_before.zip precede the UI edits. The user's request authorizes the automatic default, superseding the previous manual-default UI decision. It does not establish independent aesthetic preference acceptance.

## Exact lifecycle

Each attempt is identified by filename, suffix and byte hash. A new upload triggers one bounded worker. Ordinary reruns, save, download preparation, detail inspection and optional adjustment reuse the cached result. Replacement or removal clears the previous prediction, controls, prepared export, saved-result notice and error. Workflow changes reset state; old workflow sessions migrate to the new default without displaying old predictions.

Failed processing clears prior images/exports and records an error for the attempted input. Unrelated reruns retain the error without repeatedly spawning workers. The explicit retry in Advanced options reruns that attempt; a different upload starts a fresh attempt. Original files are not overwritten, and saving still requires an explicit click. JSON records the automatic UI version and selected backend alongside actual current strength/source/pixel/export identities.

Existing optional manual, native recipe, protected-natural and legacy workflows remain available. Their rendering and export helpers are unchanged. Backend formulas, weights, source bounds, format support, worker deadlines, clipping guards and RAW contracts are unchanged. The default remains a beta and does not claim semantic scene segmentation, photographic intent inference or recovered detail. Sharpening remains outside this work.

## Verification

The real JPEG upload-triggered AppTest begins with no input and a legacy session, then supplies a JPEG without clicking Process image or selecting parameters. It verifies the actual LUT worker, default automatic suggestion/full strength, collapsed advanced/optional settings, exact original-size displayed PNG and matching prepared PNG/JSON. A temporary-home save matches these bytes; no personal Downloads write occurs. Worker counts stay constant during repeated reruns, download preparation, saving and overall adjustment.

A second upload produces its own source hash/dimensions/pixels and resets controls/export/saved notice. Invalid replacement clears the previous result; ordinary reruns do not retry; explicit retry and a subsequent valid source recover. Removing the input clears result/error/attempt state. Retained manual mode requires an explicit process action. A separate invalid-DNG boundary check verifies dispatch to the existing native worker with `--recipe auto` and the same no-retry/error behavior; it does not claim new real-DNG output measurements.

The first focused run failed because the test accessed a nonexistent Expander attribute; the public proto's `expanded` field fixes the test, preserving the failed log/XML. The repaired focused run passes six checks in18.78s. A final session-migration check was added afterwards and is included in the whole-suite run. Final suite/environment/service/hash results are recorded in [verification.json](verification.json) and [full-suite.xml](full-suite.xml).

The prior [shadow-control acceptance](../adaptive_lut_shadows_v3/ACCEPTANCE.md) contains twelve finite tonal reviews and actual50MPworker/PNG/manual-AppTest evidence. Those backend/source/pixel records are preserved; no new photographic-quality or50MPperformance measurement is claimed for this UI task. Original data, pinned model/decoder/native sources and earlier experiments remain intact. Live browser-client/download acceptance and independent user/personal-camera preference testing remain unverified.
