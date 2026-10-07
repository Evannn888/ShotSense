# Scene-aware automatic adjustment research, 2026-10-05

This task selects a suitable implementation for a local novice photo tool. It does not modify the current automatic application or claim that scene recognition alone produces better edits.

## Frozen scope

- Review actual protected JPEG LUT/shadow and native DNG decision paths, prior17-photo defaults, installed dependencies and the macOS14.8.4 arm64 host.
- Compare native Apple Vision, MobileCLIP2, SigLIP2 and Places365 using official author documentation, released model cards, preprocessing/dependency/licensing information. Native-first selection is conditional on observed usefulness; portable alternatives remain research-only unless tested.
- Probe `VNClassifyImageRequest` and `VNDetectFaceRectanglesRequest` on unadjusted cached original PNGs from the previous17-source run. Generate aspect-preserving maximum-edge512 diagnostic thumbnails locally. DNG inputs are the existing neutral developed sRGB baseline, never expert edits or sensor RAW directly.
- Record OS/build/request revisions, full supported taxonomy, per-label score and Apple's class-specific minimum-precision predicate, face boxes, runtime and thumbnail/source hashes. Do not force top1, normalize native scores to sum1, or describe an uncalibrated score as deployment accuracy. Use precision0.90 only as a diagnostic operating point, not as new-domain90% accuracy assurance.
- Run the batch twice in one helper process to distinguish initial setup and subsequent per-image processing; separately measure three single-image helper processes on the frozen arch/portrait/sunset diagnostic cases. Report thumbnail decode+classification+faces separately from helper startup/build and full photo processing.
- Review results against previously inspected scene descriptions. These17sources are reused development diagnostics, heavily skewed to scenery/low-light with only one DNG portrait. No recognition accuracy/independent human preference conclusions or prompt/threshold tuning to this set.
- No personal-image network uploads, model training, new production Python dependencies, new sharpening, application/model/recipe/default changes or user Downloads writes. All derived thumbnails and helper binaries stay under ignored local diagnostic storage. Research source/output metadata stays under this artifact directory; protect previous runs from overwrite.

## User clarification and optional portable comparison

The user clarified personal research-only, noncommercial use during this task. MobileCLIP2 remains eligible for the research comparison rather than being excluded for product use. An isolated ignored dependency target may install pinned OpenCLIP/Timm/tokenizer support without changing the project venv or locked requirements. Prepare only the official `apple/MobileCLIP2-S0` author weights after recording revision/size/SHA256; load with `weights_only=True`, run inference offline and preserve the model license. Reuse the same17unadjusted512-edge thumbnails and fixed English descriptions in separate subject/context and lighting groups, stored before model inference. Store cosine similarities and selected-list rankings as diagnostic evidence, never calibrated class probabilities or night/intent ground truth. Compare stock author/OpenCLIP preprocessing with one predeclared full-frame letterbox view because center cropping can omit relevant people/foreground. This is a feasibility/prompt-sensitivity check, not tuned validation accuracy. Limit Torch CPU threads to1and keep text embeddings cached during measured image inference. No application integration or training.

## Native helper build amendment

The initial Swift build exceeded60s; its bounded retry produced an explicit Swift compiler/SDK mismatch and duplicate `SwiftBridging` module errors before any inference. Sources and failure logs are retained. Use the same native Vision requests through a small Objective-C/Clang helper to avoid modifying system SDK/toolchain settings or depending on Swift module compatibility. Native request revisions, thumbnails, pass counts, precision predicate and output schema are unchanged; compilation/setup is measured separately from native inference.

## Optional portable deployment feasibility probe

If strict MobileCLIP2 loading and native image inference succeed, test one FP32 image-only ONNX export (fixed batch1/3×256×256, opset17, normalized512D output) using installed ONNX/ORT. Preserve exact cached text embeddings and both fixed preprocessing views; compare ONNX and Torch normalized features/cosines/rankings on all17×2views at maximum absolute feature difference1e-4. Verify checker, source/model/prompt hashes and repeat determinism. Record file size, export/session setup, one-thread model-only p50/p95 after10warmups/50timed calls, and separately a fresh Torch-free helper's single-image preprocessing+ORT latency. These are local feasibility measurements, not recognition accuracy, full photo latency or production integration. Never install portable research packages into the production venv.

## Proposed decision structure to investigate

Subject/context evidence + source/candidate tonal/color/noise-risk measurements -> a few bounded conservative edit candidates -> reject unsafe candidates -> choose a useful adjustment or retain the natural baseline with a recorded reason. Keep artistic intent and uncertain recognition explicit. JPEG display-space color and RAW native rendering require separate contracts and acceptance. A face box is not a precise skin mask; noise proxies are not calibrated sensor noise/SNR. Do not claim cropped8-bit JPEG highlight reconstruction.

## Deliverable and acceptance

Produce a Chinese research/implementation recommendation with primary citations, a minimal integration sequence, representative acceptance coverage, local feasibility evidence and limitations. Check off research only after results/documentation/source preservation checks; production implementation and independent acceptance remain open. Full application tests are unnecessary for this isolated read-only study; execute the probe, check its output schema/reproducibility and verify frozen application/dependency/source hashes.
