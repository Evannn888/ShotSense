# First experimental JPEG enhancement version

Delivered 2026-10-04. **Functional acceptance passed. Independent image-quality/generalization acceptance has not passed.** This is a usable local low-light prototype; the previous RAW recommendation release remains separate.

## Provenance and reproduction

- Model: [Fediory/HVI-CIDNet-Generalization](https://huggingface.co/Fediory/HVI-CIDNet-Generalization), author-published MIT checkpoint.
- Source: [official HVI-CIDNet repository](https://github.com/Fediory/HVI-CIDNet), revision `eb43d7d91e9a336c66856824ff9e4603ae41f408`.
- Weights revision: `51481ef2546f870060c43eb6d6525399f5b3d2d3`.
- Weights SHA256: `2291407125e809cc9c0614cc2d010d21d309a66eb3da33e1ee2386a68fa05894`.
- License and modifications: `src/vendor/hvi/LICENSE` and `NOTICE.md`. Architecture defaults match the model configuration. Package-relative imports, removed optional hub download mixin, and equivalent native Torch reshapes are the only upstream inference changes, alongside whitespace normalization.
- New runtime dependency: Safetensors0.6.2; existing Torch/OpenCV/Pillow reused. No training framework, hosted-photo API or einops/huggingface_hub dependency.

Prepare with `venv/bin/python scripts/prepare_jpeg_restoration.py`; the download is revision-pinned, checked before atomic installation and skipped for an existing matching file. Large weights remain local. Photo inference does not download or access the network. `manifest.json` records exact identity and architecture.

## First-version behavior

Select **AI low-light enhancement (experimental)**, upload JPG/JPEG and click **Process image**. EXIF orientation and ICC-to-sRGB decoding follow the existing JPEG contract, including rejection of malformed ICC/unprofiled CMYK. Inputs retain128MB/40MP limits. Output preserves aspect ratio and has at most960px per edge; model input is display-sRGB float RGB, padded to multiples of8. Reflect padding falls back to replication for tiny dimensions. Author inference controls: gamma1, alpha_s1, alpha_i1, both transform gates enabled; no reference/GT correction, training or model selection.

A serialized CPU subprocess has a60-second timeout. Strength blends the cached original with the restoration in display sRGB without rerunning the model. Zero preserves the exact decoded/resized pixels;100% uses the clipped model output. PNG declares stable sRGB. JSON describes the actual strength, output dimensions, model/source revisions, input hash and PNG hash. No Lightroom parameter predictions are invented. RAW1600px tone previews and experimental legacy JPEG estimates remain available separately.

## Verification

- Full suite: **53 passed in25.83s**; `pip check` passed. Existing optional colour-science Matplotlib warning only.
- New real-backend test covers odd-size EXIF rotation/ICC, strict matching checkpoint, deterministic replay, checksum/missing-weight errors, finite tiny/black output, invalid strength, exact strength-zero PNG pixels, isolated network-forbidden CLI, real application worker, and offline Streamlit AppTest before/after plus matching PNG/JSON at100/50/0%.
- Six previously inspected600x400 LOL diagnostic examples: finite, same-size, clearly changed outputs; CPU forward1.53–1.81s. See `verification.json`; reproduce with `venv/bin/python scripts/verify_jpeg_restoration.py` using retained local inputs.
-1440x960 synthetic daylight input produced960x640; forward4.43s, isolated CLI6.64s including import/decode/cache, maximum RSS1,619,116,032bytes (~1.62GB). Single measured run; not a latency guarantee.
- Existing RAW and legacy JPEG application tests passed. Production RAW ONNX is unchanged. No eval15 access.
- Verification used offline Streamlit AppTest and actual local workers; no browser visual verification is claimed.

## Visual findings and remaining limits

Reviewed fixed examples27/102/113/254/565/6: interior objects, books and foliage become more visible; extreme darkness still has conspicuous colored grain, and some shadows/colors look unnatural. Fine-detail preservation is not certified. A synthetic daylight control has mean absolute sRGB change0.2274 at100%, showing that normal bright images can be excessively altered. Reduce strength or leave enhancement at0 for unsuitable images. This release has no automatic low-light detection or denoising guarantee.

The author's complete checkpoint training provenance is unavailable. Previously inspected LOL data may overlap its training; examples are diagnostic, not independent benchmarks. No PSNR improvement, unseen accuracy or restoration of genuinely missing detail is claimed. Output is a resized enhanced image, not a full-resolution RAW renderer. Broader independent photos, dedicated noise/color evaluation and larger-resolution/memory optimization remain future work.
