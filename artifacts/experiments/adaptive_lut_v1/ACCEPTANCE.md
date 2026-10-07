# Paired sRGB adaptive-LUT pilot ·2026-10-05

The pinned Image-Adaptive-3DLUT checkpoint runs locally on CPU, preserves full dimensions and exports matching sRGB PNGs. **Reject this checkpoint as a natural automatic adjustment/default.** Its observed dark-detail loss, strong color changes and gray-ramp reversals fail the frozen photographic gate. Keep the isolated comparison/CLI; no website integration, default change or training. This decision concerns this checkpoint and adapter on these diagnostics, not all adaptive-LUT methods.

## Implementation and provenance

Use the author's matched paired-sRGB `classifier.pth`/`LUTs.pth`, combined2,377,713bytes, at revision `b491f6df64a588864739a157db271e5c848e1805`. Exact expected sizes/Git blob IDs/SHA256 are checked before strict `torch.load(weights_only=True, map_location='cpu')`; variant, keys, shapes and finite values are checked. Retain the author sources and Apache2.0 license, with modifications attributed in `src/adaptive_lut.py`. Weight/training-data rights for a later distributed product are not independently established by a repository declaration.

Reuse installed Torch/NumPy/Pillow and the existing EXIF/ICC JPEG decoder/strength/PNG renderer. No old Torch extension/training stack, dependency additions or inference-time network requests. The256×256 bilinear classifier mixes three basis tables with the author's unconstrained coefficients. Bounded input sampling and128-row current Torch interpolation avoid full-image neural activations and preserve the author's B,G,R volume axes and1.0001 endpoint interval.

The CLI currently accepts JPEG only. The seven DNG comparisons reuse frozen neutral camera-WB sRGB development; they do not implement a new live RAW workflow or alter RAW training/preprocessing. JPEG128MiB/64MP bounds, existing DNG40MP rules and current renderer remain intact. Full-resolution output does not reconstruct blur, compression artifacts or clipped detail.

## Numerical and export checks

Extracted author CPU interpolation agrees within1.43e-6 float maximum absolute difference in the recorded dim33 check (dim17:7.15e-7). Tests cover asymmetric channel tables, endpoints and random input/table values; identity-table8-bit ramps remain exact. Same-input adapted/original classifier outputs are bit-identical. The final small/ordinary predictor checks show resize difference at most1.79e-7 and weight difference3.58e-7; actual50MP checks are recorded separately in `predictor-parity-highres.json`. These are numerical parity checks, not photographic quality scores.

Retain the initial C++ header build failure and first2pass/1fail regression. Tiny uniform images exposed rounding amplified by InstanceNorm; bounded small inputs now use native Torch resize. The corrected focused run passes3tests in2.21s. Final full suite: **72passed in82.82s**, one existing optional Matplotlib warning. `pip check` and `git diff --check` pass. All ten source hashes, current app/engine/JPEG/four recipe hashes and seven core RAW/model files are preserved; the core files matchHEAD.

## Actual high-resolution workers

Fresh sequential CPU CLI workers use the existing app's60-second parent process/cleanup helper. The worker time includes decode, model setup/inference and full-array NPZ serialization; PNG rendering is additional. The comparison's4.44/4.17s LUT times have the model already loaded and are not full workflow times.

| Source | Native size | Worker call | PNG render | Complete diagnostic check | Sampled tree RSS |
|---|---:|---:|---:|---:|---:|
| Terraces |8192×6144|16.74s|7.23s|42.81s|1.46GiB|
| Lake |8688×5712|12.68s|5.06s|31.25s|1.19GiB|

The complete checks also load arrays, verify full PNG pixels/hash/sRGB/dimensions, independently verify a50% native crop, and verify exact0% original identity. Both match the frozen full-strength PNGs. The250ms RSS process-tree sum includes the sampling command, can count shared pages twice and miss short peaks; it is not a physical-memory guarantee. No automatic Downloads write or browser-delivery claim.

## Photographic decision

Review original/currentv5 shadows/LUT100 at overview and three native420×280 regions per each of ten predeclared photos. The three user-downloaded JPEGs and seven previously inspected FiveK DNGs are diagnostics, with possible external training overlap; there is no independent holdout, expert target scoring or recorded user preference for these new results.

| Scene | Observed LUT100 result |
|---|---|
| User terraces | Much stronger yellow/green and road contrast; dark terrace rows lose tonal separation.2.668% newly all-black pixels. |
| User lake | Stronger blue/gold separation; dark water/foliage become heavier and existing compression blocks more conspicuous.2.217% newly all-black pixels. |
| User stone arch | Sky changes little; rock/arch shadows darken severely, opposing the requested clearer foreground.5.881% newly all-black pixels. |
| Alpine valley | Stronger grass contrast; bright clipped cloud areas become flat gray without restored texture, while trees darken. |
| Yarn | More vivid red/purple with source softness preserved; global color change is substantial. |
| Acacia | Bluer sky and darker shrubs; native sky/grain becomes more visible. |
| Overcast lake | Stronger yellow-green foliage; muted original atmosphere changes. |
| Portrait | Warmer/brighter skin and more saturated background; pre-existing cyan/magenta eyeglass fringes stand out more. No claimed skin accuracy. |
| Sunset | Stronger orange glow with heavier dark foreground; stylistic intent remains unknown. |
| Ruin | Darker stone and deeper warm contrast lose subtle wall layering compared with currentv5. |

Counts supplement visual observations; newly all-black means an outputRGB(0,0,0) pixel whose input was notRGB(0,0,0), not sensor exposure ground truth. Preserve unclamped below-zero/above-one counts in `results.json`; quantized255 counts alone conceal clipped float values rounded below255.

The independent50% blends on all three user photos reduce casts/crushing and have no new full-value channels. They still darken the arch foreground; they are not promoted as a fixed automatic style. Synthetic neutral ramp shows9/14/9 decreasing R/G/B steps, maximum3-code step reversal and4-code neutral channel spread, with visible high-tone irregularity. The dark-grain control's per-channel standard deviation rises from2.00 to4.58–4.86 at100% (3.23–3.65 at50%); this is stronger noise contrast, not denoising. Edge geometry remains fixed; inspected patches do not establish universal absence of artifacts.

Next work should address restrained strength, dark-detail/monotonic-tone protection and explicit preferred style examples before reconsidering automatic integration. SepLUT or another checkpoint remains a separate conditional comparison, not demonstrated better by this pilot.

## Reproduction and retained evidence

From the repository root:

```sh
venv/bin/python scripts/prepare_adaptive_lut.py
venv/bin/python -m src.adaptive_lut /absolute/path/input.jpg --output result.json --preview-source result.npz
venv/bin/python -m pytest -q tests/test_adaptive_lut.py
```

The inference command produces full original/adjusted arrays and LUT metadata, with no file overwrite of the input. PNG comparisons use existing `render_photo` and remain under local `data/user_photo_diagnostics/lut-v1/`. Frozen10-photo comparison, synthetic/half diagnostics and actual-worker scripts are alongside this report. `trilinear-reference.c` is test-only; on this Mac build with `xcrun clang -O2 -shared -fPIC artifacts/experiments/adaptive_lut_v1/trilinear-reference.c -o data/external_models/image_adaptive_3dlut/trilinear-reference.dylib`. No compiler is needed for ordinary inference. Model-dependent tests explicitly skip when the local bundle/reference is absent.

Retain protocol/input/baseline/model manifests, upstream source/license, initial failures, parity reports, full/half/synthetic results, actual-worker reports and final verification/source snapshot. Personal full-resolution PNGs and diagnostic sheets stay local; `overview-1.png`, `overview-2.png`, `user-photos-strength-comparison.png`, `synthetic-controls.png` and each photo's `native.png` expose the recorded failures for review. Current website/service/settings/model/recipes are preserved.
