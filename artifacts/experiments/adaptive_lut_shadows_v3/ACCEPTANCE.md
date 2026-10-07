# Independent natural-color shadow control — 2026-10-05

The user approved stronger foreground tonal separation and an independent shadow control, excluding sharpening. The opt-in JPEG Natural color workflow now adds a brightness-dependent shadow curve after the existing fixed natural-color result. The pretrained predictor, basis LUTs and previous natural-color pixel calculation are preserved. Manual remains the default workflow.

The [frozen protocol](PROTOCOL.md) preceded implementation. Twelve previously inspected sources and their previous natural-color PNGs were hash checked before the new stage. The [conditional UI amendment](UI_AMENDMENT.md) followed finite scene, synthetic and runtime checks. This is development regression evidence; it does not establish independent photographic preference or general camera acceptance.

## Behavior

- **Shadow lift** selects 0–100%, initially using a source-brightness suggestion capped at 75%. Suggestions are a provisional histogram rule, not calibrated confidence or scene understanding.
- Shadow 0% preserves the previous natural-color base, including its mild brightness adjustment. The separate overall **Adjustment strength** 0% restores the decoded original. Intermediate overall strengths blend the actual shadow-adjusted result with the original.
- Every shadow change starts from the same cached original-resolution natural base. One current result is cached; repeated slider changes do not accumulate edits or rerun model inference.
- The new stage uses one common RGB gain per pixel. Deepest blacks and luminance at or above 0.55 remain unchanged by this stage; upper midtones taper and channel headroom limits saturated colors. It is a global luminance mapping, without semantic masks or spatial processing. Equally dark clouds can brighten along with foreground; artistic silhouettes may need a lower setting.
- JSON records natural-base model/settings/hash separately from actual shadow strength, selection, version and input/output pixel hashes. Display, prepared PNG, native crop and explicit local save all use the current adjusted pixels and overall strength.
- Input/workflow changes clear shadow state. Processing failures and old results lacking the current natural base/version cannot display or export a stale adjusted result. JPG/JPEG uploader support and existing 64MP/128MiB bounds remain unchanged; the WebP diagnostic is offline only.

## Finite tonal review

All twelve original/previous-natural/new-auto full images and native regions were inspected, along with neutral ramps, color/skin patches and neutral edge/grain controls. Foreground rock, grass and tree layering improves without conspicuous new hue casts or spatial halos on this inspected set. No new all-black pixels or full-value channel pixels occur in these results. The neutral ramp remains ordered and neutral; the sampled continuous shadow curve has minimum slope 0.51214. Common RGB gain preserves channel ratios to quantization. Tonal mapping can change the visibility of existing noise; it does not reconstruct missing information.

On the supplied cloudy-tree photo, default shadow lift is 50%. Fixed rectangle mean display-sRGB luminance changes from previous natural to new auto are foreground 35.40→47.61 (+34.5%), tree rectangle 40.68→47.76 (+17.4%) and sky 104.32→107.75 (+3.3%). These are descriptive averages including backgrounds, not semantic masks, physical exposure measurements or quality scores. The arch suggests 75%, portrait 30%, overcast lake 10%, and bright sunset/WebP 0%. Full per-photo results are in [results.json](results.json); synthetic checks are in [controls.json](controls.json).

Several sources may overlap the external checkpoint's FiveK training data, and all twelve scenes were already inspected. Source blur, white clipping, noise and unknown photographic intent remain limitations. Sharpening was excluded from this work and evaluation. No automatic-default promotion or independent user preference acceptance is claimed.

## Actual high-resolution delivery

Two fresh serial CPU JPEG model workers fit the existing 60-second parent deadline. Both preserve source hashes, dimensions, source-to-base-to-shadow hash chains, exact frozen PNG pixels and independently computed 50%/native-crop/zero-strength results.

| Source | Dimensions | Worker | Native PNG | Sampled process-tree RSS |
|---|---:|---:|---:|---:|
| Terraces | 8192×6144 | 33.48s | 7.39s | 1.39GiB |
| Mountain lake | 8688×5712 | 26.56s | 5.28s | 1.44GiB |

Worker time includes decoding, model/protection/shadow processing and compressed pixel cache, but excludes PNG export. Additional full/half/zero verification takes 60.31/45.34 seconds for the complete diagnostic scripts, outside the single-worker timeout. RSS is a 250ms sampled sum that may double count shared pages and miss brief peaks; it is not a precise isolated peak allocation. See [first worker](worker-1.json) and [second worker](worker-2.json).

An isolated AppTest consumes the actual 8192×6144 worker result, verifies the automatic output, then adjusts shadow to 40% with overall strength 50%. Full displayed/downloaded/saved PNG pixels and JSON match independently computed pixels; a native 320px crop is exact. The manual shadow stage takes 11.71s and the complete diagnostic 52.97s, including repeated PNG encodes/checks. The tested current PNG is 75,987,829 bytes. Local save uses a temporary home, without writing to personal Downloads. This is original-resolution processing, not an instantaneous low-resolution slider preview. See [app report](app-report.json).

## Functional verification and limits

Focused tests pass (4 in 7.10s). They exercise the real JPEG button/worker, automatic/manual shadow strengths, base preservation, no model rerun, no accumulation, overall 0/50/100%, native crop, exact PNG/JSON/local save, export invalidation, controlled failure, prior-version quarantine, input/workflow reset and invalid inputs. Final whole-suite and environment results are recorded in [verification.json](verification.json) and [full-suite.xml](full-suite.xml).

[Integrity checks](integrity.json) retain twelve source hashes, native engine/decoder/PP3/dependency hashes and seven core RAW/model files matching HEAD. Current implementation and relevant existing sources are preserved in source_snapshot.zip; pre-edit snapshots and previous v1/v2 experiments remain intact.

This validates programmatic rendering and export construction. Live browser download/client behavior and independent personal-camera/user preference acceptance remain unverified. No new dependencies, training, RAW-label changes, extra uploader formats or automatic photo transfer were introduced.
