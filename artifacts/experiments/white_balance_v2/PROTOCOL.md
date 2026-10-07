# WB v2: joint evidence / preserve warm temperature

Frozen 2026-10-06 before v2 actual-photo candidate inference. Failure-driven revision; no independent preference/default promotion. User approved. No sharpening/training/new weights/dependencies.

## Exact decision and rendering

Unchanged offline DeepWB AWB model/384px preview/11-term polynomial/rank/condition/row chunks/caps/gamut/skin-color proxy. Scene v2 and LUT/tone/shadow recipes unchanged. Unavailable recognition/document/median<.06 preserve exact source. Night skip requires night ranked first in BOTH views AND original median<.25. Similarities are not probabilities or intent.

Warm protection uses source preview candidate mask Y[.15,.90], channel range<=.22, >=128 pixels and >=2%. Mean candidate R-B>.02 AND either sunset within .04 of best in ANY view OR indoor_artificial within .02 of best in BOTH views restricts correction to tint-only. This intentionally sacrifices some true temperature-cast removal to preserve likely warm light; no Kelvin/illuminant claim.

Tint-only projects the model's luminance-neutral delta onto [1,-(.2126+.0722)/.7152,1]. It preserves R-B before quantization and original luminance; final quantization R-B drift <=1 code. Blend .5, max channel .04 (.025 at full skin-color proxy), original black/headroom and luma<=.501 code/new endpoints=0 unchanged.

Candidate order: when unprotected, full then tint-only; when warm-protected, tint-only only. Render each candidate on the rounded original 384px preview using the SAME bounded mapping as native, then run existing neutral_evidence against these actual quantized candidate pixels: >=10% presumed-neutral mean-chroma reduction, source>=.01, correction norm>=.003, >=2 supported quadrants (>=64 pixels each), all cosine>=.5. Preserve exact source if none passes. Record all attempted evidence/mode/warm-gate facts. Do not add a source gray-world fallback or retune gates after viewing results.

## Frozen inputs and review

Preserve WB v1, scene v1/v2, legacy, labels/Catalog/splits/LUT/recognition/AWB hashes. Same109 diagnostics and twelve fixed review IDs, all previous13 applied IDs, a0341/a4485/candle/flash/night/portrait/flower controls. Freeze new public titles before download and inference, exclude prior source identities:

1. window_portrait: Portrait at the window.jpg
2. restaurant_window: Man in suit interacts in a restaurant.jpg
3. restaurant_people: People in a restaurant (52378138372).jpg
4. night_market: Night at street market ,people selling shoes and clothes.jpg
5. gold_sunset: Golden Sunset Silhouette.jpg
6. green_landscape: Rocky landscape with green trees and cloudy sky.jpg

Acquire exact Commons originals once with metadata/license/author/revision/SHA1; <=32MiB each. Respect HTTP cooldown; no automatic retries/replacements. Unavailable originals remain unavailable, no claiming six acquired if fewer succeed. Decode sRGB/EXIF, prepare <=960px tagged quality100/subsampling0 JPEG diagnostic views and source hashes before candidate inference. Original public pictures may already be edited and are not WB ground truth. Retain unmodified originals. Compare actual prior WB v1 / no-WB / v2 pipelines, inspect every applied v2 + previous13 + fixed/negative + every fresh photo. Expert C distances are descriptive only and never decision inputs.

## Gates

Run meaningful projection/mood/actual-bounded-evidence/failure tests, exact default route and same classification, repeated deterministic outputs, tagged native PNG and decoded-zero, actual positive Streamlit AppTest PNG/JSON/cache with new version. Perform actual20MP native worker and nonzero mapping controls; record timings as observations only. Keep artifacts immutable after completion. Visual review can reject new severe failures; failed frozen variant remains recorded. The scene beta stays default-off; retain unresolved a0341/mixed-light/colored-object/skin-proxy/intent limits. Integration only after these functional and fixed diagnostic gates.
