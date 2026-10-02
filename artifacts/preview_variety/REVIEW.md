# Multi-scene Preview Check (2026-10-02)

Current production model and linear-raw-protected-v2 with highlight protection enabled. Compare baseline/100%/75%. Selected scenes visually from fixed-seed candidates before inference, without expert labels, retraining, or model selection. Eight training photos are exploratory checks; four formal test photos are additional checks, not a population accuracy estimate.

| Scene | Split | Recommended EV | Observation |
|---|---|---:|---|
| portrait | train | 0.76 | Face and clothing are brighter; 100% looks slightly flat, while 75% is more conservative. No obvious new color cast at display size. |
| dark-interior | train | 2.12 | The 2.12 EV recommendation substantially improves subject visibility in the dark interior. Background also brightens; skin-tone correction is outside this check. |
| night-building | train | 1.12 | Staircase and architectural detail are clearer; colored lighting and the nighttime atmosphere remain visible. |
| backlit-ruins | train | 0.19 | Sky is darker but the building lifts only slightly; subject improvement is limited. |
| snow-mountain | train | 0.64 | Snow and dark rocks brighten while mountain texture remains visible. Local contrast is slightly weaker at 100%. |
| sunset | train | 0.39 | Warm color and silhouette atmosphere remain, with small changes. Existing solar clipping must not be described as recovered detail. |
| colorful-textiles | train | 0.62 | Textiles brighten without an obvious new hue shift. Validated WB/saturation adjustment was not applied. |
| cloudy-lake | train | 0.06 | Exposure recommendation is near zero, so changes are small; clouds darken slightly. |
| test-portrait | test | 0.67 | Face visibility improves. 100% looks slightly flat; 75% retains more of the original tonal separation. |
| test-backlit-sky | test | 1.01 | Dark trees improve slightly and the sky retains layers, but foreground remains dark, showing the limits of global adjustment. |
| test-night-street | test | 1.04 | Facades and pavement are clearer without obvious new light spreading or color cast at display size. |
| test-mountain-valley | test | -0.01 | Large pre-existing clipped cloud areas turn gray/flat when darkened without recovering texture. This case is not a sufficient visual improvement. |

All 12 inference jobs/aspect-preserving previews/two strengths succeeded. Zero-strength PNGs matched baseline byte-for-byte. Neither strength introduced full-channel pixels; this clipping diagnostic does not prove texture recovery. Runtime approximately 2.53–3.52 seconds per photo including multiple PNG/comparison writes.

Limitations: backlit subjects improve only partly; darkening pre-existing clipped clouds may make them gray. Global curves cannot replace local masks, real RAW reconstruction, or validated WB. No obvious new halos at display size; pixel-level noise and Lightroom-reference checks were not performed.

Reproduce: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python -m scripts.check_preview_variety`. See report.json for measurements/paths. Overview columns are baseline/100%/75%. Original per-photo PNGs are retained locally.
