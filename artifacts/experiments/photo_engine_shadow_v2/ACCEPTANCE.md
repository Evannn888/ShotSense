# Foreground correction — 2026-10-04

Decision: add **Lift shadows** as an explicit experimental recipe and **Apply shadow lift** as a direct same-source action. The user's screenshot showed that gentle0.35EV global lift was inadequate; their review of Shadows45 requested a clearer foreground. Select native RawTherapee5.11 tone equalizer80/55/20/0/0/0 with zero global exposure, regularization0/pivot0, layered on the existing neutral/camera-WB configuration. No new sharpening, denoising, WB/saturation adjustment, model, dependency or training. Automatic selection, its0.1% clipping guard and the manual default stay unchanged.

## Evidence and decision

The [initial protocol](PROTOCOL.md), [initial manifest](manifest.json) and [initial report](report.json) freeze11 inputs: a screenshot-derived proxy, three previously inspected public JPEGs, three previously inspected RAWs and four synthetic controls. Compare the unchanged gentle profile to native RGB Shadows45 and equalizer50/30/10. Preserve all profiles, reports and pre-change runtime/test snapshots. User review then requested more foreground clarity; [the follow-up protocol](foreground-feedback/PROTOCOL.md) froze stronger Shadows65 and equalizer80/55/20 on the same input hashes/regions before further scoring. [The follow-up decision](foreground-feedback/DECISION.md) was written before integration. These are targeted development regressions, not fresh reserved acceptance images or new model-test metrics.

The attached screenshot's left-image interior was color-managed from its embedded Display profile into sRGB, cropped580×460, then encoded as a high-quality JPEG through the existing decoder. [Provenance](proxy-provenance.json) retains screenshot/profile/crop/proxy hashes. It cannot establish original JPEG processing, native full-resolution texture, or restored missing detail. User-photo derivatives/media remain local and ignored; no external upload occurred. Browser asset reads failed on repeated CDP focus timeouts; no workaround or live-browser verification is claimed.

Fixed region diagnostics use mean encoded-sRGB weighted luma, not linear EV or photographic scores:

| Screenshot proxy | Foreground mean | Sky mean |
|---|---:|---:|
| Decoded proxy | 0.1045 | 0.5610 |
| Existing gentle lift | 0.1214 | 0.6256 |
| First Shadows45 | 0.1761 | 0.6223 |
| Stronger Shadows65 | 0.2271 | 0.6478 |
| Selected equalizer80/55/20 | 0.2154 | 0.5844 |

Selected foreground is+106.2% versus proxy original and+22.4% versus the first reviewed Shadows45 result; sky+4.2%. Select it for useful foreground visibility with less sky alteration than Shadows65. These ratios are descriptive display-luma changes, not proof of recovered texture or independent preference. The user's exact original JPEG remains unavailable.

Viewed both rounds' three full-frame/three native-detail sheets and proxy pairs. No conspicuous new broad cast, severe detail destruction or edge halo was observed on these limited views. Existing colored grain becomes clearer in dark interiors and the public portrait eye. The wool center crop is source-defocused and cannot validate sharp-knit detail. Sunset foreground remains relatively dark. Reject Shadows65 for this integration because its sky shift and portrait flattening are greater. Shared working-RGB tone gain precedes output color conversion, so exact output hue and unclipped highlights are not guaranteed.

The selected proxy has zero new full-value-channel pixels. The inspected public JPEG/RAW fixtures remain below0.1%, while the neutral ramp introduces one new full-value column (0.390625%). This prevents a zero-clipping claim and automatic promotion. Black stays black; dimensions/bounds, exact zero strength and all11 immutable input hashes pass. [Follow-up report](foreground-feedback/report.json) records full per-image/profile/input/output identities.

Local visual artifacts: [selected screenshot-proxy pair](foreground-feedback/proxy-equalizer-comparison.png); full frames [1](foreground-feedback/overview-1.jpg), [2](foreground-feedback/overview-2.jpg), [3](foreground-feedback/overview-3.jpg); native crops [1](foreground-feedback/native-crops-1.jpg), [2](foreground-feedback/native-crops-2.jpg), [3](foreground-feedback/native-crops-3.jpg). PNG/JPEG media remain ignored local files.

## Implementation and functional evidence

Runtime version `rawtherapee-recipes-v2` exposes `--recipe shadows` and the corresponding UI selection. The new profile is byte-identical to the selected frozen candidate; it layers neutral plus shadows, never the gentle exposure profile. Existing explicit actions share one callback/worker, verify source SHA before reprocessing, reset100% strength, and clear stale results on recipe changes. JPEG zero strength restores decoded color-managed pixels; RAW zero restores neutral camera-WB engine development. Current result captions identify the actual recipe. PNG display/download bytes and matching JSON hash/strength/dimensions remain coupled.

Focused functional check: two tests passed28.13s, covering a textured JPEG shadow/sky/black fixture through the actual bounded worker with deterministic engine replay, stronger dark-tone response than gentle, same-size PNG/hash/zero strength, and the real same-source RAW UI shadow action with profile isolation, strength reset and export agreement. Full suite: **61 passed**, no failures/skips,69.14s. A historical test snapshot was initially collected under a test_ filename; its old session-key check failed against current code. Renamed the snapshot without changing bytes and reran full discovery; see [the collection correction](test-collection-correction.md). Complete-suite result and final source identities are recorded in [verification](verification.json). Test success establishes routing/export contracts, not photographic quality. No new dependencies; the existing optional Matplotlib warning remains.

The pinned PP3 keys and native algorithms were checked against official5.11 sources: [profile serialization](https://github.com/RawTherapee/RawTherapee/blob/5.11/rtengine/procparams.cc), [tone equalizer](https://github.com/RawTherapee/RawTherapee/blob/5.11/rtengine/iptoneequalizer.cc), [Shadows/Highlights](https://github.com/RawTherapee/RawTherapee/blob/5.11/rtengine/ipshadowshighlights.cc). Their local hashes are recorded in verification. Native tonal filtering is not segmentation, sharpening, detail reconstruction or denoising.

Independent original-file eyes/skin/fur/mixed-light/backlight/noisy/intentional-dark personal-camera review remains required before broad naturalness claims or automatic-default selection. The screenshot proves the visible weakness and permits targeted proxy diagnosis; it does not close that acceptance gate.
