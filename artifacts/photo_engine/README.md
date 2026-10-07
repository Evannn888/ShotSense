# Pinned photo engine

`manifest.json` records the official RawTherapee5.11 universal Mac archive, CLI and application-tree hashes. `scripts/prepare_photo_engine.py` verifies the archive and full installed application tree; inference verifies the pinned CLI checksum. Re-run preparation to audit an installation that was changed outside ShotSense. The CLI dynamically uses `/Applications/RawTherapee.app`; this adapter is specific to that Mac layout. Original binaries remain separate, unmodified and excluded from Git.

Upstream RawTherapee is GPL-3.0-or-later. The separately installed app includes its own notices; this repository does not vendor its executable or source. See the [pinned upstream source](https://github.com/RawTherapee/RawTherapee/tree/5.11) and [license](https://github.com/RawTherapee/RawTherapee/blob/5.11/LICENSE). These partial PP3 files are ShotSense configuration, not predictions of Adobe parameters.

- `neutral.pp3`: explicit sRGB output, no JPEG WB/auto exposure/look table/sharpening/denoising/crop/resize. Unspecified modules use the pinned5.11 neutral defaults; do not add `-d` or `-s` to load user defaults or input sidecars.
- `natural.pp3`: adds0.35EV, engine HighlightCompr40 and threshold80. Those controls are not Adobe HighlightRecovery.
- `chroma.pp3`: adds manual Lab chroma filtering5, luminance filtering0, AutoGainfalse. This is optional because scene statistics do not provide a validated noise estimate.
- DNG adds an ephemeral override enabling camera WB and cameraICC input. The neutral development is the comparison baseline, and can itself clip or fail to match a camera's preferred rendition.

The adapter copies input into a temporary folder and uses fresh `RT_SETTINGS`/`RT_CACHE` paths. EXIF/ICC-normalized JPEG pixels are supplied as a lossless sRGB PNG. Engine outputs are checked for format, dimensions and pixel bounds. PP3 file hashes appear in each result; changing them changes the recipe contract and requires a new photo review. The frozen provisional auto selector and clipping guard are documented in the [pilot protocol](../experiments/photo_engine_v1/PROTOCOL.md).
