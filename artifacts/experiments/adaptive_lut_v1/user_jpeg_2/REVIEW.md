# User cloudy-trees JPEG trial ·2026-10-05

Unchanged pinned paired-sRGB adaptive-LUT JPEG pipeline tested on Downloads/images (2).jpeg. Native678×452RGB,25833bytes; EXIF/ICC handled by the existing decoder. Frozen source and current code identities in input.json precede inference; no resizing, sharpening, training or source overwrite.

Save separate original/50/100%sRGBPNG under data/user_photo_diagnostics/lut-jpeg-2/. Verify exact PNG pixels against each strength blend, native dimensions/mode/sRGB/output hashes,0%original identity, input and app/engine/JPEG/LUT code preservation. Prior72-test evidence retained without a fresh suite run for this diagnostic. Inference with model setup0.0967s is not a web workflow benchmark.

Visual assessment:50%brightens cloud tones and gives restrained warmer grass, with trees still dark;100%strengthens blue/cloud and gold/green separation while making the tree interiors more solid silhouettes. New all-black pixels4.0502%at100% and0.3567%at50%; no new full-value channels at100%. These are quantized output counts, not exposure ground truth. Reviewer provisionally favors50%for natural adjustment;100%is a dramatic style with shadow-detail loss. No claimed reconstruction, new user preference or automatic promotion. Report retains coefficients/model/LUT/pixel identities and unclamped diagnostics.
