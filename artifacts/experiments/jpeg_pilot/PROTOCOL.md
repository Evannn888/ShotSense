# Small paired-image dataset pilot

Fixed before pair extraction/evaluation, 2026-10-02.

The FiveK adaptive-LUT author's Google Drive folder and OneDrive listing returned 404 during this attempt. Use the accessible LOL author-linked archive as a bounded low-light data/renderer diagnostic. This is a development pilot, not the planned general FiveK retouching experiment.

Download cap: 400 MiB. Extract only 24 low/normal pairs from `our485`, selected by sorted matching filenames and NumPy seed 20261003 without examining pixels/targets. Do not extract or inspect `eval15`. Reject missing, duplicated, non-RGB or dimension-incompatible pairs. Record archive/member hashes, native PNG bit depth/mode and ICC assumptions; source PNGs remain original and separate from quality-95 JPEG input derivatives.

Compare identity, current RAW-trained experimental JPEG estimates rendered through v3, and a fixed manual +1 EV/highlight-protected v3 baseline. Additionally fit the v3 exposure/highlight curve to each pair on a fixed grid as a **target-informed diagnostic upper envelope**, never a learned model or deployable quality result. This bounds only the selected grid and current renderer, not all possible global curves. Measure paired display-sRGB MSE/PSNR, mean brightness and newly saturated pixels; verify zero-strength identity and deterministic repetition. PNG-to-JPEG encoding is measured separately. These metrics are low-light reference matching, not universal aesthetic preference or denoising acceptance.

Produce a pair manifest, per-image metrics and representative comparison sheets selected before results. Preserve production code/models. No training, hyperparameter selection on the official test set, or deployment promotion is authorized by this pilot's results. Follow-up training requires a separately fixed split/task/model protocol.
