# Separate denoising and illumination: bounded development pilot

Fixed before new image extraction or training, 2026-10-02. The user approved the next step after the spatial residual pilot failed its noise/visual gates. This is isolated JPEG research, not a RAW parameter model change.

## Data and controls

Use the verified local author-linked LOL archive. Preserve the previous 120 inspected `our485` pairs for training/regression and add 48 filenames selected without pixels using seed 20261010. Carry prior scene groups forward; review all new low/reference thumbnails and all hash candidates against old images before splitting. Keep old IDs and related scene groups training-only. Reuse the prior native edge-phase alignment screen: response >=0.2 and native shift >2px quarantines the pair; low responses remain explicitly uncertain and are reviewed. No warping or official eval15 pixels. Freeze about 16 new validation pairs in at least eight groups using seed 20261011. Stop if insufficient eligible groups; do not silently change sampling or split.

Freeze the noise-aware seed42 shared-curve/bilateral pipeline and original training statistics before any new scores. Native image controls: identity, existing-v3, frozen shared base, and fixed denoised base. Apply OpenCV fastNlMeansDenoisingColored to the encoded base with h=7, hColor=10, templateWindowSize=7, searchWindowSize=21. No new dependencies or search over filter settings. This denoiser is a fixed recipe, not physically calibrated noise removal.

Train a matched global scalar illumination control and one small spatial illumination candidate on the same crops/objective/optimizer/budget. The control has one scalar; the candidate uses three 3x3 convolutions (3->8->8->1) with ReLU, zero final weights and 881 parameters. Input is only the denoised RGB, area-downsampled by four; bilinearly upsample the scalar field. Its nominal low-resolution context spans roughly 28 native pixels, not a full semantic scene model. No additive RGB offset or learned denoiser. Treat the frozen denoiser and predicted illumination separately.

## Rendering and loss

Illumination multiplier g=exp(log(4)*tanh(logit)), bounded [0.25,4]. Apply the shared shoulder function output=1-(1-base)^g, preserving exact base zero/one endpoints and neutral gain one. Local gain is spatially smoothed by low-resolution prediction/interpolation but does not guarantee noise reduction, texture recovery or global monotonicity across different images. Strength zero returns original source exactly; blend for intermediate strength.

Compute source/base/denoising at native resolution before five fixed 96px crops. Exclude a 16px loss border to reduce crop-padding/context differences. Use the inherited RGB/gradient/chroma/flat-variation loss plus 0.1 times absolute RGB error where reference luminance <0.05, normalized by mask occupancy. This explicitly penalizes inaccurate lifted black regions rather than forbidding legitimate shadow detail. No feature normalization is fitted for this CNN/control; parent statistics remain frozen.

CPU, one Torch/OpenCV thread. AdamW lr=0.001, weight_decay=0.0001, batch16, maximum30 epochs/patience6, validation-crop objective selection including neutral epoch0. Train global seed42 and spatial seed42 first. Train spatial43/44 only when all seed42 numerical gates pass. No accelerator retrial or settings grid. Save exact parent/data/source/configuration/checkpoint identities and histories; score full native images on CPU.

## Fixed gates and review

On the new grouped development set require spatial mean per-image PSNR >= frozen base +0.5dB and >= trained global control -0.25dB; dark median-residual proxy <=90% frozen base; target chroma MSE <= frozen base; target luminance-gradient error <=110% frozen base; mean new full-channel fraction <=0.5%. The proxy also includes texture/edges; no calibrated noise or independent scene significance claim. Confirm >=2/3 spatial seed means improve PSNR if three seeds run.

Review first eight sorted new validation IDs as full/native-center/native-edge comparisons; also old102/27 regressions excluded from validation. Reject visible grain/casts, gray raised shadows, highlight/color clipping, seams and fine-detail loss even if numerical gates pass. Do not relax gates or fit more settings on the inspected set. Independent unseen personal images, final frozen evaluation, export/runtime/latency and UI acceptance precede any integration. Preserve existing production app/model/renderer and browser restrictions.
