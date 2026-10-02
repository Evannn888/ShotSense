# ShotSense

A local research prototype that recommends photo adjustment parameters from unedited DNG files with valid camera white balance information. ShotSense combines EfficientNet-B0 semantic features with 132-dimensional linear Lab statistics to predict the arithmetic mean of five FiveK experts' parameters.

Recommendations are absolute values for legacy Camera Raw **PV2003**. `HighlightRecovery` is not modern `Highlights`. Only exposure and highlight recovery passed the recommendation gate; contrast, saturation, temperature, and tint remain experimental.

## Run locally

From the project root:

```sh
venv/bin/python -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501/, upload an unedited DNG, or select **Use project sample**. Review parameters and download JSON. Inputs are limited to 128 MB and 40 megapixels, with a 60-second timeout. The app processes one RAW job at a time locally.

The page shows baseline RAW development and an approximate adjusted preview side by side, preserving the original aspect ratio with a maximum edge of 1600 pixels. Highlight protection is enabled by default. Adjustment strength ranges from 0–100%; applied values and clipping diagnostics appear on the page and in JSON. PNG downloads reflect the current settings. Only validated exposure and highlight compression are simulated. This custom preview is not equivalent to Lightroom and cannot guarantee recovery of clipped detail.

## Environment and reproduction

Verified on macOS arm64 with Python 3.9.8. Set up the development and training environment:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.lock.txt
venv/bin/python -m pip check
venv/bin/python -m pytest -q
```

The full test suite requires the locally retained dataset and caches. Original data live in `data/raw/dngs/` and `data/raw/fivek_dataset/raw_photos/fivek.lrcat`. The Catalog is read-only, invalid labels are quarantined, and previous labels are preserved as immutable snapshots. Training requires pretrained weights at `artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`; preparing those weights initially requires network access. ONNX inference runs offline.

```sh
venv/bin/python -m src.extract_labels
venv/bin/python -m src.preprocess --limit 20 --workers 1
venv/bin/python -m src.train --pilot --metadata data/processed/pilot/metadata.npz --output-dir artifacts/pilot_model --epochs 30
venv/bin/python -m src.export_onnx --checkpoint artifacts/pilot_model/best.pt --metadata data/processed/pilot/metadata.npz --output-dir artifacts/pilot_model
venv/bin/python -m scripts.prepare_data --workers 6
venv/bin/python -m src.train
venv/bin/python -m src.export_onnx
```

Caches are identified by processing-code, dependency, label, and configuration hashes. Repeated runs resume and rebuild corrupt files. Configuration changes require a new output directory or explicit `--rebuild`. Pilot and full data, splits, and models are separate. The full photo-ID split uses seed 42 and an 80/10/10 ratio. Feature standardization uses only training data. Burst/near-duplicate grouping is unavailable, so this split does not establish that all similar-scene leakage has been excluded.

## Command-line inference

```sh
venv/bin/python -m src.inference path/to/input.dng --output result.json --preview input.jpg
```

JSON separates `recommended_absolute` and `experimental_absolute`, with units, legacy process semantics, model/data versions, and timing. No Delta is returned without a baseline using the same parameter version. ONNX inference does not import torch/torchvision; it still requires NumPy, ONNX Runtime, rawpy, colour-science, OpenCV, and Pillow.

## Acceptance evidence

- `data/intermediate/label_audit.json`: field sources, evidence for omitted defaults, invalid IDs, 43 repaired fields, process versions, and label hashes.
- `data/intermediate/color_audit.json`: linear/D50 configuration and pixel comparisons between shared and separate RAW contexts.
- `data/intermediate/preprocessing_benchmark.json`: throughput and peak memory for 1/2/6 workers on the same pilot.
- `data/processed/manifest.json` and `splits.json`: accounting, versions, and mutually exclusive splits. Of 5000 inputs, 4946 succeeded, 3 had invalid labels, and 51 lacked valid camera WB.
- `data/processed/input_support_audit.json`: per-file RAW-header support audit and original failure report. Unknown processing errors still fail; automatic WB is not silently enabled.
- [Model evaluation](artifacts/model/evaluation.json): constant/physical-linear/semantic/dual comparisons, final test metrics, expert disagreement, and camera/Catalog-tag/tail/As-Shot WB strata.
- [Deployment report](artifacts/model/deployment_report.json): PyTorch/ORT parity and CPU forward p50/p95, reported separately from RAW end-to-end time.

Beating a baseline on a fixed validation set is evidence of research effectiveness; acceptable error for practical use remains uncalibrated. Six parameters cannot fully reproduce expert development. Modern Lightroom mapping, faithful high-resolution development, additional RAW formats, and edited JPEG inputs are outside the verified scope. Final test macro-average error was 18.7% below the constant baseline; exposure MAE was 0.281 EV and highlight recovery MAE was 7.428. FP32 CPU forward p95 was 19.39 ms. All 26 checks passed after English localization. See [acceptance](artifacts/model/ACCEPTANCE.md) and the [master plan and development log](SHOTSENSE_MASTER_PLAN.md).

## Model improvement roadmap

The [detailed model improvement plan](MODEL_IMPROVEMENT_PLAN.md) defines grouped evaluation, validation-only experiment selection, matched baselines, aspect-preserving semantic inputs, a bounded head search, and conditional partial backbone fine-tuning. It includes three-seed confirmation, proposed promotion gates, deployment checks, and development records. Planning is complete; these experiments have not started. White-balance context, expert styles, and regional editing are conditional later work.

## Preview development

The [preview improvement plan](PREVIEW_IMPROVEMENT_PLAN.md) records each stage. Three-photo v2 results are retained in [the v2 report](artifacts/preview_v2/acceptance.json). Current reproduction scripts call the current renderer; historical results identify their original source hashes.

The current renderer is `linear-raw-shoulder-v3`: normal midtones retain linear exposure gain, highlights use a continuous shoulder, and baseline channel-ceiling diagnostics appear in the page and JSON. These diagnostics cannot establish sensor overexposure or texture recovery. See the [16-photo v2/v3 review](artifacts/preview_v3/REVIEW.md). Reproduce with `venv/bin/python -m scripts.compare_preview_v3` after preparing local data. Browser acceptance of the latest preview remains incomplete because saved browser permissions block localhost access.

![Baseline, v2 and v3 comparisons](artifacts/preview_v3/overview-3.jpg)

## Repository contents

The repository includes code, pinned dependencies, the master plan/development log, the production ONNX model and training checkpoint, acceptance JSON, comparison JPGs, and an English snapshot/provenance record. Original FiveK DNG/Catalog files, extracted labels, preprocessing/semantic/pretrained-backbone caches, the virtual environment, and duplicate PNG previews remain local.

The included production model can process your own supported DNG without retraining. **Use project sample** requires a DNG under local `data/raw/dngs/`. Full data tests, training, and comparison reproduction require separately prepared data and caches. `data/` audit paths and individual PNGs in reports refer to locally retained evidence.

Original historical ZIP snapshots and the initial Chinese-interface screenshot are preserved locally and in earlier Git history, rather than altered and presented as original evidence. Their hashes and purpose are recorded in [snapshot provenance](artifacts/SNAPSHOT_PROVENANCE.md). Current documentation and application text use English; prior commits remain intact.
