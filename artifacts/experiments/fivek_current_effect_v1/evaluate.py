"""Compare unchanged real automatic JPEG processing on99prepared development pairs."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.lut_natural import process_jpeg

DATA = ROOT / "data/external/fivek_current_effect_v1"
PRIOR = ROOT / "artifacts/experiments/fivek_paired_targets_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    with path.open("x") as f:
        f.write(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def stats(rgb, reference):
    pooled = cv2.resize(rgb.astype(np.float32) / 255, (8, 8), interpolation=cv2.INTER_AREA)
    target = cv2.resize(reference.astype(np.float32) / 255, (8, 8), interpolation=cv2.INTER_AREA)
    y = rgb.astype(np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32) / 255
    return {"pooled_rgb_reference_mae": float(np.abs(pooled - target).mean()),
            "display_luma_mean": float(y.mean()), "display_luma_q30": float(np.percentile(y, 30))}


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    DATA.mkdir(exist_ok=False)
    for name in ("input_jpeg", "original", "automatic", "comparisons"):
        (DATA / name).mkdir()
    (HERE / "records").mkdir(exist_ok=False)
    baseline = json.loads((HERE / "baseline.json").read_text())
    pairs = json.loads((PRIOR / "eligible_pairs.json").read_text())["photos"]
    assert sha(PRIOR / "eligible_pairs.json") == baseline["eligible_manifest_sha256"]
    icc = (PRIOR / "srgb.icc").read_bytes()
    started = time.perf_counter()
    records = []
    for index, pair in enumerate(pairs, 1):
        identity = Path(pair["source_id"]).stem
        assert sha(ROOT / pair["source_view_path"]) == pair["source_view_sha256"]
        assert sha(ROOT / pair["target_view_path"]) == pair["target_view_sha256"]
        with Image.open(ROOT / pair["source_view_path"]) as im:
            source = np.array(im)
            jpeg = DATA / "input_jpeg" / (identity + ".jpg")
            im.save(jpeg, quality=100, subsampling=0, icc_profile=icc)
        with Image.open(ROOT / pair["target_view_path"]) as im:
            target = np.array(im)
        metadata, pixels = process_jpeg(jpeg)
        original, adjusted = pixels["original"], pixels["adjusted"]
        assert original.shape == adjusted.shape == target.shape == source.shape
        assert original.dtype == adjusted.dtype == np.uint8
        assert np.isfinite(adjusted).all()
        hashes = {}
        for folder, rgb in (("original", original), ("automatic", adjusted)):
            path = DATA / folder / (identity + ".png")
            Image.fromarray(rgb).save(path, icc_profile=icc)
            with Image.open(path) as decoded:
                assert np.array_equal(np.array(decoded), rgb) and decoded.info["icc_profile"] == icc
            hashes[str(path.relative_to(ROOT))] = sha(path)
        black = int((np.all(adjusted == 0, axis=2) & ~np.all(original == 0, axis=2)).sum())
        full = int((np.any(adjusted == 255, axis=2) & ~np.any(original == 255, axis=2)).sum())
        assert black == full == 0
        record = {"source_id": pair["source_id"], "split": pair["split"], "attributes": pair["attributes"],
                  "jpeg_sha256": sha(jpeg), "encoder": "Pillow quality100/subsampling0/taggedsRGB",
                  "jpeg_decoded_vs_source_png_mae_codes": float(np.abs(original.astype(np.int16) - source.astype(np.int16)).mean()),
                  "size": [original.shape[1], original.shape[0]], "new_black_pixels": black, "new_full_pixels": full,
                  "original": stats(original, target), "automatic": stats(adjusted, target), "target": stats(target, target),
                  "source_preserved": True, "exports_sha256": hashes,
                  "target_view_sha256": pair["target_view_sha256"], "actual_processing": metadata}
        save(HERE / "records" / (identity + ".json"), record)
        records.append(record)
        if index % 10 == 0:
            print("Processed", index, "/", len(pairs), flush=True)
    first = pairs[0]
    metadata, repeated = process_jpeg(DATA / "input_jpeg" / (Path(first["source_id"]).stem + ".jpg"))
    with Image.open(DATA / "automatic" / (Path(first["source_id"]).stem + ".png")) as im:
        assert np.array_equal(np.array(im), repeated["adjusted"])
    by_id = {r["source_id"]: r for r in pairs}
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    for page in range(0, 12, 4):
        sheet = Image.new("RGB", (1560, 1320), "white")
        draw = ImageDraw.Draw(sheet)
        for n, name in enumerate(baseline["review_ids"][page:page + 4]):
            identity = Path(name).stem
            pair = by_id[name]
            draw.text((10, n * 330 + 5), name, font=font, fill="black")
            for column, (path, label) in enumerate(((DATA / "original" / (identity + ".png"), "Decoded input"),
                    (DATA / "automatic" / (identity + ".png"), "Current ShotSense automatic"),
                    (ROOT / pair["target_view_path"], "Expert C reference"))):
                with Image.open(path) as im:
                    view = ImageOps.contain(im, (500, 275))
                    sheet.paste(view, (column * 520 + (520 - view.width) // 2, n * 330 + 35 + (275 - view.height) // 2))
                draw.text((column * 520 + 10, n * 330 + 308), label, font=font, fill="black")
        sheet.save(DATA / "comparisons" / ("review-%d.jpg" % (page // 4 + 1)), quality=95)
    for path, expected in baseline["protected_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for pair in pairs:
        assert sha(ROOT / pair["source_view_path"]) == pair["source_view_sha256"]
        assert sha(ROOT / pair["target_view_path"]) == pair["target_view_sha256"]
    summary = {"count": len(records), "zero_new_black_full_count": len(records), "fixed_input_repeat_exact": True,
               "original_reference_error_mean": float(np.mean([r["original"]["pooled_rgb_reference_mae"] for r in records])),
               "automatic_reference_error_mean": float(np.mean([r["automatic"]["pooled_rgb_reference_mae"] for r in records])),
               "lower_pooled_reference_error_count": sum(r["automatic"]["pooled_rgb_reference_mae"] < r["original"]["pooled_rgb_reference_mae"] for r in records),
               "higher_pooled_reference_error_count": sum(r["automatic"]["pooled_rgb_reference_mae"] > r["original"]["pooled_rgb_reference_mae"] for r in records),
               "shadow_strength_min_max": [min(r["actual_processing"]["shadow_adjustment"]["strength"] for r in records),
                                             max(r["actual_processing"]["shadow_adjustment"]["strength"] for r in records)],
               "jpeg_source_mae_codes_mean": float(np.mean([r["jpeg_decoded_vs_source_png_mae_codes"] for r in records])),
               "protected_files_unchanged": len(baseline["protected_sha256"]),
               "elapsed_seconds": time.perf_counter() - started,
               "scope": "Unchanged actual JPEG route on <=960px RAW-developed FiveK development views. Coarse reference proximity is not independent photographic quality/preference or native alignment. No training, tuning or sharpening."}
    save(HERE / "results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
