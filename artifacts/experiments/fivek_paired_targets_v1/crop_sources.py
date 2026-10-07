"""Source-only DNG default-crop comparison; preserve the initial full-area audit."""
import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image, TiffImagePlugin
import rawpy

from audit import COMMON_RAW_PARAMS, SRGB_PARAMS, SRGB_ICC, geometry, small
from acquire import HERE, ROOT, DATA, digest, load, save_new
from src.preprocess import encode_semantic_image


def rotate(rgb, flip):
    if flip == 0:
        return rgb
    codes = {3: cv2.ROTATE_180, 5: cv2.ROTATE_90_COUNTERCLOCKWISE, 6: cv2.ROTATE_90_CLOCKWISE}
    if flip not in codes:
        raise ValueError("Unsupported original LibRaw flip")
    return cv2.rotate(rgb, codes[flip])


def source_crop(path):
    with Image.open(path) as im:
        offsets = im.tag_v2.get(330, ())
    found = []
    with path.open("rb") as stream:
        header = stream.read(8)
        for offset in offsets:
            directory = TiffImagePlugin.ImageFileDirectory_v2(header)
            stream.seek(offset)
            directory.load(stream)
            if directory.get(50720) is not None:
                assert tuple(directory.get(50718)) == (1, 1), "Non-unit DefaultScale"
                origin, size = tuple(directory[50719]), tuple(directory[50720])
                assert all(float(v).is_integer() for v in origin + size)
                found.append(tuple(int(v) for v in origin + size))
    assert len(found) == 1
    return found[0]


if __name__ == "__main__":
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261005)
    folder = DATA / "default_crop_views"
    folder.mkdir(exist_ok=False)
    records_path = HERE / "default_crop_audit"
    records_path.mkdir(exist_ok=False)
    selection = load(HERE / "selection.json")
    saved = []
    for index, row in enumerate(selection["photos"], 1):
        started = time.perf_counter()
        try:
            path = ROOT / row["source_path"]
            assert digest(path) == row["source_sha256"]
            x, y, w, h = source_crop(path)
            with rawpy.imread(str(path)) as raw:
                flip = raw.sizes.flip
                visible_size = (raw.sizes.height, raw.sizes.width)
                pixels = raw.postprocess(**dict(COMMON_RAW_PARAMS, user_flip=0), **SRGB_PARAMS)
            assert pixels.shape[:2] == visible_size, "Developed area differs from RAW visible area"
            assert digest(ROOT / row["cache_path"]) == hashlib.sha256(encode_semantic_image(rotate(pixels, flip))).hexdigest()
            assert 0 <= x < pixels.shape[1] and 0 <= y < pixels.shape[0]
            assert w > 0 and h > 0 and x + w <= pixels.shape[1] and y + h <= pixels.shape[0]
            cropped = rotate(pixels[y:y + h, x:x + w], flip)
            native_size = [cropped.shape[1], cropped.shape[0]]
            view = small(cropped)
            del cropped, pixels
            target_path = DATA / "views" / (row["public_id"] + "-target.png")
            with Image.open(target_path) as im:
                target = np.array(im)
            initial = load(HERE / "audit" / (row["public_id"] + ".json"))
            result = geometry(view, target)
            output = folder / (row["public_id"] + "-input.png")
            Image.fromarray(view).save(output, icc_profile=SRGB_ICC)
            record = {"source_id": row["source_id"], "public_id": row["public_id"], "split": row["split"],
                      "source_sha256": row["source_sha256"], "source_default_crop_xywh": [x, y, w, h],
                      "libraw_flip": flip, "crop_native_size": native_size,
                      "target_native_size": initial["target_native_size"],
                      "native_size_matches": native_size == initial["target_native_size"],
                      "uncropped_cache_recipe_matches": True, "geometry": result,
                      "view_path": str(output.relative_to(ROOT)), "view_sha256": digest(output),
                      "seconds": time.perf_counter() - started}
            save_new(records_path / (row["public_id"] + ".json"), record)
            saved.append(record)
            print(index, row["source_id"], result["status"], "native_size_matches", record["native_size_matches"], flush=True)
        except Exception as error:
            with (HERE / "default_crop_failures.jsonl").open("a") as f:
                f.write(json.dumps({"source_id": row["source_id"], "error": type(error).__name__ + ": " + str(error)}) + "\n")
            print("FAILED", row["source_id"], str(error), flush=True)
    print("Source-only crop audit completed:", len(saved), "/ 100", flush=True)
