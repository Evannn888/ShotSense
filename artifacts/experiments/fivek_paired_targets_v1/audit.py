"""Offline ICC/geometry audit of frozen Expert C pairs; no training or parameter labels."""
import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time
import zlib

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps
import rawpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src.color_pipeline import COMMON_RAW_PARAMS, SRGB_PARAMS, PIPELINE_CONFIG
from acquire import DATA, digest, load, save_new

LIBRARY = Path("/opt/homebrew/opt/little-cms2/lib/liblcms2.dylib")
LCMS = ctypes.CDLL(str(LIBRARY))
for name, args, result in (
    ("cmsOpenProfileFromMem", [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_void_p),
    ("cmsCreate_sRGBProfile", [], ctypes.c_void_p),
    ("cmsCloseProfile", [ctypes.c_void_p], ctypes.c_int),
    ("cmsCreateTransform", [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32], ctypes.c_void_p),
    ("cmsDoTransform", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32], None),
    ("cmsDeleteTransform", [ctypes.c_void_p], None),
    ("cmsGetEncodedCMMversion", [], ctypes.c_uint32),
):
    function = getattr(LCMS, name)
    function.argtypes, function.restype = args, result
SRGB_PROFILE_PATH = HERE / "srgb.icc"
if not SRGB_PROFILE_PATH.exists():
    with SRGB_PROFILE_PATH.open("xb") as f:
        f.write(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
SRGB_ICC = SRGB_PROFILE_PATH.read_bytes()
RGB16 = (4 << 16) | (3 << 3) | 2  # Installed lcms2.h: PT_RGB/CHANNELS_SH/BYTES_SH.
CONTRACT = {"version": 2, "intent": "relative colorimetric", "intent_value": 1, "flags": 256,
            "destination_icc_sha256": digest(SRGB_PROFILE_PATH),
            "black_point_compensation": False, "optimizer": False,
            "library_sha256": digest(LIBRARY), "library_version": LCMS.cmsGetEncodedCMMversion(),
            "source_recipe": PIPELINE_CONFIG["common_raw_params"],
            "source_output": PIPELINE_CONFIG["outputs"]["srgb8"],
            "analysis_max_edge": 960, "geometry_gate": {"aspect_difference_max": .005,
            "minimum_inliers": 25, "minimum_inlier_ratio": .4,
            "corner_displacement_max_pixels": 3, "median_identity_error_max_pixels": 2},
            "geometry_limits": "Resized-view feature diagnostic; not native subpixel alignment proof or independent quality."}
CONTRACT = json.loads(json.dumps(CONTRACT))  # Compare persisted JSON types, including RAW tuple options.


def convert16(rgb, profile):
    assert rgb.dtype == np.uint16 and rgb.ndim == 3 and rgb.shape[-1] == 3
    source = np.ascontiguousarray(rgb)
    buffer = ctypes.create_string_buffer(profile)
    src = LCMS.cmsOpenProfileFromMem(buffer, len(profile))
    destination_buffer = ctypes.create_string_buffer(SRGB_ICC)
    dst = LCMS.cmsOpenProfileFromMem(destination_buffer, len(SRGB_ICC))
    transform = None
    try:
        if not src or not dst:
            raise ValueError("Invalid ICC profile")
        transform = LCMS.cmsCreateTransform(src, RGB16, dst, RGB16, 1, 256)
        if not transform:
            raise ValueError("ICC transform creation failed")
        output = np.empty_like(source)
        for row in range(0, len(source), 128):
            a, b = source[row:row + 128], output[row:row + 128]
            LCMS.cmsDoTransform(transform, a.ctypes.data, b.ctypes.data, a.shape[0] * a.shape[1])
        return output
    finally:
        if transform:
            LCMS.cmsDeleteTransform(transform)
        if src:
            LCMS.cmsCloseProfile(src)
        if dst:
            LCMS.cmsCloseProfile(dst)


def png16(path, rgb):
    assert not path.exists()
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok and encoded[24] == 16
    payload = encoded.tobytes()
    iccp = b"sRGB\0\0" + zlib.compress(SRGB_ICC)
    chunk = struct.pack(">I", len(iccp)) + b"iCCP" + iccp + struct.pack(">I", zlib.crc32(b"iCCP" + iccp) & 0xffffffff)
    with path.open("xb") as f:
        f.write(payload[:33] + chunk + payload[33:])
    decoded = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, rgb)
    with Image.open(path) as im:
        assert im.info["icc_profile"] == SRGB_ICC


def small(rgb):
    scale = min(1, 960 / max(rgb.shape[:2]))
    return cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def geometry(a, b):
    aspect_error = abs((a.shape[1] / a.shape[0]) / (b.shape[1] / b.shape[0]) - 1)
    result = {"relative_aspect_difference": aspect_error, "status": "quarantined", "reason": "aspect or feature correspondence uncertain"}
    if aspect_error > .005:
        return result
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    detector = cv2.ORB_create(nfeatures=2500, fastThreshold=8)
    k1, d1 = detector.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    k2, d2 = detector.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None:
        return result
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .75 * m[1].distance]
    result["ratio_filtered_matches"] = len(good)
    if len(good) < 25:
        return result
    left = np.float32([k1[m.queryIdx].pt for m in good])
    right = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(left, right, cv2.RANSAC, 2)
    if matrix is None or mask is None:
        return result
    inliers = mask.ravel().astype(bool)
    corners = np.float32([[[0, 0], [a.shape[1] - 1, 0], [a.shape[1] - 1, a.shape[0] - 1], [0, a.shape[0] - 1]]])
    displacement = np.linalg.norm(cv2.perspectiveTransform(corners, matrix) - corners, axis=2)
    identity_error = np.linalg.norm(left[inliers] - right[inliers], axis=1)
    result.update({"inliers": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
                   "homography_source_to_target_view": matrix.tolist(), "corner_displacement_max_pixels": float(displacement.max()),
                   "median_identity_error_pixels": float(np.median(identity_error))})
    if inliers.sum() >= 25 and inliers.mean() >= .4 and displacement.max() <= 3 and np.median(identity_error) <= 2:
        result.update(status="provisionally_aligned", reason="Fixed diagnostic thresholds passed; native precision and human preference remain unverified")
    return result


def checks(profile):
    rng = np.random.default_rng(20261005)
    chart = rng.integers(0, 65536, (32, 32, 3), dtype=np.uint16)
    identity = convert16(chart, SRGB_ICC)
    error = int(np.abs(identity.astype(np.int32) - chart.astype(np.int32)).max())
    assert error <= 1, error
    gray = np.repeat(np.linspace(0, 65535, 1024, dtype=np.uint16)[None, :, None], 3, axis=2)
    converted = convert16(gray, profile)
    assert np.all(np.diff(converted.astype(np.int32), axis=1) >= 0)
    spread = int(np.ptp(converted.astype(np.int32), axis=2).max())
    assert spread <= 8, spread
    eight = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    precision = convert16(eight.astype(np.uint16) * 257, profile)
    rounded = ((precision.astype(np.uint32) + 128) // 257).astype(np.uint8)
    pillow = np.array(ImageCms.profileToProfile(Image.fromarray(eight), ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                      ImageCms.createProfile("sRGB"), renderingIntent=1, outputMode="RGB", flags=256))
    parity = int(np.abs(rounded.astype(np.int16) - pillow.astype(np.int16)).max())
    assert parity <= 1, parity
    source = rng.integers(0, 256, (300, 450, 3), dtype=np.uint8)
    identity_geometry = geometry(source, source)
    cropped_geometry = geometry(source, source[:, 60:-60])
    assert identity_geometry["status"] == "provisionally_aligned"
    assert cropped_geometry["status"] == "quarantined"
    return {"srgb16_identity_max_error_codes": error, "prophoto_neutral_max_channel_spread_codes": spread,
            "neutral_ramp_monotone": True, "pillow8_parity_max_error_codes": parity,
            "identity_geometry_passed": True, "cropped_geometry_rejected": True}


def audit_row(row):
    record_path = HERE / "audit" / (row["public_id"] + ".json")
    if record_path.exists():
        record = load(record_path)
        for path, expected in record["derived_sha256"].items():
            assert digest(ROOT / path) == expected, path
        return record
    started = time.perf_counter()
    path = ROOT / row["target_path"]
    acquired = load(HERE / "acquisition" / (row["public_id"] + ".json"))
    assert digest(path) == acquired["sha256"]
    source_path = ROOT / row["source_path"]
    assert digest(source_path) == row["source_sha256"]
    with Image.open(path) as image:
        tags = image.tag_v2
        assert image.format == "TIFF" and tuple(tags[258]) == (16, 16, 16)
        assert tags[262] == 2 and tags[277] == 3 and tags.get(284, 1) == 1 and tags.get(339, (1,))[0] == 1
        orientation = int(tags.get(274, 1))
        assert orientation == 1, "Nontrivial TIFF orientation requires a new explicit audit"
        icc = image.info.get("icc_profile")
        assert icc, "Missing embedded profile"
        profile_name = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
        native_size = list(image.size)
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert bgr is not None and bgr.dtype == np.uint16 and bgr.shape == (native_size[1], native_size[0], 3)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    del bgr
    converted = convert16(original_rgb, icc)
    del original_rgb
    target16 = DATA / "srgb16" / (row["public_id"] + ".png")
    png16(target16, converted)
    endpoint_fraction = float(np.any((converted == 0) | (converted == 65535), axis=2).mean())
    target_view = ((small(converted).astype(np.uint32) + 128) // 257).astype(np.uint8)
    del converted
    with rawpy.imread(str(source_path)) as raw:
        wb = np.array(raw.camera_whitebalance[:3])
        assert np.isfinite(wb).all() and np.all(wb > .00001)
        srgb = raw.postprocess(**COMMON_RAW_PARAMS, **SRGB_PARAMS)
        camera = {"raw_size": [raw.sizes.raw_width, raw.sizes.raw_height], "flip": raw.sizes.flip,
                  "visible_size": [raw.sizes.width, raw.sizes.height], "camera_wb": wb.tolist()}
    source_size = [srgb.shape[1], srgb.shape[0]]
    source_view = small(srgb)
    # Confirm the development branch against the frozen, existing semantic-cache recipe.
    from src.preprocess import encode_semantic_image
    cache_recipe_matches = digest(ROOT / row["cache_path"]) == hashlib.sha256(encode_semantic_image(srgb)).hexdigest()
    assert cache_recipe_matches, "New source development differs from original pipeline cache"
    del srgb
    alignment = geometry(source_view, target_view)
    derived = {}
    for label, pixels in (("input", source_view), ("target", target_view)):
        output = DATA / "views" / (row["public_id"] + "-" + label + ".png")
        assert not output.exists()
        Image.fromarray(pixels).save(output, icc_profile=SRGB_ICC)
        derived[str(output.relative_to(ROOT))] = digest(output)
    derived[str(target16.relative_to(ROOT))] = digest(target16)
    record = {"source_id": row["source_id"], "public_id": row["public_id"], "split": row["split"], "attributes": row["attributes"],
              "target_sha256": acquired["sha256"], "source_sha256": row["source_sha256"],
              "target_native_size": native_size, "source_native_size": source_size,
              "target_bit_depth": 16, "icc_bytes": len(icc), "icc_sha256": hashlib.sha256(icc).hexdigest(),
              "icc_description": profile_name, "orientation": orientation, "camera": camera,
              "cache_recipe_matches": cache_recipe_matches, "target_srgb_endpoint_pixel_fraction": endpoint_fraction,
              "geometry": alignment, "derived_sha256": derived, "seconds": time.perf_counter() - started}
    save_new(record_path, record)
    return record


def overview(rows):
    font = ImageFont.load_default(size=16)
    (DATA / "comparisons").mkdir(exist_ok=True)
    for page in range(0, len(rows), 10):
        sheet = Image.new("RGB", (1520, 1220), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[page:page + 10]):
            x, y = (index % 2) * 760, (index // 2) * 244
            record = load(HERE / "audit" / (row["public_id"] + ".json"))
            draw.text((x + 6, y + 2), row["source_id"] + " | " + row["split"], font=font, fill="black")
            for column, label in enumerate(("input", "target")):
                with Image.open(DATA / "views" / (row["public_id"] + "-" + label + ".png")) as im:
                    view = ImageOps.contain(im, (370, 180))
                    sheet.paste(view, (x + column * 380 + (370 - view.width) // 2, y + 27 + (180 - view.height) // 2))
            draw.text((x + 6, y + 211), "Camera-WB input / Expert C | " + record["geometry"]["status"], font=font, fill="black")
        sheet.save(DATA / "comparisons" / ("page-%02d.jpg" % (page // 10 + 1)), quality=94)


def main(limit):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261005)
    for name in ("srgb16", "views"):
        (DATA / name).mkdir(exist_ok=True)
    (HERE / "audit").mkdir(exist_ok=True)
    selection = load(HERE / "selection.json")
    contract_path = HERE / "color_geometry_contract.json"
    if contract_path.exists():
        assert load(contract_path) == CONTRACT
    else:
        save_new(contract_path, CONTRACT)
    rows = selection["photos"][:limit]
    with Image.open(ROOT / rows[0]["target_path"]) as im:
        test_result = checks(im.info["icc_profile"])
    if not (HERE / "checks.json").exists():
        save_new(HERE / "checks.json", test_result)
    records = []
    for row in rows:
        try:
            record = audit_row(row)
            records.append(record)
            print(len(records), row["source_id"], record["geometry"]["status"], round(record["seconds"], 2), flush=True)
        except Exception as error:
            with (HERE / "audit_failures.jsonl").open("a") as f:
                f.write(json.dumps({"source_id": row["source_id"], "error": type(error).__name__ + ": " + str(error)}) + "\n")
            print("FAILED", row["source_id"], str(error), flush=True)
    overview([r for r in rows if (HERE / "audit" / (r["public_id"] + ".json")).exists()])
    print("Audited", len(records), "of", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    assert 1 <= args.limit <= 100
    main(args.limit)
