"""Deterministic, resumable RAW preprocessing. Run: python -m src.preprocess."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import platform
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.color_pipeline import PIPELINE_CONFIG, develop_dng_outputs, prophoto16_to_lab_d50
from src.parameters import PARAMS, validate_parameters

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 95


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_bytes(payload)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def extract_132d_features(lab_img):
    lab = np.asarray(lab_img, dtype=np.float32)
    if lab.ndim != 3 or lab.shape[2] != 3 or lab.shape[0] < 3 or lab.shape[1] < 1:
        raise ValueError("Expected Lab image H>=3, W>=1, channels=3")
    if not np.isfinite(lab).all():
        raise ValueError("Non-finite Lab values")
    histograms = []
    for channel, bounds in enumerate(((0, 100), (-128, 128), (-128, 128))):
        counts, _ = np.histogram(np.clip(lab[:, :, channel], *bounds), bins=32, range=bounds)
        histograms.extend(counts / lab.shape[0] / lab.shape[1])
    features = list(histograms)
    for region in np.array_split(lab, 3, axis=0):
        pixels = region.reshape(-1, 3)
        features.extend(pixels.mean(axis=0))
        features.extend(pixels.std(axis=0))
    lightness = lab[:, :, 0]
    for mask in (lightness <= 33.3, (lightness > 33.3) & (lightness <= 66.7), lightness > 66.7):
        pixels = lab[mask]
        features.extend(pixels.mean(axis=0) if len(pixels) else [0.0] * 3)
        features.extend(pixels.std(axis=0) if len(pixels) else [0.0] * 3)
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (132,) or not np.isfinite(result).all():
        raise ValueError("Invalid physical features")
    return result


def encode_semantic_image(srgb):
    if srgb.dtype != np.uint8 or srgb.ndim != 3 or srgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB semantic output")
    small = cv2.resize(srgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError("JPEG encoding failed")
    return encoded.tobytes()


def extract_inputs(dng_path):
    """Shared with inference: physical features and exact JPEG bytes."""
    prophoto, srgb = develop_dng_outputs(str(dng_path))
    # Resize linear floats rather than quantizing an intermediate thumbnail.
    linear = cv2.resize(prophoto.astype(np.float32) / 65535.0, IMAGE_SIZE,
                        interpolation=cv2.INTER_AREA)
    lab = prophoto16_to_lab_d50(linear)
    outside = np.mean((lab < [0, -128, -128]) | (lab > [100, 128, 128]), axis=(0, 1))
    return extract_132d_features(lab), encode_semantic_image(srgb), outside.astype(np.float32)


def target_statistics(experts):
    if set(experts) != set("ABCDE"):
        raise ValueError("Expected exactly experts A–E")
    matrix = validate_parameters([[experts[expert][key] for key in PARAMS] for expert in "ABCDE"])
    return matrix.mean(axis=0).astype(np.float32), matrix.std(axis=0).astype(np.float32)


def source_signature(path):
    stat = Path(path).stat()
    return np.asarray([stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def read_cached(cache_path, image_path, image_id, config_hash, source):
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["id"]) != image_id or str(cached["config_hash"]) != config_hash:
                return None
            if not np.array_equal(cached["source"], source):
                return None
            if str(cached["image_sha256"]) != sha256_file(image_path):
                return None
            arrays = {key: cached[key].copy() for key in ("X", "Y", "Y_std", "lab_outside_fraction")}
        for key, shape in (("X", (132,)), ("Y", (6,)), ("Y_std", (6,)), ("lab_outside_fraction", (3,))):
            if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
                return None
        validate_parameters(arrays["Y"])
        if (arrays["Y_std"] < 0).any() or ((arrays["lab_outside_fraction"] < 0) | (arrays["lab_outside_fraction"] > 1)).any():
            return None
        with Image.open(image_path) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != IMAGE_SIZE:
                return None
            image.load()
        return arrays
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def process_image(job):
    dng_path, experts, output_dir, config_hash, rebuild = job
    image_id = Path(dng_path).name.lower()
    image_path = Path(output_dir) / "images" / (Path(image_id).stem + ".jpg")
    cache_path = Path(output_dir) / "features" / (Path(image_id).stem + ".npz")
    started = time.perf_counter()
    try:
        cv2.setNumThreads(1)
        source = source_signature(dng_path)
        if not rebuild:
            cached = read_cached(cache_path, image_path, image_id, config_hash, source)
            if cached is not None:
                return {"id": image_id, "status": "cached", "seconds": time.perf_counter() - started, **cached}
        mean, std = target_statistics(experts)
        features, jpeg, outside = extract_inputs(dng_path)
        if not np.array_equal(source_signature(dng_path), source):
            raise OSError("RAW changed while processing")
        atomic_bytes(image_path, jpeg)
        atomic_npz(cache_path, X=features, Y=mean, Y_std=std, id=np.asarray(image_id),
                   config_hash=np.asarray(config_hash), source=source,
                   image_sha256=np.asarray(hashlib.sha256(jpeg).hexdigest()), lab_outside_fraction=outside)
        return {"id": image_id, "status": "processed", "seconds": time.perf_counter() - started,
                "X": features, "Y": mean, "Y_std": std, "lab_outside_fraction": outside}
    except Exception as error:
        return {"id": image_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def preprocessing_config(labels_path):
    implementation = {name: sha256_file(Path(__file__).with_name(name))
                      for name in ("preprocess.py", "color_pipeline.py", "parameters.py")}
    return {"schema_version": 1, "parameters": list(PARAMS), "physical_dimensions": 132,
            "pipeline": PIPELINE_CONFIG, "size": list(IMAGE_SIZE), "jpeg_quality": JPEG_QUALITY,
            "interpolation": "cv2.INTER_AREA", "opencv": cv2.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "labels_sha256": sha256_file(labels_path),
            "implementation_sha256": implementation}


def run_preprocessing(raw_dir, labels_path, output_dir, limit=None, workers=1, rebuild=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The kernel releases this lock after a crash; no stale-lock recovery needed.
    with (output_dir / ".preprocess.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another preprocessing run owns this output directory") from error
        return _run_preprocessing(raw_dir, labels_path, output_dir, limit, workers, rebuild)


def _run_preprocessing(raw_dir, labels_path, output_dir, limit=None, workers=1, rebuild=False):
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    raw_dir, labels_path, output_dir = Path(raw_dir), Path(labels_path), Path(output_dir)
    labels = json.loads(labels_path.read_text())
    audit_path = labels_path.with_name("label_audit.json")
    audit = json.loads(audit_path.read_text())
    config = preprocessing_config(labels_path)
    if audit.get("labels_sha256") != config["labels_sha256"]:
        raise ValueError("Label audit does not match labels; rerun extraction")
    filtered_by_audit = audit["filtered_images"]
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    files = sorted((p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dng"),
                   key=lambda p: p.name.lower())
    ids = [p.name.lower() for p in files]
    if len(ids) != len(set(ids)):
        raise ValueError("Case-insensitive RAW filename collision")
    if not files:
        raise ValueError("No DNG inputs")
    files = files[:limit] if limit else files
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.json"
    if limit and manifest_path.exists() and json.loads(manifest_path.read_text()).get("scope") == "full":
        raise ValueError("Pilot cannot replace full metadata; use a separate output directory")
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("config_hash") != config_hash and not rebuild:
            raise ValueError("Preprocessing configuration changed; use --rebuild or another output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("images", "features"):
        (output_dir / directory).mkdir(exist_ok=True)
    atomic_bytes(config_path, json.dumps({"config_hash": config_hash, "config": config}, indent=2).encode())
    results, filtered, jobs = [], {}, []
    for path in files:
        image_id = path.name.lower()
        if image_id in filtered_by_audit:
            filtered[image_id] = filtered_by_audit[image_id]
        elif image_id not in labels:
            results.append({"id": image_id, "status": "failed", "error": "Missing audited label"})
        else:
            try:
                target_statistics(labels[image_id])
            except (KeyError, ValueError, TypeError) as error:
                results.append({"id": image_id, "status": "failed", "error": f"Invalid audited label: {error}"})
            else:
                jobs.append((str(path), labels[image_id], str(output_dir), config_hash, rebuild))
    started = time.perf_counter()
    if workers == 1:
        for index, job in enumerate(jobs, 1):
            results.append(process_image(job))
            if index % 20 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)}", flush=True)
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, result in enumerate(pool.imap_unordered(process_image, jobs), 1):
                results.append(result)
                if index % 20 == 0 or index == len(jobs):
                    print(f"Processed {index}/{len(jobs)}", flush=True)
    successes = sorted((r for r in results if r["status"] != "failed"), key=lambda r: r["id"])
    failures = {r["id"]: r["error"] for r in results if r["status"] == "failed"}
    report = {"config_hash": config_hash, "scope": "pilot" if limit else "full",
              "status": "complete" if successes and not failures else "failed",
              "input_count": len(files), "success_count": len(successes), "filtered_count": len(filtered),
              "failed_count": len(failures), "cached_count": sum(r["status"] == "cached" for r in successes),
              "workers": workers, "elapsed_seconds": time.perf_counter() - started,
              "filtered_images": filtered, "failed_images": failures, "config": config}
    assert report["input_count"] == len(successes) + len(filtered) + len(failures)
    metadata_path = output_dir / "metadata.npz"
    if successes:
        atomic_npz(metadata_path, **{key: np.stack([r[key] for r in successes]) for key in ("X", "Y", "Y_std")},
                   ids=np.asarray([r["id"] for r in successes]), config_hash=np.asarray(config_hash))
        report["metadata_sha256"] = sha256_file(metadata_path)
        report["lab_outside_fraction_mean"] = np.mean([r["lab_outside_fraction"] for r in successes], axis=0).tolist()
    else:
        # A previous successful artifact must not masquerade as this failed run.
        metadata_path.unlink(missing_ok=True)
    atomic_bytes(output_dir / "manifest.json", json.dumps(report, indent=2, sort_keys=True).encode())
    print(json.dumps({key: report[key] for key in ("status", "input_count", "success_count", "filtered_count",
                                                 "failed_count", "cached_count", "elapsed_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/dngs")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/intermediate/expert_labels.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or ROOT / ("data/processed/pilot" if args.limit else "data/processed")
    report = run_preprocessing(args.raw_dir, args.labels, output, args.limit, args.workers, args.rebuild)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
