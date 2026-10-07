"""Freeze the approved 100-source selection, then resume bounded original TIFF fetches."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/external/fivek_expert_c_pilot_v1"
FIELDS = ("subject", "light", "location", "time")
QUOTAS = {
    "subject": {"person(s)": 30, "nature": 25, "man-made object": 25, "animal(s)": 15, "unknown": 5},
    "light": {"sun or sky": 55, "mixed": 25, "artificial": 20},
    "location": {"outdoors": 65, "indoors": 30, "unknown": 5},
    "time": {"day": 60, "dawn or dusk": 15, "night": 10, "unknown": 15},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    path = HERE / "selection.json"
    if path.exists():
        raise FileExistsError("Selection is already frozen; use acquire to resume.")
    historical = load(ROOT / "data/processed/splits.json")["splits"]
    grouped = load(ROOT / "data/processed/model_vnext/splits.json")["splits"]
    groups = load(ROOT / "data/processed/model_vnext/groups.json")
    excluded = {groups[name] for name in historical["test"] + grouped["test"]}
    annotations = load(ROOT / "artifacts/experiments/dataset_research_scene_v2/fivek_public_annotations.json")
    public = {(r["id"] + ".dng").casefold(): r for r in annotations}
    raw_files = list((ROOT / "data/raw/dngs").glob("*.dng"))
    raw = {p.name.casefold(): p for p in raw_files}
    assert len(raw) == len(raw_files) == 5000
    selected = []
    chosen_groups = set()
    for split, amount in (("train", 80), ("val", 20)):
        names = sorted(set(historical[split]) & set(grouped[split]))
        names = [n for n in names if groups[n] not in excluded]
        random.Random(20261005 + amount).shuffle(names)
        tally = {field: Counter() for field in FIELDS}
        for _ in range(amount):
            def priority(name):
                row = public[name]
                return sum(max(0, QUOTAS[f].get(row[f], 0) * amount / 100 - tally[f][row[f]])
                           / max(1, QUOTAS[f].get(row[f], 0) * amount / 100) for f in FIELDS)
            eligible = [n for n in names if groups[n] not in chosen_groups]
            name = max(eligible, key=priority)
            row = public[name]
            chosen_groups.add(groups[name])
            cache = ROOT / "data/processed/images" / (Path(name).stem + ".jpg")
            selected.append({"source_id": name, "public_id": row["id"], "group": groups[name],
                             "split": split, "attributes": {f: row[f] for f in FIELDS},
                             "source_path": str(raw[name].relative_to(ROOT)),
                             "source_sha256": digest(raw[name]),
                             "cache_path": str(cache.relative_to(ROOT)), "cache_sha256": digest(cache),
                             "expert": "C", "target_url": row["expert_tiff_urls"]["c"],
                             "target_path": str((DATA / "original_tiff" / (row["id"] + ".tif")).relative_to(ROOT))})
            for field in FIELDS:
                tally[field][row[field]] += 1
    assert len(selected) == len(chosen_groups) == 100
    assert not chosen_groups & excluded
    baseline = load(ROOT / "artifacts/experiments/scene_analysis_public_v1/baseline.json")
    protected = list(baseline) + ["src/color_pipeline.py", "src/preprocess.py",
        "data/processed/splits.json", "data/processed/model_vnext/splits.json",
        "data/processed/model_vnext/groups.json", "data/intermediate/expert_labels.json",
        "data/intermediate/label_audit.json",
        "artifacts/model/model.onnx", "artifacts/model/model.json",
        "data/external_models/image_adaptive_3dlut/pretrained_models/sRGB/classifier.pth",
        "data/external_models/image_adaptive_3dlut/pretrained_models/sRGB/LUTs.pth"]
    protected = [p for p in protected if (ROOT / p).is_file()]
    catalog = list((ROOT / "data/raw").rglob("*.lrcat"))
    protected += [str(p.relative_to(ROOT)) for p in catalog]
    frozen = {p: digest(ROOT / p) for p in protected}
    for p, expected in baseline.items():
        assert frozen[p] == expected, p
    selection = {"seed": 20261005, "count": 100, "expert": "C", "requested_metadata_quotas": QUOTAS,
                 "selection": "Greedy metadata coverage; 80 train/20 development validation, intersection of both split versions; no target pixels.",
                 "excluded_test_groups": len(excluded), "protected_sha256": frozen,
                 "coverage": {f: dict(Counter(r["attributes"][f] for r in selected)) for f in FIELDS},
                 "photos": selected, "limits": {"max_file_bytes": 128 * 1024**2, "max_total_bytes": 8 * 1024**3, "workers": 3}}
    save_new(path, selection)
    DATA.mkdir(parents=True, exist_ok=False)
    for name in ("LicenseAdobe.txt", "LicenseAdobeMIT.txt"):
        shutil.copyfile(ROOT / "artifacts/experiments/dataset_research_scene_v2/sources" / name, DATA / name)
    save_new(DATA / "NOTICE.json", {"source": "https://data.csail.mit.edu/graphics/fivek/", "expert": "C",
                                  "purpose": "Personal non-commercial research; preserve original photographs and agreements.",
                                  "selection_sha256": digest(path)})
    print(json.dumps({"selection_count": 100, "coverage": selection["coverage"], "protected_files": len(frozen)}, indent=2), flush=True)


def fetch(row):
    dest = ROOT / row["target_path"]
    record = HERE / "acquisition" / (row["public_id"] + ".json")
    if record.exists():
        result = load(record)
        assert dest.is_file() and digest(dest) == result["sha256"]
        return row["source_id"], "verified cached", result["bytes"]
    if dest.exists():
        raise FileExistsError("Unrecorded target cannot be overwritten: " + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(1, 4):
        started = time.perf_counter()
        temp = dest.with_suffix(".part")
        try:
            request = urllib.request.Request(row["target_url"], headers={"User-Agent": "ShotSense-personal-research/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response, temp.open("xb") as output:
                length = int(response.headers.get("Content-Length", 0))
                assert 0 < length <= 128 * 1024**2, "Missing/excessive Content-Length"
                assert response.status == 200 and "image/tiff" in response.headers.get("Content-Type", "")
                received = 0
                h = hashlib.sha256()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    assert received <= length and time.perf_counter() - started < 240, "Transfer bound exceeded"
                    output.write(block)
                    h.update(block)
                assert received == length, "Truncated transfer"
                final_url = response.url
            with temp.open("rb") as f:
                assert f.read(4) in (b"II*\x00", b"MM\x00*"), "Not a TIFF"
            result = {"source_id": row["source_id"], "expert": "C", "url": row["target_url"],
                      "final_url": final_url, "bytes": received, "sha256": h.hexdigest(),
                      "content_type": "image/tiff", "status": 200, "attempt": attempt,
                      "seconds": time.perf_counter() - started, "earlier_errors": errors}
            temp.rename(dest)
            save_new(record, result)
            return row["source_id"], "acquired", received
        except Exception as error:
            failed = {"attempt": attempt, "error": type(error).__name__ + ": " + str(error),
                      "seconds": time.perf_counter() - started, "partial_bytes": temp.stat().st_size if temp.exists() else 0}
            errors.append(failed)
            temp.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(5 * attempt)
    return row["source_id"], errors, 0


def acquire(limit):
    selection = load(HERE / "selection.json")
    assert selection["count"] == 100
    (HERE / "acquisition").mkdir(exist_ok=True)
    # The per-file bound makes the maximum 100-file batch <=8GiB only with actual headers;
    # cap the planned HEAD sum before issuing body transfers.
    rows = selection["photos"][:limit]
    total = 0
    for count, row in enumerate(rows, 1):
        with urllib.request.urlopen(urllib.request.Request(row["target_url"], method="HEAD"), timeout=30) as r:
            size = int(r.headers["Content-Length"])
            assert r.status == 200 and 0 < size <= selection["limits"]["max_file_bytes"]
            total += size
        if count % 10 == 0:
            print("HEAD", count, "/", len(rows), flush=True)
    assert total <= selection["limits"]["max_total_bytes"]
    print("HEAD bounds verified:", len(rows), "files,", total, "bytes", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(fetch, row) for row in rows]
        for count, future in enumerate(as_completed(tasks), 1):
            name, status, size = future.result()
            if not isinstance(status, str):
                with (HERE / "acquisition_failures.jsonl").open("a") as f:
                    f.write(json.dumps({"source_id": name, "errors": status}) + "\n")
                status = "failed"
            print(count, "/", len(rows), name, status, size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "acquire"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    assert 1 <= args.limit <= 100
    prepare() if args.mode == "prepare" else acquire(args.limit)
