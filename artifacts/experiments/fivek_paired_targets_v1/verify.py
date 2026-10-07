"""Close acquisition/encoding/provenance gates without reading reserved test pixels."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import urllib.request

from acquire import ROOT, HERE, DATA, digest, load, save_new

selection = load(HERE / "selection.json")
historical = load(ROOT / "data/processed/splits.json")["splits"]
grouped = load(ROOT / "data/processed/model_vnext/splits.json")["splits"]
groups = load(ROOT / "data/processed/model_vnext/groups.json")
excluded = {groups[name] for name in historical["test"] + grouped["test"]}
rows = selection["photos"]
assert len(rows) == len({r["source_id"] for r in rows}) == len({r["group"] for r in rows}) == 100
assert not {r["group"] for r in rows} & excluded
assert Counter(r["split"] for r in rows) == {"train": 80, "val": 20}
assert all(r["source_id"] in historical[r["split"]] and r["source_id"] in grouped[r["split"]] for r in rows)
assert len(list((HERE / "acquisition").glob("*.json"))) == len(list((HERE / "audit").glob("*.json"))) == 100
assert len(list((DATA / "original_tiff").glob("*.tif"))) == 100
assert not list((DATA / "original_tiff").glob("*.part"))
checks = load(HERE / "checks.json")
assert checks["srgb16_identity_max_error_codes"] == 0
assert checks["pillow8_parity_max_error_codes"] <= 1
assert checks["neutral_ramp_monotone"] and checks["identity_geometry_passed"] and checks["cropped_geometry_rejected"]
implementation = load(HERE / "implementation_baseline-v2.json")
implementation.update(load(HERE / "source_crop_implementation_baseline.json"))
for name, expected in implementation.items():
    assert digest(HERE / name) == expected, name
protected = dict(selection["protected_sha256"])
protected.update(load(HERE / "label_baseline_addendum.json")["protected_sha256"])
for name, expected in protected.items():
    assert digest(ROOT / name) == expected, name
records, acquisitions = [], []
for index, row in enumerate(rows, 1):
    assert digest(ROOT / row["source_path"]) == row["source_sha256"]
    assert digest(ROOT / row["cache_path"]) == row["cache_sha256"]
    acquired = load(HERE / "acquisition" / (row["public_id"] + ".json"))
    record = load(HERE / "audit" / (row["public_id"] + ".json"))
    assert acquired["url"] == row["target_url"] and acquired["expert"] == "C"
    assert acquired["bytes"] == (ROOT / row["target_path"]).stat().st_size
    assert digest(ROOT / row["target_path"]) == record["target_sha256"] == acquired["sha256"]
    assert record["source_id"] == row["source_id"] and record["split"] == row["split"]
    assert record["source_sha256"] == row["source_sha256"] and record["cache_recipe_matches"]
    assert record["target_bit_depth"] == 16 and record["icc_bytes"] > 0 and record["orientation"] == 1
    assert record["geometry"]["status"] in ("provisionally_aligned", "quarantined")
    for path, expected in record["derived_sha256"].items():
        assert digest(ROOT / path) == expected, path
    records.append(record)
    acquisitions.append(acquired)
    if index % 20 == 0:
        print("Verified", index, "/ 100", flush=True)
total = sum(r["bytes"] for r in acquisitions)
assert total <= selection["limits"]["max_total_bytes"]
assert max(r["bytes"] for r in acquisitions) <= selection["limits"]["max_file_bytes"]
with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as response:
    health = {"status": response.status, "body": response.read(64).decode()}
assert health == {"status": 200, "body": "ok"}
assert subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True).returncode == 0
crop_records = [load(HERE / "default_crop_audit" / (r["public_id"] + ".json")) for r in rows]
assert len(crop_records) == 100
for row, record in zip(rows, crop_records):
    assert record["source_id"] == row["source_id"]
    assert record["source_sha256"] == row["source_sha256"] and record["uncropped_cache_recipe_matches"]
    assert digest(ROOT / record["view_path"]) == record["view_sha256"]
    assert record["geometry"]["status"] in ("provisionally_aligned", "quarantined")
result = {"acquired": 100, "audited": 100, "source_only_crop_audited": len(crop_records), "original_target_bytes": total,
          "geometry_counts": dict(Counter(r["geometry"]["status"] for r in records)),
          "split_geometry_counts": {s: dict(Counter(r["geometry"]["status"] for r in records if r["split"] == s)) for s in ("train", "val")},
          "icc_descriptions": dict(Counter(r["icc_description"] for r in records)),
          "icc_hashes": dict(Counter(r["icc_sha256"] for r in records)),
          "native_size_equal_count": sum(r["source_native_size"] == r["target_native_size"] for r in records),
          "default_crop_geometry_counts": dict(Counter(r["geometry"]["status"] for r in crop_records)),
          "default_crop_native_size_match_count": sum(r["native_size_matches"] for r in crop_records),
          "default_crop_split_geometry_counts": {s: dict(Counter(r["geometry"]["status"] for r in crop_records if r["split"] == s)) for s in ("train", "val")},
          "protected_identity_count": len(protected), "originals_and_derived_hashes_match": True,
          "checks": checks, "application_health": health, "diff_check_passed": True,
          "trained_model": False, "controller_integrated": False, "test_pixels_read": False,
          "native_pixel_alignment_accepted": False, "independent_photo_quality_accepted": False,
          "whole_suite_rerun": False,
          "limitations": "Provisional resized-view geometry only; reviewed grouping is incomplete, pretrained FiveK overlap exists, RAW developer differs from Adobe and phone JPEGs. No training or automatic-quality claim."}
save_new(HERE / "verification.json", result)
print(json.dumps(result, indent=2), flush=True)
