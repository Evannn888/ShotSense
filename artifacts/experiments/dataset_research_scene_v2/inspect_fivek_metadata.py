"""Reproduce the saved metadata/ID audit offline; never read photo pixels."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
URL = "https://data.csail.mit.edu/graphics/fivek/"
FIELDS = ("subject", "light", "location", "time")


def load(name):
    return json.loads((HERE / name).read_text())


def counts(records):
    return {field: dict(Counter(r[field] for r in records)) for field in FIELDS}


raw_html = (HERE / "sources/fivek_official.html").read_bytes()
records = []
for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", raw_html.decode(), re.S | re.I):
    source = re.search(r'href="(img/dng/[^"<>]+\.dng)"', row)
    if source is None:
        continue
    record = {"id": Path(source[1]).stem, "source_dng_url": urljoin(URL, source[1])}
    for field in FIELDS:
        value = re.search(field + r":<br\s*/?>(.*?)</td>", row, re.S | re.I)
        assert value is not None, (record["id"], field)
        record[field] = html.unescape(re.sub(r"<[^>]+>", "", value[1])).strip()
    targets = re.findall(r'href="(img/tiff16_([a-e])/[^"<>]+\.tif)"', row)
    record["expert_tiff_urls"] = {expert: urljoin(URL, path) for path, expert in targets}
    assert set(record["expert_tiff_urls"]) == set("abcde"), record["id"]
    records.append(record)

assert len(records) == 5000 and records == load("fivek_public_annotations.json")
by_name = {(r["id"] + ".dng").casefold(): r for r in records}
assert len(by_name) == len(records)
splits_path = ROOT / "data/processed/splits.json"
split_data = json.loads(splits_path.read_text())["splits"]
names = [name.casefold() for group in split_data.values() for name in group]
assert len(names) == len(set(names)) == 4946
matched = [by_name[name] for name in names]
mapping = load("local_annotation_mapping.json")
assert mapping["matching_public_annotation_count"] == len(matched)
assert mapping["supported_annotation_counts"] == counts(matched)
assert mapping["split_match_counts"] == {key: len(group) for key, group in split_data.items()}
summary = load("fivek_annotation_summary.json")
assert summary["annotation_counts"] == counts(records)
assert summary["source_html_bytes"] == len(raw_html)
assert summary["source_html_sha256"] == hashlib.sha256(raw_html).hexdigest()
print(json.dumps({"public_records": len(records), "matched_local_sources": len(matched),
                  "split_counts": mapping["split_match_counts"],
                  "split_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
                  "scope": "Offline metadata only; no photo pixel access or writes."}, indent=2))
