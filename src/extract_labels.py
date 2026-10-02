"""Read-only FiveK Catalog extraction with explicit source and omission auditing."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np

try:
    from .parameters import PARAMS, validate_parameters
except ImportError:
    from parameters import PARAMS, validate_parameters

COLLECTIONS = {"A": 918542, "B": 923976, "C": 930899, "D": 936482, "E": 954927}
DEFAULT_HISTORY_NAMES = {"Saturation": "Saturation", "HighlightRecovery": "Highlight Recovery"}
VERSION = "catalog-v2-top-level-audited"
PROCESS_REFERENCES = [
    "https://helpx.adobe.com/lightroom-classic/desktop/process-and-develop-photos/develop-module-options.html",
    "https://www.adobe.com/special/photoshop/camera_raw/Camera_Raw_5.7_ReadMe.pdf",
]


def parse_settings(text):
    """Read only top-level Lua table fields; never execute the Catalog text."""
    match = re.fullmatch(r"\s*s\s*=\s*\{(.*)\}\s*", text, re.S)
    if not match:
        raise ValueError("Invalid settings table")
    body = match[1]
    fields, start, depth, quote, escaped = [], 0, 0, None, False
    for i, char in enumerate(body):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced settings table")
        elif char == "," and depth == 0:
            fields.append(body[start:i])
            start = i + 1
    if quote or depth:
        raise ValueError("Unclosed string or nested table")
    fields.append(body[start:])
    result = {}
    for field in fields:
        if not field.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*", field, re.S)
        if not match or match[1] in result:
            raise ValueError("Invalid or duplicate top-level field")
        key, raw = match.groups()
        if raw.startswith("{"):
            value = None  # Nested settings are deliberately not label candidates.
        elif raw.startswith(('"', "'")):
            value = ast.literal_eval(raw)
            if not isinstance(value, str):
                raise ValueError("Invalid string field")
        elif raw in ("true", "false", "nil"):
            value = {"true": True, "false": False, "nil": None}[raw]
        elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw):
            value = float(raw)
        else:
            raise ValueError("Unsupported scalar in " + key)
        result[key] = value
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_catalog(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.row_factory = sqlite3.Row
    return connection


def audit_default_rules(connection):
    """Prove the two omission rules against named edits and their recorded values."""
    evidence = {}
    for parameter, name in DEFAULT_HISTORY_NAMES.items():
        counts, examples = Counter(), []
        rows = connection.execute(
            "SELECT id_local, image, valueString, text FROM Adobe_libraryImageDevelopHistoryStep WHERE name = ?",
            (name,),
        )
        for row in rows:
            settings = parse_settings(row["text"])
            expected = float(row["valueString"])
            omitted = parameter not in settings
            actual = 0.0 if omitted else settings[parameter]
            counts["total"] += 1
            counts["omitted_zero" if omitted else "explicit"] += 1
            if not math.isfinite(expected) or actual != expected:
                raise ValueError("Catalog history contradicts omission rule for " + parameter)
            if omitted and len(examples) < 3:
                examples.append({"history_id": row["id_local"], "image_id": row["image"],
                                 "valueString": row["valueString"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()})
        if not counts["omitted_zero"] or not counts["explicit"]:
            raise ValueError("Insufficient Catalog evidence for omitted " + parameter)
        evidence[parameter] = {"value": 0.0, "basis": "same Catalog named history edits; omitted field iff recorded value is zero",
                               "counts": dict(counts), "examples": examples}
    return evidence


def process_metadata(settings):
    process = settings.get("ProcessVersion")
    version = settings.get("Version")
    if process == "5.0":
        return {"raw": process, "effective": "5.0", "family": "PV2003", "source": "explicit ProcessVersion"}
    if process is None and version in ("4.5", "5.3", "5.4"):
        return {"raw": None, "effective": "5.0", "family": "PV2003", "source": "inferred from pre-5.7 Camera Raw Version " + version}
    raise ValueError("Unverified process/version combination: " + repr((process, version)))


def extract_record(settings, default_rules):
    values, sources, errors = {}, {}, []
    for parameter in PARAMS:
        key = parameter
        if parameter in ("Temperature", "Tint") and settings.get("WhiteBalance") == "Custom":
            if "Custom" + parameter in settings:
                key = "Custom" + parameter
        if key in settings:
            value = settings[key]
            source = "develop_settings." + key
        elif parameter in default_rules:
            value = default_rules[parameter]["value"]
            source = "catalog_verified_omitted_zero." + parameter
        else:
            value, source = None, "missing_absolute_field"
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            errors.append("missing/non-numeric absolute " + parameter)
            value = None
        values[parameter], sources[parameter] = value, source
    try:
        validate_parameters([values[p] for p in PARAMS])
    except ValueError as exc:
        errors.append(str(exc))
    return values, sources, errors


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def extract_labels(db_path=Path("data/raw/fivek_dataset/raw_photos/fivek.lrcat"), out_dir=Path("data/intermediate")):
    db_path, out_dir = Path(db_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "expert_labels.json"
    old, snapshot = {}, None
    if output.exists():
        old_bytes = output.read_bytes()
        old = json.loads(old_bytes)
        digest = hashlib.sha256(old_bytes).hexdigest()
        snapshot = out_dir / ("expert_labels.snapshot." + digest + ".json")
        if not snapshot.exists():
            with snapshot.open("xb") as handle:
                handle.write(old_bytes)
            snapshot.chmod(0o444)
        elif sha256_file(snapshot) != digest:
            raise ValueError("Existing immutable snapshot hash mismatch")
    source_hash = sha256_file(db_path)
    results, provenance, filtered = {}, {}, defaultdict(list)
    counts, distributions = Counter(), defaultdict(Counter)
    with open_catalog(db_path) as connection:
        default_rules = audit_default_rules(connection)
        for expert, collection_id in COLLECTIONS.items():
            name = connection.execute("SELECT name FROM AgLibraryCollection WHERE id_local=?", (collection_id,)).fetchone()
            if name is None or name[0] != expert:
                raise ValueError("Unexpected expert collection: " + expert)
            rows = connection.execute("""
                SELECT file.id_local AS root_file_id, file.lc_idx_filename AS filename,
                       image.id_local AS image_id, dev.id_local AS develop_id,
                       dev.historySettingsID AS history_id, dev.text AS text
                FROM AgLibraryCollectionImage AS collection
                JOIN Adobe_images AS image ON image.id_local = collection.image
                JOIN AgLibraryFile AS file ON file.id_local = image.rootFile
                LEFT JOIN Adobe_imageDevelopSettings AS dev ON dev.image = image.id_local
                WHERE collection.collection = ? ORDER BY file.id_local, dev.id_local
            """, (collection_id,))
            grouped = defaultdict(list)
            for row in rows:
                counts["query_rows"] += 1
                grouped[(row["root_file_id"], row["filename"])].append(row)
            for (root_file, filename), records in grouped.items():
                image_id = filename.lower()
                if expert in results.setdefault(image_id, {}):
                    raise ValueError("Canonical filename collision: " + image_id)
                nonempty = [row for row in records if row["text"] and row["text"].strip()]
                counts["empty_records"] += len(records) - len(nonempty)
                if len(nonempty) != 1:
                    filtered[image_id].append(expert + ": expected exactly one nonempty record, found " + str(len(nonempty)))
                    results[image_id][expert] = dict.fromkeys(PARAMS)
                    continue
                row = nonempty[0]
                counts["nonempty_records"] += 1
                record = {"root_file_id": root_file, "image_id": row["image_id"], "develop_id": row["develop_id"],
                          "history_id": row["history_id"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
                try:
                    settings = parse_settings(row["text"])
                    values, sources, errors = extract_record(settings, default_rules)
                    record.update({"fields": sources, "WhiteBalance": settings.get("WhiteBalance"),
                                   "CameraProfile": settings.get("CameraProfile"), "Version": settings.get("Version")})
                    for key in ("ProcessVersion", "Version", "WhiteBalance", "CameraProfile"):
                        distributions[key][str(settings.get(key, "omitted"))] += 1
                    for source in sources.values():
                        counts[source] += 1
                    try:
                        record["process_version"] = process_metadata(settings)
                    except ValueError as exc:
                        errors.append(str(exc))
                    filtered[image_id].extend(expert + ": " + error for error in errors)
                except (ValueError, SyntaxError) as exc:
                    values = dict.fromkeys(PARAMS)
                    filtered[image_id].append(expert + ": " + str(exc))
                results[image_id][expert] = values
                provenance.setdefault(image_id, {})[expert] = record
    filtered = {key: reasons for key, reasons in filtered.items() if reasons}
    for image_id, experts in results.items():
        if set(experts) != set(COLLECTIONS):
            filtered.setdefault(image_id, []).append("Missing expert collections")
    valid = {image_id: experts for image_id, experts in results.items() if image_id not in filtered}
    statistics = {}
    if valid:
        labels = np.array([[[experts[e][p] for p in PARAMS] for e in COLLECTIONS] for experts in valid.values()])
        means = labels.mean(axis=1)
        validate_parameters(labels)
        validate_parameters(means)
        for j, parameter in enumerate(PARAMS):
            statistics[parameter] = dict(zip(("mean_min", "mean_p1", "mean_p99", "mean_max"),
                                             map(float, np.percentile(means[:, j], [0, 1, 99, 100]))))
            statistics[parameter].update({"expert_min": float(labels[:, :, j].min()), "expert_max": float(labels[:, :, j].max())})
    changes = []
    for image_id, experts in results.items():
        for expert, values in experts.items():
            for parameter, value in values.items():
                previous = old.get(image_id, {}).get(expert, {}).get(parameter)
                if previous != value:
                    changes.append({"id": image_id, "expert": expert, "parameter": parameter, "old": previous, "new": value})
    if sha256_file(db_path) != source_hash:
        raise ValueError("Source Catalog changed during extraction")
    write_json(output, results)
    audit = {"extraction_version": VERSION, "source_catalog": str(db_path.resolve()), "source_sha256": source_hash,
             "extractor_sha256": sha256_file(Path(__file__)), "labels_sha256": sha256_file(output),
             "parameter_order": list(PARAMS), "old_snapshot": str(snapshot.resolve()) if snapshot else None,
             "counts": {**dict(counts), "images": len(results), "valid_images": len(valid), "filtered_images": len(filtered)},
             "default_rules": default_rules, "process_references": PROCESS_REFERENCES,
             "process_note": "Omission is resolved only for observed pre-5.7 Camera Raw versions, not as a generic missing-key default. Catalog history step 1555242 explicitly identifies PV2010 as 5.7.",
             "xmp_note": "AdditionalMetadata.xmp may be stale; it is not used as a fallback for expert settings.",
             "distributions": {key: dict(value) for key, value in distributions.items()},
             "filtered_images": filtered, "statistics": statistics,
             "differences": {"removed_images": sorted(set(old) - set(results)), "added_images": sorted(set(results) - set(old)), "changed_fields": changes},
             "records": provenance}
    write_json(out_dir / "label_audit.json", audit)
    print(json.dumps({"labels": str(output), "audit": str(out_dir / "label_audit.json"), "images": len(results),
                      "valid_images": len(valid), "filtered_images": len(filtered), "changed_fields": len(changes)}))
    return results, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/raw/fivek_dataset/raw_photos/fivek.lrcat"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/intermediate"))
    args = parser.parse_args()
    extract_labels(args.catalog, args.output_dir)
