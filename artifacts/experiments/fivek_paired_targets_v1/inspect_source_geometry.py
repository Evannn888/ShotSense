"""Reproduce source-only RAW geometry metadata; do not read embedded expert XMP."""
import json
from pathlib import Path

from PIL import Image, TiffImagePlugin
from acquire import HERE, ROOT, load, save_new

records = []
for row in load(HERE / "selection.json")["photos"]:
    source = ROOT / row["source_path"]
    with Image.open(source) as image:
        offsets = image.tag_v2.get(330, ())
    record = {"source_id": row["source_id"], "raw_subifds": []}
    with source.open("rb") as stream:
        header = stream.read(8)
        for offset in offsets:
            directory = TiffImagePlugin.ImageFileDirectory_v2(header)
            stream.seek(offset)
            directory.load(stream)
            if directory.get(50720) is not None:
                record["raw_subifds"].append({str(tag): str(directory.get(tag)) for tag in
                    (256, 257, 274, 50718, 50719, 50720, 50829)})
    assert len(record["raw_subifds"]) == 1, row["source_id"]
    records.append(record)
assert len(records) == 100
result = {"scope": "Only source RAW SubIFD geometry tags; no embedded expert XMP/WB/pixels used.", "photos": records}
path = HERE / "source_raw_crop_metadata.json"
if path.exists():
    assert load(path) == result
else:
    save_new(path, result)
print("Source-only crop metadata reproduced for all 100 frozen sources.")
