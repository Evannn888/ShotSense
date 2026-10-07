"""One isolated decode-to-full-PNG measurement; run under /usr/bin/time -l."""
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
started = time.perf_counter()
from src.photo_engine import process_photo, render_photo

result, source = process_photo(Path(sys.argv[1]))
before, after, metadata = render_photo(source)
with tempfile.TemporaryDirectory(prefix='shotsense-benchmark-') as directory:
    folder = Path(directory)
    (folder/'before.png').write_bytes(before)
    (folder/'after.png').write_bytes(after)
    (folder/'adjustment.json').write_text(json.dumps(dict(result,export=metadata)))
print(json.dumps({'input_sha256':result['input_sha256'],'size':metadata['size'],
                  'selected_recipe':result['selected_recipe'],'output_png_sha256':metadata['output_png_sha256'],
                  'seconds_including_import_decode_recipe_png_and_temp_exports':time.perf_counter()-started}))
