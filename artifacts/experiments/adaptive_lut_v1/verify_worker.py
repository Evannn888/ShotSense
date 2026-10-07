"""Fresh, sequential actual50MP LUT worker/export check using the existing60s parent."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import sys
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from app.streamlit_app import run_worker
from src.photo_engine import render_photo

name = sys.argv[1]
frozen = json.loads((Path(__file__).parent / 'inputs.json').read_text())
entry = next(case for case in frozen['cases'] if case['id'] == name)
source = Path(entry['path'])
folder = ROOT / 'data/user_photo_diagnostics/lut-v1' / ('worker-' + name)
folder.mkdir(exist_ok=True)
report = {'input': entry, 'status': 'running', 'parent_timeout_seconds': 60,
          'memory_semantics': '250ms sampled root/process-tree RSS sum includes sampling command; shared pages may count twice and short peaks may be missed. macOS rusage high-water bytes recorded separately.'}
rss = {'tree_peak_bytes': 0, 'root_peak_bytes': 0, 'samples': 0}
stop = threading.Event()
def monitor():
    while not stop.is_set():
        output = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True)
        rows = [tuple(map(int, line.split())) for line in output.stdout.splitlines()]
        ids = {os.getpid()}
        while True:
            expanded = ids | {pid for pid, parent, _ in rows if parent in ids}
            if ids == expanded: break
            ids = expanded
        rss['tree_peak_bytes'] = max(rss['tree_peak_bytes'], sum(size*1024 for pid, _, size in rows if pid in ids))
        rss['root_peak_bytes'] = max(rss['root_peak_bytes'], sum(size*1024 for pid, _, size in rows if pid == os.getpid()))
        rss['samples'] += 1; stop.wait(.25)
thread = threading.Thread(target=monitor, daemon=True); thread.start()
started = time.perf_counter()
try:
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
    run_worker([sys.executable, '-m', 'src.adaptive_lut', str(source), '--output', str(folder/'model.json'),
                '--preview-source', str(folder/'pixels.npz')])
    report['worker_call_seconds'] = time.perf_counter() - started
    metadata = json.loads((folder/'model.json').read_text())
    assert metadata['input_sha256'] == entry['sha256'] and metadata['size'] == entry['header_size']
    with np.load(folder/'pixels.npz', allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in ('original', 'adjusted')}
    assert arrays['original'].dtype == arrays['adjusted'].dtype == np.uint8
    assert arrays['original'].shape == arrays['adjusted'].shape
    for key in ('original', 'adjusted'):
        expected = metadata['input_pixels_sha256' if key == 'original' else 'output_pixels_sha256']
        assert hashlib.sha256(arrays[key]).hexdigest() == expected
    render_started = time.perf_counter()
    before, after, export = render_photo(arrays)
    export['semantics'] = 'Frozen pretrained LUT result blended with original sRGB baseline.'
    report['png_render_seconds'] = time.perf_counter() - render_started
    assert hashlib.sha256(after).hexdigest() == export['output_png_sha256']
    assert after == (ROOT/'data/user_photo_diagnostics/lut-v1'/name/'lut100.png').read_bytes()
    (folder/'result.png').write_bytes(after)
    with Image.open(folder/'result.png') as image:
        assert image.format == 'PNG' and image.mode == 'RGB'
        assert list(image.size) == entry['header_size'] and 'srgb' in image.info
        np.testing.assert_array_equal(np.asarray(image), arrays['adjusted'])
    del before, after
    half_before, half_after, half_export = render_photo(arrays, .5)
    half_export['semantics'] = export['semantics']
    # Native crop bytes match the independently row-blended output at50%.
    with Image.open(io.BytesIO(half_after)) as image:
        expected = np.rint(arrays['original'][100:380,100:520].astype(np.float32)*.5
                           + arrays['adjusted'][100:380,100:520].astype(np.float32)*.5).astype(np.uint8)
        np.testing.assert_array_equal(np.asarray(image.crop((100,100,520,380))), expected)
    del half_before, half_after
    zero_before, zero_after, zero_export = render_photo(arrays, 0)
    assert zero_before == zero_after and zero_export['changed_pixel_fraction'] == 0
    del zero_before, zero_after
    assert hashlib.sha256(source.read_bytes()).hexdigest() == entry['sha256']
    report.update(status='passed', lut=metadata, export=export, half_export=half_export,
                  source_preserved=True, matches_frozen_full_png=True, native_half_crop_matches=True,
                  zero_strength_identity=True)
except Exception as error:
    report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
finally:
    stop.set(); thread.join(timeout=2)
    report.update(total_seconds=time.perf_counter()-started, rss=rss,
                  root_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  children_high_water_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (folder/'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report.get(key) for key in ('status','worker_call_seconds','png_render_seconds','total_seconds','rss','error')}), flush=True)
sys.exit(0 if report['status'] == 'passed' else 1)
