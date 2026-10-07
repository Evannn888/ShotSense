"""Reproduce one frozen personal JPEG through the ordinary isolated app worker."""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from app.streamlit_app import run_job
from src.photo_engine import render_photo
from src.preview import render_linear_preview

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image', type=Path)
parser.add_argument('--workflow', choices=('natural', 'manual'), default='natural')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

stop = threading.Event()
rss = {'root_peak_bytes': 0, 'tree_peak_bytes': 0, 'samples': 0}

def monitor():
    while not stop.is_set():
        result = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True)
        rows = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
        descendants = {os.getpid()}
        while True:
            expanded = descendants | {pid for pid, parent, _ in rows if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        root = sum(size * 1024 for pid, _, size in rows if pid == os.getpid())
        total = sum(size * 1024 for pid, _, size in rows if pid in descendants)
        rss['root_peak_bytes'] = max(root, rss['root_peak_bytes'])
        rss['tree_peak_bytes'] = max(total, rss['tree_peak_bytes'])
        rss['samples'] += 1
        stop.wait(.25)

thread = threading.Thread(target=monitor, daemon=True)
thread.start()
started = time.perf_counter()
report = {'input': str(args.image), 'workflow': args.workflow,
          'input_sha256_before': hashlib.sha256(args.image.read_bytes()).hexdigest(),
          'memory_semantics': 'Observed process-tree RSS sum at 250 ms; shared pages can be counted more than once. macOS rusage high-water RSS is process-only, in bytes.'}
try:
    payload, source = run_job(args.image.read_bytes(), args.image.suffix,
                              natural=args.workflow == 'natural', recipe='shadows')
    report['app_worker_seconds'] = time.perf_counter() - started
    report['payload'] = payload
    if args.workflow == 'natural':
        rendered = time.perf_counter()
        before, after, metadata = render_photo(source, 1.)
        report['render_100_seconds'] = time.perf_counter() - rendered
        for name, data in [('before', before), ('after', after)]:
            (args.output / (name + '.png')).write_bytes(data)
            with Image.open(io.BytesIO(data)) as image:
                assert image.format == 'PNG' and list(image.size) == payload['output_size']
                assert 'srgb' in image.info
                np.testing.assert_array_equal(np.asarray(image), source['original' if name == 'before' else 'adjusted'])
            report[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'size': payload['output_size']}
        report['export'] = metadata
        del before, after
        gc.collect()
        rendered = time.perf_counter()
        before, after, zero = render_photo(source, 0.)
        assert before == after and zero['changed_pixel_fraction'] == 0
        with Image.open(io.BytesIO(after)) as image:
            np.testing.assert_array_equal(np.asarray(image), source['original'])
        report['zero_identity_verified'] = True
        report['render_0_seconds'] = time.perf_counter() - rendered
        del before, after
    else:
        before, after, metadata = render_linear_preview(source, {'Exposure': 0., 'HighlightRecovery': 0.})
        assert before == after
        report['manual_preview'] = metadata
        report['zero_identity_verified'] = True
    assert hashlib.sha256(args.image.read_bytes()).hexdigest() == report['input_sha256_before']
    report['source_preserved'] = True
    report['status'] = 'passed'
except Exception as error:
    report['status'] = 'failed'
    report['error'] = type(error).__name__ + ': ' + str(error)
finally:
    stop.set()
    thread.join(timeout=2)
    report['total_seconds'] = time.perf_counter() - started
    report['rss'] = rss
    report['rusage_root_high_water_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['rusage_children_high_water_bytes'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('workflow', 'status', 'total_seconds', 'rss')}), flush=True)
sys.exit(0 if report['status'] == 'passed' else 1)
