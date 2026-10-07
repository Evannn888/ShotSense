"""One predeclared original-size public portrait functional JPEG worker/export check."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.photo_engine import render_photo

items = json.loads((ROOT/'artifacts/experiments/scene_analysis_public_v1/inputs.json').read_text())
item = next(i for i in items if i['id']=='day_portrait')
source = Path(item['downloaded_path'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == item['downloaded_sha256']
folder = ROOT/'data/external/white_balance_v2/native'
folder.mkdir(exist_ok=False)
frozen = {'id': item['id'], 'source_sha256': item['downloaded_sha256'],
          'source_size': item['commons_original_size'],
          'worker_contract': 'Existing JPEG CLI plus --scene-aware --white-balance;60s bound; exact native PNG/zero-strength check only, not preference or latency acceptance.',
          'source_sha256s': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), ROOT/'src/scene_policy.py', ROOT/'src/lut_natural.py', ROOT/'src/white_balance.py']}}
with (HERE/'native_final_baseline.json').open('x') as f:
    f.write(json.dumps(frozen, indent=2)+'\n')
started = time.perf_counter()
command = [sys.executable, '-m', 'src.lut_natural', str(source), '--scene-aware', '--white-balance',
           '--output', str(folder/'result.json'), '--preview-source', str(folder/'pixels.npz')]
run = subprocess.run(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                     capture_output=True, text=True, timeout=60)
(HERE/'native_final_worker_stderr.log').write_text(run.stderr)
assert run.returncode == 0, run.stderr
result = json.loads((folder/'result.json').read_text())
with np.load(folder/'pixels.npz', allow_pickle=False) as cache:
    pixels = {key: cache[key].copy() for key in ('original','natural_base','adjusted')}
assert result['input_sha256'] == item['downloaded_sha256']
assert result['output_size'] == item['commons_original_size'] == [5472,3648]
before, after, metadata = render_photo(pixels)
_, zero, _ = render_photo(pixels, 0.)
assert zero == before
with Image.open(io.BytesIO(after)) as im:
    assert im.size == (5472,3648) and im.info['srgb'] == 0
    assert np.array_equal(np.array(im), pixels['adjusted'])
(folder/'adjusted.png').write_bytes(after)
assert hashlib.sha256(source.read_bytes()).hexdigest() == item['downloaded_sha256']
for path, expected in frozen['source_sha256s'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected
verification = {'source': frozen, 'result': result, 'export': metadata, 'zero_strength_exact_original': True,
                'native_png_exact': True, 'output_png_sha256': hashlib.sha256(after).hexdigest(),
                'elapsed_seconds_including_export': time.perf_counter()-started}
with (HERE/'native_final_verification.json').open('x') as f:
    f.write(json.dumps(verification,indent=2,allow_nan=False)+'\n')
print(json.dumps({'size': result['output_size'], 'worker_seconds':result['timing']['end_to_end_seconds'],
                  'total_seconds':verification['elapsed_seconds_including_export'],
                  'reasons':result['scene_policy']['reasons'], 'exact_png':True},indent=2))
