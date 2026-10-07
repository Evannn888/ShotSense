"""Bounded native feasibility probe; original photos and current app remain unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).parent
PREVIOUS = ROOT / 'artifacts/experiments/automatic_photo_validation_v1'
OUTPUT = ROOT / 'data/user_photo_diagnostics/scene-analysis-research-v1'
OUTPUT.mkdir(exist_ok=True)
assert not (FOLDER / 'vision_results.json').exists(), 'Preserve completed probe; use another run folder'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Preserve existing evidence: ' + str(path)
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


baseline = json.loads((PREVIOUS / 'baseline.json').read_text())
assert all(sha(ROOT / name) == expected for name, expected in baseline.items())
save_json(FOLDER / 'baseline.json', baseline)
cases = json.loads((PREVIOUS / 'inputs.json').read_text())['cases']
inputs = []
for case in cases:
    source = Path(case['path'])
    assert sha(source) == case['sha256']
    cached = ROOT / 'data/user_photo_diagnostics/automatic-validation-v1' / case['id'] / 'original.png'
    with Image.open(cached) as image:
        assert image.mode == 'RGB' and image.format == 'PNG'
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        thumbnail = OUTPUT / (case['id'] + '.png')
        image.save(thumbnail)
        pixels = np.asarray(image, dtype=np.float32) / 255
        y = pixels @ np.array([.2126, .7152, .0722], dtype=np.float32)
        inputs.append(dict(id=case['id'], path=str(thumbnail), thumbnail_sha256=sha(thumbnail),
                           thumbnail_size=list(image.size), original_render_sha256=sha(cached),
                           source_path=str(source), source_sha256=case['sha256'],
                           source_quantiles=[float(x) for x in np.quantile(y, [.05, .30, .50, .95])]))
save_json(FOLDER / 'inputs.json', inputs)
requests = [dict(id=x['id'], path=x['path']) for x in inputs]
save_json(OUTPUT / 'request.json', requests)
binary = OUTPUT / 'vision_probe'
build_started = time.perf_counter()
built = subprocess.run(['xcrun', 'clang', '-O2', '-fobjc-arc', str(FOLDER / 'vision_probe.m'),
                        '-framework', 'Foundation', '-framework', 'Vision', '-framework', 'ImageIO',
                        '-framework', 'CoreGraphics', '-o', str(binary)],
                       capture_output=True, text=True, timeout=60)
save_json(FOLDER / 'build-native.json', dict(seconds=time.perf_counter()-build_started,
                                    exit_code=built.returncode, stdout=built.stdout, stderr=built.stderr))
assert built.returncode == 0, built.stderr


def run(request):
    before = time.perf_counter()
    completed = subprocess.run([str(binary), str(request)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    result['subprocess_wall_seconds'] = time.perf_counter()-before
    result['stderr'] = completed.stderr
    return result


report = run(OUTPUT / 'request.json')
assert len(report['rows']) == 2*len(cases) and len(report['taxonomy']) > 1000
for row in report['rows']:
    assert len(row['labels']) == len(report['taxonomy'])
    assert all(np.isfinite(item['score']) and 0 <= item['score'] <= 1 for item in row['labels'])
    assert all(np.isfinite(row[key]) and row[key] >= 0
               for key in ('total_ms', 'classification_ms', 'face_ms', 'decode_ms'))
save_json(FOLDER / 'vision_results.json', report)
single = []
for identity in ('nir-himi-1P1yaGS_Gek-unsplash', 'a2132-IMG_4947', 'a2270-_DSC0033'):
    request = OUTPUT / (identity + '.json')
    save_json(request, [x for x in requests if x['id'] == identity])
    result = run(request)
    save_json(FOLDER / ('single-' + identity + '.json'), result)
    single.append(dict(id=identity, subprocess_wall_seconds=result['subprocess_wall_seconds'],
                       first_total_ms=result['rows'][0]['total_ms'],
                       second_total_ms=result['rows'][1]['total_ms']))

repeated_differences = []
for first, second in zip(report['rows'][:len(cases)], report['rows'][len(cases):]):
    assert first['id'] == second['id']
    a = {x['identifier']: x['score'] for x in first['labels']}
    b = {x['identifier']: x['score'] for x in second['labels']}
    assert a.keys() == b.keys()
    repeated_differences.append(dict(id=first['id'], max_label_score_difference=max(abs(a[k]-b[k]) for k in a),
                                    first_face_count=len(first['faces']), second_face_count=len(second['faces'])))
assert all(sha(ROOT / name) == expected for name, expected in baseline.items())
assert all(sha(Path(case['path'])) == case['sha256'] for case in cases)
save_json(FOLDER / 'verification.json', dict(
    status='feasibility_probe_completed_not_production_acceptance',
    app_dependency_hashes_unchanged=True, all17_original_source_hashes_unchanged=True,
    case_count=len(cases), passes=2, binary_sha256=sha(binary),
    native_source_sha256=sha(FOLDER / 'vision_probe.m'), python_sha256=sha(Path(__file__)),
    thumbnail_limit=512, repeats=repeated_differences, single_processes=single,
    warm_total_ms_quantiles=[float(x) for x in np.quantile(
        [row['total_ms'] for row in report['rows'][len(cases):]], [.50, .95])],
    native_model_downloads=0, native_new_python_dependencies=0, sharpening=False))
print('Completed17scene probes,2passes,3single-image processes; application and sources unchanged.')
