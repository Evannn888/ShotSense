"""Verify retained inputs/results and unchanged application/model identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
from PIL import Image

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
PREVIOUS = ROOT / 'artifacts/experiments/scene_analysis_research_v1'
MODEL = ROOT / 'data/user_photo_diagnostics/scene-analysis-research-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((FOLDER / name).read_text())


baseline = read('baseline.json')
application = {p: sha(ROOT / p) == value for p, value in baseline.items()}
assert all(application.values())
inputs, results, native = read('inputs.json'), read('results.json'), read('vision_results.json')
assert len(inputs) == 10 and len(results['rows']) == 40 and len(native['rows']) == 20
assert sha(FOLDER / 'expected_labels.json') == results['expected_labels_sha256']
assert sha(FOLDER / 'inputs.json') == results['source_manifest_sha256']
preserved = {i['id']: sha(i['path']) == i['thumbnail_sha256'] for i in inputs}
downloads = {i['id']: sha(i['downloaded_path']) == i['downloaded_sha256']
             for i in inputs if not i.get('synthetic')}
assert all(preserved.values()) and all(downloads.values())
original_sha1 = {i['id']: hashlib.sha1(Path(i['downloaded_path']).read_bytes()).hexdigest() == i['commons_original_sha1']
                 for i in inputs if i.get('original_sha1_verified')}
assert len(original_sha1) == 4 and all(original_sha1.values())
controls = {}
for item in inputs:
    if not item.get('synthetic'):
        continue
    parent = next(i for i in inputs if i['id'] == item['parent_id'])
    assert item['parent_thumbnail_sha256'] == parent['thumbnail_sha256']
    with Image.open(parent['path']) as im:
        srgb = np.asarray(im.convert('RGB'), dtype=np.float64) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4) / 8
    transformed = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
    encoded = np.rint(np.clip(transformed, 0, 1) * 255).astype(np.uint8)
    with Image.open(item['path']) as im:
        controls[item['id']] = bool(np.array_equal(encoded, np.asarray(im)))
assert all(controls.values())
assert sha(MODEL / 'mobileclip2_s0_image_fp32.onnx') == results['model_sha256']
assert sha(MODEL / 'text_embeddings.npy') == results['text_sha256']
assert sha(PREVIOUS / 'mobileclip_prompts.json') == results['prompt_sha256']
assert sha(MODEL / 'vision_probe') == native['helper_sha256']
text = np.load(MODEL / 'text_embeddings.npy', allow_pickle=False)
for row in results['rows']:
    assert np.isfinite(row['embedding']).all() and np.isfinite(row['cosine_scores']).all()
    assert np.allclose(np.asarray(row['embedding'], dtype=np.float32) @ text.T,
                       row['cosine_scores'], atol=1e-6, rtol=0)
native_by_run = {(r['id'], r['pass']): r for r in native['rows']}
native_differences = []
for item in inputs:
    a, b = native_by_run[(item['id'], 0)], native_by_run[(item['id'], 1)]
    for row in (a, b):
        assert len(row['labels']) == len(native['taxonomy'])
        assert all(np.isfinite(v['score']) for v in row['labels'])
        assert all(np.isfinite(v['box_bottom_left']).all() and 0 <= v['score'] <= 1 for v in row['faces'])
    aa = {v['identifier']: v['score'] for v in a['labels']}
    bb = {v['identifier']: v['score'] for v in b['labels']}
    native_differences.append(max(abs(aa[k] - bb[k]) for k in aa))
assert read('preprocessing_parity.json')['stock_pil_matches_torchvision_uint8_exact']
previous_sources = json.loads((PREVIOUS / 'inputs.json').read_text())
previous_preserved = {i['id']: sha(i['source_path']) == i['source_sha256'] and sha(i['path']) == i['thumbnail_sha256']
                      for i in previous_sources}
assert all(previous_preserved.values())
core_paths = json.loads((PREVIOUS / 'final_verification.json').read_text())['preserved_raw_model_core_matches_HEAD']
core = {p: (ROOT / p).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)
        for p in core_paths}
assert all(core.values())
pip = subprocess.check_output([str(ROOT / 'venv/bin/python'), '-m', 'pip', 'check'], text=True).strip()
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=10) as response:
    health = dict(http_status=response.status, body=response.read().decode())
assert health['http_status'] == 200 and health['body'] == 'ok'
verification = dict(status='small_recognition_diagnostic_complete_not_edit_quality_acceptance',
    counts=dict(public_photos=8, verified_original_downloads=4, published_preview_downloads=4,
                synthetic_controls=2, mobileclip_view_passes=40, native_image_passes=20),
    application_dependency_preserved=application, input_thumbnails_preserved=preserved,
    downloaded_files_preserved=downloads, original_commons_sha1_verified=original_sha1,
    dark_controls_exact_pixel_replay=controls, previous_sources_and_thumbnails_preserved=previous_preserved,
    preserved_raw_model_core_matches_HEAD=core, model_text_prompts_helper_preserved=True,
    label_and_input_manifests_preserved=True, stock_preprocessing_pixel_parity=True,
    finite_outputs_and_embedding_score_replay=True,
    mobileclip_repeat_max_cosine_difference=results['repeat_max_cosine_difference'],
    native_repeat_max_classification_difference=max(native_differences),
    native_classification_revision=native['classification_revision'], face_revision=native['face_revision'],
    pip_check=pip, git_diff_check=True, local_service_health=health,
    application_changed=False, production_weights_changed=False, sharpening=False,
    independent_scene_accuracy_passed=False, new_edit_quality_passed=False,
    source_record_files={p: sha(FOLDER / p) for p in ('PROTOCOL.md', 'inputs.json', 'expected_labels.json',
                        'results.json', 'vision_results.json', 'summary.json', 'REPORT.zh-CN.md')})
(FOLDER / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print(json.dumps({'status': verification['status'], 'health': health,
                  'native_repeat_max_difference': max(native_differences), 'pip_check': pip}))
