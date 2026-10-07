"""Frozen image-only MobileCLIP/native-face diagnostics, without photo edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
PREVIOUS = ROOT / 'artifacts/experiments/scene_analysis_research_v1'
MODEL_LOCAL = ROOT / 'data/user_photo_diagnostics/scene-analysis-research-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def views(image):
    image = image.convert('RGB')
    w, h = image.size
    size = (256, int(256 * h / w)) if w <= h else (int(256 * w / h), 256)
    scaled = image.resize(size, Image.Resampling.BICUBIC)
    x, y = int(round((size[0] - 256) / 2)), int(round((size[1] - 256) / 2))
    return {
        'stock_center_crop': scaled.crop((x, y, x + 256, y + 256)),
        'full_frame_letterbox': ImageOps.pad(image, (256, 256), method=Image.Resampling.BICUBIC,
                                            color=(127, 127, 127), centering=(.5, .5)),
    }


def save_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    assert not (FOLDER / 'results.json').exists(), 'Preserve completed results.'
    expected = json.loads((FOLDER / 'expected_labels.json').read_text())
    inputs = json.loads((FOLDER / 'inputs.json').read_text())
    assert set(expected) == {i['id'] for i in inputs}
    baseline = json.loads((FOLDER / 'baseline.json').read_text())
    assert all(sha(ROOT / path) == value for path, value in baseline.items())
    for item in inputs:
        assert sha(item['path']) == item['thumbnail_sha256']
        if 'downloaded_path' in item:
            assert sha(item['downloaded_path']) == item['downloaded_sha256']
    model = MODEL_LOCAL / 'mobileclip2_s0_image_fp32.onnx'
    vectors = MODEL_LOCAL / 'text_embeddings.npy'
    prompt = PREVIOUS / 'mobileclip_prompts.json'
    assert sha(model) == '3719246574ad6edba0f5c700ed2cea82210d1318f29689355c3db2e165946a90'
    manifest = json.loads((PREVIOUS / 'text_embedding_manifest.json').read_text())
    assert sha(vectors) == manifest['sha256']
    assert sha(prompt) == manifest['prompt_sha256']
    labels = manifest['labels']
    text_vectors = np.load(vectors, allow_pickle=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    setup_ms = (time.perf_counter() - started) * 1000
    rows = []
    tensors = {}
    for item in inputs:
        with Image.open(item['path']) as image:
            for view, frame in views(image).items():
                tensors[(item['id'], view)] = np.ascontiguousarray(
                    np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)[None] / 255)
    for run in range(2):
        for (key, view), tensor in tensors.items():
            started = time.perf_counter()
            embedding = session.run(None, {'image': tensor})[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            assert embedding.shape == (1, 512) and np.isfinite(embedding).all()
            scores = (embedding @ text_vectors.T)[0]
            assert np.isfinite(scores).all()
            grouped = {}
            for group in ('subject', 'lighting'):
                order = sorted((i for i, pair in enumerate(labels) if pair[0] == group),
                               key=lambda i: float(scores[i]), reverse=True)
                grouped[group] = dict(top_two=[{'label': labels[i][1], 'cosine': float(scores[i])}
                                              for i in order[:2]],
                                      top_two_gap=float(scores[order[0]] - scores[order[1]]))
            rows.append(dict(id=key, view=view, run=run, groups=grouped,
                             cosine_scores=[float(v) for v in scores],
                             embedding=embedding[0].tolist(), model_ms=elapsed_ms))
    pairs = [(rows[i], rows[i + len(tensors)]) for i in range(len(tensors))]
    difference = max(float(np.max(np.abs(np.asarray(a['cosine_scores']) - b['cosine_scores'])))
                     for a, b in pairs)
    assert difference <= 1e-6
    save_json(FOLDER / 'results.json', dict(model_sha256=sha(model), text_sha256=sha(vectors),
              prompt_sha256=sha(prompt), labels=labels, expected_labels_sha256=sha(FOLDER / 'expected_labels.json'),
              source_manifest_sha256=sha(FOLDER / 'inputs.json'), providers=session.get_providers(),
              threads=1, setup_ms=setup_ms, repeat_max_cosine_difference=difference,
              score_semantics='Raw cosine similarities, not calibrated probabilities.', rows=rows))
    native_inputs = FOLDER / 'native_inputs.json'
    save_json(native_inputs, [{'id': i['id'], 'path': i['path']} for i in inputs])
    helper = MODEL_LOCAL / 'vision_probe'
    helper_sha = sha(helper)
    native = subprocess.run([str(helper), str(native_inputs)], check=True, capture_output=True, timeout=60)
    parsed = json.loads(native.stdout)
    assert len(parsed['rows']) == len(inputs) * 2
    parsed['helper_sha256'] = helper_sha
    save_json(FOLDER / 'vision_results.json', parsed)
    print(json.dumps({'inputs': len(inputs), 'views': len(tensors), 'passes': 2,
                      'repeat_max_cosine_difference': difference,
                      'native_rows': len(parsed['rows']), 'native_face_counts':
                      {r['id']: len(r['faces']) for r in parsed['rows'] if r['pass'] == 0}}))
