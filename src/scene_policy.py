"""Offline scene evidence and fixed, conservative JPEG adjustment limits."""
from functools import lru_cache
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from src.adaptive_lut import validate_rgb

VERSION = 'scene-aware-natural-v2'
# ponytail: reuse verified local research assets; package a bundle when distributing the app.
BUNDLE = Path(__file__).resolve().parents[1] / 'data/user_photo_diagnostics/scene-analysis-research-v1'
MODEL_SHA = '3719246574ad6edba0f5c700ed2cea82210d1318f29689355c3db2e165946a90'
TEXT_SHA = '9c08adcf7174338c0d6dcda27d7f8887c7a92c69b11a4048768564f18b40cee7'
SUBJECTS = ('people', 'natural_landscape', 'indoor_room', 'urban_architecture',
            'animal_pet', 'food', 'object_still_life', 'document_graphic')
LIGHTING = ('daylight', 'overcast', 'sunset_sunrise', 'night', 'indoor_artificial',
            'indoor_natural', 'underexposed_daylight')


@lru_cache(maxsize=1)
def load_scene_model():
    import onnxruntime as ort
    model, text = BUNDLE / 'mobileclip2_s0_image_fp32.onnx', BUNDLE / 'text_embeddings.npy'
    for path, expected in ((model, MODEL_SHA), (text, TEXT_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Scene asset checksum mismatch: ' + path.name)
    vectors = np.load(text, allow_pickle=False)
    if (vectors.shape != (15, 512) or vectors.dtype != np.float32
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)):
        raise ValueError('Invalid scene text vectors')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    return session, vectors


def scene_views(rgb):
    image = Image.fromarray(rgb)
    w, h = image.size
    if max(w, h) > 16 * min(w, h):
        raise ValueError('Scene crop skipped for extreme aspect ratio')
    size = (256, int(256*h/w)) if w <= h else (int(256*w/h), 256)
    scaled = image.resize(size, Image.Resampling.BICUBIC)
    x, y = int(round((size[0]-256)/2)), int(round((size[1]-256)/2))
    return {'center_crop': scaled.crop((x, y, x+256, y+256)),
            'full_frame': ImageOps.pad(image, (256, 256), method=Image.Resampling.BICUBIC,
                                      color=(127, 127, 127), centering=(.5, .5))}


def analyze_scene(rgb):
    validate_rgb(rgb)
    started = time.perf_counter()
    try:
        frames = scene_views(rgb)
        session, vectors = load_scene_model()
        views = {}
        for name, frame in frames.items():
            tensor = np.ascontiguousarray(np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)[None]/255)
            feature = session.run(None, {'image': tensor})[0]
            if (feature.shape != (1, 512) or not np.isfinite(feature).all()
                    or not np.allclose(np.linalg.norm(feature, axis=1), 1, atol=1e-4)):
                raise ValueError('Invalid scene image embedding')
            scores = (feature @ vectors.T)[0]
            if not np.isfinite(scores).all():
                raise ValueError('Invalid scene scores')
            views[name] = {'subject': dict(zip(SUBJECTS, map(float, scores[:8]))),
                           'lighting': dict(zip(LIGHTING, map(float, scores[8:])))}
        result = {'status': 'available', 'views': views, 'model': 'MobileCLIP2-S0',
                  'model_sha256': MODEL_SHA, 'text_sha256': TEXT_SHA,
                  'semantics': 'Source-only cosine similarities; not calibrated probabilities, illuminant or artistic intent.'}
    except Exception as error:
        # Recognition is optional; its failure must not prevent the existing photo workflow.
        result = {'status': 'unavailable', 'error': type(error).__name__ + ': ' + str(error)[:200],
                  'fallback': 'existing protected JPEG processing; no downloads'}
    result['seconds'] = time.perf_counter()-started
    return result


def scene_limits(rgb, evidence):
    validate_rgb(rgb)
    limits = {'color_strength': .35, 'tone_lift': .15, 'shadow_strength': .75}
    result = {'version': VERSION, 'limits': limits, 'reasons': [],
              'status': 'restricted' if evidence['status'] == 'available' else 'recognition_fallback',
              'white_balance': 'not applied; illuminant/neutral evidence is unvalidated'}
    if evidence['status'] != 'available':
        result['reasons'].append('recognition_unavailable')
        return result
    h, w = rgb.shape[:2]
    sample = rgb[::max(1, (h+255)//256), ::max(1, (w+255)//256)].astype(np.float32)/255
    y = sample @ np.array([.2126, .7152, .0722], dtype=np.float32)
    median = float(np.median(y))
    result['source_display_luma_median'] = median
    views = [evidence['views'][v] for v in ('full_frame', 'center_crop')]
    ranks = [sorted(v['lighting'], key=v['lighting'].get, reverse=True) for v in views]

    def restrict(reason, color, tone, shadow):
        result['reasons'].append(reason)
        for name, amount in zip(limits, (color, tone, shadow)):
            limits[name] = min(limits[name], amount)

    near = lambda label: any(v['lighting'][label] >= max(v['lighting'].values())-.02 for v in views)
    if all(r[0] == 'night' for r in ranks) and median < .25:
        restrict('night_mood', .10, 0., 0.)
    if near('sunset_sunrise'):
        restrict('sunset_silhouette', .15, .025, .15)
    if (ranks[0][0] != ranks[1][0] or 'underexposed_daylight' in [r[0] for r in ranks]
            or any(v['lighting'][r[0]]-v['lighting'][r[1]] < .015 for v, r in zip(views, ranks))):
        restrict('uncertain_lighting', .15, .03, .20)
    if median < .06:
        restrict('very_dark_source', .10, .02, .10)
    subject_ranks = [sorted(v['subject'], key=v['subject'].get, reverse=True) for v in views]
    if all(r[0] == 'document_graphic' and v['subject'][r[0]]-v['subject'][r[1]] >= .02
           for v, r in zip(views, subject_ranks)):
        restrict('document_preserved', 0., 0., 0.)
    if not result['reasons']:
        result['status'] = 'existing_limits'
    return result
