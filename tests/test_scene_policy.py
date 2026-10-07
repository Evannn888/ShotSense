import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from src import scene_policy
from src.lut_natural import process_jpeg, protect_rgb
from src.photo_engine import render_photo


def evidence(lighting, other=None, subject='natural_landscape'):
    def view(label):
        return {'lighting': {k: .30 if k == label else .1 for k in scene_policy.LIGHTING},
                'subject': {k: .30 if k == subject else .1 for k in scene_policy.SUBJECTS}}
    return {'status': 'available', 'views': {'full_frame': view(lighting), 'center_crop': view(other or lighting)}}


def test_scene_restraint_keeps_night_black_levels_and_handles_intent_ambiguity():
    rgb = np.full((36, 72, 3), 32, dtype=np.uint8)
    night = scene_policy.scene_limits(rgb, evidence('night'))
    assert night['limits']['tone_lift'] == night['limits']['shadow_strength'] == 0
    _, result = protect_rgb(rgb, 255-rgb, night['limits'])
    np.testing.assert_array_equal(result, rgb)
    ordinary = scene_policy.scene_limits(rgb, evidence('daylight'))
    assert ordinary['limits']['shadow_strength'] == .75
    sunset = scene_policy.scene_limits(rgb, evidence('sunset_sunrise'))
    assert sunset['limits']['shadow_strength'] == .15
    uncertain = scene_policy.scene_limits(rgb, evidence('daylight', 'overcast'))
    assert uncertain['limits']['shadow_strength'] == .20
    mistaken = scene_policy.scene_limits(rgb, evidence('underexposed_daylight'))
    assert 'uncertain_lighting' in mistaken['reasons']
    extreme = scene_policy.scene_limits(rgb//4, evidence('daylight'))
    assert extreme['limits']['shadow_strength'] == .10
    document = scene_policy.scene_limits(rgb, evidence('daylight', subject='document_graphic'))
    assert all(v == 0 for v in document['limits'].values())
    ambiguous = evidence('daylight')
    for v in ambiguous['views'].values():
        v['lighting']['night'] = .29
    assert 'night_mood' not in scene_policy.scene_limits(rgb, ambiguous)['reasons']
    assert scene_policy.scene_limits(rgb, ambiguous)['limits']['shadow_strength'] == .20
    disagreement = scene_policy.scene_limits(rgb, evidence('night', 'underexposed_daylight'))
    assert 'night_mood' not in disagreement['reasons']
    assert disagreement['limits']['shadow_strength'] == .20
    with pytest.raises(ValueError):
        protect_rgb(rgb, rgb, {'color_strength': -.1, 'tone_lift': 0})


def test_missing_scene_assets_fall_back_to_exact_existing_pixels(tmp_path, monkeypatch):
    rgb = np.random.default_rng(32).integers(0, 230, (129, 192, 3), dtype=np.uint8)
    jpeg = tmp_path/'image.jpg'
    Image.fromarray(rgb).save(jpeg, quality=100, subsampling=0)
    if not (scene_policy.BUNDLE.parents[1]/'external_models/image_adaptive_3dlut/manifest.json').exists():
        pytest.skip('Prepare pinned LUT assets')
    before, original = process_jpeg(jpeg)
    scene_policy.load_scene_model.cache_clear()
    monkeypatch.setattr(scene_policy, 'BUNDLE', tmp_path/'missing')
    try:
        metadata, current = process_jpeg(jpeg, scene_aware=True)
    finally:
        scene_policy.load_scene_model.cache_clear()
    assert metadata['scene_analysis']['status'] == 'unavailable'
    assert metadata['scene_policy']['status'] == 'recognition_fallback'
    for name in original:
        np.testing.assert_array_equal(original[name], current[name])
    assert metadata['natural_color']['natural']['settings'] == before['natural_color']['natural']['settings']
    assert metadata['shadow_adjustment']['strength'] == before['shadow_adjustment']['strength']
    _, output, export = render_photo(current)
    with Image.open(io.BytesIO(output)) as image:
        assert image.size == (192, 129) and 'srgb' in image.info
        np.testing.assert_array_equal(np.array(image), current['adjusted'])
    _, zero, _ = render_photo(current, 0)
    baseline, _, _ = render_photo(current)
    assert zero == baseline and export['output_png_sha256'] == hashlib.sha256(output).hexdigest()


def test_scene_onnx_scores_repeat_and_preprocessing_matches_prior_probe():
    if not (scene_policy.BUNDLE/'mobileclip2_s0_image_fp32.onnx').exists():
        pytest.skip('Local optional recognition assets not prepared')
    rgb = np.random.default_rng(6).integers(0, 256, (363, 549, 3), dtype=np.uint8)
    from artifacts.experiments.scene_analysis_public_v1.validate import views
    previous = views(Image.fromarray(rgb))
    frames = scene_policy.scene_views(rgb)
    np.testing.assert_array_equal(frames['center_crop'], previous['stock_center_crop'])
    np.testing.assert_array_equal(frames['full_frame'], previous['full_frame_letterbox'])
    a, b = scene_policy.analyze_scene(rgb), scene_policy.analyze_scene(rgb)
    assert a['status'] == b['status'] == 'available'
    assert a['views'] == b['views']
    assert a['model_sha256'] == scene_policy.MODEL_SHA
    for v in a['views'].values():
        assert len(v['subject']) == 8 and len(v['lighting']) == 7
        assert all(np.isfinite(x) for group in v.values() for x in group.values())
    assert scene_policy.analyze_scene(np.zeros((1, 100, 3), dtype=np.uint8))['status'] == 'unavailable'


@pytest.mark.parametrize('failure', ['clipping', 'malformed'])
def test_final_scene_guard_restores_original_and_consistent_cached_base(tmp_path, monkeypatch, failure):
    from src import adaptive_lut, lut_natural
    if not (adaptive_lut.BUNDLE/'manifest.json').exists():
        pytest.skip('Prepare pinned LUT')
    jpeg = tmp_path/'photo.jpg'
    Image.fromarray(np.full((24, 36, 3), 80, dtype=np.uint8)).save(jpeg)
    monkeypatch.setattr(scene_policy, 'analyze_scene', lambda _: evidence('daylight'))
    apply = lut_natural.apply_shadow_lift
    calls = []
    def damaged(rgb, strength):
        metadata, adjusted = apply(rgb, strength)
        calls.append(strength)
        if len(calls) == 1:
            adjusted = np.full_like(rgb, 255) if failure == 'clipping' else adjusted[:, :2]
        return metadata, adjusted
    monkeypatch.setattr(lut_natural, 'apply_shadow_lift', damaged)
    metadata, pixels = process_jpeg(jpeg, scene_aware=True)
    assert metadata['scene_policy']['status'] == 'candidate_rejected'
    assert metadata['shadow_adjustment']['strength'] == 0
    assert metadata['shadow_adjustment']['selection']['suggested_strength'] == 0
    assert metadata['natural_color']['natural']['applied'] is False
    for name in ('natural_base', 'adjusted'):
        np.testing.assert_array_equal(pixels[name], pixels['original'])
    assert metadata['natural_color']['natural']['output_pixels_sha256'] == hashlib.sha256(pixels['natural_base']).hexdigest()
