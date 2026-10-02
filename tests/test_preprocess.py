import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.preprocess import extract_132d_features, encode_semantic_image, process_image, run_preprocessing


def label():
    return {expert: dict(zip(('Exposure', 'Contrast', 'Saturation', 'Temperature', 'Tint', 'HighlightRecovery'),
                            [0., -2., 0., 15000., 0., 0.])) for expert in 'ABCDE'}


def fixture_data(tmp_path, names=('a.dng', 'b.dng'), filtered=None):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for name in names:
        (raw / name).write_bytes(b'fake raw')
    labels = tmp_path / 'expert_labels.json'
    labels.write_text(json.dumps({name: label() for name in names}))
    labels.with_name('label_audit.json').write_text(json.dumps({
        'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest(), 'filtered_images': filtered or {}}))
    return raw, labels, tmp_path / 'out'


def fake_inputs(_):
    rgb = np.full((12, 18, 3), [220, 70, 30], dtype=np.uint8)
    return np.arange(132, dtype=np.float32), encode_semantic_image(rgb), np.zeros(3, dtype=np.float32)


def test_histogram_mass_and_empty_brightness_regions():
    lab = np.full((6, 4, 3), [20., 250., -250.], dtype=np.float32)
    features = extract_132d_features(lab)
    np.testing.assert_allclose(features[:96].reshape(3, 32).sum(axis=1), 1)
    assert features[32 + 31] == 1 and features[64] == 1
    assert features[97] == 250 and features[98] == -250
    np.testing.assert_array_equal(features[-12:], 0)
    for bad in (lab[:2], np.full_like(lab, np.nan)):
        with pytest.raises(ValueError):
            extract_132d_features(bad)


def test_resuming_damage_and_source_change(tmp_path):
    raw, labels, out = fixture_data(tmp_path)
    with patch('src.preprocess.extract_inputs', side_effect=fake_inputs) as develop:
        first = run_preprocessing(raw, labels, out)
        assert first['success_count'] == 2 and first['cached_count'] == 0
        second = run_preprocessing(raw, labels, out)
        assert second['cached_count'] == 2 and develop.call_count == 2
        (out / 'images/a.jpg').write_bytes(b'broken image')
        third = run_preprocessing(raw, labels, out)
        assert third['cached_count'] == 1 and develop.call_count == 3
        (raw / 'b.dng').write_bytes(b'changed source')
        fourth = run_preprocessing(raw, labels, out)
        assert fourth['cached_count'] == 1 and develop.call_count == 4
    with np.load(out / 'metadata.npz', allow_pickle=False) as meta:
        assert meta['ids'].tolist() == ['a.dng', 'b.dng']
        assert meta['X'].dtype == np.float32 and meta['X'].shape == (2, 132)
        assert meta['Y'][0, 1] == -2 and meta['Y'][0, 3] == 15000


def test_filter_failures_and_zero_success(tmp_path):
    raw, labels, out = fixture_data(tmp_path, filtered={'a.dng': ['no absolute WB']})
    with patch('src.preprocess.extract_inputs', side_effect=OSError('decode failed')):
        report = run_preprocessing(raw, labels, out)
    assert report['status'] == 'failed'
    assert (report['input_count'], report['filtered_count'], report['failed_count']) == (2, 1, 1)
    assert not (out / 'metadata.npz').exists()
    assert 'decode failed' in report['failed_images']['b.dng']


def test_limit_is_repeatable_and_configuration_rejected(tmp_path):
    raw, labels, out = fixture_data(tmp_path, names=('b.dng', 'a.dng'))
    with patch('src.preprocess.extract_inputs', side_effect=fake_inputs):
        report = run_preprocessing(raw, labels, out, limit=1)
        assert report['scope'] == 'pilot' and report['input_count'] == 1
        with np.load(out / 'metadata.npz') as meta:
            assert meta['ids'].tolist() == ['a.dng']
        assert run_preprocessing(raw, labels, out, limit=1)['cached_count'] == 1
        config = json.loads((out / 'config.json').read_text())
        config['config_hash'] = 'stale'
        (out / 'config.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='configuration changed'):
            run_preprocessing(raw, labels, out, limit=1)
        assert run_preprocessing(raw, labels, out, limit=1, rebuild=True)['cached_count'] == 0
