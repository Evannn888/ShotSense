import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.dataset import ParameterNormalizer
from src.model import ShotSenseModel
from src.preprocess import sha256_file
from src.train import final_evaluate, metrics, train
from scripts.audit_photo_groups import candidate_pairs


def test_validation_only_fixed_split_and_explicit_final_access(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    ids = np.array(['synthetic-%02d.dng' % i for i in range(20)])
    metadata = tmp_path / 'metadata.npz'
    y = np.tile([.5, 4., 2., 5500., 1., 15.], (20, 1)).astype(np.float32)
    y[:, 0] += rng.uniform(-.3, .3, 20)
    y[:, 5] += rng.uniform(-5, 5, 20)
    np.savez(metadata, X=rng.normal(size=(20, 132)).astype(np.float32), Y=y,
             Y_std=np.zeros_like(y), ids=ids, config_hash=np.array('synthetic-v1'))
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'scope': 'full', 'config_hash': 'synthetic-v1',
        'metadata_sha256': sha256_file(metadata), 'config': {'fixture': True}}))
    mapping = {photo: 'group-%02d' % (i // 2) for i, photo in enumerate(ids)}
    groups_path = tmp_path / 'groups.json'
    groups_path.write_text(json.dumps(mapping))
    split_path = tmp_path / 'grouped-split.json'
    monkeypatch.setattr('src.train.ShotSenseModel',
                        lambda *args, **kwargs: ShotSenseModel(*args, pretrained=False))
    embeddings = rng.normal(size=(20, 1280)).astype(np.float32)
    monkeypatch.setattr('src.train.semantic_features', lambda *args, **kwargs: embeddings)
    partitions = []
    def diagnostics(dataset, predictions, groups, train_ids, metric):
        partitions.append(set(groups))
        for prediction in predictions.values():
            selected = np.concatenate(list(groups.values()))
            assert np.isfinite(prediction[selected]).all()
            others = np.setdiff1d(np.arange(20), selected)
            assert np.isnan(prediction[others]).all()
        return {'requested_partitions': sorted(groups)}
    monkeypatch.setattr('src.train.diagnostics', diagnostics)
    reports = []
    for seed in (42, 43):
        reports.append(train(metadata, tmp_path / ('seed-%d' % seed), epochs=1,
                             seed=seed, split_path=split_path, groups_path=groups_path))
    assert partitions == [{'val'}, {'val'}]
    assert all('test' not in report and report['validated_parameters'] == [] for report in reports)
    assert reports[0]['config']['split_seed'] == reports[1]['config']['split_seed'] == 42
    assert reports[0]['config']['split_sha256'] == reports[1]['config']['split_sha256']
    split = json.loads(split_path.read_text())['splits']
    membership = {photo: name for name, photos in split.items() for photo in photos}
    assert all(membership[ids[i]] == membership[ids[i+1]] for i in range(0, 20, 2))
    run = tmp_path / 'seed-42'
    with np.load(run / 'validation_predictions.npz', allow_pickle=False) as values:
        assert values['ids'].tolist() == split['val']
    before = sha256_file(run / 'best.pt')
    with pytest.raises(FileExistsError): train(metadata, run, epochs=1)
    assert sha256_file(run / 'best.pt') == before
    final = final_evaluate(metadata, run)
    assert partitions[-1] == {'test'} and 'test' in final
    assert json.loads((run / 'final_test_access.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError): final_evaluate(metadata, run)
    assert partitions == [{'val'}, {'val'}, {'test'}]
    altered = tmp_path / 'seed-43'
    with (altered / 'controls.pt').open('ab') as handle: handle.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'): final_evaluate(metadata, altered)
    assert not (altered / 'final_test_access.json').exists()


def test_p2_units_and_failed_run_record(tmp_path):
    truth = np.tile([0., 4., 2., 5500., 1., 15.], (2, 1))
    prediction = truth.copy(); prediction[:, 0] += 1; prediction[:, 5] += 10
    result = metrics(ParameterNormalizer.normalize(prediction), ParameterNormalizer.normalize(truth))
    assert result['p2'] == pytest.approx(.5 * (1 / 8 + 10 / 100))
    run = tmp_path / 'failed'
    with pytest.raises(ValueError, match='Invalid epochs'):
        train(tmp_path / 'missing.npz', run, epochs=0)
    status = json.loads((run / 'run_status.json').read_text())
    assert status['status'] == 'failed' and status['error_type'] == 'ValueError'


def test_group_audit_proposes_pairs_without_merging():
    difference = np.array([[0]*8, [0]*8, [255]*8], dtype=np.uint8)
    perceptual = difference.copy()
    moment = datetime(2026, 10, 2)
    capture = [('camera', 1, moment), None, ('camera', 1, moment+timedelta(seconds=2))]
    pairs = candidate_pairs(difference, perceptual, capture)
    assert set(pairs) == {(0, 1), (0, 2)}
    assert pairs[0, 1]['reasons'] == ['visual_hash']
    assert pairs[0, 2]['reasons'] == ['same_camera_capture_within_3_seconds']
    assert pairs[0, 2]['dhash_distance'] == 64
