"""Ensure bounded search cannot silently reuse a differently configured run."""
import json

import pytest

from scripts.train_head_grid import CONTROL, check_run


def test_head_grid_requires_exact_configuration_and_preserved_artifacts(tmp_path):
    report = check_run(CONTROL, 42, .001, .01)
    assert report['evaluation_scope'] == 'validation_only'
    assert report['backbone_unchanged']
    config = json.loads((CONTROL / 'config.json').read_text())
    config['seed'] = 43
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='configuration mismatch'):
        check_run(tmp_path, 42, .001, .01)
