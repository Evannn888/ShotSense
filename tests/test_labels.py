import json
from pathlib import Path
import numpy as np
import pytest
from src.extract_labels import parse_settings, extract_record, process_metadata, sha256_file
from src.parameters import PARAMS, validate_parameters


def settings():
    return dict(zip(PARAMS, [0., -10., 0., 15000., 0., 20.]), WhiteBalance='As Shot', Version='5.3')


def test_top_level_fields_do_not_match_nested_or_prefixed_keys():
    parsed = parse_settings('s = { IncrementalTemperature = -3, CustomTint = 20, Nested = { Saturation = 99, Temperature = 8000 }, Exposure = -1e-2, WhiteBalance = "As Shot" }')
    assert parsed['Exposure'] == -.01 and 'Temperature' not in parsed and 'Saturation' not in parsed
    assert parsed['Nested'] is None
    for malformed in ('s = { Exposure=1, Exposure=2 }', 's = { Nested={a=1 }', 's={Exposure=__import__("os")}', 's={Exposure=true'):
        with pytest.raises(ValueError):
            parse_settings(malformed)


def test_wb_mode_and_missing_values():
    s = settings(); s.update(CustomTemperature=5000., CustomTint=12.)
    values, _, errors = extract_record(s, {})
    assert not errors and values['Temperature'] == 15000 and values['Tint'] == 0
    s['WhiteBalance'] = 'Custom'
    values, _, errors = extract_record(s, {})
    assert not errors and values['Temperature'] == 5000 and values['Tint'] == 12
    del s['Temperature']; del s['CustomTemperature']; s['IncrementalTemperature'] = 3
    values, _, errors = extract_record(s, {})
    assert values['Temperature'] is None and errors
    s = settings(); del s['Saturation']
    assert extract_record(s, {})[2]
    values, sources, errors = extract_record(s, {'Saturation': {'value': 0.}})
    assert not errors and values['Saturation'] == 0 and 'verified_omitted' in sources['Saturation']


def test_ranges_and_version_require_evidence():
    validate_parameters(np.array([0., -50., -100., 50000., -150., 100.]))
    with pytest.raises(ValueError):
        validate_parameters([0, 0, 0, np.nan, 0, 0])
    assert process_metadata({'ProcessVersion':'5.0'})['family'] == 'PV2003'
    assert process_metadata({'Version':'5.3'})['effective'] == '5.0'
    with pytest.raises(ValueError):
        process_metadata({'Version':'8.0'})


def test_real_extraction_audit_matches_artifact():
    path = Path(__file__).resolve().parents[1] / 'data/intermediate/label_audit.json'
    if not path.exists():
        pytest.skip('Catalog audit has not been generated')
    audit = json.loads(path.read_text())
    assert audit['labels_sha256'] == sha256_file(path.with_name('expert_labels.json'))
    assert audit['counts']['images'] == 5000 and audit['counts']['valid_images'] == 4997
    assert set(audit['filtered_images']) == {'a3131-ke_.dng','a3214-ke_-8375.dng','a3741-ke_-8337.dng'}
    for rule in audit['default_rules'].values():
        assert rule['counts']['explicit'] > 0 and rule['counts']['omitted_zero'] > 0
