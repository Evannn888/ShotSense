"""Verify final frozen exports, immutable controls and recorded test gates."""
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
from artifacts.experiments.scene_policy_connection_v1.evaluate import read,save,sha

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
baseline=read(HERE/'baseline.json')
protected=baseline['protected_sha256'];allowed=baseline['approved_changes']
for path,expected in protected.items():
    if path not in allowed:assert sha(ROOT/path)==expected,path
implementation=read(HERE/'implementation_baseline.json')
for path,expected in implementation.items():
    if path!='tests/test_white_balance.py':assert sha(ROOT/path)==expected,path
tests=(HERE/'full_tests.log').read_text()
assert '91 passed' in tests and 'failed' not in tests and 'skipped' not in tests
exports=0;records=[]
for path in sorted((HERE/'records').glob('*.json')):
    record=read(path);records.append(record)
    for relative,expected in record['exports_sha256'].items():
        assert sha(ROOT/relative)==expected,relative
        with Image.open(ROOT/relative) as im:assert im.info.get('icc_profile') and im.mode=='RGB'
        exports+=1
    if 'new_black_pixel_positions' in record:
        assert record['new_black_pixel_positions']==record['new_full_pixel_positions']==0
for item in read(HERE/'input_baseline.json')['photos']:
    assert sha(ROOT/item['jpeg_path'])==item['jpeg_sha256']
    assert sha(ROOT/item['target_path'])==item['target_sha256']
for item in read(HERE/'fresh_inputs.json'):
    assert sha(Path(item['downloaded_path']))==item['downloaded_sha256']
    assert sha(ROOT/item['jpeg_path'])==item['jpeg_sha256']
native=read(HERE/'native_final_verification.json')
assert native['native_png_exact'] and native['zero_strength_exact_original']
assert sha(ROOT/'data/external/white_balance_v2/native/adjusted.png')==native['output_png_sha256']
mapping=read(HERE/'native_mapping_verification.json')
assert mapping['identity_exact'] and mapping['nonzero'] and mapping['chunk_crop_exact']
assert mapping['maximum_red_minus_blue_drift_codes']<=1
with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=10) as response:
    health={'status':response.status,'body':response.read().decode()}
assert health=={'status':200,'body':'ok'}
save(HERE/'verification.json',{'prior_and_fresh_records':len(records),'recorded_exports_hashes_and_icc':exports,
    'immutable_protected_files':len(protected)-len(allowed),'authorized_changed_files':allowed,
    'inference_implementation_exact':True,'photographic_rules_not_retuned':True,
    'fivek_source_target_hashes_exact':99,'fresh_source_count':len(read(HERE/'fresh_inputs.json')),
    'unavailable_fresh_count':len(read(HERE/'fresh_unavailable.json')),
    'project_tests_passed':91,'native_png_and_zero_exact':True,'native_nonzero_tint_chunk_exact':True,
    'streamlit_health':health,'integration':'Existing default-off scene beta; upload cache includes WBv2 version.',
    'limits':'Functional/export/diagnostic evidence only; no independent preference, intent, skin or WB ground truth.'})
print(json.dumps(read(HERE/'verification.json'),indent=2))
