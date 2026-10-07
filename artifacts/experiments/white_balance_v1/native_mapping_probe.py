"""Reproduce the recorded native-size fixed-affine functional control, not preference."""
import hashlib
import json
from pathlib import Path

import numpy as np
from src.white_balance import apply_mapping

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
with np.load(ROOT/'data/external/white_balance_v1/native/pixels.npz',allow_pickle=False) as cache:
    rgb=cache['original']
identity=np.zeros((11,3),dtype=np.float32);identity[:3]=np.eye(3)
output,_=apply_mapping(rgb,identity)
assert np.array_equal(output,rgb)
identity[-1]=[-.03,.04,-.02]
output,guards=apply_mapping(rgb,identity)
for y,x in ((0,0),(1200,1500),(3400,5200)):
    crop,_=apply_mapping(rgb[y:y+64,x:x+96],identity)
    assert np.array_equal(crop,output[y:y+64,x:x+96])
record=json.loads((HERE/'native_mapping_verification.json').read_text())
assert hashlib.sha256(output).hexdigest()==record['output_pixels_sha256']
assert guards==record['guards']
print('Recorded20MPidentity/nonzero/chunk/crop mapping checks reproduced')
