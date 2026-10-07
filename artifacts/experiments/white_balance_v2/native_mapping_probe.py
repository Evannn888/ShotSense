"""20MP identity/nonzero tint/chunk control, not learned quality evidence."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from src.white_balance import apply_mapping

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
with np.load(ROOT/'data/external/white_balance_v1/native/pixels.npz',allow_pickle=False) as cache:
    rgb=cache['original']
coefficients=np.zeros((11,3),dtype=np.float32);coefficients[:3]=np.eye(3)
started=time.perf_counter()
identity,_=apply_mapping(rgb,coefficients,tint_only=True)
assert np.array_equal(identity,rgb)
coefficients[-1]=[-.03,.04,-.02]
output,guards=apply_mapping(rgb,coefficients,tint_only=True)
assert not np.array_equal(output,rgb)
maximum=0
for row in range(0,len(rgb),128):
    change=output[row:row+128].astype(np.int16)-rgb[row:row+128]
    maximum=max(maximum,int(np.abs(change[...,0]-change[...,2]).max()))
assert maximum<=1
for y,x in ((0,0),(1200,1500),(3400,5200)):
    crop,_=apply_mapping(rgb[y:y+64,x:x+96],coefficients,tint_only=True)
    assert np.array_equal(crop,output[y:y+64,x:x+96])
record={'size':[rgb.shape[1],rgb.shape[0]],'identity_exact':True,'nonzero':True,'chunk_crop_exact':True,
        'maximum_red_minus_blue_drift_codes':maximum,'guards':guards,
        'output_pixels_sha256':hashlib.sha256(output).hexdigest(),
        'seconds_including_identity_and_crop':time.perf_counter()-started,
        'scope':'Fixed affine functional control, not learned or photographic preference evidence.'}
with (HERE/'native_mapping_verification.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
