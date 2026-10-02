import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.inference import ROOT, load_bundle, predict_arrays, predict_dng
from src.image_io import decode_semantic_image
from src.preprocess import extract_inputs


def test_prediction_rejects_misaligned_or_nonfinite_inputs():
    image = np.zeros((1,3,224,224), dtype=np.float32)
    physical = np.zeros((1,132), dtype=np.float32)
    for bad_image, bad_physical in ((image.astype(np.float64),physical),
                                    (image,physical[:,:131]),
                                    (image,physical+np.nan)):
        with pytest.raises(ValueError):
            predict_arrays(None,bad_image,bad_physical)


@pytest.mark.parametrize('scope',['pilot','full'])
def test_real_raw_offline_online_and_onnx_batch_parity(scope):
    bundle = ROOT / ('artifacts/pilot_model' if scope=='pilot' else 'artifacts/model')
    if not (bundle/'model.json').exists():
        pytest.skip('Complete the documented training and export first')
    processed=ROOT/('data/processed/pilot' if scope=='pilot' else 'data/processed')
    with np.load(processed/'metadata.npz') as data:
        first_id, expected_physical = str(data['ids'][0]),data['X'][0]
    raw = next(path for path in (ROOT/'data/raw/dngs').glob('*') if path.name.lower()==first_id)
    physical,jpeg,_ = extract_inputs(raw)
    np.testing.assert_array_equal(physical,expected_physical)
    assert jpeg == (processed/'images'/(Path(first_id).stem+'.jpg')).read_bytes()
    image = decode_semantic_image(io.BytesIO(jpeg))
    session,_ = load_bundle(bundle)
    single = predict_arrays(session,image[None],physical[None])[0]
    batch = predict_arrays(session,np.stack([image,image]),np.stack([physical,physical]))
    np.testing.assert_allclose(batch,np.stack([single,single]),rtol=0,atol=1e-5)
    result,_ = predict_dng(raw,bundle)
    assert len(result['recommended_absolute'])+len(result['experimental_absolute'])==6
    if scope=='pilot': assert not result['recommended_absolute']
    assert {'Temperature','Tint'}.issubset(result['experimental_absolute'])
    # Prove the complete runtime path works with Torch and network imports disabled.
    script = """
import sys,socket,json
sys.modules['torch']=None
sys.modules['torchvision']=None
def no_network(*args,**kwargs): raise AssertionError('network access')
socket.create_connection=no_network
from src.inference import predict_dng
result,_=predict_dng(sys.argv[1],sys.argv[2])
assert sys.modules['torch'] is None and sys.modules['torchvision'] is None
print(json.dumps(result))
"""
    process = subprocess.run([sys.executable,'-c',script,str(raw),str(bundle)],cwd=ROOT,
                             capture_output=True,text=True,timeout=60,check=True)
    isolated = json.loads(process.stdout)
    for name,value in result['experimental_absolute'].items():
        assert isolated['experimental_absolute'][name]==pytest.approx(value,abs=1e-5)


def test_tampered_model_bundle_is_rejected(tmp_path):
    from src.inference import _session
    fake = tmp_path/'model.onnx'; fake.write_bytes(b'not an onnx model')
    with pytest.raises(ValueError,match='hash'):
        _session(str(fake),'incorrect',1)
