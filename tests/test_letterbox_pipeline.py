import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.image_io import decode_semantic_image
from src.inference import ROOT, predict_dng
from src.preprocess import extract_inputs
from src.semantic_candidate import extract_candidate_inputs, validate_candidate_cache


def test_real_candidate_physical_and_cached_online_parity(monkeypatch):
    pilot=ROOT/'data/processed/model_vnext/letterbox-pilot'
    if not (pilot/'manifest.json').exists(): pytest.skip('Prepare the candidate pilot first')
    dataset=validate_candidate_cache(pilot)
    photo=str(dataset.ids[0])
    raw=next(p for p in (ROOT/'data/raw/dngs').iterdir() if p.name.lower()==photo)
    physical,jpeg,_=extract_candidate_inputs(raw)
    original,old_jpeg,_=extract_inputs(raw)
    np.testing.assert_array_equal(physical,original)
    np.testing.assert_array_equal(physical,dataset.X[0])
    assert jpeg==(pilot/'images'/(Path(photo).stem+'.jpg')).read_bytes()
    assert jpeg!=old_jpeg
    np.testing.assert_array_equal(decode_semantic_image(io.BytesIO(jpeg)),dataset[0][0].numpy())
    config=json.loads((ROOT/'artifacts/model/model.json').read_text())
    config['pipeline_config']['semantic_geometry']=json.loads((pilot/'manifest.json').read_text())['config']['semantic_geometry']
    config['validated_parameters']=[]
    monkeypatch.setattr('src.inference.load_bundle',lambda _: (object(),config))
    def capture(session,image,features):
        np.testing.assert_array_equal(image[0],dataset[0][0].numpy())
        np.testing.assert_array_equal(features[0],dataset.X[0])
        return np.zeros((1,6),dtype=np.float32)
    monkeypatch.setattr('src.inference.predict_arrays',capture)
    result,online_jpeg=predict_dng(raw)
    assert online_jpeg==jpeg and not result['recommended_absolute']


def test_candidate_validation_rejects_mixed_image_and_geometry(tmp_path):
    pilot=ROOT/'data/processed/model_vnext/letterbox-pilot'
    if not (pilot/'manifest.json').exists(): pytest.skip('Prepare the candidate pilot first')
    destination=tmp_path/'candidate'; shutil.copytree(pilot,destination)
    dataset=validate_candidate_cache(destination)
    image=destination/'images'/(Path(dataset.ids[0]).stem+'.jpg')
    from src.preprocess import encode_semantic_image
    image.write_bytes(encode_semantic_image(np.zeros((224,224,3),dtype=np.uint8)))
    with pytest.raises(ValueError,match='Mixed or damaged'): validate_candidate_cache(destination)
    manifest=json.loads((destination/'manifest.json').read_text())
    manifest['config']['semantic_geometry']['version']='stretch'
    (destination/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='geometry'): validate_candidate_cache(destination)
