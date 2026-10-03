import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.image_io import decode_semantic_image
from src.inference import ROOT, load_bundle
from src.preprocess import encode_semantic_image, source_signature, sha256_file
from src.semantic_candidate import SEMANTIC_GEOMETRY, letterbox_rgb
from scripts.preprocess_letterbox import process_candidate


@pytest.mark.parametrize('shape', [(100,200),(200,100),(100,100)])
def test_letterbox_preserves_circle_and_full_frame(shape):
    h,w=shape
    rgb=np.full((h,w,3),10,dtype=np.uint8)
    cv2.circle(rgb,(w//2,h//2),20,(255,0,0),-1)
    rgb[:10,:10]=[0,255,0]; rgb[-10:,-10:]=[0,0,255]
    boxed=letterbox_rgb(rgb)
    red=np.argwhere((boxed[:,:,0]>200)&(boxed[:,:,1]<20))
    assert abs(np.ptp(red[:,0])-np.ptp(red[:,1]))<=1
    assert ((boxed[:,:,1]>200)&(boxed[:,:,0]<20)).any()
    assert ((boxed[:,:,2]>200)&(boxed[:,:,0]<20)).any()
    if h!=w: np.testing.assert_array_equal(boxed[0,0],SEMANTIC_GEOMETRY['padding_rgb'])
    jpeg=encode_semantic_image(boxed)
    assert decode_semantic_image(io.BytesIO(jpeg)).shape==(3,224,224)
    assert letterbox_rgb(rgb).tobytes()==boxed.tobytes()
    if h==w: assert jpeg==encode_semantic_image(rgb)


def test_candidate_cache_resume_damage_and_wrong_physical_branch(tmp_path,monkeypatch):
    raw=tmp_path/'sample.dng'; raw.write_bytes(b'synthetic raw')
    out=tmp_path/'out'; (out/'features').mkdir(parents=True); (out/'images').mkdir()
    x=np.arange(132,dtype=np.float32); y=np.array([0,0,0,5500,0,10],dtype=np.float32)
    jpeg=encode_semantic_image(letterbox_rgb(np.full((100,200,3),80,dtype=np.uint8)))
    calls=[]
    def inputs(path): calls.append(path); return x.copy(),jpeg,np.zeros(3,dtype=np.float32)
    monkeypatch.setattr('scripts.preprocess_letterbox.extract_candidate_inputs',inputs)
    job=(str(raw),str(out),'candidate-version',source_signature(raw),x,y,np.zeros(6,dtype=np.float32))
    assert process_candidate(job)['status']=='processed'
    assert process_candidate(job)['status']=='cached' and len(calls)==1
    (out/'images/sample.jpg').write_bytes(encode_semantic_image(np.full((224,224,3),80,dtype=np.uint8)))
    assert process_candidate(job)['status']=='processed' and len(calls)==2
    assert process_candidate(tuple(list(job[:2])+['old-version']+list(job[3:])))['status']=='processed'
    broken=tuple(list(job[:4])+[x+1]+list(job[5:]))
    assert 'physical features differ' in process_candidate(broken)['reason']
    raw.write_bytes(b'changed source')
    assert 'source changed' in process_candidate(job)['reason']


def test_bundle_rejects_unknown_geometry_and_validates_candidate_source(tmp_path,monkeypatch):
    bundle=json.loads((ROOT/'artifacts/model/model.json').read_text())
    bundle['pipeline_config']['semantic_geometry']={'version':'unknown'}
    (tmp_path/'model.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='geometry'): load_bundle(tmp_path)
    bundle['pipeline_config']['semantic_geometry']=SEMANTIC_GEOMETRY
    (tmp_path/'model.json').write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='implementation'): load_bundle(tmp_path)
    bundle['pipeline_config']['implementation_sha256']['semantic_candidate.py']=sha256_file(ROOT/'src/semantic_candidate.py')
    (tmp_path/'model.json').write_text(json.dumps(bundle))
    sentinel=object(); monkeypatch.setattr('src.inference._session',lambda *args:sentinel)
    assert load_bundle(tmp_path)[0] is sentinel
    for bad in (np.zeros((0,5,3),dtype=np.uint8),np.zeros((5,5,3),dtype=np.float32)):
        with pytest.raises(ValueError): letterbox_rgb(bad)
