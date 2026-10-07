import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from src import scene_policy, white_balance as wb
from src.lut_natural import process_jpeg
from tests.test_scene_policy import evidence


def test_wb_identity_color_bounds_skin_proxy_luminance_and_headroom():
    rgb = np.random.default_rng(81).integers(0,256,(145,96,3),dtype=np.uint8)
    rgb[0,0]=0;rgb[0,1]=255;rgb[0,2]=[252,15,42];rgb[0,3]=[178,115,82]
    identity = np.zeros((11,3));identity[:3]=np.eye(3)
    current,guards = wb.apply_mapping(rgb,identity)
    np.testing.assert_array_equal(current,rgb)
    cast = identity.copy();cast[-1]=[-.2,.05,.2]
    current,guards = wb.apply_mapping(rgb,cast)
    assert guards['maximum_luma_change_codes']<=.501
    assert guards['maximum_channel_change_codes']<=10.501
    assert guards['new_black_pixel_positions']==guards['new_full_pixel_positions']==0
    assert wb.skin_color_weight(rgb[0,3].astype(np.float32)[None]/255)[0]>.99
    assert np.abs(current[0,3].astype(float)-rgb[0,3]).max()<=6.5
    np.testing.assert_array_equal(current[0,0],rgb[0,0])
    tint,guards = wb.apply_mapping(rgb,cast,tint_only=True)
    difference=tint.astype(int)-rgb.astype(int)
    assert np.abs(difference[...,0]-difference[...,2]).max()<=1
    assert guards['tint_only'] and guards['maximum_luma_change_codes']<=.501
    with pytest.raises(ValueError):wb.apply_mapping(rgb,np.full((11,3),np.nan))


def test_neutral_support_rejects_identity_and_spatially_conflicting_lighting():
    neutral = np.full((64,96,3),.5,dtype=np.float32)
    assert wb.neutral_evidence(neutral,neutral)[0]=='no_supported_neutral_improvement'
    source = np.full((64,96,3),[.45,.55,.45],dtype=np.float32)
    corrected = np.full_like(source,.5)
    assert wb.neutral_evidence(source,corrected)[0] is None
    corrected[:32,:48]=[.4,.6,.4]
    assert wb.neutral_evidence(source,corrected)[0]=='spatial_neutral_disagreement'
    colorful = np.full_like(source,[.8,.1,.2])
    assert wb.neutral_evidence(colorful,corrected)[0]=='insufficient_neutral_candidates'


def test_mood_and_scene_failure_skip_wb_before_loading_assets(monkeypatch):
    monkeypatch.setattr(wb,'load_wb_model',lambda:pytest.fail('Mood skip must not infer WB'))
    rgb = np.full((32,48,3),50,dtype=np.uint8)
    scene=evidence('night');policy=scene_policy.scene_limits(rgb,scene)
    meta,current=wb.correct_white_balance(rgb,scene,policy)
    assert meta['reason']=='two_view_dark_night_retained' and not meta['applied']
    np.testing.assert_array_equal(current,rgb)
    missing={'status':'unavailable'}
    meta,current=wb.correct_white_balance(rgb,missing,scene_policy.scene_limits(rgb,missing))
    assert meta['reason']=='recognition_unavailable'
    np.testing.assert_array_equal(current,rgb)


def test_warm_color_gate_and_joint_night_evidence_do_not_equate_similarity_with_night(monkeypatch):
    sample=np.full((64,96,3),[.60,.53,.50],dtype=np.float32)
    assert wb.warm_evidence(sample,evidence('sunset_sunrise'))['protected']
    assert wb.warm_evidence(sample,evidence('indoor_artificial'))['protected']
    assert not wb.warm_evidence(sample,evidence('daylight'))['protected']
    assert not wb.warm_evidence(sample[...,::-1],evidence('sunset_sunrise'))['protected']
    calls=[]
    def missing():calls.append(1);raise FileNotFoundError('Controlled absent WB')
    monkeypatch.setattr(wb,'load_wb_model',missing)
    for rgb,scene in [(np.full((64,96,3),50,dtype=np.uint8),evidence('night','daylight')),
                      (np.full((64,96,3),100,dtype=np.uint8),evidence('night'))]:
        meta,current=wb.correct_white_balance(rgb,scene,scene_policy.scene_limits(rgb,scene))
        assert meta['status']=='fallback' and not meta['applied']
        np.testing.assert_array_equal(current,rgb)
    assert len(calls)==2


def test_supported_model_prediction_is_bounded_and_corrupt_assets_recover(tmp_path,monkeypatch):
    rng=np.random.default_rng(8)
    gray=rng.uniform(.3,.7,(96,128,1))
    source=(np.clip(gray+rng.uniform(-.04,.04,(96,128,3))+[.01,.045,-.035],0,1)*255).astype(np.uint8)
    class Model:
        def run(self,_,feed):
            a=feed['image'];y=np.einsum('nchw,c->nhw',a,wb.LUMA)[:,None]
            return [.3*a+.7*y]
    monkeypatch.setattr(wb,'load_wb_model',lambda:Model())
    scene=evidence('daylight');policy=scene_policy.scene_limits(source,scene)
    meta,current=wb.correct_white_balance(source,scene,policy)
    assert meta['applied'] and meta['guards']['maximum_luma_change_codes']<=.501
    assert current.shape==source.shape and not np.array_equal(current,source)
    assert meta['output_pixels_sha256']==hashlib.sha256(current).hexdigest()
    assert wb.wb_preview(source).shape==(288,384,3)
    monkeypatch.undo()
    wb.load_wb_model.cache_clear();monkeypatch.setattr(wb,'BUNDLE',tmp_path/'absent')
    try:meta,current=wb.correct_white_balance(source,scene,policy)
    finally:wb.load_wb_model.cache_clear()
    assert meta['status']=='fallback' and not meta['applied']
    np.testing.assert_array_equal(current,source)


def test_warm_picture_uses_tint_only_and_checks_actual_bounded_pixels(monkeypatch):
    rng=np.random.default_rng(88)
    gray=rng.uniform(.3,.7,(96,128,1))
    rgb=(np.clip(gray+rng.uniform(-.015,.015,(96,128,3))+[.05,-.04,.02],0,1)*255).astype(np.uint8)
    class Model:
        def run(self,_,feed):
            a=feed['image'];y=np.einsum('nchw,c->nhw',a,wb.LUMA)[:,None]
            return [.3*a+.7*y]
    monkeypatch.setattr(wb,'load_wb_model',lambda:Model())
    scene=evidence('sunset_sunrise');policy=scene_policy.scene_limits(rgb,scene)
    meta,current=wb.correct_white_balance(rgb,scene,policy)
    assert meta['applied'] and meta['mode']=='tint_only' and meta['warm_evidence']['protected']
    assert [a['mode'] for a in meta['candidate_attempts']]==['tint_only']
    change=current.astype(int)-rgb.astype(int)
    assert np.abs(change[...,0]-change[...,2]).max()<=1
    sample=wb.wb_preview(rgb)
    rendered,_=wb.apply_mapping(np.rint(sample*255).astype(np.uint8),
                               np.array(meta['coefficients'],dtype=np.float32),tint_only=True)
    reason,expected=wb.neutral_evidence(sample,rendered.astype(np.float32)/255)
    assert reason is None and expected==meta['neutral_evidence']


def test_actual_worker_missing_wb_assets_preserves_scene_v2(tmp_path,monkeypatch):
    jpeg=tmp_path/'image.jpg'
    Image.fromarray(np.random.default_rng(83).integers(30,190,(96,144,3),dtype=np.uint8)).save(jpeg)
    monkeypatch.setattr(scene_policy,'analyze_scene',lambda _:evidence('daylight'))
    _,prior=process_jpeg(jpeg,scene_aware=True)
    wb.load_wb_model.cache_clear();monkeypatch.setattr(wb,'BUNDLE',tmp_path/'absent')
    try:meta,current=process_jpeg(jpeg,scene_aware=True,white_balance=True)
    finally:wb.load_wb_model.cache_clear()
    assert meta['processing_version']==wb.PIPELINE_VERSION and meta['white_balance']['status']=='fallback'
    for name in current:np.testing.assert_array_equal(current[name],prior[name])
    with pytest.raises(ValueError):process_jpeg(jpeg,white_balance=True)


def test_actual_automatic_ui_applies_wb_exports_same_pixels_and_zero_is_original(tmp_path,monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    # WBv2 retains warm dining-room temperature; use a still-applied fixture for
    # the positive export contract. The original dining case stays in the audit.
    source=root/'data/external/fivek_current_effect_v1/input_jpeg/a3442-mb_060909_003.jpg'
    expected=root/'data/external/white_balance_v2/automatic/a3442-mb_060909_003.png'
    if not source.exists() or not expected.exists():pytest.skip('Local WB diagnostic photo not prepared')
    current=[None];calls=[];exports={}
    class Upload:
        name='child.jpg'
        def getvalue(self):return source.read_bytes()
    popen=subprocess.Popen;download=st.download_button
    def spawn(command,**kwargs):
        if 'src.lut_natural' in command:calls.append(command)
        return popen(command,**kwargs)
    def capture(label,data,*args,**kwargs):
        exports[label]=data
        return download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:current[0])
    monkeypatch.setattr(st,'download_button',capture)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    at=AppTest.from_file(str(root/'app/streamlit_app.py')).run(timeout=30)
    next(c for c in at.checkbox if c.label=='Use scene-aware adjustment (experimental)').check().run(timeout=30)
    current[0]=Upload();at.run(timeout=30)
    assert not at.exception and not at.error and len(calls)==1
    result,pixels=at.session_state['prediction']
    assert result['white_balance']['applied']
    assert result['scene_policy']['white_balance']['status']=='applied'
    with Image.open(expected) as im:np.testing.assert_array_equal(pixels['adjusted'],np.array(im))
    assert any('gentle color-cast' in c.value for c in at.caption)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    meta=json.loads(exports['Download adjustment JSON'])
    assert meta['white_balance']==result['white_balance']
    assert meta['export']['output_png_sha256']==hashlib.sha256(exports['Download full-resolution PNG']).hexdigest()
    with Image.open(io.BytesIO(exports['Download full-resolution PNG'])) as im:
        np.testing.assert_array_equal(np.array(im),pixels['adjusted'])
    next(s for s in at.slider if s.label=='Adjustment strength').set_value(0).run(timeout=30)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    with Image.open(io.BytesIO(exports['Download full-resolution PNG'])) as im:
        np.testing.assert_array_equal(np.array(im),pixels['original'])
    assert len(calls)==1
    identity=at.session_state['automatic_attempt']
    at.session_state['automatic_attempt']=identity[:-1]+(scene_policy.VERSION+':bounded-deepwb-v1',)
    at.run(timeout=30)
    assert not at.exception and not at.error and len(calls)==2
    assert at.session_state['automatic_attempt'][-1]==scene_policy.VERSION+':'+wb.VERSION
    at.run(timeout=30)
    assert len(calls)==2
