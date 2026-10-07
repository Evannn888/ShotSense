import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from src import photo_engine as engine


def test_native_fine_tuning_zero_axes_replay_validation_and_worker(monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare pinned photo engine')
    rgb=np.full((96,180,3),128,dtype=np.uint8); original=rgb.copy()
    metadata,zero=engine.fine_tune_photo(rgb,{})
    assert not metadata['engine_applied'] and metadata['input_pixels_sha256']==metadata['output_pixels_sha256']
    np.testing.assert_array_equal(zero,rgb)
    outputs={}
    for name,value in [('warmth',10),('warmth',-10),('tint',10),('tint',-10)]:
        _,outputs[(name,value)]=engine.fine_tune_photo(rgb,{name:value})
    assert outputs[('warmth',10)][0,0,0]>outputs[('warmth',10)][0,0,2]
    assert outputs[('warmth',-10)][0,0,2]>outputs[('warmth',-10)][0,0,0]
    assert outputs[('tint',10)][0,0,0]>outputs[('tint',10)][0,0,1]
    assert outputs[('tint',-10)][0,0,1]>outputs[('tint',-10)][0,0,0]
    settings={'warmth':6,'tint':2,'vibrance':4,'local_contrast':6}
    metadata,expected=engine.fine_tune_photo(rgb,settings)
    from app.streamlit_app import run_fine_tuning
    worker,actual=run_fine_tuning(rgb,settings)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(rgb,original)
    assert worker['profile_sha256']==metadata['profile_sha256'] and worker['settings']==metadata['settings']
    assert worker['version']==engine.FINE_TUNING_VERSION and worker['size']==[180,96]
    monkeypatch.setattr(engine.subprocess,'run',lambda *a,**k: pytest.fail('Invalid or zero tuning must not run the engine'))
    for bad in ({'unknown':1},{'warmth':21},{'tint':float('nan')},{'vibrance':True},{'local_contrast':-1},{'warmth':1.5}):
        with pytest.raises(ValueError): engine.fine_tune_photo(rgb,bad)
    _,zero=engine.fine_tune_photo(rgb,{})
    np.testing.assert_array_equal(zero,rgb)


def test_app_tuning_reapplies_base_matches_exports_resets_and_preserves_success_on_failure(tmp_path,monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare pinned photo engine')
    rgb=np.zeros((360,540,3),dtype=np.uint8)
    rgb[:,:180]=[45,100,135];rgb[:,180:360]=[110,85,55];rgb[:,360:]=[85,85,85]
    path=tmp_path/'scene.jpg';Image.fromarray(rgb).save(path)
    result,source=engine.process_photo(path,'shadows');result['input_name']='scene.jpg'
    result['timing']['local_worker_seconds']=0
    import streamlit as st
    exports={};original_download=st.download_button
    def capture(label,data,*a,**k):
        exports[label]=data
        return original_download(label,data,*a,**k)
    monkeypatch.setattr(st,'download_button',capture)
    at=AppTest.from_file(str(engine.ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    at.selectbox[0].set_value('Lift shadows').run(timeout=30)
    at.session_state['prediction']=(result,source);at.run(timeout=30)
    assert not at.exception and 'photo_fine_result' not in at.session_state
    next(s for s in at.slider if s.label=='Warmth').set_value(10)
    next(s for s in at.slider if s.label=='Vibrance').set_value(4)
    next(b for b in at.button if b.label=='Apply fine tuning').click().run(timeout=30)
    assert not at.exception and not at.error
    identity,metadata,tuned=at.session_state['photo_fine_result']
    expected_metadata,expected=engine.fine_tune_photo(source['adjusted'],{'warmth':10,'vibrance':4})
    np.testing.assert_array_equal(tuned,expected)
    assert metadata['input_pixels_sha256']==hashlib.sha256(memoryview(source['adjusted'])).hexdigest()
    np.testing.assert_array_equal(at.session_state['prediction'][1]['adjusted'],source['adjusted'])
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    payload=json.loads(exports['Download adjustment JSON'])
    assert payload['fine_tuning']['settings']==metadata['settings']
    _,expected_png,_=engine.render_photo({'original':source['original'],'adjusted':expected})
    assert exports['Download full-resolution PNG']==expected_png
    assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected_png
    # Applying the same controls starts from the preserved recipe, not the tuned result.
    next(b for b in at.button if b.label=='Apply fine tuning').click().run(timeout=30)
    assert not at.exception and not at.get('download_button')
    np.testing.assert_array_equal(at.session_state['photo_fine_result'][2],expected)
    next(s for s in at.slider if s.label=='Adjustment strength').set_value(50).run(timeout=30)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    _,half,_=engine.render_photo({'original':source['original'],'adjusted':expected},.5)
    assert exports['Download full-resolution PNG']==half
    next(c for c in at.checkbox if c.label=='Inspect detail at original size').check().run(timeout=30)
    assert not at.exception and len(at.get('imgs'))==4
    crop={'original':source['original'][20:340,110:430],'adjusted':expected[20:340,110:430]}
    _,crop_png,_=engine.render_photo(crop,.5)
    assert base64.b64decode(at.get('imgs')[3].proto.imgs[0].url.split(',',1)[1])==crop_png
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'temporary-home')
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    saved=list((tmp_path/'temporary-home/Downloads').glob('*-finetuned-*.png'))
    assert len(saved)==1 and saved[0].read_bytes()==half and at.success
    # Failed application leaves the prior successful pixels and their actual settings visible.
    popen=subprocess.Popen
    def spawn(command,**kwargs):
        if '--fine-tune' in command: raise ValueError('controlled fine-tuning failure')
        return popen(command,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    next(s for s in at.slider if s.label=='Warmth').set_value(15)
    next(b for b in at.button if b.label=='Apply fine tuning').click().run(timeout=30)
    assert not at.exception and any('controlled fine-tuning failure' in e.value for e in at.error)
    np.testing.assert_array_equal(at.session_state['photo_fine_result'][2],expected)
    assert not at.get('download_button')
    next(b for b in at.button if b.label=='Reset fine tuning').click().run(timeout=30)
    assert 'photo_fine_result' not in at.session_state and not at.exception
    assert next(s for s in at.slider if s.label=='Warmth').value==0
    _,reset_png,_=engine.render_photo(source,.5)
    assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==reset_png
    at.radio[0].set_value('Manual exposure').run(timeout=30)
    assert not at.exception and 'prediction' not in at.session_state
