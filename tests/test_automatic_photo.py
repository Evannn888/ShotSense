import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import adaptive_lut
from src.photo_engine import ROOT,render_photo
from src.scene_policy import VERSION as SCENE_VERSION
from src.white_balance import VERSION as WB_VERSION,PIPELINE_VERSION as WB_PIPELINE_VERSION


def test_upload_automatically_processes_once_exports_current_photo_and_recovers(tmp_path,monkeypatch):
    if not (adaptive_lut.BUNDLE/'manifest.json').exists():pytest.skip('Prepare pinned adaptive LUT')
    current=[None];calls=[];exports={}
    class Upload:
        name='first.jpg'
        def __init__(self,rgb):
            buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=95)
            self.data=buffer.getvalue()
        def getvalue(self):return self.data
    first=Upload(np.full((128,192,3),[38,55,74],dtype=np.uint8))
    popen=subprocess.Popen;download=st.download_button
    def spawn(command,**kwargs):
        if 'src.lut_natural' in command:calls.append(command)
        return popen(command,**kwargs)
    def capture(label,data,*args,**kwargs):
        exports[label]=data
        return download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'file_uploader',lambda *args,**kwargs:current[0])
    monkeypatch.setattr(st,'download_button',capture)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'temporary-home')
    at=AppTest.from_file(str(ROOT/'app/streamlit_app.py'))
    at.session_state['photo_workflow_v1']='Manual exposure (recommended)'
    at.session_state['prediction']=('old result','old pixels')
    at.run(timeout=30)
    assert not at.exception and not calls and at.radio[0].value=='Automatic (recommended)'
    assert 'prediction' not in at.session_state
    assert not at.get('imgs') and not at.slider
    assert next(e for e in at.expander if e.label=='Advanced options').proto.expanded is False
    current[0]=first;at.run(timeout=30)  # A new upload alone triggers the real worker.
    assert not at.exception and not at.error and len(calls)==1
    result,pixels=at.session_state['prediction']
    assert result['automatic_workflow']['backend']=='protected-natural-color'
    assert result['input_sha256']==hashlib.sha256(first.data).hexdigest()
    assert result['output_size']==[192,128]
    assert next(e for e in at.expander if e.label=='Optional adjustments').proto.expanded is False
    assert next(s for s in at.slider if s.label=='Adjustment strength').value==100
    assert next(s for s in at.slider if s.label=='Shadow lift').value==result['shadow_adjustment']['strength']*100
    _,expected,_=render_photo(pixels)
    assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected
    assert not (tmp_path/'temporary-home/Downloads').exists()
    at.run(timeout=30)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    assert len(calls)==1 and exports['Download full-resolution PNG']==expected
    metadata=json.loads(exports['Download adjustment JSON'])
    assert metadata['automatic_workflow']==result['automatic_workflow']
    assert metadata['input_sha256']==result['input_sha256']
    assert metadata['export']['output_png_sha256']==hashlib.sha256(expected).hexdigest()
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    files=list((tmp_path/'temporary-home/Downloads').glob('*.png'))
    assert len(files)==1 and files[0].read_bytes()==expected and len(calls)==1
    next(s for s in at.slider if s.label=='Adjustment strength').set_value(50).run(timeout=30)
    assert len(calls)==1 and not at.get('download_button')

    second=Upload(np.full((120,180,3),[165,172,180],dtype=np.uint8));second.name='second.jpg'
    current[0]=second;at.run(timeout=30)
    assert not at.exception and not at.error and len(calls)==2
    result2,pixels2=at.session_state['prediction']
    assert result2['input_name']=='second.jpg' and result2['input_sha256']==hashlib.sha256(second.data).hexdigest()
    assert next(s for s in at.slider if s.label=='Adjustment strength').value==100
    assert next(s for s in at.slider if s.label=='Shadow lift').value==result2['shadow_adjustment']['strength']*100
    _,expected2,_=render_photo(pixels2)
    assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected2
    assert not at.get('download_button') and not at.success
    second.data=b'invalid JPEG'
    at.run(timeout=30)
    assert not at.exception and at.error and len(calls)==3
    assert 'prediction' not in at.session_state and not at.get('imgs') and not at.get('download_button')
    at.run(timeout=30)
    assert at.error and len(calls)==3  # A failed upload does not cause an infinite retry loop.
    next(b for b in at.button if b.label=='Process image').click().run(timeout=30)
    assert not at.exception and at.error and len(calls)==4
    current[0]=first;at.run(timeout=30)
    assert not at.exception and not at.error and len(calls)==5
    current[0]=None;at.run(timeout=30)
    assert not at.exception and not at.error and not at.get('imgs') and 'prediction' not in at.session_state
    assert 'automatic_attempt' not in at.session_state and 'automatic_error' not in at.session_state

    current[0]=first;at.run(timeout=30)
    assert len(calls)==6
    next(c for c in at.checkbox if c.label=='Use scene-aware adjustment (experimental)').check().run(timeout=30)
    assert not at.exception and not at.error and len(calls)==7
    scene_result,scene_pixels=at.session_state['prediction']
    assert '--scene-aware' in calls[-1]
    assert '--white-balance' in calls[-1]
    assert scene_result['automatic_workflow']['backend']=='scene-aware-protected-color'
    assert scene_result['scene_policy']['version']==SCENE_VERSION
    assert scene_result['processing_version']==WB_PIPELINE_VERSION
    _,scene_png,_=render_photo(scene_pixels)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    assert len(calls)==7 and exports['Download full-resolution PNG']==scene_png
    scene_json=json.loads(exports['Download adjustment JSON'])
    assert scene_json['scene_policy']==scene_result['scene_policy']
    assert scene_json['white_balance']==scene_result['white_balance']
    # An older policy result for the same upload must be regenerated exactly once.
    identity=at.session_state['automatic_attempt']
    assert identity[-1]==SCENE_VERSION+':'+WB_VERSION
    at.session_state['automatic_attempt']=identity[:-1]+('scene-aware-natural-v2',)
    at.run(timeout=30)
    assert not at.exception and not at.error and len(calls)==8
    assert at.session_state['automatic_attempt'][-1]==SCENE_VERSION+':'+WB_VERSION
    at.run(timeout=30)
    assert len(calls)==8
    next(s for s in at.slider if s.label=='Adjustment strength').set_value(50).run(timeout=30)
    assert len(calls)==8 and not at.get('download_button')
    next(c for c in at.checkbox if c.label=='Use scene-aware adjustment (experimental)').uncheck().run(timeout=30)
    assert len(calls)==9 and 'scene_policy' not in at.session_state['prediction'][0]
    np.testing.assert_array_equal(at.session_state['prediction'][1]['adjusted'],pixels['adjusted'])
    at.radio[0].set_value('Manual exposure').run(timeout=30)
    assert len(calls)==9 and 'prediction' not in at.session_state and not at.get('imgs')
    at.run(timeout=30)
    assert len(calls)==9  # The retained manual workflow still requires an explicit process click.


def test_automatic_dng_dispatch_keeps_existing_conservative_worker_contract(monkeypatch):
    current=[None];commands=[]
    class Upload:
        name='input.dng'
        def getvalue(self):return b'controlled invalid DNG'
    popen=subprocess.Popen
    def spawn(command,**kwargs):
        if 'src.photo_engine' in command:commands.append(command)
        return popen(command,**kwargs)
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:current[0])
    monkeypatch.setattr(subprocess,'Popen',spawn)
    at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
    current[0]=Upload();at.run(timeout=30)
    assert not at.exception and at.error and len(commands)==1
    assert commands[0][commands[0].index('--recipe')+1]=='auto'
    assert 'prediction' not in at.session_state and not at.get('imgs')
    at.run(timeout=30)
    assert len(commands)==1 and at.error
