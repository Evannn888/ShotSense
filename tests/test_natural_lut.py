import io
import numpy as np
import pytest
from PIL import Image

from src.lut_natural import LUMA,natural_settings,protect_rgb,shadow_settings,shadow_curve,apply_shadow_lift
from src.photo_engine import render_photo


def test_protection_preserves_neutral_tone_order_dark_detail_black_white_and_gamut():
    ramp = np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    # A deliberately destructive LUT is dark, tinted, nonmonotone and clipped.
    raw = np.stack([np.arange(255,-1,-1),np.full(256,255),np.zeros(256)],axis=-1)[None].astype(np.uint8)
    metadata,result = protect_rgb(ramp,raw)
    np.testing.assert_array_equal(result[:,:,0],result[:,:,1])
    np.testing.assert_array_equal(result[:,:,1],result[:,:,2])
    assert np.all(np.diff(result[0,:,0].astype(np.int16))>=0)
    np.testing.assert_array_equal(result[0,[0,255]],ramp[0,[0,255]])
    assert np.all(result>=ramp) and metadata['new_black_pixel_fraction']==metadata['new_full_channel_fraction']==0
    rgb = np.random.default_rng(12).integers(0,256,(129,321,3),dtype=np.uint8)
    rgb[0,:6] = [[255,0,0],[0,255,0],[0,0,255],[254,254,254],[1,0,0],[0,0,0]]
    saved = rgb.copy(); adversarial = 255-rgb
    metadata,result = protect_rgb(rgb,adversarial)
    assert result.shape==rgb.shape and result.dtype==np.uint8
    np.testing.assert_array_equal(rgb,saved)
    assert metadata['minimum_shadow_luminance_delta_codes']>=-.501
    assert not np.any(np.all(result==0,axis=-1)&~np.all(rgb==0,axis=-1))
    assert not np.any(np.any(result==255,axis=-1)&~np.any(rgb==255,axis=-1))


def test_policy_adapts_to_color_change_and_exports_native_identity_and_blend():
    rng = np.random.default_rng(24); rgb = rng.integers(10,240,(149,293,3),dtype=np.uint8)
    mild = rgb.copy(); strong = rgb.copy(); strong[:,:,0]=255;strong[:,:,1]=0
    assert natural_settings(rgb,strong)['color_strength']<natural_settings(rgb,mild)['color_strength']<=.35
    metadata,result = protect_rgb(rgb,strong)
    before,after,export = render_photo({'original':rgb,'adjusted':result},.5)
    with Image.open(io.BytesIO(after)) as image:
        assert image.size==(293,149) and 'srgb' in image.info
        np.testing.assert_array_equal(np.asarray(image),np.rint(rgb.astype(np.float32)*.5+result.astype(np.float32)*.5).astype(np.uint8))
    _,zero,identity = render_photo({'original':rgb,'adjusted':result},0)
    assert zero==before and identity['changed_pixel_fraction']==0
    with pytest.raises(ValueError): protect_rgb(rgb,strong[:,:20])
    with pytest.raises(ValueError): protect_rgb(rgb.astype(np.float32),strong)


def test_shadow_curve_dark_layering_neutral_order_highlight_identity_hue_and_no_accumulation():
    continuous=np.linspace(0,1,65537)
    ramp=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    for strength in (0,.5,1):
        curve=shadow_curve(continuous,strength)
        assert np.all(np.diff(curve)>0) and np.all(curve>=continuous)
        np.testing.assert_array_equal(curve[continuous>=.55],continuous[continuous>=.55])
        metadata,adjusted=apply_shadow_lift(ramp,strength)
        assert np.all(np.diff(adjusted[0,:,0].astype(np.int16))>=0)
        np.testing.assert_array_equal(adjusted[0,:2],ramp[0,:2])
        np.testing.assert_array_equal(adjusted[0,141:],ramp[0,141:])
        assert np.all(adjusted==adjusted[:,:,:1]) and np.all(adjusted>=ramp)
        assert metadata['new_black_pixel_fraction']==metadata['new_full_channel_fraction']==0
        if strength: assert adjusted[0,40,0]>=ramp[0,40,0]+10*strength
    rgb=np.random.default_rng(17).integers(0,256,(129,217,3),dtype=np.uint8);saved=rgb.copy()
    _,a=apply_shadow_lift(rgb,.75);_,b=apply_shadow_lift(rgb,.75)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(rgb,saved)
    assert np.all(a>=rgb) and not np.any(np.any(a==255,axis=-1)&~np.any(rgb==255,axis=-1))
    source=rgb.astype(np.int32);result=a.astype(np.int32)
    # Common RGB scaling preserves ratios up to the independent half-code rounding.
    assert np.all(np.abs(result[:,:,0]*source[:,:,1]-result[:,:,1]*source[:,:,0])<=(source[:,:,0]+source[:,:,1])*.501)
    _,zero=apply_shadow_lift(rgb,0);np.testing.assert_array_equal(zero,rgb)
    dim=shadow_settings(np.full((40,60,3),45,dtype=np.uint8))
    bright=shadow_settings(np.full((40,60,3),190,dtype=np.uint8))
    black=shadow_settings(np.zeros((40,60,3),dtype=np.uint8))
    assert 0<dim['suggested_strength']<=.75 and bright['suggested_strength']==black['suggested_strength']==0
    for bad in (-.01,1.01,float('nan'),True,'0.5'):
        with pytest.raises(ValueError):apply_shadow_lift(rgb,bad)


def test_natural_color_real_button_worker_png_json_blend_detail_save_and_failure(tmp_path,monkeypatch):
    import base64
    import hashlib
    import json
    from pathlib import Path
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src import adaptive_lut
    from src.photo_engine import ROOT
    from src.lut_natural import process_jpeg
    if not (adaptive_lut.BUNDLE/'manifest.json').exists(): pytest.skip('Prepare pinned adaptive LUT')
    rgb=np.full((360,540,3),[50,75,100],dtype=np.uint8);rgb[:140]=[170,175,180]
    rgb[200:320,50:250]=[8,10,6]
    path=tmp_path/'color.jpg';Image.fromarray(rgb).save(path,quality=95)
    class Upload:
        name='color.jpg'
        data=path.read_bytes()
        def getvalue(self): return self.data
    upload=Upload();exports={};download=st.download_button
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:upload)
    def capture(label,data,*a,**k):
        exports[label]=data
        return download(label,data,*a,**k)
    monkeypatch.setattr(st,'download_button',capture)
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'temporary-home')
    at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
    assert not at.exception and at.radio[0].value=='Automatic (recommended)'
    at.radio[0].set_value('Natural color (experimental)').run(timeout=30)
    assert next(b for b in at.button if b.label=='Use project sample').disabled
    next(b for b in at.button if b.label=='Process image').click().run(timeout=30)
    assert not at.exception and not at.error and not at.get('download_button')
    result,pixels=at.session_state['prediction']
    expected_result,expected_pixels=process_jpeg(path)
    assert result['input_status']=='experimental_natural_lut' and result['output_size']==[540,360]
    assert result['natural_color']['natural']['settings']==expected_result['natural_color']['natural']['settings']
    np.testing.assert_array_equal(pixels['adjusted'],expected_pixels['adjusted'])
    np.testing.assert_array_equal(pixels['natural_base'],expected_pixels['natural_base'])
    automatic=round(result['shadow_adjustment']['selection']['suggested_strength']*100)
    assert next(s for s in at.slider if s.label=='Shadow lift').value==automatic
    assert not any(s.label=='Warmth' for s in at.slider)
    for strength in (100,50,0):
        next(s for s in at.slider if s.label=='Adjustment strength').set_value(strength).run(timeout=30)
        assert not at.get('download_button')
        next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
        before,expected,_=render_photo(pixels,strength/100)
        assert exports['Download full-resolution PNG']==expected
        assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected
        payload=json.loads(exports['Download adjustment JSON'])
        assert payload['export']['strength']==strength/100 and payload['selected_recipe']=='natural-color'
        assert payload['shadow_adjustment']['strength']==automatic/100
        assert 'RawTherapee' not in payload['export']['semantics']
        assert payload['export']['output_png_sha256']==hashlib.sha256(expected).hexdigest()
        if strength==0: assert before==expected
    # Changing shadow strength uses the cached color base, without model inference or accumulating edits.
    import subprocess
    popen=subprocess.Popen
    def spawn(command,**kwargs):
        if 'src.lut_natural' in command:pytest.fail('Shadow slider must reuse the cached color base')
        return popen(command,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(subprocess,'Popen',spawn)
        next(s for s in at.slider if s.label=='Adjustment strength').set_value(100).run(timeout=30)
        repeated=None
        for level in (0,100,25,100):
            next(s for s in at.slider if s.label=='Shadow lift').set_value(level).run(timeout=30)
            assert not at.exception and not at.error and not at.get('download_button')
            expected_shadow,adjusted=apply_shadow_lift(pixels['natural_base'],level/100)
            _,expected,_=render_photo({'original':pixels['original'],'adjusted':adjusted})
            assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected
            if level==100:
                if repeated is not None:assert repeated==expected
                repeated=expected
            next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
            payload=json.loads(exports['Download adjustment JSON'])
            assert payload['shadow_adjustment']['strength']==level/100
            assert payload['shadow_adjustment']['output_pixels_sha256']==expected_shadow['output_pixels_sha256']
            assert payload['natural_color']==result['natural_color'] and exports['Download full-resolution PNG']==expected
            np.testing.assert_array_equal(at.session_state['prediction'][1]['natural_base'],pixels['natural_base'])
        next(s for s in at.slider if s.label=='Shadow lift').set_value(automatic).run(timeout=30)
        assert 'photo_shadow_result' not in at.session_state
    next(s for s in at.slider if s.label=='Adjustment strength').set_value(50).run(timeout=30)
    next(c for c in at.checkbox if c.label=='Inspect detail at original size').check().run(timeout=30)
    crop={key:value[20:340,110:430] for key,value in pixels.items()}
    _,expected_crop,_=render_photo(crop,.5)
    assert base64.b64decode(at.get('imgs')[3].proto.imgs[0].url.split(',',1)[1])==expected_crop
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    _,half,_=render_photo(pixels,.5)
    saved=list((tmp_path/'temporary-home/Downloads').glob('shotsense-natural-color-*.png'))
    assert len(saved)==1 and saved[0].read_bytes()==half and at.success
    from src import lut_natural
    def failed(*args):raise ValueError('controlled shadow failure')
    with monkeypatch.context() as patch:
        patch.setattr(lut_natural,'apply_shadow_lift',failed)
        next(s for s in at.slider if s.label=='Shadow lift').set_value(15).run(timeout=30)
        assert not at.exception and any('controlled shadow failure' in e.value for e in at.error)
        assert not at.get('imgs') and not at.get('download_button') and 'photo_shadow_result' not in at.session_state
    next(s for s in at.slider if s.label=='Shadow lift').set_value(automatic).run(timeout=30)
    assert not at.exception and not at.error and at.get('imgs')
    # A previous version must reprocess instead of exporting an incomplete/stale result.
    old_pixels={key:pixels[key] for key in ('original','adjusted')}
    at.session_state['prediction']=(result,old_pixels);at.run(timeout=30)
    assert not at.exception and not at.get('imgs') and not at.get('download_button')
    assert any('older natural-color version' in item.value for item in at.info)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    assert 'prediction' not in at.session_state and 'photo_shadow_strength' not in at.session_state and not at.get('imgs')
    at.radio[0].set_value('Natural color (experimental)').run(timeout=30)
    at.session_state['prediction']=(result,pixels);at.run(timeout=30)
    upload.data=b'controlled invalid JPEG'
    next(b for b in at.button if b.label=='Process image').click().run(timeout=30)
    assert not at.exception and at.error and 'prediction' not in at.session_state and not at.get('imgs')
    from app.streamlit_app import run_job
    with pytest.raises(ValueError,match='JPG and JPEG only'): run_job(b'dng','.dng',color=True)
