import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageCms
import pytest

from src import photo_engine as engine


def test_real_engine_orientation_color_full_resolution_identity_and_replay(tmp_path, monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    path = tmp_path / 'oriented.jpeg'
    rgb = np.zeros((41, 65, 3), dtype=np.uint8)
    rgb[:] = [65, 80, 95]; rgb[:10, :10] = [4, 3, 2]
    image = Image.fromarray(rgb); exif = Image.Exif(); exif[274] = 6
    image.save(path, quality=95, exif=exif,
               icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    original_bytes = path.read_bytes()
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Inference must be offline'))
    result, source = engine.process_photo(path)
    _, repeated = engine.process_photo(path)
    assert path.read_bytes() == original_bytes
    assert result['selected_recipe'] == 'natural'
    assert result['jpeg_color']['embedded_icc_converted_to_srgb']
    assert result['engine']['version'] == '5.11' and result['output_size'] == [41, 65]
    assert source['original'].shape == (65, 41, 3)
    np.testing.assert_array_equal(source['adjusted'], repeated['adjusted'])
    before, after, zero = engine.render_photo(source, 0)
    assert before == after and zero['changed_pixel_fraction'] == 0
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(after))), source['original'])
    before, after, half = engine.render_photo(source, .5)
    assert before != after and half['output_png_sha256'] == hashlib.sha256(after).hexdigest()
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(after))),
        np.rint(source['original'].astype(np.float32) * .5 + source['adjusted'].astype(np.float32) * .5).astype(np.uint8))
    assert Image.open(io.BytesIO(after)).size == (41, 65)
    assert 'srgb' in Image.open(io.BytesIO(after)).info
    for strength in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError): engine.render_photo(source, strength)

    output, cache = tmp_path / 'result.json', tmp_path / 'source.npz'
    offline = """
import sys,runpy,socket,urllib.request
socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network'))
urllib.request.urlopen=socket.create_connection
sys.argv=['src.photo_engine']+sys.argv[1:]
runpy.run_module('src.photo_engine',run_name='__main__')
"""
    subprocess.run([sys.executable, '-c', offline, str(path), '--output', str(output),
                    '--preview-source', str(cache)], cwd=engine.ROOT, check=True, capture_output=True, timeout=60)
    assert json.loads(output.read_text())['input_sha256'] == result['input_sha256']
    with np.load(cache, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['adjusted'], source['adjusted'])

    for color in ([0,0,0], [8,6,4], [190,180,160]):
        control = tmp_path / 'control.jpg'; Image.new('RGB', (3, 2), tuple(color)).save(control)
        payload, pixels = engine.process_photo(control)
        assert payload['selected_recipe'] == 'unchanged' and payload['timing']['engine_runs'] == []
        np.testing.assert_array_equal(pixels['original'], pixels['adjusted'])
    with pytest.raises(ValueError, match='Unknown'): engine.process_photo(path, 'unknown')
    fake = tmp_path / 'fake.jpg'; fake.write_bytes(b'not an image')
    with pytest.raises((ValueError, OSError)): engine.process_photo(fake)
    missing = tmp_path / 'absent'; monkeypatch.setattr(engine, 'CLI', missing)
    with pytest.raises(ValueError, match='missing'): engine.process_photo(path)
    missing.write_bytes(b'broken executable')
    with pytest.raises(ValueError, match='checksum'): engine.process_photo(path)


def test_engine_subprocess_failure_and_timeout_propagate(tmp_path, monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    path = tmp_path / 'input.jpg'; Image.new('RGB', (33, 25), (65,80,95)).save(path)
    def clipped(*args, **kwargs):
        command = args[0]
        Image.new('RGB', (33,25), (255,255,255)).save(command[command.index('-O')+1])
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(engine.subprocess, 'run', clipped)
    payload, pixels = engine.process_photo(path)
    assert payload['selected_recipe'] == 'unchanged' and 'clipping guard' in payload['selection']['reason']
    np.testing.assert_array_equal(pixels['original'], pixels['adjusted'])
    monkeypatch.setattr(engine.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a[0], 3, '', 'bad profile'))
    with pytest.raises(ValueError, match='bad profile'): engine.process_photo(path)
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(engine.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired): engine.process_photo(path)


def test_real_dng_engine_is_separate_and_preserves_dimensions(tmp_path):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    path = next(iter(sorted((engine.ROOT / 'data/raw/dngs').glob('*.dng'))), None)
    if path is None: pytest.skip('Locally retained DNG required')
    payload, pixels = engine.process_photo(path, 'neutral')
    assert payload['input_format'] == 'DNG'
    assert payload['recommended_absolute'] == payload['experimental_absolute'] == {}
    assert len(payload['timing']['engine_runs']) == 1
    np.testing.assert_array_equal(pixels['original'], pixels['adjusted'])
    assert max(payload['output_size']) > 1600
    _, after, metadata = engine.render_photo(pixels, 0)
    assert list(Image.open(io.BytesIO(after)).size) == payload['output_size'] == metadata['size']


def test_real_engine_neutral_ramp_and_dark_color_patches(tmp_path):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    ramp=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    path=tmp_path/'ramp.jpg'; Image.fromarray(np.repeat(ramp,65,axis=0)).save(path,quality=100)
    payload,pixels=engine.process_photo(path,'natural')
    assert (pixels['adjusted'].max(axis=-1)-pixels['adjusted'].min(axis=-1)).max()<=1
    patches=np.full((192,256,3),[65,80,95],dtype=np.uint8)
    patches[:64,:64]=[0,0,0]; patches[:64,64:128]=[8,6,4]
    path=tmp_path/'dark-colors.jpg'; Image.fromarray(patches).save(path,quality=100)
    payload,pixels=engine.process_photo(path,'natural')
    np.testing.assert_array_equal(pixels['adjusted'][32,32],[0,0,0])
    r,g,b=pixels['adjusted'][32,96]
    assert r>=g>=b and r<20
    # These limited controls exclude a new offset/tint here; they do not validate a real eye/photo.


def test_app_engine_downloads_strength_and_stale_results(tmp_path, monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from app.streamlit_app import run_job
    exports = {}
    original_download = st.download_button
    def capture(label, data, *args, **kwargs):
        exports[label] = data
        return original_download(label, data, *args, **kwargs)
    monkeypatch.setattr(st, 'download_button', capture)
    path = tmp_path / 'input.jpg'; Image.new('RGB', (1733,37), (65,80,95)).save(path)
    payload, pixels = run_job(path.read_bytes(), '.jpg', natural=True)
    assert payload['selected_recipe']=='natural' and payload['output_size']==[1733,37]
    at = AppTest.from_file(str(engine.ROOT / 'app/streamlit_app.py')).run(timeout=30)
    assert at.radio[0].value=='Automatic (recommended)' and not at.exception
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    at.session_state['prediction']=(payload,pixels)
    at.run(timeout=30)
    assert not at.exception and not at.error and len(at.get('imgs'))==2
    assert not at.get('download_button')
    for strength in (100,50,0):
        at.slider[0].set_value(strength).run(timeout=30)
        assert not at.get('download_button')
        next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
        assert {item.proto.label for item in at.get('download_button')}=={'Download full-resolution PNG','Download adjustment JSON'}
        before, expected, metadata = engine.render_photo(pixels,strength/100)
        assert exports['Download full-resolution PNG']==expected
        assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==expected
        saved=json.loads(exports['Download adjustment JSON'])
        assert saved['export']==metadata and saved['input_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert Image.open(io.BytesIO(expected)).size==(1733,37)
        if strength==0: assert before==expected
    at.selectbox[0].set_value('Gentle lift + chroma denoising').run(timeout=30)
    assert 'prediction' not in at.session_state and not at.get('imgs')
    at.session_state['prediction']=(payload,pixels)
    at.run(timeout=30)
    at.radio[0].set_value('Manual exposure').run(timeout=30)
    assert 'prediction' not in at.session_state and not at.get('imgs')
    # Exercise the button's real RAW worker route rather than only injecting a cached result.
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    next(button for button in at.button if button.label=='Use project sample').click().run(timeout=60)
    assert not at.exception and not at.error
    raw_result, raw_pixels=at.session_state['prediction']
    assert raw_result['input_format']=='DNG' and max(raw_result['output_size'])>1600
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    assert Image.open(io.BytesIO(exports['Download full-resolution PNG'])).size==tuple(raw_result['output_size'])
    at.slider[0].set_value(0).run(timeout=30)
    assert not at.get('download_button')
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(exports['Download full-resolution PNG']))),raw_pixels['original'])


def test_app_timeout_kills_the_worker_process_group(tmp_path, monkeypatch):
    from app.streamlit_app import run_job
    import app.streamlit_app as app
    killed=[]; calls=[]
    class TimedOutWorker:
        pid=123456
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def communicate(self,timeout=None):
            calls.append(timeout)
            if timeout is not None: raise subprocess.TimeoutExpired('worker',timeout)
            return '', ''
    def spawn(command,**kwargs):
        assert kwargs['start_new_session'] and '--recipe' in command
        return TimedOutWorker()
    monkeypatch.setattr(app.subprocess,'Popen',spawn)
    monkeypatch.setattr(app.os,'killpg',lambda pid,signal: killed.append((pid,signal)))
    with pytest.raises(subprocess.TimeoutExpired): run_job(b'bounded worker input','.jpeg',natural=True)
    assert calls==[60,None] and killed==[(123456,app.signal.SIGKILL)]


def test_app_pixel_limit_error_reports_the_input_format(monkeypatch):
    import app.streamlit_app as app
    class RejectedWorker:
        returncode=1
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def communicate(self,timeout=None): return '', 'ValueError: image exceeds the megapixel limit'
    monkeypatch.setattr(app.subprocess,'Popen',lambda *a,**k: RejectedWorker())
    for suffix,label in (('.jpeg','JPEG exceeds the 64-megapixel'),('.dng','DNG exceeds the 40-megapixel')):
        with pytest.raises(ValueError,match=label):
            app.run_job(b'rejected image',suffix,natural=True)


def test_app_explains_unchanged_and_applies_gentle_to_the_same_source(monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    exports={}; download=st.download_button
    def capture(label,data,*args,**kwargs):
        exports[label]=data
        return download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture)
    at=AppTest.from_file(str(engine.ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    next(b for b in at.button if b.label=='Use project sample').click().run(timeout=60)
    payload,pixels=at.session_state['prediction']; original_hash=payload['input_sha256']
    # Controlled rejection state exercises the exact explanatory branch reported by the user.
    payload=dict(payload,selected_recipe='unchanged',selection={'reason':'candidate rejected by clipping guard',
                 'candidate_new_full_channel_fraction':.00312683})
    at.session_state['prediction']=(payload,dict(original=pixels['original'],adjusted=pixels['original'].copy()))
    at.run(timeout=30)
    assert not at.exception and any('0.31%' in item.value and '0.10%' in item.value for item in at.info)
    next(b for b in at.button if b.label=='Apply gentle lift').click().run(timeout=60)
    assert not at.exception and not at.error and at.selectbox[0].value=='Gentle lift'
    result,pixels=at.session_state['prediction']
    assert result['requested_recipe']==result['selected_recipe']=='natural' and result['input_sha256']==original_hash
    assert 'pending_recipe' not in at.session_state and np.any(pixels['adjusted']!=pixels['original'])
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    saved=json.loads(exports['Download adjustment JSON'])
    assert saved['requested_recipe']=='natural' and saved['export']['output_png_sha256']==hashlib.sha256(exports['Download full-resolution PNG']).hexdigest()
    assert not any('Automatic choice preserved' in item.value for item in at.info)
    at.slider[0].set_value(0).run(timeout=30)
    assert any('Strength is 0%' in item.value for item in at.info)
    assert not any(b.label=='Apply gentle lift' for b in at.button)
    # The stronger recipe reuses the same action/worker and never stacks the gentle profile.
    next(b for b in at.button if b.label=='Apply shadow lift').click().run(timeout=60)
    assert not at.exception and not at.error and at.selectbox[0].value=='Lift shadows'
    result,pixels=at.session_state['prediction']
    assert result['requested_recipe']==result['selected_recipe']=='shadows' and result['input_sha256']==original_hash
    assert at.slider[0].value==100 and 'natural.pp3' not in result['profile_sha256']
    assert result['profile_sha256']['shadows.pp3']==hashlib.sha256((engine.BUNDLE/'shadows.pp3').read_bytes()).hexdigest()
    assert not at.get('download_button')
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    saved=json.loads(exports['Download adjustment JSON'])
    assert saved['selected_recipe']=='shadows' and saved['export']['output_png_sha256']==hashlib.sha256(exports['Download full-resolution PNG']).hexdigest()
    assert not any(b.label=='Apply shadow lift' for b in at.button)
    at.slider[0].set_value(0).run(timeout=30)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(exports['Download full-resolution PNG']))),pixels['original'])
    # Reject a changed source before starting a worker, rather than editing a different file.
    at.session_state['pending_recipe']=(True,'mismatched-input-hash')
    at.run(timeout=30)
    assert not at.exception and any('source has changed' in item.value for item in at.error)
    assert 'prediction' not in at.session_state


def test_shadow_recipe_targets_dark_tones_through_real_jpeg_worker(tmp_path):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    from app.streamlit_app import run_job
    rgb=np.full((256,513,3),[180,185,190],dtype=np.uint8)
    rgb[128:]=[30,35,25];rgb[144:224:8,32:480:8]=[40,45,35]
    rgb[200:224,480:]=0
    path=tmp_path/'backlight.jpg';Image.fromarray(rgb).save(path,quality=100,subsampling=0)
    original_bytes=path.read_bytes()
    result,pixels=run_job(original_bytes,'.jpg',natural=True,recipe='shadows')
    _,replayed=engine.process_photo(path,'shadows')
    _,gentle=engine.process_photo(path,'natural')
    np.testing.assert_array_equal(pixels['adjusted'],replayed['adjusted'])
    assert result['requested_recipe']==result['selected_recipe']=='shadows'
    assert result['output_size']==[513,256] and path.read_bytes()==original_bytes
    assert set(result['profile_sha256'])=={'neutral.pp3','shadows.pp3'}
    assert (engine.BUNDLE/'shadows.pp3').read_bytes()==(engine.ROOT/'artifacts/experiments/photo_engine_quality_v5/candidate-shadows.pp3').read_bytes()
    delta=pixels['adjusted'].astype(np.float32)-pixels['original'].astype(np.float32)
    sky=float(delta[:120].mean());foreground=float(delta[144:224,:480].mean())
    assert 0<=sky<13 and 8<foreground<25 and foreground>3*sky
    assert float((pixels['adjusted'][144:224,:480].astype(np.float32)-gentle['adjusted'][144:224,:480]).mean())>3
    np.testing.assert_array_equal(pixels['adjusted'][210,500],[0,0,0])
    before,after,zero=engine.render_photo(pixels,0)
    assert before==after and zero['changed_pixel_fraction']==0
    _,after,metadata=engine.render_photo(pixels)
    assert Image.open(io.BytesIO(after)).size==(513,256) and metadata['output_png_sha256']==hashlib.sha256(after).hexdigest()


def test_actual_png_image_and_download_file_bindings_and_detail_crop(tmp_path, monkeypatch):
    if not engine.CLI.exists(): pytest.skip('Prepare the pinned RawTherapee engine')
    from streamlit.testing.v1 import AppTest
    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
    from streamlit.runtime.media_file_storage import MediaFileKind
    stored={};load=MemoryMediaFileStorage.load_and_get_id
    def capture(self,*args,**kwargs):
        file_id=load(self,*args,**kwargs)
        stored[file_id]=self.get_file(file_id)
        return file_id
    monkeypatch.setattr(MemoryMediaFileStorage,'load_and_get_id',capture)
    path=tmp_path/'detail.jpg'
    rgb=np.full((91,1801,3),[55,70,80],dtype=np.uint8);rgb[::2,::2]=[100,110,130]
    Image.fromarray(rgb).save(path,quality=100,subsampling=0)
    result,pixels=engine.process_photo(path,'shadows')
    result.update(input_name='detail.jpg');result['timing']['local_worker_seconds']=0
    at=AppTest.from_file(str(engine.ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    at.selectbox[0].set_value('Lift shadows').run(timeout=30)
    at.session_state['prediction']=(result,pixels);at.run(timeout=30)
    assert not at.exception and not at.get('download_button')
    def displayed(index):
        url=at.get('imgs')[index].proto.imgs[0].url
        assert url.startswith('data:image/png;base64,')
        return base64.b64decode(url.split(',',1)[1])
    before,after,_=engine.render_photo(pixels)
    assert displayed(0)==before and displayed(1)==after
    assert Image.open(io.BytesIO(displayed(1))).size==(1801,91)
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    downloads={b.proto.label:b.proto for b in at.get('download_button')}
    png_proto=downloads['Download full-resolution PNG'];json_proto=downloads['Download adjustment JSON']
    assert png_proto.url.endswith('.png') and json_proto.url.endswith('.json') and png_proto.url!=json_proto.url
    png=stored[Path(png_proto.url).stem];parameters=stored[Path(json_proto.url).stem]
    assert png.kind==parameters.kind==MediaFileKind.DOWNLOADABLE
    assert png.mimetype=='image/png' and png.filename.endswith('.png') and png.content==after==displayed(1)
    assert png.content.startswith(b'\x89PNG\r\n\x1a\n') and png_proto.ignore_rerun and json_proto.ignore_rerun
    saved=json.loads(parameters.content)
    assert parameters.mimetype=='application/json' and parameters.filename.endswith('.json')
    assert saved['export']['output_png_sha256']==hashlib.sha256(png.content).hexdigest()
    assert any(item.label=='Adjustment metadata (JSON)' for item in at.expander)
    at.slider[0].set_value(50).run(timeout=30)
    assert not at.get('download_button')
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    new_png=next(b.proto for b in at.get('download_button') if b.proto.label=='Download full-resolution PNG')
    assert new_png.url!=png_proto.url and stored[Path(new_png.url).stem].content==displayed(1)
    next(c for c in at.checkbox if c.label=='Inspect detail at original size').check().run(timeout=30)
    next(s for s in at.slider if s.label=='Detail horizontal position').set_value(100).run(timeout=30)
    next(s for s in at.slider if s.label=='Detail vertical position').set_value(0).run(timeout=30)
    assert Image.open(io.BytesIO(displayed(3))).size==(320,91)
    full=np.asarray(Image.open(io.BytesIO(displayed(1))))
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(displayed(3)))),full[:91,-320:])
    changed=dict(result,input_sha256='different-source-with-identical-pixels')
    at.session_state['prediction']=(changed,pixels);at.run(timeout=30)
    assert not at.get('download_button') and not at.exception


def test_small_png_overview_preserves_pixels_and_marks_old_shadow_cache(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    calls=[]; show=st.image
    def capture(data,*args,**kwargs):
        calls.append(kwargs)
        return show(data,*args,**kwargs)
    monkeypatch.setattr(st,'image',capture)
    rgb=np.full((452,563,3),[25,30,35],dtype=np.uint8)
    pixels={'original':rgb,'adjusted':rgb+10}
    result={'input_status':'experimental_photo_engine','input_format':'JPEG','input_name':'small.jpg',
            'processing_version':'rawtherapee-recipes-v2','requested_recipe':'shadows','selected_recipe':'shadows',
            'input_sha256':'small-fixture','engine':{'version':'5.11'},'selection':{},
            'timing':{'local_worker_seconds':0}}
    at=AppTest.from_file(str(engine.ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    at.session_state['prediction']=(result,pixels);at.run(timeout=30)
    assert not at.exception and all(call['width']=='content' for call in calls)
    assert any('563 × 452' in item.value and 'missing detail' in item.value for item in at.info)
    assert any('older, stronger shadow lift' in item.value for item in at.warning)
    assert next(b for b in at.button if b.label=='Reprocess with gentler shadows').disabled
    result=dict(result,processing_version=engine.VERSION)
    at.session_state['prediction']=(result,pixels);at.run(timeout=30)
    assert not at.warning and not any(b.label=='Reprocess with gentler shadows' for b in at.button)
    shown=base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])
    assert shown==engine.render_photo(pixels)[1] and Image.open(io.BytesIO(shown)).size==(563,452)


def test_local_save_button_delivers_current_png_and_reports_failure(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.streamlit_app import save_png_file
    home=tmp_path/'local-home';home.mkdir()
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:home))
    rgb=np.full((39,1801,3),[65,80,95],dtype=np.uint8)
    pixels={'original':rgb,'adjusted':np.clip(rgb.astype(int)+20,0,255).astype(np.uint8)}
    result={'input_status':'experimental_photo_engine','input_format':'JPEG','input_name':'local.jpg',
            'input_sha256':'local-fixture','requested_recipe':'shadows','selected_recipe':'shadows',
            'engine':{'version':'5.11'},'timing':{'local_worker_seconds':0}}
    at=AppTest.from_file(str(engine.ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
    at.session_state['prediction']=(result,pixels);at.run(timeout=30)
    assert not (home/'Downloads').exists() and not at.exception
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    files=list((home/'Downloads').glob('*.png'));assert len(files)==1
    saved=files[0];before,expected,_=engine.render_photo(pixels)
    assert saved.read_bytes()==expected and Image.open(saved).size==(1801,39)
    assert any(str(saved) in message.value for message in at.success)
    assert base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])==saved.read_bytes()
    original_mtime=saved.stat().st_mtime_ns
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    assert saved.stat().st_mtime_ns==original_mtime and len(list((home/'Downloads').glob('*.png')))==1
    at.slider[0].set_value(50).run(timeout=30)
    assert not at.success
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    _,half,_=engine.render_photo(pixels,.5)
    assert len(list((home/'Downloads').glob('*.png')))==2
    assert any(path.read_bytes()==half for path in (home/'Downloads').glob('*.png'))
    # A conflicting existing file is neither overwritten nor reported as success.
    saved.write_bytes(b'existing unrelated content')
    at.slider[0].set_value(100).run(timeout=30)
    next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=30)
    assert saved.read_bytes()==b'existing unrelated content' and not at.success
    assert not at.exception and any('PNG was not saved' in error.value for error in at.error)
    with pytest.raises(ValueError):save_png_file(b'{"not":"PNG"}','wrong.png')
    with pytest.raises(ValueError):save_png_file(expected,'../wrong.png')
    linked=home/'Downloads'/'linked.png';linked.symlink_to(saved)
    with pytest.raises(FileExistsError):save_png_file(expected,'linked.png')
    assert saved.read_bytes()==b'existing unrelated content' and linked.is_symlink()
