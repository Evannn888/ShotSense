"""Actual50MP auto/manual shadow, overall blend, PNG/JSON/crop/local-save AppTest."""
import base64,hashlib,io,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.lut_natural import apply_shadow_lift
from src.photo_engine import render_photo
folder=ROOT/'data/user_photo_diagnostics/lut-shadows-v3/worker-ahmet-yuksek-7lK0bpUj8AA-unsplash'
worker=json.loads((folder/'report.json').read_text());assert worker['status']=='passed'
result=dict(worker['lut'],input_name=Path(worker['input']['path']).name)
result['timing']['local_worker_seconds']=worker['worker_call_seconds']
with np.load(folder/'pixels.npz',allow_pickle=False) as stored:
    source={key:stored[key].copy() for key in ('original','natural_base','adjusted')}
exports={};download=st.download_button

def capture(label,data,*args,**kwargs):
    exports[label]=data
    return download(label,data,*args,**kwargs)
report={'scope':'Actual50MPworker result, manual shadow and mixed overall strength in isolated AppTest; real button/worker route and no model reprocessing covered by functional regression. No live-browser client acceptance.'}
started=time.perf_counter()
try:
    with tempfile.TemporaryDirectory(prefix='shotsense-shadow-ui-') as home,patch.object(st,'download_button',capture):
        at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
        at.radio[0].set_value('Natural color (experimental)').run(timeout=30)
        at.session_state['prediction']=(result,source);at.run(timeout=60)
        assert not at.exception and not at.error and not at.get('download_button')
        def displayed(index):return base64.b64decode(at.get('imgs')[index].proto.imgs[0].url.split(',',1)[1])
        assert displayed(1)==(folder/'result.png').read_bytes()
        assert next(s for s in at.slider if s.label=='Shadow lift').value==60
        next(s for s in at.slider if s.label=='Shadow lift').set_value(40)
        next(s for s in at.slider if s.label=='Adjustment strength').set_value(50)
        next(c for c in at.checkbox if c.label=='Inspect detail at original size').check()
        next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=60)
        assert not at.exception and not at.error
        png=exports['Download full-resolution PNG'];assert displayed(1)==png
        metadata=json.loads(exports['Download adjustment JSON'])
        assert metadata['export']['size']==[8192,6144] and metadata['export']['strength']==.5
        assert metadata['export']['output_png_sha256']==hashlib.sha256(png).hexdigest()
        assert metadata['shadow_adjustment']['strength']==.4 and metadata['natural_color']==result['natural_color']
        _,manual=apply_shadow_lift(source['natural_base'],.4)
        assert metadata['shadow_adjustment']['output_pixels_sha256']==hashlib.sha256(manual).hexdigest()
        with Image.open(io.BytesIO(png)) as image:
            assert image.format=='PNG' and image.size==(8192,6144) and 'srgb' in image.info
            actual=np.asarray(image).copy()
        for y in range(0,len(actual),128):
            end=y+128
            expected=np.rint(source['original'][y:end].astype(np.float32)*.5+manual[y:end].astype(np.float32)*.5).astype(np.uint8)
            np.testing.assert_array_equal(actual[y:end],expected)
        crop={'original':source['original'][2912:3232,3936:4256],'adjusted':manual[2912:3232,3936:4256]}
        _,crop_png,_=render_photo(crop,.5);assert displayed(3)==crop_png
        del actual,manual
        with patch.object(Path,'home',return_value=Path(home)):
            next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=60)
        assert not at.exception and not at.error and at.success
        saved=list((Path(home)/'Downloads').glob('*.png'));assert len(saved)==1 and saved[0].read_bytes()==png
        report.update(status='passed',auto_matches_actual_worker=True,manual_shadow_mixed_overall_pixels_exact=True,
            display_download_local_save_exact=True,matching_json=True,original_size_native_crop_exact=True,
            manual_stage_seconds=metadata['shadow_adjustment']['seconds'],no_actual_downloads_write=True,size=[8192,6144],png_bytes=len(png))
except Exception as error:report.update(status='failed',error=type(error).__name__+': '+str(error))
report['seconds']=time.perf_counter()-started
(Path(__file__).parent/'app-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
sys.exit(0 if report['status']=='passed' else 1)
