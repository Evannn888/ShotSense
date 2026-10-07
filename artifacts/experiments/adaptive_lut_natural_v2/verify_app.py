"""AppTest of actual50MP protected worker result; no live-browser acceptance claim."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.photo_engine import render_photo
folder=ROOT/'data/user_photo_diagnostics/lut-natural-v2/worker-ahmet-yuksek-7lK0bpUj8AA-unsplash'
worker=json.loads((folder/'report.json').read_text());assert worker['status']=='passed'
result=dict(worker['lut'],input_name=Path(worker['input']['path']).name)
result['timing']['local_worker_seconds']=worker['worker_call_seconds']
with np.load(folder/'pixels.npz',allow_pickle=False) as stored:
    source={key:stored[key].copy() for key in ('original','adjusted')}
exports={};download=st.download_button

def capture(label,data,*a,**k):
    exports[label]=data
    return download(label,data,*a,**k)
report={'scope':'Actual50MPworker result in isolated AppTest; real JPEG button/worker route covered by functional regression. Live-browser delivery remains unverified.'}
started=time.perf_counter()
try:
    with tempfile.TemporaryDirectory(prefix='shotsense-natural-ui-') as home,patch.object(st,'download_button',capture):
        at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
        at.radio[0].set_value('Natural color (experimental)').run(timeout=30)
        at.session_state['prediction']=(result,source);at.run(timeout=60)
        assert not at.exception and not at.error and not at.get('download_button')
        expected=(folder/'result.png').read_bytes()
        def displayed(index):return base64.b64decode(at.get('imgs')[index].proto.imgs[0].url.split(',',1)[1])
        assert displayed(1)==expected
        next(c for c in at.checkbox if c.label=='Inspect detail at original size').check()
        next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=60)
        assert not at.exception and not at.error and exports['Download full-resolution PNG']==expected
        metadata=json.loads(exports['Download adjustment JSON'])
        assert metadata['export']['size']==[8192,6144] and metadata['export']['output_png_sha256']==hashlib.sha256(expected).hexdigest()
        assert metadata['natural_color']['natural']==result['natural_color']['natural']
        cw,ch=320,320;x,y=3936,2912
        crop={key:pixels[y:y+ch,x:x+cw] for key,pixels in source.items()}
        _,crop_png,_=render_photo(crop)
        assert displayed(3)==crop_png
        with patch.object(Path,'home',return_value=Path(home)):
            next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=60)
        assert not at.exception and not at.error and at.success
        saved=list((Path(home)/'Downloads').glob('*.png'))
        assert len(saved)==1 and saved[0].read_bytes()==expected
        with Image.open(io.BytesIO(expected)) as image:
            assert image.format=='PNG' and image.size==(8192,6144) and 'srgb' in image.info
            np.testing.assert_array_equal(np.asarray(image),source['adjusted'])
        report.update(status='passed',display_download_local_save_exact=True,matching_json=True,
            original_size_native_crop_exact=True,no_actual_downloads_write=True,size=[8192,6144],png_bytes=len(expected))
except Exception as error:report.update(status='failed',error=type(error).__name__+': '+str(error))
report['seconds']=time.perf_counter()-started
(Path(__file__).parent/'app-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
sys.exit(0 if report['status']=='passed' else 1)
