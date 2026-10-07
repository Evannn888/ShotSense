"""Actual 50 MP form submission, matching export and private local-save verification."""
import base64
import hashlib
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

base=ROOT/'data/user_photo_diagnostics/highres-v6/ahmet-natural'
worker=json.loads((base/'report.json').read_text());result=dict(worker['payload'],input_name=Path(worker['input']).name)
source={}
for key,file in [('original','before.png'),('adjusted','after.png')]:
    with Image.open(base/file) as image:source[key]=np.asarray(image,dtype=np.uint8).copy()
expected=json.loads((ROOT/'data/user_photo_diagnostics/finetune-v7/worker-ahmet-natural/report.json').read_text())
exports={};original_download=st.download_button
def capture(label,data,*args,**kwargs):
    exports[label]=data
    return original_download(label,data,*args,**kwargs)
report={'scope':'Isolated AppTest of actual50MP fine form/export/save, not the live browser.'};started=time.perf_counter()
try:
    with tempfile.TemporaryDirectory(prefix='shotsense-fine-ui-') as directory,patch.object(st,'download_button',capture):
        at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
        at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
        at.selectbox[0].set_value('Lift shadows').run(timeout=30)
        at.session_state['prediction']=(result,source);at.run(timeout=90)
        assert not at.exception and not at.error
        next(s for s in at.slider if s.label=='Vibrance').set_value(-4)
        next(s for s in at.slider if s.label=='Local contrast').set_value(6)
        next(b for b in at.button if b.label=='Apply fine tuning').click().run(timeout=120)
        assert not at.exception and not at.error
        identity,metadata,pixels=at.session_state['photo_fine_result']
        assert metadata['output_pixels_sha256']==expected['fine_tuning']['output_pixels_sha256']
        displayed=base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])
        assert hashlib.sha256(displayed).hexdigest()==expected['export']['output_png_sha256']
        next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=90)
        assert exports['Download full-resolution PNG']==displayed
        payload=json.loads(exports['Download adjustment JSON'])
        assert payload['fine_tuning']['settings']==metadata['settings']
        assert payload['export']['output_png_sha256']==expected['export']['output_png_sha256']
        with patch.object(Path,'home',return_value=Path(directory)):
            next(b for b in at.button if b.label=='Save PNG to Downloads').click().run(timeout=90)
        assert not at.exception and not at.error and at.success
        files=list((Path(directory)/'Downloads').glob('*.png'))
        assert len(files)==1 and files[0].read_bytes()==displayed and '-finetuned-' in files[0].name
        report.update(status='passed',form_actual_50MP_worker=True,display_png_json_match=True,
                      private_save_matches=True,no_actual_downloads_write=True,size=metadata['size'],
                      output_png_sha256=expected['export']['output_png_sha256'],fine_settings=metadata['settings'])
except Exception as error:
    report.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    report['seconds']=time.perf_counter()-started
    (ROOT/'data/user_photo_diagnostics/finetune-v7/app-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
sys.exit(0 if report['status']=='passed' else 1)
