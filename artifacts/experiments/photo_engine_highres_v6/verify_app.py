"""Render the actual 50 MP worker result and exercise local PNG save in an isolated test session."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

directory=ROOT/'data/user_photo_diagnostics/highres-v6/ahmet-natural'
worker=json.loads((directory/'report.json').read_text())
result=dict(worker['payload'],input_name=Path(worker['input']).name)
source={}
for name,filename in (('original','before.png'),('adjusted','after.png')):
    with Image.open(directory/filename) as image:
        source[name]=np.asarray(image,dtype=np.uint8).copy()
report={'scope':'Isolated AppTest of actual 50 MP worker result; does not establish live-browser delivery.'}
started=time.perf_counter()
try:
    with tempfile.TemporaryDirectory(prefix='shotsense-highres-ui-') as home:
        at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
        at.radio[0].set_value('Natural adjustment (experimental)').run(timeout=30)
        at.selectbox[0].set_value('Lift shadows').run(timeout=30)
        at.session_state['prediction']=(result,source)
        at.run(timeout=90)
        assert not at.exception and not at.error
        images=at.get('imgs')
        assert len(images)==2
        for item,key in zip(images,('before','after')):
            data=base64.b64decode(item.proto.imgs[0].url.split(',',1)[1])
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
            assert hashlib.sha256(data).hexdigest()==worker[key]['sha256']
        report['display_matches_actual_native_png']=True
        with patch.object(Path,'home',return_value=Path(home)):
            next(button for button in at.button if button.label=='Save PNG to Downloads').click().run(timeout=90)
        assert not at.exception and not at.error and at.success
        files=list((Path(home)/'Downloads').glob('*.png'))
        assert len(files)==1
        data=files[0].read_bytes()
        assert hashlib.sha256(data).hexdigest()==worker['after']['sha256']
        with Image.open(files[0]) as image:
            assert image.format=='PNG' and list(image.size)==worker['after']['size']
        report['ordinary_button_saved_matching_native_png']=True
        report['temporary_save_bytes']=len(data)
        report['save_size']=worker['after']['size']
        report['no_actual_downloads_write']=True
        report['status']='passed'
except Exception as error:
    report['status']='failed'
    report['error']=type(error).__name__+': '+str(error)
finally:
    report['seconds']=time.perf_counter()-started
    (directory.parent/'app-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
sys.exit(0 if report['status']=='passed' else 1)
