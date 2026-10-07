"""Reproduce the pinned framework's PNG-to-JPEG display conversion; no browser interaction."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from PIL import Image
import streamlit
from streamlit.elements.lib import image_utils
from streamlit.elements.lib.layout_utils import LayoutConfig
from src.photo_engine import render_photo

out=Path(__file__).parent
rgb=np.tile(np.array([[16,240]*900],dtype=np.uint8)[...,None],(65,1,3))
png=render_photo({'original':rgb,'adjusted':rgb})[1]
format=image_utils._validate_image_format_string(png,'auto')
old=image_utils._ensure_image_size_and_format(png,LayoutConfig(width='stretch'),format)
uri='data:image/png;base64,'+base64.b64encode(png).decode('ascii')
assert image_utils.image_to_url(uri,LayoutConfig(width='stretch'),False,'RGB','auto','diagnostic')==uri
assert base64.b64decode(uri.split(',',1)[1])==png
def describe(data):
    with Image.open(io.BytesIO(data)) as image:
        return {'format':image.format,'size':list(image.size),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
assert describe(old)['size']==[1460,52] and describe(old)['format']=='JPEG'
report={'scope':'Synthetic display transfer reproduction, not original user photo/download acceptance',
        'streamlit_version':streamlit.__version__,'source_sha256':hashlib.sha256(Path(image_utils.__file__).read_bytes()).hexdigest(),
        'input_png':describe(png),'old_image_payload':describe(old),'new_image_payload':describe(png),
        'new_data_url_byte_identity':True}
destination=out/'display-reproduction.json'
if destination.exists(): raise ValueError('Preserve the frozen reproduction; use another output directory')
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
