"""Reproduce diagnostic examples and bounded-resolution CPU checks; no benchmark claim."""
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from src.jpeg_restoration import BUNDLE, restore_jpeg, render_restoration

rows, records = [], []
for stem in ('27','102','113','254','565','6'):
    path = ROOT / 'data/external/lol_pilot' / (stem + '.jpg')
    if not path.exists():
        raise ValueError('Retained diagnostic input missing: ' + stem)
    started = time.perf_counter()
    result, source = restore_jpeg(path)
    before, after, meta = render_restoration(source)
    original = Image.open(io.BytesIO(before)); restored = Image.open(io.BytesIO(after))
    row = Image.new('RGB', (900, 324), 'white')
    for x, image in ((0, original),(450, restored)):
        image.thumbnail((450,300)); row.paste(image,(x,24))
    ImageDraw.Draw(row).text((8,5), stem + ' | Before / HVI-CIDNet 100%', fill='black')
    rows.append(row)
    records.append({'id':stem, 'timing':result['timing'], 'script_seconds':time.perf_counter()-started,
                    'changed_pixel_fraction':meta['changed_pixel_fraction'],
                    'source_mean':float(source['original'].mean()/255), 'output_mean':float(source['restored'].mean())})
# Large synthetic daylight control exposes enhancement outside its intended domain.
control = BUNDLE / 'daylight-control.jpg'
x = np.linspace(.2,1.,1440,dtype=np.float32)[None,:,None]
rgb = np.broadcast_to(x,(960,1440,3)).copy(); rgb[:,:,2] *= .85
Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(control,quality=95)
result, source = restore_jpeg(control)
assert source['original'].shape == (640,960,3)
before, zero, meta = render_restoration(source,0)
assert before == zero
records.append({'id':'synthetic-daylight-1440x960','output_size':result['output_size'],'timing':result['timing'],
                'mean_absolute_change_srgb':float(np.abs(source['restored']-source['original']/255).mean()),
                'strength_zero_exact':before==zero})
contact = Image.new('RGB',(900,324*len(rows)), 'white')
for i,row in enumerate(rows): contact.paste(row,(0,i*324))
contact.save(BUNDLE / 'diagnostics.jpg',quality=95)
(BUNDLE / 'verification.json').write_text(json.dumps({'scope':'Diagnostic previously inspected LOL examples; pretrained overlap unknown. No eval15 access or unseen benchmark.',
                                                       'cases':records},indent=2)+'\n')
print(json.dumps(records,indent=2))
