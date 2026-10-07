"""Frozen original/current-native/pretrained-LUT diagnostic comparison."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
from src.adaptive_lut import adjust_rgb, load_model
from src.photo_engine import render_photo
from src.jpeg_inference import decode_jpeg

FOLDER = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/user_photo_diagnostics/lut-v1'
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
model = load_model()
frozen = json.loads((FOLDER / 'inputs.json').read_text())
previous = {item['id']:item for item in json.loads((ROOT/'data/user_photo_diagnostics/finetune-v7/report.json').read_text())['cases']}
records = []; sheets = []
for entry in frozen['cases']:
    started = time.perf_counter(); name = entry['id']; path = Path(entry['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    cached = ROOT / 'data/user_photo_diagnostics/finetune-v7' / name
    prior = previous[name]
    assert prior['baseline']['input_sha256'] == entry['sha256'] and prior['baseline']['processing_version'] == 'rawtherapee-recipes-v5'
    original = np.asarray(Image.open(cached/'original.png'), dtype=np.uint8).copy()
    baseline = np.asarray(Image.open(cached/'baseline.png'), dtype=np.uint8).copy()
    assert hashlib.sha256(baseline).hexdigest() == prior['candidates'][0]['tuning']['input_pixels_sha256']
    assert [original.shape[1],original.shape[0]] == prior['baseline']['output_size'] and baseline.shape == original.shape
    if path.suffix.lower() != '.dng':
        decoded, _ = decode_jpeg(path); np.testing.assert_array_equal(original, decoded); del decoded
    metadata, adjusted = adjust_rgb(original, model)
    before, after, export = render_photo({'original':original,'adjusted':adjusted})
    export['semantics'] = 'Frozen pretrained LUT result blended with original sRGB baseline.'
    directory = OUTPUT/name; directory.mkdir(exist_ok=True); (directory/'lut100.png').write_bytes(after)
    panels = []; h,w = original.shape[:2]
    for label,pixels in [('Original / neutral RAW',original),('Current v5 shadows',baseline),('Pretrained LUT / 100%',adjusted)]:
        image = Image.fromarray(pixels); image.thumbnail((400,280),Image.Resampling.LANCZOS); panels.append((label,image))
    canvas = Image.new('RGB',(1200,330),'#eeeeee'); draw = ImageDraw.Draw(canvas)
    for col,(label,image) in enumerate(panels):
        draw.text((col*400+5,5),name,fill='black');draw.text((col*400+5,22),label,fill='black');canvas.paste(image,(col*400,46))
    canvas.save(directory/'overview.png'); sheets.append(canvas)
    boxes = ([(1120,2380),(1665,2300),(w//2,h//2)] if name=='a2132-IMG_4947' else
             [(max(0,w//2-210),max(0,h//2-140)),(max(0,w//3-210),max(0,h*4//5-140)),(max(0,w//2-210),max(0,h//8-140))])
    detail = Image.new('RGB',(1290,930),'#eeeeee'); draw = ImageDraw.Draw(detail)
    for row,(x,y) in enumerate(boxes):
        x=min(x,w-420);y=min(y,h-280)
        for col,(label,pixels) in enumerate([('Original',original),('Current v5',baseline),('LUT100',adjusted)]):
            draw.text((col*430+5,row*310+3),f'{label} x={x} y={y}',fill='black')
            detail.paste(Image.fromarray(pixels[y:y+280,x:x+420]),(col*430+5,row*310+25))
    detail.save(directory/'native.png')
    records.append({'id':name,'source':entry,'baseline':prior['baseline'],'lut':metadata,'export':export,
                    'png_bytes':len(after),'source_preserved':hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],
                    'total_seconds':time.perf_counter()-started})
    (OUTPUT/'report.json').write_text(json.dumps({'scope':frozen['scope'],'comparison':'Original/currentv5/frozen paired sRGB LUT100; known-inspected diagnostics, no preference score.','cases':records},indent=2)+'\n')
    print(name,round(metadata['seconds'],2),'LUT s',round(records[-1]['total_seconds'],2),'total s',
          round(metadata['new_full_channel_fraction']*100,3),'%new full',round(metadata['new_black_pixel_fraction']*100,3),'%new black',flush=True)
    del original,baseline,adjusted,before,after
for start in (0,5):
    canvas=Image.new('RGB',(1200,1650),'#eeeeee')
    for row,sheet in enumerate(sheets[start:start+5]):canvas.paste(sheet,(0,row*330))
    canvas.save(OUTPUT/f'overview-{start//5+1}.png')
print('Completed10cases',flush=True)
