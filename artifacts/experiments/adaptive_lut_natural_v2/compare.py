"""Fixed twelve-scene protected-LUT review from verified frozen original/raw arrays."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.lut_natural import protect_rgb
from src.photo_engine import render_photo

FOLDER = Path(__file__).resolve().parent
OUTPUT = ROOT/'data/user_photo_diagnostics/lut-natural-v2'; OUTPUT.mkdir(exist_ok=True)
old = {case['id']:case for case in json.loads((FOLDER.parent/'adaptive_lut_v1/results.json').read_text())['cases']}
records = []; sheets = []
for entry in json.loads((FOLDER/'inputs.json').read_text())['cases']:
    started = time.perf_counter(); name = entry['id']
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    if name in old:
        original_path = ROOT/'data/user_photo_diagnostics/finetune-v7'/name/'original.png'
        raw_path = ROOT/'data/user_photo_diagnostics/lut-v1'/name/'lut100.png'
        expected_original = old[name]['input_pixels_sha256']; expected_raw = old[name]['output_pixels_sha256']
    else:
        sub = 'user_webp_1' if name=='user-webp-1' else 'user_jpeg_2'
        report = json.loads((FOLDER.parent/'adaptive_lut_v1'/sub/'report.json').read_text())
        paths = {item['label']:Path(item['path']) for item in report['exports']}
        original_path,raw_path = paths['original'],paths['lut100']
        expected_original = report['lut']['input_pixels_sha256']; expected_raw = report['lut']['output_pixels_sha256']
    with Image.open(original_path) as image: original = np.asarray(image).copy()
    with Image.open(raw_path) as image: raw = np.asarray(image).copy()
    assert hashlib.sha256(original).hexdigest()==expected_original and hashlib.sha256(raw).hexdigest()==expected_raw
    metadata,adjusted = protect_rgb(original,raw)
    before,after,export = render_photo({'original':original,'adjusted':adjusted})
    export['semantics'] = metadata['semantics']
    directory = OUTPUT/name; directory.mkdir(exist_ok=True); (directory/'natural.png').write_bytes(after)
    canvas = Image.new('RGB',(1200,330),'#eeeeee'); draw = ImageDraw.Draw(canvas)
    for col,(label,pixels) in enumerate([('Original',original),('Author LUT100',raw),('Protected natural',adjusted)]):
        image = Image.fromarray(pixels); image.thumbnail((400,280),Image.Resampling.LANCZOS)
        draw.text((col*400+5,5),name,fill='black'); draw.text((col*400+5,22),label,fill='black')
        canvas.paste(image,(col*400,46))
    canvas.save(directory/'overview.png'); sheets.append(canvas)
    h,w = original.shape[:2]; cw,ch = min(420,w),min(280,h)
    boxes = ([(1120,2380),(1665,2300),(w//2,h//2)] if name=='a2132-IMG_4947' else
             [(max(0,w//2-cw//2),max(0,h//2-ch//2)),(max(0,w//3-cw//2),max(0,h*4//5-ch//2)),(max(0,w//2-cw//2),max(0,h//8-ch//2))])
    detail = Image.new('RGB',(1290,930),'#eeeeee'); draw = ImageDraw.Draw(detail)
    for row,(x,y) in enumerate(boxes):
        x=min(x,w-cw); y=min(y,h-ch)
        for col,(label,pixels) in enumerate([('Original',original),('Author LUT100',raw),('Protected natural',adjusted)]):
            draw.text((col*430+5,row*310+3),f'{label} x={x} y={y}',fill='black')
            detail.paste(Image.fromarray(pixels[y:y+ch,x:x+cw]),(col*430+5,row*310+25))
    detail.save(directory/'native.png')
    records.append({'id':name,'source':entry,'natural':metadata,'export':export,'png_bytes':len(after),
        'original_png':str(original_path),'raw_png':str(raw_path),'source_preserved':True,'total_seconds':time.perf_counter()-started})
    (OUTPUT/'report.json').write_text(json.dumps({'scope':'Frozen inspected12-source review; no independent photographic quality or user preference claim.','cases':records},indent=2)+'\n')
    print(name,round(metadata['settings']['color_strength']*100,1),'%color',round(metadata['settings']['tone_lift'],3),'lift',
          round(metadata['minimum_shadow_luminance_delta_codes'],3),'min shadow codes',round(records[-1]['total_seconds'],2),'s',flush=True)
    del original,raw,adjusted,before,after
for start in (0,4,8):
    canvas = Image.new('RGB',(1200,1320),'#eeeeee')
    for row,sheet in enumerate(sheets[start:start+4]): canvas.paste(sheet,(0,row*330))
    canvas.save(OUTPUT/f'overview-{start//4+1}.png')
print('Completed12protected comparisons',flush=True)
