"""Reuse the frozen v1 neutral/color/edge/grain control pixels and predictions."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.lut_natural import protect_rgb
from src.photo_engine import render_photo
out=ROOT/'data/user_photo_diagnostics/lut-natural-v2'
reports=[];canvas=Image.new('RGB',(792,870),'#eeeeee');draw=ImageDraw.Draw(canvas)
for index,name in enumerate(('Neutral ramp','Color / skin patches','Edge / subtle grain')):
    folder=ROOT/'data/user_photo_diagnostics/lut-v1'/f'synthetic-{index}'
    with Image.open(folder/'original.png') as image: original=np.asarray(image).copy()
    with Image.open(folder/'full.png') as image: raw=np.asarray(image).copy()
    metadata,adjusted=protect_rgb(original,raw)
    _,png,export=render_photo({'original':original,'adjusted':adjusted})
    (out/f'control-{index}.png').write_bytes(png)
    record={'name':name,'protection':metadata,'export':export}
    if index==0:
        record.update(decreasing_channel_steps=np.sum(np.diff(adjusted[0].astype(np.int16),axis=0)<0,axis=0).tolist(),maximum_neutral_channel_spread=int(np.ptp(adjusted.astype(np.int16),axis=2).max()))
        assert record['decreasing_channel_steps']==[0,0,0] and record['maximum_neutral_channel_spread']==0
    if index==1:
        record['patches']={label:[pixels[i//4*64,i%4*64].tolist() for i in range(16)] for label,pixels in [('original',original),('raw',raw),('natural',adjusted)]}
    if index==2:
        record['grain_std_channels']={label:pixels[160:230,20:100].std(axis=(0,1)).tolist() for label,pixels in [('original',original),('raw',raw),('natural',adjusted)]}
        # Pointwise color/tone processing cannot introduce spatial overshoot around this neutral step.
        assert np.all(np.diff(adjusted[0,:,0].astype(np.int16))>=0)
    for col,(label,pixels) in enumerate([('Original',original),('Author100',raw),('Natural',adjusted)]):
        draw.text((col*264+4,index*290+3),name+' / '+label,fill='black')
        canvas.paste(Image.fromarray(pixels),(col*264+4,index*290+25))
    reports.append(record)
canvas.save(out/'controls.png')
(Path(__file__).parent/'controls.json').write_text(json.dumps(reports,indent=2)+'\n')
print('Neutral/color/skin/edge/grain controls passed')
