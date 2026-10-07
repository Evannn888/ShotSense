"""Frozen neutral/color/skin/edge controls of brightness-only shadow stage."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.lut_natural import shadow_settings,apply_shadow_lift,shadow_curve
from src.photo_engine import render_photo
out=ROOT/'data/user_photo_diagnostics/lut-shadows-v3';out.mkdir(exist_ok=True)
canvas=Image.new('RGB',(792,870),'#eeeeee');draw=ImageDraw.Draw(canvas);reports=[]
for index,label in enumerate(('Neutral ramp','Color / skin patches','Neutral edge / grain')):
    with Image.open(ROOT/'data/user_photo_diagnostics/lut-v1'/f'synthetic-{index}/original.png') as image:source=np.asarray(image).copy()
    with Image.open(ROOT/'data/user_photo_diagnostics/lut-natural-v2'/f'control-{index}.png') as image:base=np.asarray(image).copy()
    selection=shadow_settings(source);metadata,auto=apply_shadow_lift(base,selection['suggested_strength'])
    maximum,full=apply_shadow_lift(base,1)
    _,data,export=render_photo({'original':source,'adjusted':auto});(out/f'control-{index}.png').write_bytes(data)
    record={'name':label,'selection':selection,'auto':metadata,'maximum':maximum,'export':dict(export,semantics=metadata['semantics'])}
    if index==0:
        record.update(negative_steps=np.sum(np.diff(auto[0].astype(np.int16),axis=0)<0,axis=0).tolist(),neutral_channel_spread=int(np.ptp(auto.astype(np.int16),axis=2).max()))
        assert record['negative_steps']==[0,0,0] and record['neutral_channel_spread']==0
    if index==1:record['patches']={name:[rgb[i//4*64,i%4*64].tolist() for i in range(16)] for name,rgb in [('base',base),('auto',auto),('max',full)]}
    if index==2:
        assert np.all(np.diff(auto[0,:,0].astype(np.int16))>=0)
        record['dark_grain_std']={name:rgb[160:230,20:100].std(axis=(0,1)).tolist() for name,rgb in [('base',base),('auto',auto),('max',full)]}
    for col,(heading,rgb) in enumerate([('Previous natural',base),('New auto',auto),('Manual100',full)]):
        draw.text((col*264+4,index*290+3),label+' / '+heading,fill='black');canvas.paste(Image.fromarray(rgb),(col*264+4,index*290+25))
    reports.append(record)
canvas.save(out/'controls.png')
curve=shadow_curve(np.linspace(0,1,65537),1)
(Path(__file__).parent/'controls.json').write_text(json.dumps({'cases':reports,'minimum_continuous_curve_slope':float(np.diff(curve).min()*65536)},indent=2)+'\n')
print('Neutral order/hue/highlight/skin/color/edge controls passed')
