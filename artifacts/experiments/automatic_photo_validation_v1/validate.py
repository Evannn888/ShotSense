"""Frozen actual upload/default UI and native photo diagnostics; no tuning."""
import base64
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import streamlit as st
from streamlit.testing.v1 import AppTest

FOLDER=Path(__file__).parent
OUTPUT=ROOT/'data/user_photo_diagnostics/automatic-validation-v1'
OUTPUT.mkdir(exist_ok=True)
if (OUTPUT/'results.json').exists():raise ValueError('Preserve existing validation; use another run folder')
CASES=json.loads((FOLDER/'inputs.json').read_text())['cases']
FONT=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
LUMA=np.array([.2126,.7152,.0722],dtype=np.float32)
records=[];sheets=[];downloads={};commands=[];holder=[None]
original_download=st.download_button;original_popen=subprocess.Popen


def capture(label,data,*args,**kwargs):
    downloads[label]=data
    return original_download(label,data,*args,**kwargs)


def spawn(command,**kwargs):
    if any(module in command for module in ('src.lut_natural','src.photo_engine')):
        commands.append(command)
    return original_popen(command,**kwargs)


def pixels_equal(png,expected):
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode=='RGB' and image.format=='PNG' and 'srgb' in image.info
        assert image.size==(expected.shape[1],expected.shape[0])
        actual=np.asarray(image)
        for row in range(0,len(expected),128):
            np.testing.assert_array_equal(actual[row:row+128],expected[row:row+128])


def diagnostic_images(name,before,after):
    originals=[Image.open(io.BytesIO(data)).copy() for data in (before,after)]
    overview=Image.new('RGB',(1280,455),'#eeeeee');draw=ImageDraw.Draw(overview)
    draw.text((12,8),name,fill='black',font=FONT)
    for col,(label,image) in enumerate(zip(('Original / neutral RAW','Default automatic'),originals)):
        draw.text((col*640+12,33),label,fill='black',font=FONT)
        thumbnail=image.copy();thumbnail.thumbnail((620,385),Image.Resampling.LANCZOS)
        overview.paste(thumbnail,(col*640+10,65))
    directory=OUTPUT/name;directory.mkdir(exist_ok=True)
    overview.save(directory/'overview.png');sheets.append(overview)
    w,h=originals[0].size;cw,ch=min(420,w),min(280,h)
    boxes=([(1120,2420),(1665,2300),(1380,2100)] if name=='a2132-IMG_4947' else
           [(max(0,(w-cw)//2),max(0,(h-ch)//2)),
            (max(0,w//3-cw//2),max(0,h*4//5-ch//2)),
            (max(0,(w-cw)//2),max(0,h//8-ch//2))])
    native=Image.new('RGB',(900,990),'#eeeeee');draw=ImageDraw.Draw(native)
    for row,(x,y) in enumerate(boxes):
        x=min(x,w-cw);y=min(y,h-ch)
        for col,(label,image) in enumerate(zip(('Original','Automatic'),originals)):
            draw.text((col*450+10,row*330+8),f'{label} x={x} y={y}',fill='black',font=FONT)
            native.paste(image.crop((x,y,x+cw,y+ch)),(col*450+10,row*330+40))
    native.save(directory/'native.png')
    del originals


with patch.object(st,'file_uploader',lambda *a,**k:holder[0]),patch.object(st,'download_button',capture),patch.object(subprocess,'Popen',spawn):
    for entry in CASES:
        name=entry['id'];path=Path(entry['path']);suffix=path.suffix.lower()
        directory=OUTPUT/name;directory.mkdir(exist_ok=True)
        record={'id':name,'source':entry,'backend_expected':'conservative-native-raw' if suffix=='.dng' else 'protected-natural-color'}
        try:
            data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==entry['sha256']
            holder[0]=None;downloads.clear()
            at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
            assert not at.exception and at.radio[0].value=='Automatic (recommended)'
            class Upload:
                name=path.name
                def getvalue(self):return data
            holder[0]=Upload();count=len(commands);started=time.perf_counter()
            at.run(timeout=120)
            record['upload_to_page_seconds']=time.perf_counter()-started
            assert not at.exception and not at.error, str(at.exception)+str(at.error)
            assert len(commands)==count+1
            result,pixels=at.session_state['prediction']
            assert result['automatic_workflow']['backend']==record['backend_expected']
            assert result['input_sha256']==entry['sha256'] and result['input_name']==path.name
            assert next(s for s in at.slider if s.label=='Adjustment strength').value==100
            if suffix!='.dng':
                assert next(s for s in at.slider if s.label=='Shadow lift').value==round(result['shadow_adjustment']['selection']['suggested_strength']*100)
            before=base64.b64decode(at.get('imgs')[0].proto.imgs[0].url.split(',',1)[1])
            after=base64.b64decode(at.get('imgs')[1].proto.imgs[0].url.split(',',1)[1])
            pixels_equal(before,pixels['original']);pixels_equal(after,pixels['adjusted'])
            assert result['output_size']==[pixels['original'].shape[1],pixels['original'].shape[0]]
            started=time.perf_counter()
            next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=120)
            record['prepare_page_seconds']=time.perf_counter()-started
            assert not at.exception and not at.error and len(commands)==count+1
            assert downloads['Download full-resolution PNG']==after
            metadata=json.loads(downloads['Download adjustment JSON'])
            assert metadata['export']['strength']==1 and metadata['export']['size']==result['output_size']
            assert metadata['export']['output_png_sha256']==hashlib.sha256(after).hexdigest()
            assert metadata['input_sha256']==entry['sha256'] and metadata['automatic_workflow']==result['automatic_workflow']
            (directory/'original.png').write_bytes(before);(directory/'automatic.png').write_bytes(after)
            (directory/'export.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
            new_black=new_full=0
            for row in range(0,len(pixels['original']),128):
                a=pixels['original'][row:row+128];b=pixels['adjusted'][row:row+128]
                new_black+=int((np.all(b==0,axis=-1)&~np.all(a==0,axis=-1)).sum())
                new_full+=int((np.any(b==255,axis=-1)&~np.any(a==255,axis=-1)).sum())
            count_pixels=pixels['original'].shape[0]*pixels['original'].shape[1]
            measures={}
            for label,rgb in pixels.items():
                if label not in ('original','adjusted'):continue
                h,w=rgb.shape[:2];sample=rgb[::max(1,(h+511)//512),::max(1,(w+511)//512)].astype(np.float32)
                y=sample@LUMA
                measures[label]={'display_luma_mean_codes':float(y.mean()),'q10':float(np.percentile(y,10)),
                    'q30':float(np.percentile(y,30)),'q90':float(np.percentile(y,90)),
                    'sample_size':[sample.shape[1],sample.shape[0]]}
            record.update(status='passed',size=result['output_size'],selected_recipe=result['selected_recipe'],
                worker_seconds=result['timing']['local_worker_seconds'],shadow_strength=result.get('shadow_adjustment',{}).get('strength'),
                native_png_bytes=len(after),new_black_pixel_fraction=new_black/count_pixels,new_full_channel_fraction=new_full/count_pixels,
                sampled_luminance=measures,source_preserved=hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],
                default_setting_used=True,actual_worker_calls=1,display_download_json_native_pixels_exact=True)
            assert record['source_preserved'] and new_black==0
            assert record['new_full_channel_fraction']<=.001 if suffix=='.dng' else new_full==0
            diagnostic_images(name,before,after)
            del at,result,pixels,before,after,data;downloads.clear();holder[0]=None;gc.collect()
        except Exception as error:
            record.update(status='failed',error=type(error).__name__+': '+str(error))
        records.append(record)
        (OUTPUT/'results.json').write_text(json.dumps({'scope':'Actual default upload-driven AppTest on17 previously inspected diagnostic sources; no independent preference acceptance.','cases':records},indent=2,allow_nan=False)+'\n')
        print(name,record['status'],record.get('size'),round(record.get('upload_to_page_seconds',0),2),'s',flush=True)
for start in range(0,len(sheets),4):
    chunk=sheets[start:start+4];sheet=Image.new('RGB',(1280,455*len(chunk)),'#eeeeee')
    for row,canvas in enumerate(chunk):sheet.paste(canvas,(0,row*455))
    sheet.save(OUTPUT/f'overview-{start//4+1}.png')
(FOLDER/'results.json').write_text((OUTPUT/'results.json').read_text())
print('Completed',len(records),'cases;',sum(r['status']=='passed' for r in records),'functional passes',flush=True)
sys.exit(0 if all(r['status']=='passed' for r in records) else 1)
