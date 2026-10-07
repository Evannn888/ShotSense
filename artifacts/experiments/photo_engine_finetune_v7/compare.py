"""Frozen ten-scene native comparison; no independent aesthetic score."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.photo_engine import fine_tune_photo,process_photo,render_photo

FOLDER=Path(__file__).resolve().parent
OUTPUT=ROOT/'data/user_photo_diagnostics/finetune-v7'
OUTPUT.mkdir(parents=True,exist_ok=True)
frozen=json.loads((FOLDER/'inputs.json').read_text())
records=[]; sheets=[]
for index,entry in enumerate(frozen['cases']):
    started=time.perf_counter(); path=Path(entry['path']); name=entry['id']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    directory=OUTPUT/name; directory.mkdir(exist_ok=True)
    if index<2:
        cached=ROOT/'data/user_photo_diagnostics/highres-v6'/('ahmet-natural' if index==0 else 'martin-natural')
        worker=json.loads((cached/'report.json').read_text())
        assert worker['payload']['input_sha256']==entry['sha256']
        arrays={}
        for key,file in [('original','before.png'),('adjusted','after.png')]:
            with Image.open(cached/file) as image: arrays[key]=np.asarray(image,dtype=np.uint8).copy()
        baseline_metadata=worker['payload']
    else:
        baseline_metadata,arrays=process_photo(path,'shadows')
    panels=[]; measures=[]
    for label,pixels in [('Original / neutral RAW',arrays['original']),('Current v5 shadows',arrays['adjusted'])]:
        Image.fromarray(pixels).save(directory/('original.png' if label.startswith('Original') else 'baseline.png'))
        image=Image.fromarray(pixels); image.thumbnail((480,340),Image.Resampling.LANCZOS); panels.append((label,image))
    settings_list=[('Mild +4 vibrance / +6 contrast',{'vibrance':4,'local_contrast':6})]
    if index<3: settings_list.append(('Restrained -4 vibrance / +6 contrast',{'vibrance':-4,'local_contrast':6}))
    for label,settings in settings_list:
        metadata,pixels=fine_tune_photo(arrays['adjusted'],settings)
        filename='mild.png' if settings['vibrance']>0 else 'restrained.png'
        before_png,after_png,export=render_photo({'original':arrays['original'],'adjusted':pixels})
        assert export['size']==baseline_metadata['output_size']
        (directory/filename).write_bytes(after_png)
        new=0
        for row in range(0,len(pixels),128):
            new+=int((np.any(pixels[row:row+128]==255,axis=-1)&~np.any(arrays['adjusted'][row:row+128]==255,axis=-1)).sum())
        measures.append({'label':label,'settings':metadata['settings'],'tuning':metadata,'export':export,
                         'new_full_channel_fraction_vs_current':new/(pixels.shape[0]*pixels.shape[1])})
        image=Image.fromarray(pixels); image.thumbnail((480,340),Image.Resampling.LANCZOS); panels.append((label,image))
        # Three native locations; portrait retains previously inspected eye coordinates.
        h,w=pixels.shape[:2]
        boxes=([(1120,2380),(1665,2300),(w//2,h//2)] if name=='a2132-IMG_4947' else
               [(max(0,w//2-210),max(0,h//2-140)),(max(0,w//3-210),max(0,h*4//5-140)),(max(0,w//2-210),max(0,h//8-140))])
        detail=Image.new('RGB',(880,3*310),'#eeeeee'); draw=ImageDraw.Draw(detail)
        for row,(x,y) in enumerate(boxes):
            x=min(x,w-420);y=min(y,h-280)
            draw.text((10,row*310+3),f'Current v5 / x={x}, y={y}',fill='black')
            draw.text((450,row*310+3),label,fill='black')
            detail.paste(Image.fromarray(arrays['adjusted'][y:y+280,x:x+420]),(10,row*310+25))
            detail.paste(Image.fromarray(pixels[y:y+280,x:x+420]),(450,row*310+25))
        detail.save(directory/(filename.replace('.png','-native.png')))
        del pixels,before_png,after_png
    canvas=Image.new('RGB',(480*len(panels),390),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for column,(label,image) in enumerate(panels):
        draw.text((column*480+5,5),name,fill='black'); draw.text((column*480+5,23),label,fill='black')
        canvas.paste(image,(column*480,46))
    canvas.save(directory/'overview.png');sheets.append(canvas)
    records.append({'id':name,'source':entry,'baseline':baseline_metadata,'candidates':measures,'seconds':time.perf_counter()-started})
    (OUTPUT/'report.json').write_text(json.dumps({'scope':frozen['scope'],'cases':records},indent=2)+'\n')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    print(name,round(records[-1]['seconds'],2),flush=True)
    del arrays
for start in (0,5):
    selected=sheets[start:start+5];canvas=Image.new('RGB',(1920,len(selected)*390),'#eeeeee')
    for row,sheet in enumerate(selected):canvas.paste(sheet,(0,row*390))
    canvas.save(OUTPUT/f'overview-{start//5+1}.png')
print('Completed10cases',flush=True)
