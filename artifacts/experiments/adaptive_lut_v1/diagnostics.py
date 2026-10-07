"""Retain neutral/color/edge/grain controls and independent50% photo blends."""
import ast
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from src import adaptive_lut as engine
from src.photo_engine import render_photo

OUTPUT = ROOT/'data/user_photo_diagnostics/lut-v1'
torch.set_num_threads(1)
model = engine.load_model()
source = (engine.BUNDLE/'models.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == model[2]['files']['models.py']['sha256']
nodes = [node for node in ast.parse(source).body if getattr(node,'name','') in ('discriminator_block','Classifier')]
scope = {'nn': nn}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen-author-classifier','exec'),scope)
author = scope['Classifier']().eval(); author.load_state_dict(model[0].state_dict(),strict=True)
records = []; rng = np.random.default_rng(91)
for shape in ((1,1,3),(13,17,3),(257,256,3),(1023,801,3)):
    rgb = rng.integers(0,256,shape,dtype=np.uint8)
    full = torch.from_numpy(rgb).permute(2,0,1)[None].contiguous().float()/255
    expected = F.interpolate(full,(256,256),mode='bilinear',align_corners=False)
    actual = engine.predictor_input(rgb)
    with torch.inference_mode():
        difference = float((model[0](actual)-author(full)).abs().max())
        assert torch.equal(model[0](expected),author(expected)) and difference <= 2e-6
    records.append({'shape':list(shape),'resize_max_abs':float((actual-expected).abs().max()),
                    'weights_max_abs':difference,'architecture_exact_same_input':True})
(Path(__file__).parent/'predictor-parity-final.json').write_text(json.dumps(records,indent=2)+'\n')

ramp = np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3))
patches = np.zeros((256,256,3),np.uint8)
colors = [[0,0,0],[255,255,255],[128,128,128],[32,32,32],
          [192,144,112],[230,195,170],[138,97,78],[90,120,80],
          [70,115,150],[180,155,90],[180,185,190],[220,220,210],
          [255,0,0],[0,255,0],[0,0,255],[30,45,50]]
for index,color in enumerate(colors): patches[index//4*64:index//4*64+64,index%4*64:index%4*64+64] = color
edge = np.where(np.indices((256,256))[1][...,None] <128,40,180).repeat(3,axis=2).astype(np.uint8)
edge[128:] = np.clip(edge[128:].astype(np.int16)+np.random.default_rng(20).integers(-3,4,(128,256,1)),0,255)
controls = [('Neutral ramp',ramp),('Color / skin-tone patches',patches),('Edge / subtle grain',edge)]
canvas = Image.new('RGB',(792,870),'#eeeeee'); draw = ImageDraw.Draw(canvas); reports = []
for row,(label,rgb) in enumerate(controls):
    metadata,adjusted = engine.adjust_rgb(rgb,model)
    before,after,export = render_photo({'original':rgb,'adjusted':adjusted},.5)
    half = np.asarray(Image.open(io.BytesIO(after))).copy()
    expected = np.rint(rgb.astype(np.float32)*.5+adjusted.astype(np.float32)*.5).astype(np.uint8)
    np.testing.assert_array_equal(half,expected)
    folder = OUTPUT/('synthetic-'+str(row)); folder.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(folder/'original.png')
    (folder/'half.png').write_bytes(after)
    _,full,full_export = render_photo({'original':rgb,'adjusted':adjusted})
    (folder/'full.png').write_bytes(full)
    details = {'name':label,'lut':metadata,'half_export':export,'full_export':full_export}
    if row==0:
        details.update(gray_ramp_channels=adjusted[0].tolist(),
                       decreasing_channel_steps=np.sum(np.diff(adjusted[0].astype(np.int16),axis=0)<0,axis=0).tolist(),
                       maximum_neutral_channel_spread=int(np.ptp(adjusted.astype(np.int16),axis=2).max()))
    if row==1: details.update(input_patches=colors,output_patches=[adjusted[i//4*64,i%4*64].tolist() for i in range(16)])
    if row==2:
        details['grain_std_channels']={key:pixels[160:230,20:100].std(axis=(0,1)).tolist() for key,pixels in [('original',rgb),('half',half),('full',adjusted)]}
    reports.append(details)
    for col,(heading,pixels) in enumerate([('Original',rgb),('LUT50%',half),('LUT100%',adjusted)]):
        draw.text((col*264+4,row*290+3),label+' / '+heading,fill='black')
        canvas.paste(Image.fromarray(pixels),(col*264+4,row*290+25))
canvas.save(OUTPUT/'synthetic-controls.png')
(OUTPUT/'synthetic-report.json').write_text(json.dumps(reports,indent=2)+'\n')

photo_reports = []; canvas = Image.new('RGB',(1200,990),'#eeeeee'); draw = ImageDraw.Draw(canvas)
frozen = json.loads((Path(__file__).parent/'inputs.json').read_text())
for row,entry in enumerate(frozen['cases'][:3]):
    name = entry['id']; folder = OUTPUT/name
    original = np.asarray(Image.open(ROOT/'data/user_photo_diagnostics/finetune-v7'/name/'original.png')).copy()
    adjusted = np.asarray(Image.open(folder/'lut100.png')).copy()
    before,after,export = render_photo({'original':original,'adjusted':adjusted},.5)
    with Image.open(io.BytesIO(after)) as image: half = np.asarray(image).copy()
    (folder/'lut50.png').write_bytes(after)
    photo_reports.append({'id':name,'export':export,'new_full_channel_fraction':float((np.any(half==255,axis=-1)&~np.any(original==255,axis=-1)).mean()),
                          'new_black_pixel_fraction':float((np.all(half==0,axis=-1)&~np.all(original==0,axis=-1)).mean())})
    for col,(heading,pixels) in enumerate([('Original',original),('LUT50%',half),('LUT100%',adjusted)]):
        image = Image.fromarray(pixels); image.thumbnail((400,280),Image.Resampling.LANCZOS)
        draw.text((col*400+5,row*330+5),name,fill='black'); draw.text((col*400+5,row*330+22),heading,fill='black')
        canvas.paste(image,(col*400,row*330+46))
    del original,adjusted,half,before,after
canvas.save(OUTPUT/'user-photos-strength-comparison.png')
(OUTPUT/'half-report.json').write_text(json.dumps(photo_reports,indent=2)+'\n')
print('Final predictor parity, three synthetic controls and three independent half blends passed')
