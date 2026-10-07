"""Frozen native-engine shadow diagnostics; screenshot proxy is not the original JPEG."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageCms
from src import photo_engine as engine

OUT=Path(__file__).resolve().parent
LOCAL=ROOT/'data/user_photo_diagnostics/shadow-v2'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def luma(rgb): return rgb.astype(np.float32)@np.array([.2126,.7152,.0722],dtype=np.float32)/255
def region(rgb,box):
    h,w=rgb.shape[:2];x0,y0,x1,y1=box
    return rgb[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]

if (OUT/'manifest.json').exists(): raise ValueError('Frozen comparison already exists')
proxy=LOCAL/'screenshot-left-proxy.jpg'
icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
Image.open(LOCAL/'screenshot-left-proxy.png').convert('RGB').save(proxy,quality=100,subsampling=0,icc_profile=icc)
photos=[('screenshot-proxy',proxy)]
photos += [('LOL-'+name,ROOT/'data/external/lol_pilot'/(name+'.jpg')) for name in ('27','102','254')]
photos += [(name,ROOT/'data/raw/dngs'/name) for name in ('a0575-kme_040.dng','a2132-IMG_4947.dng','a2270-_DSC0033.dng')]
fixture=np.zeros((256,512,3),dtype=np.uint8)
fixture[:128]=[180,185,190]
fixture[128:]=[30,35,25];fixture[144:224:8,32:480:8]=[40,45,35]
for name,pixels in [('daylight',np.full((128,256,3),[190,180,160],dtype=np.uint8)),('black',np.zeros((128,256,3),dtype=np.uint8)),
                    ('gray-ramp',np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],128,axis=0).repeat(3,axis=2)),('shadow-sky-step',fixture)]:
    path=LOCAL/(name+'.jpg');Image.fromarray(pixels).save(path,quality=100,subsampling=0,icc_profile=icc);photos.append(('synthetic-'+name,path))
profiles={'gentle':ROOT/'artifacts/photo_engine/natural.pp3','shadows':OUT/'candidate-shadows.pp3','equalizer':OUT/'candidate-equalizer.pp3'}
provenance=json.loads((OUT/'proxy-provenance.json').read_text())
manifest={'scope':'Screenshot-derived JPEG proxy plus previously inspected public regressions and synthetic controls; no original-file or independent preference acceptance',
          'engine':engine.engine_identity(),'profiles':{name:{'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for name,path in profiles.items()},
          'photos':[{'id':name,'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for name,path in photos],
          'proxy_regions':provenance['regions_normalized_xyxy'],'source_sha256':sha(Path(__file__))}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records=[];sheets=[];crops=[]
bundle=engine.BUNDLE
with tempfile.TemporaryDirectory(prefix='shotsense-shadow-comparison-') as directory:
    temporary=Path(directory)
    shutil.copyfile(bundle/'manifest.json',temporary/'manifest.json');shutil.copyfile(bundle/'neutral.pp3',temporary/'neutral.pp3')
    engine.BUNDLE=temporary
    for name,path in photos:
        variants={};measures={};original=None
        for candidate,profile in profiles.items():
            shutil.copyfile(profile,temporary/'natural.pp3')
            result,source=engine.process_photo(path,'natural')
            before,after,metadata=engine.render_photo(source)
            if original is None: original=source['original'].copy();variants['Original']=Image.fromarray(original)
            assert np.array_equal(original,source['original']) and engine.render_photo(source,0)[0]==engine.render_photo(source,0)[1]
            variants[candidate]=Image.open(io.BytesIO(after)).copy()
            before_luma=luma(original);after_luma=luma(source['adjusted'])
            measures[candidate]={'result':result,'export':metadata,'mean_absolute_srgb_change':float(np.abs(source['adjusted'].astype(np.float32)-original).mean()/255),
                                'median_luma_before':float(np.median(before_luma)),'median_luma_after':float(np.median(after_luma))}
            if name=='screenshot-proxy':
                measures[candidate]['regions']={key:{'mean_before':float(luma(region(original,box)).mean()),'mean_after':float(luma(region(source['adjusted'],box)).mean())} for key,box in manifest['proxy_regions'].items()}
                (OUT/('proxy-'+candidate+'.png')).write_bytes(after)
            if name=='a2132-IMG_4947.dng':
                variants[candidate+' eye']=variants[candidate].crop((1120,2420,1450,2590))
        for panels,native in ((sheets,False),(crops,True)):
            row=Image.new('RGB',(1200,244),'white');draw=ImageDraw.Draw(row)
            for col,key in enumerate(('Original','gentle','shadows','equalizer')):
                image=variants[key].copy();w,h=image.size
                if native:
                    box=(1120,2420,1450,2590) if name=='a2132-IMG_4947.dng' else (max(0,w//2-150),max(0,h//2-100),min(w,w//2+150),min(h,h//2+100))
                    image=image.crop(box)
                else:image.thumbnail((300,200))
                row.paste(image,(col*300,40));draw.text((col*300+4,3),name[:34],fill='black');draw.text((col*300+4,21),key,fill='black')
            panels.append(row)
        if name=='screenshot-proxy':
            (OUT/'proxy-original.png').write_bytes(engine.render_photo({'original':original,'adjusted':original})[0])
            for candidate in ('shadows','equalizer'):
                panel=Image.new('RGB',(1160,488),'white');draw=ImageDraw.Draw(panel)
                panel.paste(variants['Original'],(0,28));panel.paste(variants[candidate],(580,28))
                draw.text((8,6),'Screenshot proxy / original',fill='black');draw.text((588,6),'Screenshot proxy / '+candidate,fill='black');panel.save(OUT/('proxy-'+candidate+'-comparison.png'))
        records.append({'id':name,'size':[original.shape[1],original.shape[0]],'candidates':measures})
        assert sha(path)==next(row['sha256'] for row in manifest['photos'] if row['id']==name)
        print(name,{key:round(v['mean_absolute_srgb_change'],4) for key,v in measures.items()},flush=True)
for prefix,panels in [('overview',sheets),('native-crops',crops)]:
    for start in range(0,len(panels),4):
        sheet=Image.new('RGB',(1200,244*len(panels[start:start+4])),'white')
        for row,panel in enumerate(panels[start:start+4]):sheet.paste(panel,(0,row*244))
        sheet.save(OUT/(prefix+'-'+str(start//4+1)+'.jpg'),quality=95)
(OUT/'report.json').write_text(json.dumps({'manifest':manifest,'cases':records},indent=2)+'\n')
