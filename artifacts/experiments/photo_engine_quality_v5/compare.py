"""Native source-only diagnostics; no detail recovery or independent acceptance."""
import hashlib,io,json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src import photo_engine as engine
OUT=Path(__file__).parent;LOCAL=ROOT/'data/user_photo_diagnostics/quality-v5'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (OUT/'report.json').exists():raise ValueError('Comparison already exists')
inputs=json.loads((OUT/'inputs.json').read_text())
photos=[(r['id'],ROOT/r['copy_path']) for r in inputs]
old=json.loads((ROOT/'artifacts/experiments/photo_engine_shadow_v2/manifest.json').read_text())
photos += [(r['id'],ROOT/r['path']) for r in old['photos'] if r['id'] in ('synthetic-black','synthetic-gray-ramp','synthetic-shadow-sky-step','a2132-IMG_4947.dng')]
rng=np.random.default_rng(20261004);noise=np.repeat(np.clip(rng.normal(12,1.2,(192,256)),0,255).round().astype(np.uint8)[:,:,None],3,axis=2)
noise_path=LOCAL/'dark-grain.jpg';Image.fromarray(noise).save(noise_path,quality=95,subsampling=0);photos.append(('synthetic-dark-grain',noise_path))
manifest={'scope':'Three actual small user JPEGs plus previously inspected portrait/synthetic regressions; source-only no reference target', 'engine':engine.engine_identity(), 'profiles':{p:sha(OUT/p) for p in ('shadows-before.pp3','candidate-shadows.pp3')},'inputs':[{'id':name,'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for name,path in photos]}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records=[]
with tempfile.TemporaryDirectory() as directory:
 bundle=Path(directory)
 for p in ('neutral.pp3','manifest.json'):shutil.copyfile(engine.BUNDLE/p,bundle/p)
 engine.BUNDLE=bundle
 for name,path in photos:
  variants={};measures={};original=None
  for label,profile,strength in [('Previous100','shadows-before.pp3',1),('Revised100','candidate-shadows.pp3',1)]:
   shutil.copyfile(OUT/profile,bundle/'shadows.pp3')
   result,source=engine.process_photo(path,'shadows');before,after,meta=engine.render_photo(source,strength)
   if original is None:original=source['original'];variants['Original']=Image.fromarray(original)
   np.testing.assert_array_equal(original,source['original']);assert engine.render_photo(source,0)[0]==engine.render_photo(source,0)[1]
   variants[label]=Image.open(io.BytesIO(after)).copy()
   if label=='Previous100':
    _,half,_=engine.render_photo(source,.5);variants['Previous50']=Image.open(io.BytesIO(half)).copy()
   px=source['adjusted'].astype(np.float32);inp=original.astype(np.float32);luma=px@np.array([.2126,.7152,.0722]);lin=inp@np.array([.2126,.7152,.0722])
   dark=lin<40
   measures[label]={'processing':result,'export':meta,'dark_mean_srgb':float(luma[dark].mean()) if dark.any() else None,'dark_display_std':float(luma[dark].std()) if dark.any() else None,'dark_p05_srgb':float(np.quantile(luma[dark],.05)) if dark.any() else None,'sky_delta_srgb':float((px[:min(120,len(px))]-inp[:min(120,len(px))]).mean()),'new_full_channel_fraction':float((np.any(px==255,axis=-1)&~np.any(inp==255,axis=-1)).mean())}
   if name in ('lake','dusk','rocks'):(LOCAL/(name+'-'+label+'.png')).write_bytes(after)
   if name=='lake':
    # Fixed regions correspond to the tree line/water and upper sky; not semantic masks.
    for key,box in {'forest':[20,195,540,260],'water':[10,280,545,440],'sky':[10,10,530,130]}.items():
     x0,y0,x1,y1=box;a=px[y0:y1,x0:x1];b=inp[y0:y1,x0:x1]
     measures[label][key]={'mean_rgb_before':b.mean(axis=(0,1)).tolist(),'mean_rgb_after':a.mean(axis=(0,1)).tolist(),'p05_before':float(np.quantile(b,.05)),'p05_after':float(np.quantile(a,.05))}
  if name in ('lake','dusk','rocks'):
   w,h=variants['Original'].size
   canvas=Image.new('RGB',(w*4,h+30),'white');draw=ImageDraw.Draw(canvas)
   for i,key in enumerate(('Original','Previous100','Previous50','Revised100')):canvas.paste(variants[key],(w*i,30));draw.text((w*i+5,6),key+' / native '+str(w)+'x'+str(h),fill='black')
   canvas.save(LOCAL/(name+'-comparison.png'))
  if name=='a2132-IMG_4947.dng':
   canvas=Image.new('RGB',(330*3,200),'white');draw=ImageDraw.Draw(canvas)
   for i,key in enumerate(('Original','Previous100','Revised100')):canvas.paste(variants[key].crop((1120,2420,1450,2590)),(i*330,30));draw.text((i*330+5,6),key+' / native eye',fill='black')
   canvas.save(LOCAL/'portrait-eye.png')
  if name.startswith('synthetic'):
   w,h=variants['Original'].size;canvas=Image.new('RGB',(w*3,h+30),'white');draw=ImageDraw.Draw(canvas)
   for i,key in enumerate(('Original','Previous100','Revised100')):canvas.paste(variants[key],(i*w,30));draw.text((i*w+5,6),key,fill='black')
   canvas.save(LOCAL/(name+'-comparison.png'))
  assert sha(path)==next(r['sha256'] for r in manifest['inputs'] if r['id']==name)
  records.append({'id':name,'size':list(variants['Original'].size),'candidates':measures});print(name,{k:(v['dark_mean_srgb'],v['dark_p05_srgb']) for k,v in measures.items()},flush=True)
(OUT/'report.json').write_text(json.dumps({'manifest':manifest,'cases':records},indent=2)+'\n')
