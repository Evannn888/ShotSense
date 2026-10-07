"""Frozen source/previous-natural/new-auto comparison; no pretrained inference repeated."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from src.lut_natural import shadow_settings,apply_shadow_lift
from src.photo_engine import render_photo
FOLDER=Path(__file__).resolve().parent
OUTPUT=ROOT/'data/user_photo_diagnostics/lut-shadows-v3';OUTPUT.mkdir(exist_ok=True)
old={row['id']:row for row in json.loads((FOLDER.parent/'adaptive_lut_natural_v2/results.json').read_text())['cases']}
entries=json.loads((FOLDER/'inputs.json').read_text())['cases']
priority=['user-cloudy-trees','nir-himi-1P1yaGS_Gek-unsplash','a2132-IMG_4947']
entries.sort(key=lambda row:priority.index(row['id']) if row['id'] in priority else 3)
records=[];sheets=[]
for entry in entries:
    started=time.perf_counter();name=entry['id'];previous=old[name]
    assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    with Image.open(previous['original_png']) as image:original=np.asarray(image).copy()
    with Image.open(ROOT/'data/user_photo_diagnostics/lut-natural-v2'/name/'natural.png') as image:base=np.asarray(image).copy()
    assert hashlib.sha256(original).hexdigest()==previous['natural']['input_pixels_sha256']
    assert hashlib.sha256(base).hexdigest()==previous['natural']['output_pixels_sha256']
    selection=shadow_settings(original);metadata,adjusted=apply_shadow_lift(base,selection['suggested_strength'])
    assert np.all(adjusted>=base)
    assert not np.any(np.any(adjusted==255,axis=-1)&~np.any(original==255,axis=-1))
    before,after,export=render_photo({'original':original,'adjusted':adjusted})
    export['semantics']=metadata['semantics'];directory=OUTPUT/name;directory.mkdir(exist_ok=True)
    (directory/'natural.png').write_bytes(after)
    canvas=Image.new('RGB',(1200,330),'#eeeeee');draw=ImageDraw.Draw(canvas)
    panels=[('Original',original),('Previous natural',base),(f"New auto {selection['suggested_strength']:.0%}",adjusted)]
    for col,(label,pixels) in enumerate(panels):
        image=Image.fromarray(pixels);image.thumbnail((400,280),Image.Resampling.LANCZOS)
        draw.text((col*400+5,5),name,fill='black');draw.text((col*400+5,22),label,fill='black');canvas.paste(image,(col*400,46))
    canvas.save(directory/'overview.png');sheets.append(canvas)
    h,w=original.shape[:2];cw,ch=min(420,w),min(280,h)
    boxes=[(1120,2380),(1665,2300),(w//2,h//2)] if name=='a2132-IMG_4947' else [(max(0,w//2-cw//2),max(0,h//2-ch//2)),(max(0,w//3-cw//2),max(0,h*4//5-ch//2)),(max(0,w//2-cw//2),max(0,h//8-ch//2))]
    detail=Image.new('RGB',(1290,930),'#eeeeee');draw=ImageDraw.Draw(detail)
    for row,(x,y) in enumerate(boxes):
        x=min(x,w-cw);y=min(y,h-ch)
        for col,(label,pixels) in enumerate(panels):
            draw.text((col*430+5,row*310+3),f'{label} x={x} y={y}',fill='black');detail.paste(Image.fromarray(pixels[y:y+ch,x:x+cw]),(col*430+5,row*310+25))
    detail.save(directory/'native.png')
    record={'id':name,'source':entry,'shadow_adjustment':dict(metadata,selection=selection),'export':export,
        'base_color_strength':previous['natural']['settings']['color_strength'],'base_pixels_match_frozen':True,
        'source_preserved':True,'seconds':time.perf_counter()-started}
    if name=='user-cloudy-trees':
        luma=np.array([.2126,.7152,.0722]);regions={'sky':(slice(0,230),slice(0,678)),'tree group':(slice(265,355),slice(55,590)),'foreground':(slice(370,430),slice(0,678))}
        record['display_luma_codes']={label:{region:float((pixels[ys,xs]@luma).mean()) for region,(ys,xs) in regions.items()} for label,pixels in panels}
    records.append(record)
    (OUTPUT/'report.json').write_text(json.dumps({'scope':'12previously inspected diagnostic sources; no independent quality/user preference claim.','cases':records},indent=2)+'\n')
    print(name,selection['suggested_strength'],round(record['seconds'],2),'s',flush=True)
    del original,base,adjusted,before,after
for start in (0,4,8):
    sheet=Image.new('RGB',(1200,1320),'#eeeeee')
    for row,canvas in enumerate(sheets[start:start+4]):sheet.paste(canvas,(0,row*330))
    sheet.save(OUTPUT/f'overview-{start//4+1}.png')
print('Completed12frozen natural-shadow comparisons',flush=True)
