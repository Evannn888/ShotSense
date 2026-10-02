"""Fixed three-photo preview acceptance; does not select or retrain the model."""
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

from src.inference import predict_dng
from src.preview import prepare_preview_source,render_linear_preview
from src.preprocess import ROOT,sha256_file


def main():
    output=ROOT/'artifacts/preview_v2'; output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('old_preview',ROOT/'artifacts/preview_v2/preview_v1_reference.py')
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    with np.load(ROOT/'data/processed/metadata.npz',allow_pickle=False) as data:
        ids=data['ids'].copy(); brightness=data['X'][:,[96,102,108]].mean(axis=1)
    order=np.argsort(brightness,kind='stable')
    chosen=[str(ids[0]),str(ids[order[round(.05*(len(ids)-1))]]),str(ids[order[round(.95*(len(ids)-1))]])]
    paths={p.name.lower():p for p in (ROOT/'data/raw/dngs').iterdir() if p.suffix.lower()=='.dng'}
    entries=[]
    font=ImageFont.load_default(size=20)
    for photo in chosen:
        started=time.perf_counter()
        result,jpeg=predict_dng(paths[photo])
        parameters={k:v for k,v in result['recommended_absolute'].items() if k in ('Exposure','HighlightRecovery')}
        source=prepare_preview_source(paths[photo])
        baseline,protected,metadata=render_linear_preview(source,parameters)
        _,direct,direct_metadata=render_linear_preview(source,parameters,protect_highlights=False)
        legacy=old.render_preview(jpeg,parameters)
        variants={'before':baseline,'old-jpeg':legacy,'direct':direct,'protected':protected}
        stem=Path(photo).stem
        for name,payload in variants.items(): (output/(stem+'-'+name+'.png')).write_bytes(payload)
        sheet=Image.new('RGB',(1920,430),'#f3f4f6'); draw=ImageDraw.Draw(sheet)
        captions=['RAW baseline','Old JPEG preview','Direct exposure','Protected preview']
        for column,((name,payload),caption) in enumerate(zip(variants.items(),captions)):
            image=ImageOps.contain(Image.open(io.BytesIO(payload)).convert('RGB'),(470,355))
            sheet.paste(image,(column*480+(480-image.width)//2,45+(355-image.height)//2))
            draw.text((column*480+12,12),caption,font=font,fill='#202020')
        draw.text((12,405),photo,font=font,fill='#202020')
        sheet.save(output/(stem+'-comparison.jpg'),quality=94)
        before_rgb=np.asarray(Image.open(io.BytesIO(baseline)),dtype=np.float32)
        after_rgb=np.asarray(Image.open(io.BytesIO(protected)),dtype=np.float32)
        dark=source.max(axis=2)<.3
        entries.append({'photo_id':photo,'parameters':parameters,'protected':metadata,'direct':direct_metadata,
                        'dark_region_mean_rgb_before':float(before_rgb[dark].mean()) if dark.any() else None,
                        'dark_region_mean_rgb_after':float(after_rgb[dark].mean()) if dark.any() else None,
                        'elapsed_seconds':time.perf_counter()-started,'comparison':stem+'-comparison.jpg'})
        print(json.dumps(entries[-1]),flush=True)
    report={'selection':'first ID, input Lab brightness p5/p95; fixed model; no expert renderings or model selection',
            'photos':entries,'preview_source_sha256':sha256_file(ROOT/'src/preview.py'),
            'old_reference_sha256':sha256_file(output/'preview_v1_reference.py'),
            'browser_acceptance':'not performed: saved browser permission blocks localhost access'}
    (output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
