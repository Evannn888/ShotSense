"""Fixed v2/v3 rendering regression plus four unseen randomly selected test photos."""
import importlib.util
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scripts.check_preview_variety import CASES
from src.inference import predict_dng
from src.preview import prepare_preview_source, render_linear_preview
from src.preprocess import ROOT, sha256_file


def main():
    out=ROOT/'artifacts/preview_v3'
    spec=importlib.util.spec_from_file_location('preview_v2',out/'preview_v2_reference.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    splits=json.loads((ROOT/'data/processed/splits.json').read_text())['splits']
    viewed=set()
    for name in ('candidates.json','heldout-candidates.json'):
        viewed.update(json.loads((ROOT/'artifacts/preview_variety'/name).read_text()))
    unseen=sorted(set(splits['test'])-viewed)
    selected=np.random.default_rng(20261002).choice(unseen,4,replace=False).tolist()
    cases=CASES+[(f'unseen-test-{i+1}',pid.split('-')[0]) for i,pid in enumerate(selected)]
    paths=list((ROOT/'data/raw/dngs').glob('*.dng'));font=ImageFont.load_default(size=20);entries=[]
    for scene,prefix in cases:
        matches=[p for p in paths if p.name.lower().startswith(prefix+'-')];assert len(matches)==1
        path=matches[0];started=time.perf_counter();result,_=predict_dng(path)
        params={k:result['recommended_absolute'][k] for k in ('Exposure','HighlightRecovery')}
        source=prepare_preview_source(path)
        before,v3,meta=render_linear_preview(source,params)
        _,v2,old_meta=old.render_linear_preview(source,params)
        _,soft,soft_meta=render_linear_preview(source,params,.75)
        assert meta['new_full_channel_fraction']==soft_meta['new_full_channel_fraction']==0
        variants=[('before',before),('v2',v2),('v3',v3)]
        sheet=Image.new('RGB',(1500,430),'#f3f4f6');draw=ImageDraw.Draw(sheet)
        for i,(name,payload) in enumerate(variants):
            (out/(scene+'-'+name+'.png')).write_bytes(payload)
            im=ImageOps.contain(Image.open(io.BytesIO(payload)).convert('RGB'),(490,355))
            sheet.paste(im,(500*i+(500-im.width)//2,45+(355-im.height)//2))
            draw.text((500*i+12,12),name,font=font,fill='#202020')
        (out/(scene+'-v3-75pct.png')).write_bytes(soft)
        draw.text((12,405),scene+' | '+path.name,font=font,fill='#202020')
        sheet.save(out/(scene+'-comparison.jpg'),quality=94)
        entry={'scene':scene,'photo_id':path.name,'parameters':params,'v2':old_meta,'v3':meta,'v3_75pct':soft_meta,
               'split':next(k for k,v in splits.items() if path.name.lower() in v),
               'elapsed_seconds':time.perf_counter()-started,'comparison':scene+'-comparison.jpg'}
        entries.append(entry);print(scene,flush=True)
    for group in range(4):
        sheet=Image.new('RGB',(1500,1720),'white')
        for row,e in enumerate(entries[group*4:group*4+4]):sheet.paste(Image.open(out/e['comparison']),(0,430*row))
        sheet.save(out/f'overview-{group+1}.jpg',quality=94)
    report={'selection':'12 previously reviewed regression scenes plus four unseen test IDs drawn before inference with NumPy seed 20261002; no expert labels consulted.',
            'unseen_test_ids':selected,'photos':entries,
            'v2_sha256':sha256_file(out/'preview_v2_reference.py'),'v3_sha256':sha256_file(ROOT/'src/preview.py'),
            'model_sha256':sha256_file(ROOT/'artifacts/model/model.onnx'),
            'limits':'Global custom approximation; source ceiling is after display gamut clipping, not sensor clipping; no Lightroom reference or population quality estimate.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
