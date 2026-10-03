"""Frozen-model regression for bounded dark amplification, not a new quality test."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.test_paired_jpeg_pilot import metrics
from src.jpeg_curve_candidate import CurveCandidate, apply_curve, source_features
from src.jpeg_inference import decode_jpeg
from src.jpeg_shadow_guard import guarded_curve, MAX_GAIN
from src.preprocess import ROOT, sha256_file


def run():
    parent=ROOT/'artifacts/experiments/jpeg_curve';out=ROOT/'artifacts/experiments/jpeg_shadow_guard'
    if (out/'report.json').exists():raise FileExistsError('Preserve completed diagnostic')
    c=json.loads((parent/'config.json').read_text());prior=json.loads((parent/'report.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text());records={p['id']:p for p in manifest['photos']}
    for path,expected in c['source_sha256'].items():assert sha256_file(ROOT/path)==expected
    for name in ('manifest','split'):assert sha256_file(parent/(name+'.json'))==c[name+'_sha256']
    assert sha256_file(ROOT/'artifacts/model/model.onnx')==c['production_model_sha256']
    torch.set_num_threads(1);entries=[];started=time.perf_counter();font=ImageFont.load_default(size=15)
    models={}
    for seed in (42,43,44):
        label=f'candidate-{seed}';checkpoint=parent/(label+'.pt')
        assert sha256_file(checkpoint)==prior['runs'][label]['checkpoint_sha256']
        model=CurveCandidate(c['feature_mean'],c['feature_scale'])
        model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval();models[label]=model
    for index,old in enumerate(prior['validation']):
        p=records[old['id']]
        for a in p['sources'].values():assert sha256_file(ROOT/a['path'])==a['sha256']
        assert sha256_file(ROOT/p['jpeg_path'])==p['jpeg_sha256']
        rgb,_=decode_jpeg(ROOT/p['jpeg_path'])
        with Image.open(ROOT/p['sources']['high']['path']) as im:target=np.array(im)
        mask=(rgb.max(-1)>0)&(rgb.max(-1)<=16)
        median=cv2.medianBlur(rgb,3);features=torch.tensor(source_features(rgb)[None])
        variants={};params={}
        for label,model in models.items():
            with torch.no_grad():gamma=model.gamma(features)[0].numpy()
            assert np.max(np.abs(gamma-old['gamma'][label]))<1e-6
            variants[label+'/original']=apply_curve(rgb,gamma)
            variants[label+'/bounded']=guarded_curve(rgb,gamma)
            variants[label+'/median-bounded']=guarded_curve(median,gamma)
            assert np.array_equal(guarded_curve(rgb,gamma,0),rgb)
            assert np.array_equal(guarded_curve(rgb,gamma),variants[label+'/bounded'])
            params[label]=gamma.tolist()
        measured={}
        for label,image in variants.items():
            measured[label]=metrics(image,target,rgb)
            residual=np.abs(image.astype(float)-cv2.medianBlur(image,3).astype(float)).mean(-1)
            measured[label]['dark_median_residual_codes']=float(residual[mask].mean()) if mask.any() else None
            if label.endswith('/original'):
                assert abs(measured[label]['mse']-old['metrics'][label.split('/')[0]]['mse'])<1e-12
        entries.append({'id':p['id'],'gamma':params,'dark_mask_fraction':float(mask.mean()),'metrics':measured})
        if index<8:
            shown=[('Original gamma',variants['candidate-42/original']),
                   ('Bounded gamma',variants['candidate-42/bounded']),
                   ('Median + bounded',variants['candidate-42/median-bounded']),('Reference',target)]
            sheet=Image.new('RGB',(1200,270),'white');draw=ImageDraw.Draw(sheet)
            for col,(name,image) in enumerate(shown):
                im=ImageOps.contain(Image.fromarray(image),(290,200));sheet.paste(im,(col*300,35))
                draw.text((col*300+5,7),name,font=font,fill='black')
            draw.text((5,245),p['id']+' | inspected development regression',font=font,fill='black')
            sheet.save(out/(Path(p['id']).stem+'-comparison.jpg'),quality=94)
            if index==0:
                crop=Image.new('RGB',(1200,330),'white');draw=ImageDraw.Draw(crop)
                for col,(name,image) in enumerate(shown):
                    tile=Image.fromarray(image).crop((300,100,500,300)).resize((300,300),Image.Resampling.NEAREST)
                    crop.paste(tile,(300*col,30));draw.text((300*col+5,6),name,font=font,fill='black')
                crop.save(out/'102-dark-crop.jpg',quality=94)
    summary={}
    for variant in ('original','bounded','median-bounded'):
        scores=[e['metrics'][f'candidate-{seed}/'+variant] for e in entries for seed in (42,43,44)]
        summary[variant]={m:float(np.mean([s[m] for s in scores if s[m] is not None])) for m in scores[0]}
    record={'count':len(entries),'max_gain':MAX_GAIN,'frozen_checkpoint_replay':True,
            'parent_report_sha256':sha256_file(parent/'report.json'),
            'checkpoints':{label:sha256_file(parent/(label+'.pt')) for label in models},
            'source_sha256':{p:sha256_file(ROOT/p) for p in ('src/jpeg_shadow_guard.py','scripts/test_shadow_guard.py')},
            'summary':summary,'photos':entries,'elapsed_seconds':time.perf_counter()-started,
            'scope':'Previously inspected development regression; no retraining, official test access or release acceptance.'}
    (out/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    overview=Image.new('RGB',(1200,2160),'white')
    for i,e in enumerate(entries[:8]):
        with Image.open(out/(Path(e['id']).stem+'-comparison.jpg')) as row:overview.paste(row,(0,270*i))
    overview.save(out/'overview.jpg',quality=94)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
