"""Prepare reviewed scene groups, then run a bounded CPU JPEG curve experiment."""
import argparse
import copy
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.audit_photo_groups import image_hashes, candidate_pairs
from scripts.test_paired_jpeg_pilot import metrics
from src.jpeg_curve_candidate import CurveCandidate, apply_curve, source_features
from src.jpeg_inference import predict_jpeg, decode_jpeg
from src.preprocess import ROOT, sha256_file
from src.preview import apply_tone, encode_png


OUT = ROOT / 'artifacts/experiments/jpeg_curve'
DATA = ROOT / 'data/external/lol_pilot'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    destination = OUT / 'manifest.json'
    if destination.exists():
        raise FileExistsError('Preserve prepared manifest')
    prior = json.loads((OUT.parent/'jpeg_pilot/report.json').read_text())
    previous = {p['id'] for p in prior['photos']}
    archive = DATA / 'LOLdataset.zip'
    assert sha256_file(archive) == prior['source_archive_sha256']
    records = []; images = {}; hashes = {'low': [], 'high': []}
    font = ImageFont.load_default(size=14)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        available = sorted(Path(n).name for n in names if n.startswith('our485/low/') and n.endswith('.png'))
        assert len(available) == 485 and len(names) == len(set(names))
        added = np.random.default_rng(20261004).choice(sorted(set(available)-previous), 24, replace=False).tolist()
        ids = sorted(previous | set(added))
        for name in ids:
            audits = {}
            for branch in ('low', 'high'):
                member = 'our485/' + branch + '/' + name
                payload = zipped.read(member)
                path = DATA/branch/name
                path.parent.mkdir(exist_ok=True)
                if path.exists(): assert path.read_bytes() == payload
                else: path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as image:
                    assert image.format == 'PNG' and image.mode == 'RGB' and not image.info.get('icc_profile')
                    assert payload[24] == 8
                    rgb = np.asarray(image).copy()
                images[name, branch] = rgb
                hashes[branch].append(image_hashes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
                audits[branch] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256_file(path)}
            assert images[name, 'low'].shape == images[name, 'high'].shape
            jpeg = DATA/(Path(name).stem+'.jpg')
            if not jpeg.exists(): Image.fromarray(images[name, 'low']).save(jpeg, quality=95, subsampling=0)
            records.append({'id': name, 'previously_inspected': name in previous,
                            'sources': audits, 'jpeg_path': str(jpeg.relative_to(ROOT)),
                            'jpeg_sha256': sha256_file(jpeg)})
    pairs = {}
    for branch in ('low', 'high'):
        differences, perceptual = zip(*hashes[branch])
        found = candidate_pairs(np.array(differences), np.array(perceptual), [None]*len(ids), 10, 12)
        for (i,j), values in found.items():
            pair = pairs.setdefault((i,j), {'left': ids[i], 'right': ids[j], 'matches': {}, 'decision': 'unreviewed'})
            pair['matches'][branch] = values
    for branch, prefix in (('high','scenes'),('low','inputs')):
        for page in range(2):
            sheet = Image.new('RGB', (1200, 720), 'white'); draw = ImageDraw.Draw(sheet)
            for slot, name in enumerate(ids[page*24:(page+1)*24]):
                x, y = slot%6*200, slot//6*180
                image = ImageOps.contain(Image.fromarray(images[name,branch]), (195,150))
                sheet.paste(image,(x,y+24)); draw.text((x+4,y+4),name+(' prior' if name in previous else ' new'),font=font,fill='black')
            sheet.save(OUT/f'{prefix}-{page+1}.jpg',quality=94)
    if pairs:
        sheet = Image.new('RGB', (600, 180*len(pairs)), 'white'); draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(pairs.values()):
            for col, name in enumerate((pair['left'],pair['right'])):
                image=ImageOps.contain(Image.fromarray(images[name,'high']), (290,145))
                sheet.paste(image,(300*col,row*180+26));draw.text((300*col+4,row*180+4),name,font=font,fill='black')
        sheet.save(OUT/'group-candidates.jpg',quality=94)
    write_json(OUT/'group_review.json', {'status':'pending', 'candidate_pairs':list(pairs.values()),
                                        'additional_groups':[], 'review_notes':''})
    write_json(destination, {'source_archive_sha256':prior['source_archive_sha256'], 'photos':records,
                            'scope':'48 our485 development pairs; no eval15 pixels extracted'})
    print('Prepared',len(records),'pairs;',len(pairs),'candidate scene pairs; review before training.')


def create_split(records, review):
    assert review['status'] == 'reviewed' and review['review_notes']
    ids = sorted(p['id'] for p in records); parent = {p:p for p in ids}
    def root(p):
        while parent[p] != p: p = parent[p]
        return p
    def merge(group):
        assert set(group) <= set(ids)
        for p in group[1:]: parent[root(p)] = root(group[0])
    for pair in review['candidate_pairs']:
        assert pair['decision'] in ('merge','reject')
        if pair['decision']=='merge': merge([pair['left'],pair['right']])
    for group in review['additional_groups']: merge(group)
    groups = {}
    for p in ids: groups.setdefault(root(p), []).append(p)
    previous = {p['id'] for p in records if p['previously_inspected']}
    eligible = sorted(g for g, members in groups.items() if not set(members)&previous)
    validation = []
    for g in np.random.default_rng(20261005).permutation(eligible):
        if sum(len(groups[p]) for p in validation) >= 16: break
        validation.append(str(g))
    assert len(validation) >= 8, 'Insufficient independent validation groups for the protocol'
    val = sorted(p for g in validation for p in groups[g]); train = sorted(set(ids)-set(val))
    assert not set(val)&previous and set(train)|set(val)==set(ids)
    return {'groups':groups, 'train':train, 'validation':val, 'validation_groups':validation,
            'grouping_limit':'Manual thumbnail/hash grouping is conservative and incomplete.'}


def train():
    if (OUT/'report.json').exists(): raise FileExistsError('Preserve completed experiment')
    manifest = json.loads((OUT/'manifest.json').read_text())
    review = json.loads((OUT/'group_review.json').read_text())
    records = manifest['photos']; by_id = {p['id']:p for p in records}
    split = create_split(records, review)
    write_json(OUT/'split.json',split)  # Freeze before model initialization/predictions.
    torch.set_num_threads(1)
    inputs={};targets={};features={};small_inputs={};small_targets={}
    for p in records:
        for source in p['sources'].values(): assert sha256_file(ROOT/source['path'])==source['sha256']
        assert sha256_file(ROOT/p['jpeg_path'])==p['jpeg_sha256']
        rgb,_=decode_jpeg(ROOT/p['jpeg_path'])
        with Image.open(ROOT/p['sources']['high']['path']) as im: target=np.array(im)
        assert rgb.shape==target.shape
        inputs[p['id']]=rgb;targets[p['id']]=target;features[p['id']]=source_features(rgb)
        h,w=rgb.shape[:2];size=(round(w*96/max(h,w)),round(h*96/max(h,w)))
        small_inputs[p['id']]=cv2.resize(rgb,size,interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/255
        small_targets[p['id']]=cv2.resize(target,size,interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/255
    assert len({x.shape for x in small_inputs.values()})==1
    def tensors(ids):
        return (torch.tensor(np.stack([small_inputs[p] for p in ids])),
                torch.tensor(np.stack([features[p] for p in ids])),
                torch.tensor(np.stack([small_targets[p] for p in ids])))
    train_x,train_f,train_y=tensors(split['train']);val_x,val_f,val_y=tensors(split['validation'])
    mean=train_f.mean(0);scale=train_f.std(0,unbiased=False).clamp_min(1e-6)
    config={'seeds':[42,43,44],'max_epochs':100,'patience':15,'batch':16,'lr':.003,'weight_decay':.0001,
            'device':'CPU','threads':1,'selection':'small validation MSE including identity epoch 0',
            'manifest_sha256':sha256_file(OUT/'manifest.json'),'review_sha256':sha256_file(OUT/'group_review.json'),
            'split_sha256':sha256_file(OUT/'split.json'), 'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),
            'source_sha256':{p:sha256_file(ROOT/p) for p in ('src/jpeg_curve_candidate.py','scripts/train_jpeg_curve.py')},
            'production_model_sha256':sha256_file(ROOT/'artifacts/model/model.onnx')}
    write_json(OUT/'config.json',config)
    predictions={};runs={};started=time.perf_counter()
    for label,seed,constant in [('constant',42,True)]+[(f'candidate-{s}',s,False) for s in config['seeds']]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=CurveCandidate(mean,scale,constant)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
        with torch.no_grad(): best_loss=float(torch.mean((model(val_x,val_f)-val_y)**2))
        best_state=copy.deepcopy(model.state_dict());best_epoch=0;history=[]
        for epoch in range(1,101):
            for batch in np.array_split(rng.permutation(len(train_x)),int(np.ceil(len(train_x)/16))):
                optimizer.zero_grad();loss=torch.mean((model(train_x[batch],train_f[batch])-train_y[batch])**2)
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
            with torch.no_grad():
                validation_loss=float(torch.mean((model(val_x,val_f)-val_y)**2))
                training_loss=float(torch.mean((model(train_x,train_f)-train_y)**2))
            history.append({'epoch':epoch,'train_mse':training_loss,'validation_mse':validation_loss})
            if validation_loss<best_loss:best_loss=validation_loss;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=15:break
        model.load_state_dict(best_state);model.eval()
        torch.save(model.state_dict(),OUT/(label+'.pt'))
        outputs={}
        with torch.no_grad():
            for p in split['validation']:
                gamma=model.gamma(torch.tensor(features[p][None]))[0].numpy()
                outputs[p]=(apply_curve(inputs[p],gamma),gamma.tolist())
        predictions[label]=outputs
        runs[label]={'seed':seed,'constant':constant,'parameter_count':sum(p.numel() for p in model.parameters()),
                     'best_epoch':best_epoch,'best_small_validation_mse':best_loss,'history':history,
                     'checkpoint_sha256':sha256_file(OUT/(label+'.pt'))}
        print(label,'best epoch',best_epoch,'/',len(history),'validation MSE',round(best_loss,6),flush=True)
    evaluated=[];font=ImageFont.load_default(size=15)
    for index,p in enumerate(split['validation']):
        rgb=inputs[p];target=targets[p]
        pred,linear=predict_jpeg(ROOT/by_id[p]['jpeg_path'])
        params={k:pred['experimental_absolute'][k] for k in ('Exposure','HighlightRecovery')}
        _,identity=encode_png(linear);assert np.array_equal(identity,rgb)
        experimental=encode_png(apply_tone(linear,params)[0])[1]
        manual=encode_png(apply_tone(linear,{'Exposure':1.,'HighlightRecovery':0.})[0])[1]
        variants={'identity':rgb,'experimental-v3':experimental,'fixed-1ev':manual}
        variants.update({label:outputs[p][0] for label,outputs in predictions.items()})
        evaluated.append({'id':p,'metrics':{label:metrics(image,target,rgb) for label,image in variants.items()},
                          'gamma':{label:outputs[p][1] for label,outputs in predictions.items()}})
        if index<8:
            shown=[('identity',rgb),('experimental-v3',experimental),('constant',variants['constant']),
                   ('candidate-42',variants['candidate-42']),('target',target)]
            sheet=Image.new('RGB',(1500,270),'white');draw=ImageDraw.Draw(sheet)
            for col,(label,image) in enumerate(shown):
                im=ImageOps.contain(Image.fromarray(image),(290,200));sheet.paste(im,(col*300,35))
                draw.text((col*300+5,7),label,font=font,fill='black')
            draw.text((5,245),p+' | scene-grouped development validation',font=font,fill='black')
            sheet.save(OUT/(Path(p).stem+'-validation.jpg'),quality=94)
    summary={label:{m:float(np.mean([p['metrics'][label][m] for p in evaluated])) for m in evaluated[0]['metrics'][label]}
             for label in evaluated[0]['metrics']}
    candidate_mean=float(np.mean([summary[f'candidate-{s}']['psnr_db'] for s in config['seeds']]))
    better=sum(all(summary[f'candidate-{s}']['psnr_db']>summary[b]['psnr_db'] for b in ('constant','experimental-v3'))
               for s in config['seeds'])
    gates={'gain_over_both_at_least_half_db':all(candidate_mean>=summary[b]['psnr_db']+.5 for b in ('constant','experimental-v3')),
           'at_least_two_seeds_better':better>=2,
           'new_full_channels_at_most_half_percent':all(summary[f'candidate-{s}']['new_full_channel_fraction']<=.005 for s in config['seeds'])}
    write_json(OUT/'report.json',{'split':split,'runs':runs,'validation':evaluated,'summary':summary,
                                'candidate_mean_psnr':candidate_mean,'better_seed_count':better,'quantitative_gates':gates,
                                'elapsed_seconds':time.perf_counter()-started,
                                'scope':'Small development validation; no official test or production promotion; visual acceptance recorded separately.'})
    print(json.dumps(summary,indent=2));print('Gates',gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train'])
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    prepare() if args.stage=='prepare' else train()
