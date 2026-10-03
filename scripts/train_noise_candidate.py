"""New-development-set pilot; native noise-aware crops and fixed matched controls."""
import argparse
import copy
import io
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps

from scripts.audit_photo_groups import image_hashes,candidate_pairs
from scripts.train_jpeg_curve import create_split,write_json
from scripts.test_paired_jpeg_pilot import metrics
from src.jpeg_curve_candidate import CurveCandidate,apply_curve,source_features
from src.jpeg_noise_candidate import NoiseCandidate,prefilter,render_noise_candidate,quality_loss
from src.jpeg_inference import decode_jpeg,predict_jpeg
from src.preprocess import ROOT,sha256_file
from src.preview import apply_tone,encode_png


OUT=ROOT/'artifacts/experiments/jpeg_noise_v1'
OLD=OUT.parent/'jpeg_curve'
DATA=ROOT/'data/external/lol_pilot'


def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve prepared manifest')
    old=json.loads((OLD/'manifest.json').read_text());previous={e['id'] for e in old['photos']}
    archive=DATA/'LOLdataset.zip';assert sha256_file(archive)==old['source_archive_sha256']
    records=[];images={};hashes={'low':[],'high':[]}
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist();assert len(names)==len(set(names))
        low={Path(n).name for n in names if n.startswith('our485/low/') and n.endswith('.png')}
        high={Path(n).name for n in names if n.startswith('our485/high/') and n.endswith('.png')}
        assert low==high and len(low)==485
        added=np.random.default_rng(20261006).choice(sorted(low-previous),32,replace=False).tolist()
        ids=sorted(previous|set(added))
        for name in ids:
            sources={}
            for branch in ('low','high'):
                payload=zipped.read('our485/'+branch+'/'+name);path=DATA/branch/name
                if path.exists():assert path.read_bytes()==payload
                else:path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as im:
                    assert im.format=='PNG' and im.mode=='RGB' and payload[24]==8 and not im.info.get('icc_profile')
                    rgb=np.array(im)
                images[name,branch]=rgb
                hashes[branch].append(image_hashes(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)))
                sources[branch]={'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)}
            assert images[name,'low'].shape==images[name,'high'].shape
            jpeg=DATA/(Path(name).stem+'.jpg')
            if not jpeg.exists():Image.fromarray(images[name,'low']).save(jpeg,quality=95,subsampling=0)
            records.append({'id':name,'previously_inspected':name in previous,'sources':sources,
                            'jpeg_path':str(jpeg.relative_to(ROOT)),'jpeg_sha256':sha256_file(jpeg)})
    pairs={}
    for branch in ('low','high'):
        d,p=zip(*hashes[branch])
        for (i,j),scores in candidate_pairs(np.array(d),np.array(p),[None]*len(ids),10,12).items():
            pair=pairs.setdefault((i,j),{'left':ids[i],'right':ids[j],'matches':{},'decision':'unreviewed'})
            pair['matches'][branch]=scores
    old_split=json.loads((OLD/'split.json').read_text())
    review={'status':'pending','candidate_pairs':list(pairs.values()),
            'additional_groups':[g for g in old_split['groups'].values() if len(g)>1],'review_notes':''}
    write_json(OUT/'group_review.json',review)
    write_json(OUT/'manifest.json',{'photos':records,'source_archive_sha256':old['source_archive_sha256'],
                                   'selection_seed':20261006,'prior_manifest_sha256':sha256_file(OLD/'manifest.json')})
    font=ImageFont.load_default(size=14)
    for branch in ('low','high'):
        for page in range(4):
            sheet=Image.new('RGB',(1200,600),'white');draw=ImageDraw.Draw(sheet)
            for slot,name in enumerate(ids[page*20:(page+1)*20]):
                x,y=slot%5*240,slot//5*150
                im=ImageOps.contain(Image.fromarray(images[name,branch]),(235,120));sheet.paste(im,(x,y+25))
                draw.text((x+5,y+4),name+(' old' if name in previous else ' new'),font=font,fill='black')
            sheet.save(OUT/f'{branch}-scenes-{page+1}.jpg',quality=94)
    if pairs:
        sheet=Image.new('RGB',(600,180*len(pairs)),'white');draw=ImageDraw.Draw(sheet)
        for i,pair in enumerate(pairs.values()):
            for col,name in enumerate((pair['left'],pair['right'])):
                im=ImageOps.contain(Image.fromarray(images[name,'high']),(290,145));sheet.paste(im,(col*300,180*i+26))
                draw.text((col*300+5,180*i+5),name,font=font,fill='black')
        sheet.save(OUT/'group-candidates.jpg',quality=94)
    print('Prepared',len(records),'pairs and',len(pairs),'related-scene candidates.',flush=True)


def crops(image):
    h,w=image.shape[:2];assert min(h,w)>=96
    for y,x in ((0,0),(0,w-96),(h-96,0),(h-96,w-96),((h-96)//2,(w-96)//2)):
        yield image[y:y+96,x:x+96].transpose(2,0,1).astype(np.float32)/255


def native_metrics(image,target,source):
    score=metrics(image,target,source)
    value=image.astype(np.float32)/255;reference=target.astype(np.float32)/255
    weights=np.array([.2126,.7152,.0722],dtype=np.float32)
    luma=value@weights;target_luma=reference@weights
    score['chroma_mse']=float(np.mean(((value-luma[...,None])-(reference-target_luma[...,None]))**2))
    score['luma_gradient_l1']=float((np.abs(np.diff(luma,axis=0)-np.diff(target_luma,axis=0)).mean()
                                   +np.abs(np.diff(luma,axis=1)-np.diff(target_luma,axis=1)).mean())/2)
    mask=(source.max(-1)>0)&(source.max(-1)<=16)
    residual=np.abs(image.astype(float)-cv2.medianBlur(image,3).astype(float)).mean(-1)
    score['dark_median_residual_codes']=float(residual[mask].mean()) if mask.any() else None
    return score


def train():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve completed experiment')
    manifest=json.loads((OUT/'manifest.json').read_text());review=json.loads((OUT/'group_review.json').read_text())
    # Reuse reviewed union/group logic, with this protocol's independent split seed.
    base=create_split(manifest['photos'],review)
    previous={e['id'] for e in manifest['photos'] if e['previously_inspected']}
    eligible=sorted(g for g,members in base['groups'].items() if not set(members)&previous)
    selected=[]
    for g in np.random.default_rng(20261007).permutation(eligible):
        if sum(len(base['groups'][p]) for p in selected)>=16:break
        selected.append(str(g))
    assert len(selected)>=8
    validation=sorted(p for g in selected for p in base['groups'][g])
    split={'groups':base['groups'],'validation_groups':selected,'validation':validation,
           'train':sorted(set(e['id'] for e in manifest['photos'])-set(validation)),'split_seed':20261007}
    assert not previous&set(validation)
    write_json(OUT/'split.json',split);torch.set_num_threads(1)
    rgb={};targets={};feature={};arrays={};records={e['id']:e for e in manifest['photos']}
    for e in manifest['photos']:
        for a in e['sources'].values():assert sha256_file(ROOT/a['path'])==a['sha256']
        assert sha256_file(ROOT/e['jpeg_path'])==e['jpeg_sha256']
        source,_=decode_jpeg(ROOT/e['jpeg_path'])
        with Image.open(ROOT/e['sources']['high']['path']) as im:target=np.array(im)
        assert source.shape==target.shape
        rgb[e['id']]=source;targets[e['id']]=target;feature[e['id']]=source_features(source)
        arrays[e['id']]=[np.stack(list(crops(i))) for i in (source,prefilter(source),target)]
    def batch_arrays(ids):
        raw,filtered,target=[torch.tensor(np.concatenate([arrays[p][i] for p in ids])) for i in range(3)]
        f=torch.tensor(np.repeat(np.stack([feature[p] for p in ids]),5,axis=0))
        return raw,filtered,target,f
    train_raw,train_filtered,train_target,train_f=batch_arrays(split['train'])
    val_raw,val_filtered,val_target,val_f=batch_arrays(validation)
    distinct=torch.tensor(np.stack([feature[p] for p in split['train']]))
    mean=distinct.mean(0);scale=distinct.std(0,unbiased=False).clamp_min(1e-6)
    config={'max_epochs':60,'patience':10,'batch':16,'lr':.003,'weight_decay':.0001,'seeds':[42,43,44],
            'cpu_threads':1,'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),
            'protocol_sha256':sha256_file(OUT/'PROTOCOL.md'),
            'source_sha256':{p:sha256_file(ROOT/p) for p in ('scripts/train_noise_candidate.py','src/jpeg_noise_candidate.py',
                            'src/jpeg_curve_candidate.py','scripts/train_jpeg_curve.py')},
            'manifest_sha256':sha256_file(OUT/'manifest.json'),'split_sha256':sha256_file(OUT/'split.json'),
            'review_sha256':sha256_file(OUT/'group_review.json'),'production_model_sha256':sha256_file(ROOT/'artifacts/model/model.onnx')}
    write_json(OUT/'config.json',config)
    outputs={};parameters={};runs={};started=time.perf_counter()
    cases=[('gamma-control',42,'gamma'),('constant',42,'constant')]+[(f'candidate-{s}',s,'candidate') for s in (42,43,44)]
    for label,seed,kind in cases:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=CurveCandidate(mean,scale) if kind=='gamma' else NoiseCandidate(mean,scale,kind=='constant')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
        def loss_for(raw,filtered,target,f):
            prediction=model(raw if kind=='gamma' else filtered,f)
            return (prediction-target).square().mean() if kind=='gamma' else quality_loss(prediction,target,raw,model.curve_parameters(f)[1])
        with torch.no_grad():best=float(loss_for(val_raw,val_filtered,val_target,val_f))
        best_state=copy.deepcopy(model.state_dict());best_epoch=0;history=[];run_start=time.perf_counter()
        for epoch in range(1,61):
            for offset in range(0,len(train_raw),16):
                if offset==0:order=rng.permutation(len(train_raw))
                b=order[offset:offset+16];optimizer.zero_grad()
                loss=loss_for(train_raw[b],train_filtered[b],train_target[b],train_f[b]);assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
            with torch.no_grad():v=float(loss_for(val_raw,val_filtered,val_target,val_f));t=float(loss_for(train_raw,train_filtered,train_target,train_f))
            history.append({'epoch':epoch,'train_loss':t,'validation_loss':v})
            if v<best:best=v;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=10:break
        model.load_state_dict(best_state);model.eval();torch.save(best_state,OUT/(label+'.pt'))
        outputs[label]={};parameters[label]={}
        with torch.no_grad():
            for p in validation+['102.png']:
                f=torch.tensor(feature[p][None])
                if kind=='gamma':
                    gamma=model.gamma(f)[0].numpy();image=apply_curve(rgb[p],gamma);values={'gamma':gamma.tolist()}
                else:
                    g,a=model.curve_parameters(f);gamma=float(g[0,0]);color=a[0].numpy()
                    image=render_noise_candidate(rgb[p],gamma,color);values={'gamma':gamma,'color':color.tolist()}
                outputs[label][p]=image;parameters[label][p]=values
        runs[label]={'seed':seed,'kind':kind,'parameter_count':sum(p.numel() for p in model.parameters()),
                     'best_epoch':best_epoch,'best_loss':best,'history':history,'elapsed_seconds':time.perf_counter()-run_start,
                     'checkpoint_sha256':sha256_file(OUT/(label+'.pt'))}
        print(label,'epoch',best_epoch,'/',len(history),'loss',round(best,6),flush=True)
    entries=[];font=ImageFont.load_default(size=15)
    for index,p in enumerate(validation+['102.png']):
        pred,linear=predict_jpeg(ROOT/records[p]['jpeg_path'])
        params={k:pred['experimental_absolute'][k] for k in ('Exposure','HighlightRecovery')}
        variants={'identity':rgb[p],'experimental-v3':encode_png(apply_tone(linear,params)[0])[1],
                  'fixed-1ev':encode_png(apply_tone(linear,{'Exposure':1.,'HighlightRecovery':0.})[0])[1]}
        variants.update({label:images[p] for label,images in outputs.items()})
        entry={'id':p,'partition':'validation' if p in validation else 'old regression training',
               'parameters':{label:values[p] for label,values in parameters.items()},
               'metrics':{label:native_metrics(image,targets[p],rgb[p]) for label,image in variants.items()}}
        entries.append(entry)
        if index<8 or p=='102.png':
            shown=[('Existing v3',variants['experimental-v3']),('Matched gamma',variants['gamma-control']),
                   ('Shared constant',variants['constant']),('Noise candidate42',variants['candidate-42']),('Reference',targets[p])]
            sheet=Image.new('RGB',(1500,270),'white');draw=ImageDraw.Draw(sheet)
            crop=Image.new('RGB',(1500,330),'white');cd=ImageDraw.Draw(crop)
            for col,(name,image) in enumerate(shown):
                im=ImageOps.contain(Image.fromarray(image),(290,200));sheet.paste(im,(col*300,35));draw.text((col*300+5,7),name,font=font,fill='black')
                h,w=image.shape[:2];tile=Image.fromarray(image).crop((w//2-96,h//2-96,w//2+96,h//2+96)).resize((300,300),Image.Resampling.NEAREST)
                crop.paste(tile,(col*300,30));cd.text((col*300+5,7),name,font=font,fill='black')
            draw.text((5,245),p+' | '+entry['partition'],font=font,fill='black')
            sheet.save(OUT/(Path(p).stem+'-comparison.jpg'),quality=94);crop.save(OUT/(Path(p).stem+'-crop.jpg'),quality=94)
    val_entries=[e for e in entries if e['partition']=='validation']
    summary={label:{m:float(np.mean([e['metrics'][label][m] for e in val_entries if e['metrics'][label][m] is not None]))
                    for m in val_entries[0]['metrics'][label]} for label in val_entries[0]['metrics']}
    avg={m:float(np.mean([summary[f'candidate-{s}'][m] for s in (42,43,44)])) for m in summary['candidate-42']}
    gates={'psnr_over_v3_by_half_db':avg['psnr_db']>=summary['experimental-v3']['psnr_db']+.5,
           'psnr_within_half_db_of_constant':avg['psnr_db']>=summary['constant']['psnr_db']-.5,
           'two_seeds_beat_v3':sum(summary[f'candidate-{s}']['psnr_db']>summary['experimental-v3']['psnr_db'] for s in (42,43,44))>=2,
           'dark_residual_at_most_70pct_gamma_control':avg['dark_median_residual_codes']<=.7*summary['gamma-control']['dark_median_residual_codes'],
           'chroma_mse_at_most_110pct_gamma_control':avg['chroma_mse']<=1.1*summary['gamma-control']['chroma_mse'],
           'new_full_channel_at_most_half_percent':avg['new_full_channel_fraction']<=.005}
    write_json(OUT/'report.json',{'split':split,'runs':runs,'photos':entries,'summary':summary,'candidate_average':avg,'numeric_gates':gates,
                                 'elapsed_seconds':time.perf_counter()-started,'scope':'New grouped development set; reference grouping reviewed; no final test or promotion. Visual decision recorded separately.'})
    print(json.dumps(summary,indent=2));print('Gates',gates)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train']);a=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);prepare() if a.stage=='prepare' else train()
