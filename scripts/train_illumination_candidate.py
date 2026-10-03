"""Separate fixed denoising/learned illumination on fresh grouped development data."""
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
from scripts.train_noise_candidate import crops,native_metrics
from scripts.train_spatial_candidate import alignment,parent_model,DATA
from src.jpeg_curve_candidate import source_features
from src.jpeg_noise_candidate import render_noise_candidate,quality_loss
from src.jpeg_illumination_candidate import IlluminationCandidate,denoise_base,predict_illumination
from src.jpeg_inference import decode_jpeg,predict_jpeg
from src.preprocess import ROOT,sha256_file
from src.preview import apply_tone,encode_png

OUT=ROOT/'artifacts/experiments/jpeg_illumination_v1'
PRIOR=OUT.parent/'jpeg_spatial_v1'
PARENT=OUT.parent/'jpeg_noise_v1'


def contact(ids,images,path,labels):
    sheet=Image.new('RGB',(1200,150*((len(ids)+4)//5)),'white');d=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=14)
    for i,p in enumerate(ids):
        x,y=i%5*240,i//5*150
        sheet.paste(ImageOps.contain(Image.fromarray(images[p]),(235,120)),(x,y+25))
        d.text((x+4,y+4),labels[p],font=font,fill='black')
    sheet.save(path,quality=94)


def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve prepared pilot')
    old=json.loads((PRIOR/'manifest.json').read_text());previous={r['id'] for r in old['photos']}
    archive=DATA/'LOLdataset.zip';assert sha256_file(archive)==old['source_archive_sha256']
    images={'low':{},'high':{}};hashes={'low':[],'high':[]};records=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))
        low={Path(n).name for n in names if n.startswith('our485/low/') and n.endswith('.png')}
        high={Path(n).name for n in names if n.startswith('our485/high/') and n.endswith('.png')}
        assert low==high and len(low)==485
        added=np.random.default_rng(20261010).choice(sorted(low-previous),48,replace=False).tolist();ids=sorted(previous|set(added))
        for p in ids:
            sources={}
            for branch in ('low','high'):
                payload=z.read('our485/'+branch+'/'+p);path=DATA/branch/p
                if path.exists():assert path.read_bytes()==payload
                else:path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as im:
                    assert im.format=='PNG' and im.mode=='RGB' and payload[24]==8 and not im.info.get('icc_profile')
                    rgb=np.array(im)
                images[branch][p]=rgb;hashes[branch].append(image_hashes(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)))
                sources[branch]={'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)}
            assert images['low'][p].shape==images['high'][p].shape
            jpeg=DATA/(Path(p).stem+'.jpg')
            if not jpeg.exists():Image.fromarray(images['low'][p]).save(jpeg,quality=95,subsampling=0)
            records.append({'id':p,'previously_inspected':p in previous,'sources':sources,
                            'jpeg_path':str(jpeg.relative_to(ROOT)),'jpeg_sha256':sha256_file(jpeg),
                            'alignment':alignment(images['low'][p],images['high'][p])})
    pairs={}
    for branch in ('low','high'):
        d,p=zip(*hashes[branch])
        for (i,j),scores in candidate_pairs(np.array(d),np.array(p),[None]*len(ids),10,12).items():
            pair=pairs.setdefault((i,j),{'left':ids[i],'right':ids[j],'matches':{},'decision':'unreviewed'})
            pair['matches'][branch]=scores
    carried=json.loads((PRIOR/'split.json').read_text())['groups_all']
    write_json(OUT/'group_review.json',{'status':'pending','candidate_pairs':list(pairs.values()),
               'additional_groups':[g for g in carried.values() if len(g)>1],'review_notes':''})
    write_json(OUT/'manifest.json',{'photos':records,'selection_seed':20261010,'source_archive_sha256':old['source_archive_sha256'],
               'prior_manifest_sha256':sha256_file(PRIOR/'manifest.json')})
    labels={p:p+(' old' if p in previous else ' new') for p in ids}
    for branch in ('low','high'):
        for page in range((len(ids)+19)//20):contact(ids[page*20:(page+1)*20],images[branch],OUT/f'{branch}-scenes-{page+1}.jpg',labels)
    special=[r['id'] for r in records if r['alignment']['status']!='screen_pass']
    for branch in ('low','high'):
        if special:contact(special,images[branch],OUT/f'alignment-{branch}.jpg',labels)
    # Every hash candidate's reference pair, with IDs for review against carried groups.
    for page in range((len(pairs)+9)//10):
        selected=list(pairs.values())[page*10:(page+1)*10]
        keys=[p for pair in selected for p in (pair['left'],pair['right'])]
        sheet=Image.new('RGB',(600,180*len(selected)),'white');d=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=14)
        for i,p in enumerate(keys):
            x,y=i%2*300,i//2*180;sheet.paste(ImageOps.contain(Image.fromarray(images['high'][p]),(290,145)),(x,y+26))
            d.text((x+4,y+4),labels[p],font=font,fill='black')
        sheet.save(OUT/f'group-candidates-{page+1}.jpg',quality=94)
    print('Prepared',len(records),'pairs;',len(pairs),'hash candidates; alignment',
          {s:sum(r['alignment']['status']==s for r in records) for s in ('screen_pass','uncertain','quarantine_shift')},flush=True)


def objective(model,x,target,original):
    value=model(x)[...,16:-16,16:-16];ref=target[...,16:-16,16:-16];raw=original[...,16:-16,16:-16]
    weights=value.new_tensor([.2126,.7152,.0722])[None,:,None,None];dark=(ref*weights).sum(1,keepdim=True)<.05
    black=(torch.abs(value-ref)*dark).sum()/(dark.sum()*3).clamp_min(1)
    return quality_loss(value,ref,raw,value.new_zeros((len(x),3)))+.1*black


def averages(scores):
    return {k:float(np.mean([s[k] for s in scores if s[k] is not None])) for k in scores[0]}


def numeric_gates(candidate,base,global_mean):
    return {'psnr_plus_half_db':candidate['psnr_db']>=base['psnr_db']+.5,
            'within_quarter_db_of_global':candidate['psnr_db']>=global_mean['psnr_db']-.25,
            'dark_variation_below_90pct':candidate['dark_median_residual_codes']<=.9*base['dark_median_residual_codes'],
            'chroma_not_worse':candidate['chroma_mse']<=base['chroma_mse'],
            'gradient_within_110pct':candidate['luma_gradient_l1']<=1.1*base['luma_gradient_l1'],
            'new_full_channels_below_half_pct':candidate['new_full_channel_fraction']<=.005}


def train():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve completed pilot')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    manifest=json.loads((OUT/'manifest.json').read_text());review=json.loads((OUT/'group_review.json').read_text())
    all_groups=create_split(manifest['photos'],review)['groups'];old={r['id'] for r in manifest['photos'] if r['previously_inspected']}
    excluded={r['id']:r['alignment'] for r in manifest['photos'] if r['alignment']['status']=='quarantine_shift'}
    groups={g:[p for p in m if p not in excluded] for g,m in all_groups.items()};groups={g:m for g,m in groups.items() if m}
    eligible=sorted(g for g in groups if not set(all_groups[g])&old);chosen=[]
    for g in np.random.default_rng(20261011).permutation(eligible):
        if sum(len(groups[g]) for g in chosen)>=16:break
        chosen.append(str(g))
    assert len(chosen)>=8,'Insufficient fresh scene groups; stop before training'
    val=sorted(p for g in chosen for p in groups[g]);ids={p for m in groups.values() for p in m}
    split={'groups_all':all_groups,'groups':groups,'train':sorted(ids-set(val)),'validation':val,
           'validation_groups':chosen,'excluded_alignment':excluded,'split_seed':20261011}
    assert not set(val)&old;write_json(OUT/'split.json',split)
    parent=parent_model();parent_state=copy.deepcopy(parent.state_dict());images={};bases={};clean={};targets={};cropped={}
    started=time.perf_counter();records={r['id']:r for r in manifest['photos']}
    for i,p in enumerate(sorted(ids)):
        r=records[p]
        for s in r['sources'].values():assert sha256_file(ROOT/s['path'])==s['sha256']
        assert sha256_file(ROOT/r['jpeg_path'])==r['jpeg_sha256'];rgb,_=decode_jpeg(ROOT/r['jpeg_path'])
        with Image.open(ROOT/r['sources']['high']['path']) as im:target=np.array(im)
        with torch.no_grad():g,a=parent.curve_parameters(torch.tensor(source_features(rgb)[None]))
        base=render_noise_candidate(rgb,float(g[0,0]),a[0].numpy());denoised=denoise_base(base)
        images[p]=rgb;bases[p]=base;clean[p]=denoised;targets[p]=target
        cropped[p]=[np.stack(list(crops(im))) for im in (denoised,target,rgb)]
        if (i+1)%32==0:print('Prepared native denoising',i+1,'/',len(ids),flush=True)
    sources=('scripts/train_illumination_candidate.py','src/jpeg_illumination_candidate.py',
             'scripts/train_spatial_candidate.py','src/jpeg_noise_candidate.py','scripts/train_noise_candidate.py',
             'src/jpeg_curve_candidate.py','scripts/train_jpeg_curve.py')
    config={'device':'cpu','lr':.001,'weight_decay':.0001,'batch':16,'max_epochs':30,'patience':6,
            'denoiser':{'h':7,'hColor':10,'templateWindowSize':7,'searchWindowSize':21},
            'protocol_sha256':sha256_file(OUT/'PROTOCOL.md'),'manifest_sha256':sha256_file(OUT/'manifest.json'),
            'review_sha256':sha256_file(OUT/'group_review.json'),'split_sha256':sha256_file(OUT/'split.json'),
            'parent_config_sha256':sha256_file(PARENT/'config.json'),'parent_checkpoint_sha256':sha256_file(PARENT/'candidate-42.pt'),
            'production_model_sha256':sha256_file(ROOT/'artifacts/model/model.onnx'),
            'source_sha256':{p:sha256_file(ROOT/p) for p in sources}}
    write_json(OUT/'config.json',config)
    def tensors(selected):return tuple(torch.tensor(np.concatenate([cropped[p][i] for p in selected])) for i in range(3))
    train_x,train_y,train_raw=tensors(split['train']);val_x,val_y,val_raw=tensors(val)
    base_mean=averages([native_metrics(bases[p],targets[p],images[p]) for p in val]);runs={};outputs={};summaries={}
    for label,seed,constant in [('global42',42,True),('spatial42',42,False),('spatial43',43,False),('spatial44',44,False)]:
        if seed!=42 and not all(runs['spatial42']['numeric_gates'].values()):break
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=IlluminationCandidate(constant)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);run_start=time.perf_counter()
        with torch.no_grad():best=float(objective(model,val_x,val_y,val_raw))
        initial=best;state=copy.deepcopy(model.state_dict());best_epoch=0;history=[]
        for epoch in range(1,31):
            order=rng.permutation(len(train_x));total=0.
            for offset in range(0,len(train_x),16):
                batch=order[offset:offset+16];optimizer.zero_grad();loss=objective(model,train_x[batch],train_y[batch],train_raw[batch])
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step();total+=float(loss.detach())*len(batch)
            with torch.no_grad():v=float(objective(model,val_x,val_y,val_raw))
            history.append({'epoch':epoch,'train_loss':total/len(train_x),'validation_loss':v})
            if v<best:best=v;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
            if epoch%5==0:print(label,'epoch',epoch,'validation',round(v,6),flush=True)
            if epoch-best_epoch>=6:break
        model.load_state_dict(state);model.eval();torch.save(state,OUT/(label+'.pt'))
        outputs[label]={p:predict_illumination(model,images[p],clean[p]) for p in val+['102.png','27.png'] if p in images}
        mean=averages([native_metrics(outputs[label][p],targets[p],images[p]) for p in val]);summaries[label]=mean
        gates={} if constant else numeric_gates(mean,base_mean,summaries['global42'])
        runs[label]={'seed':seed,'constant':constant,'best_epoch':best_epoch,'epochs_run':len(history),'initial_loss':initial,
                     'best_loss':best,'history':history,'numeric_gates':gates,'elapsed_seconds':time.perf_counter()-run_start,
                     'checkpoint_sha256':sha256_file(OUT/(label+'.pt')),'parameter_count':sum(p.numel() for p in model.parameters())}
        print(label,'native PSNR',round(mean['psnr_db'],4),'gates',gates,flush=True)
    assert all(torch.equal(parent.state_dict()[k],v) for k,v in parent_state.items())
    photos=[];font=ImageFont.load_default(size=14)
    for index,p in enumerate(val+['102.png','27.png']):
        if p not in images:continue
        pred,linear=predict_jpeg(ROOT/records[p]['jpeg_path']);params={k:pred['experimental_absolute'][k] for k in ('Exposure','HighlightRecovery')}
        variants={'identity':images[p],'existing-v3':encode_png(apply_tone(linear,params)[0])[1],
                  'frozen-base':bases[p],'denoised-base':clean[p]}
        variants.update({k:v[p] for k,v in outputs.items()})
        photos.append({'id':p,'partition':'validation' if p in val else 'old training regression',
                       'alignment':records[p]['alignment'],'metrics':{k:native_metrics(v,targets[p],images[p]) for k,v in variants.items()}})
        if index<8 or p in ('102.png','27.png'):
            shown=[('Frozen base',bases[p]),('Fixed denoising',clean[p]),('Global illumination',outputs['global42'][p]),
                   ('Spatial illumination',outputs['spatial42'][p]),('Reference',targets[p])]
            sheet=Image.new('RGB',(1500,270),'white');d=ImageDraw.Draw(sheet)
            center=Image.new('RGB',(1500,320),'white');cd=ImageDraw.Draw(center)
            edge=Image.new('RGB',(1500,320),'white');ed=ImageDraw.Draw(edge)
            for i,(name,image) in enumerate(shown):
                sheet.paste(ImageOps.contain(Image.fromarray(image),(290,200)),(300*i,30));d.text((300*i+4,5),name,font=font,fill='black')
                h,w=image.shape[:2]
                for canvas,draw,tile in [(center,cd,image[h//2-48:h//2+48,w//2-48:w//2+48]),(edge,ed,image[:96,:96])]:
                    canvas.paste(Image.fromarray(tile).resize((300,300),Image.Resampling.NEAREST),(300*i,20))
                    draw.text((300*i+4,2),name,font=font,fill='black')
            d.text((5,245),p+' | '+photos[-1]['partition'],font=font,fill='black')
            for kind,canvas in [('comparison',sheet),('crop',center),('edge',edge)]:canvas.save(OUT/(Path(p).stem+'-'+kind+'.jpg'),quality=94)
    for k in ('identity','existing-v3','frozen-base','denoised-base'):
        summaries[k]=averages([r['metrics'][k] for r in photos if r['partition']=='validation'])
    write_json(OUT/'report.json',{'split':split,'runs':runs,'summary':summaries,'photos':photos,'parent_unchanged':True,
                                 'elapsed_seconds':time.perf_counter()-started,'scope':'Grouped development pilot, not final test or deployment.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);prepare() if args.stage=='prepare' else train()
