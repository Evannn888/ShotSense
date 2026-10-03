"""Alignment/group audit and bounded small spatial-restoration experiment."""
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
from src.jpeg_curve_candidate import source_features
from src.jpeg_noise_candidate import NoiseCandidate,render_noise_candidate,quality_loss
from src.jpeg_spatial_candidate import SpatialCandidate,predict_spatial
from src.jpeg_inference import decode_jpeg,predict_jpeg
from src.preprocess import ROOT,sha256_file
from src.preview import apply_tone,encode_png


OUT=ROOT/'artifacts/experiments/jpeg_spatial_v1'
PARENT=OUT.parent/'jpeg_noise_v1'
DATA=ROOT/'data/external/lol_pilot'


def alignment(low,high):
    h,w=low.shape[:2];ratio=min(1,320/max(h,w));size=(round(w*ratio),round(h*ratio))
    edges=[]
    for image in (low,high):
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        edge=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
        edges.append((edge-edge.mean())/max(float(edge.std()),1e-6))
    shift,response=cv2.phaseCorrelate(edges[0],edges[1],cv2.createHanningWindow(size,cv2.CV_32F))
    native=[float(v/ratio) for v in shift]
    status='quarantine_shift' if response>=.2 and max(abs(v) for v in native)>2 else ('uncertain' if response<.2 else 'screen_pass')
    return {'native_shift_xy':native,'response':float(response),'status':status,'limit':'Exposure/noise-dependent edge screen; no calibrated registration claim.'}


def prepare():
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve prepared pilot')
    old=json.loads((PARENT/'manifest.json').read_text());previous={p['id'] for p in old['photos']}
    archive=DATA/'LOLdataset.zip';assert sha256_file(archive)==old['source_archive_sha256']
    images={};records=[];hashes={'low':[],'high':[]}
    with zipfile.ZipFile(archive) as zipped:
        names=zipped.namelist();assert len(names)==len(set(names))
        low={Path(n).name for n in names if n.startswith('our485/low/') and n.endswith('.png')}
        high={Path(n).name for n in names if n.startswith('our485/high/') and n.endswith('.png')};assert low==high and len(low)==485
        new=np.random.default_rng(20261008).choice(sorted(low-previous),40,replace=False).tolist();ids=sorted(previous|set(new))
        for p in ids:
            sources={}
            for branch in ('low','high'):
                payload=zipped.read('our485/'+branch+'/'+p);path=DATA/branch/p
                if path.exists():assert path.read_bytes()==payload
                else:path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as im:
                    assert im.format=='PNG' and im.mode=='RGB' and payload[24]==8 and not im.info.get('icc_profile');rgb=np.array(im)
                images[p,branch]=rgb;hashes[branch].append(image_hashes(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)))
                sources[branch]={'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)}
            assert images[p,'low'].shape==images[p,'high'].shape
            jpeg=DATA/(Path(p).stem+'.jpg')
            if not jpeg.exists():Image.fromarray(images[p,'low']).save(jpeg,quality=95,subsampling=0)
            records.append({'id':p,'previously_inspected':p in previous,'sources':sources,
                            'jpeg_path':str(jpeg.relative_to(ROOT)),'jpeg_sha256':sha256_file(jpeg),
                            'alignment':alignment(images[p,'low'],images[p,'high'])})
    pairs={}
    for branch in ('low','high'):
        d,p=zip(*hashes[branch])
        for (i,j),scores in candidate_pairs(np.array(d),np.array(p),[None]*len(ids),10,12).items():
            pair=pairs.setdefault((i,j),{'left':ids[i],'right':ids[j],'matches':{},'decision':'unreviewed'});pair['matches'][branch]=scores
    old_groups=json.loads((PARENT/'split.json').read_text())['groups']
    write_json(OUT/'group_review.json',{'status':'pending','candidate_pairs':list(pairs.values()),
               'additional_groups':[g for g in old_groups.values() if len(g)>1],'review_notes':''})
    write_json(OUT/'manifest.json',{'photos':records,'source_archive_sha256':old['source_archive_sha256'],'selection_seed':20261008})
    font=ImageFont.load_default(size=14)
    def contact(selected,branch,path):
        sheet=Image.new('RGB',(1200,150*int(np.ceil(len(selected)/5))),'white');draw=ImageDraw.Draw(sheet)
        for i,p in enumerate(selected):
            x,y=i%5*240,i//5*150;im=ImageOps.contain(Image.fromarray(images[p,branch]),(235,120));sheet.paste(im,(x,y+25))
            draw.text((x+4,y+4),p+(' old' if p in previous else ' new'),font=font,fill='black')
        sheet.save(path,quality=94)
    for branch in ('low','high'):
        for page in range(6):contact(ids[page*20:(page+1)*20],branch,OUT/f'{branch}-scenes-{page+1}.jpg')
    special=[r['id'] for r in records if r['alignment']['status']=='quarantine_shift']
    special+= [r['id'] for r in records if r['alignment']['status']=='uncertain'][:6]
    if special:
        contact(special,'low',OUT/'alignment-low.jpg');contact(special,'high',OUT/'alignment-high.jpg')
    if pairs:
        sheet=Image.new('RGB',(600,180*len(pairs)),'white');draw=ImageDraw.Draw(sheet)
        for i,pair in enumerate(pairs.values()):
            for col,p in enumerate((pair['left'],pair['right'])):
                im=ImageOps.contain(Image.fromarray(images[p,'high']),(290,145));sheet.paste(im,(col*300,i*180+26));draw.text((col*300+5,i*180+5),p,font=font,fill='black')
        sheet.save(OUT/'group-candidates.jpg',quality=94)
    print('Prepared',len(records),'pairs;',len(pairs),'hash candidates; alignment statuses',
          {status:sum(r['alignment']['status']==status for r in records) for status in ('screen_pass','uncertain','quarantine_shift')},flush=True)


def parent_model():
    c=json.loads((PARENT/'config.json').read_text());r=json.loads((PARENT/'report.json').read_text())
    for path,h in c['source_sha256'].items():assert sha256_file(ROOT/path)==h
    checkpoint=PARENT/'candidate-42.pt';assert sha256_file(checkpoint)==r['runs']['candidate-42']['checkpoint_sha256']
    model=NoiseCandidate(c['feature_mean'],c['feature_scale']);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def loss(model,x,target):
    output=model(x)[...,3:-3,3:-3];raw=x[:,:3,...,3:-3,3:-3];reference=target[...,3:-3,3:-3]
    return quality_loss(output,reference,raw,output.new_zeros((len(x),3)))


def compatible_mps(x,target):
    if not torch.backends.mps.is_available():return {'available':False,'device':'cpu'}
    torch.manual_seed(42);cpu=SpatialCandidate();gpu=copy.deepcopy(cpu).to('mps')
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in (cpu,gpu)]
    max_forward=0.;max_gradient=0.
    for step in range(3):
        for m,opt,device in zip((cpu,gpu),optimizers,('cpu','mps')):
            opt.zero_grad();l=loss(m,x.to(device),target.to(device));assert torch.isfinite(l);l.backward()
        with torch.no_grad():max_forward=max(max_forward,float((cpu(x)-gpu(x.to('mps')).cpu()).abs().max()))
        for a,b in zip(cpu.parameters(),gpu.parameters()):
            assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
            max_gradient=max(max_gradient,float((a.grad-b.grad.cpu()).abs().max()))
        for opt in optimizers:opt.step()
    passed=max_forward<=1e-5 and max_gradient<=1e-5
    return {'available':True,'passed':passed,'max_forward_difference':max_forward,'max_gradient_difference':max_gradient,'steps':3,'device':'mps' if passed else 'cpu'}


def train():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve completed pilot')
    manifest=json.loads((OUT/'manifest.json').read_text());review=json.loads((OUT/'group_review.json').read_text())
    grouped=create_split(manifest['photos'],review)['groups'];old={p['id'] for p in manifest['photos'] if p['previously_inspected']}
    excluded={p['id']:p['alignment'] for p in manifest['photos'] if p['alignment']['status']=='quarantine_shift'}
    groups={g:[p for p in members if p not in excluded] for g,members in grouped.items()};groups={g:m for g,m in groups.items() if m}
    eligible=sorted(g for g in groups if not set(grouped[g])&old);chosen=[]
    for g in np.random.default_rng(20261009).permutation(eligible):
        if sum(len(groups[g]) for g in chosen)>=16:break
        chosen.append(str(g))
    assert len(chosen)>=8
    val=sorted(p for g in chosen for p in groups[g]);all_ids={p for m in groups.values() for p in m}
    split={'groups_all':grouped,'groups':groups,'train':sorted(all_ids-set(val)),'validation':val,
           'validation_groups':chosen,'excluded_alignment':excluded,'split_seed':20261009}
    assert not set(val)&old;write_json(OUT/'split.json',split);torch.set_num_threads(1)
    parent=parent_model();parent_state=copy.deepcopy(parent.state_dict());images={};bases={};targets={};cropped={}
    records={p['id']:p for p in manifest['photos']}
    for p in sorted(all_ids):
        e=records[p]
        for a in e['sources'].values():assert sha256_file(ROOT/a['path'])==a['sha256']
        assert sha256_file(ROOT/e['jpeg_path'])==e['jpeg_sha256'];rgb,_=decode_jpeg(ROOT/e['jpeg_path'])
        with Image.open(ROOT/e['sources']['high']['path']) as im:target=np.array(im)
        with torch.no_grad():g,a=parent.curve_parameters(torch.tensor(source_features(rgb)[None]))
        base=render_noise_candidate(rgb,float(g[0,0]),a[0].numpy())
        images[p]=rgb;bases[p]=base;targets[p]=target
        cropped[p]=(np.stack(list(crops(np.concatenate([rgb,base],axis=-1)))),np.stack(list(crops(target))))
    def tensors(ids):return tuple(torch.tensor(np.concatenate([cropped[p][i] for p in ids])) for i in range(2))
    train_x,train_y=tensors(split['train']);val_x,val_y=tensors(val)
    compatibility=compatible_mps(train_x[:4],train_y[:4]);device=compatibility['device'];write_json(OUT/'device_compatibility.json',compatibility)
    config={'lr':.001,'weight_decay':.0001,'batch':16,'max_epochs':30,'patience':6,'first_seed':42,'conditional_seeds':[43,44],'device':device,
            'protocol_sha256':sha256_file(OUT/'PROTOCOL.md'),'manifest_sha256':sha256_file(OUT/'manifest.json'),
            'review_sha256':sha256_file(OUT/'group_review.json'),'split_sha256':sha256_file(OUT/'split.json'),
            'parent_checkpoint_sha256':sha256_file(PARENT/'candidate-42.pt'),'parent_config_sha256':sha256_file(PARENT/'config.json'),
            'production_model_sha256':sha256_file(ROOT/'artifacts/model/model.onnx'),
            'source_sha256':{p:sha256_file(ROOT/p) for p in ('scripts/train_spatial_candidate.py','src/jpeg_spatial_candidate.py',
                'src/jpeg_noise_candidate.py','scripts/train_noise_candidate.py','src/jpeg_curve_candidate.py','scripts/train_jpeg_curve.py')}}
    write_json(OUT/'config.json',config);started=time.perf_counter();runs={};outputs={};summaries={}
    parent_metrics={p:native_metrics(bases[p],targets[p],images[p]) for p in val}
    def average(scores):return {m:float(np.mean([s[m] for s in scores if s[m] is not None])) for m in scores[0]}
    parent_mean=average(list(parent_metrics.values()))
    def gates(mean):return {'psnr_plus_half_db':mean['psnr_db']>=parent_mean['psnr_db']+.5,
                          'dark_variation_not_worse':mean['dark_median_residual_codes']<=parent_mean['dark_median_residual_codes'],
                          'chroma_not_worse':mean['chroma_mse']<=parent_mean['chroma_mse'],
                          'gradient_within_110pct':mean['luma_gradient_l1']<=1.1*parent_mean['luma_gradient_l1'],
                          'new_full_channels_below_half_pct':mean['new_full_channel_fraction']<=.005}
    for seed in (42,43,44):
        if seed!=42 and not all(runs['seed42']['numeric_gates'].values()):break
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SpatialCandidate().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        with torch.no_grad():best=float(loss(model,val_x.to(device),val_y.to(device)).cpu())
        initial=best;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=0;history=[];run_start=time.perf_counter()
        for epoch in range(1,31):
            order=rng.permutation(len(train_x));train_sum=0.
            for offset in range(0,len(train_x),16):
                ids=order[offset:offset+16];optimizer.zero_grad();l=loss(model,train_x[ids].to(device),train_y[ids].to(device))
                assert torch.isfinite(l);l.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step();train_sum+=float(l.detach().cpu())*len(ids)
            with torch.no_grad():v=float(loss(model,val_x.to(device),val_y.to(device)).cpu())
            history.append({'epoch':epoch,'train_loss':train_sum/len(train_x),'validation_loss':v})
            if v<best:best=v;best_epoch=epoch;state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            if epoch%5==0:print('seed',seed,'epoch',epoch,'validation',round(v,6),flush=True)
            if epoch-best_epoch>=6:break
        model=SpatialCandidate();model.load_state_dict(state);model.eval();torch.save(state,OUT/f'seed{seed}.pt')
        outputs[f'seed{seed}']={p:predict_spatial(model,images[p],bases[p]) for p in val+['102.png','27.png'] if p in images}
        mean=average([native_metrics(outputs[f'seed{seed}'][p],targets[p],images[p]) for p in val]);summaries[f'seed{seed}']=mean
        runs[f'seed{seed}']={'best_epoch':best_epoch,'epochs_run':len(history),'initial_loss':initial,'best_loss':best,
                            'history':history,'numeric_gates':gates(mean),'elapsed_seconds':time.perf_counter()-run_start,
                            'checkpoint_sha256':sha256_file(OUT/f'seed{seed}.pt')}
        print('Seed',seed,'native PSNR',mean['psnr_db'],'gates',gates(mean),flush=True)
    assert all(torch.equal(parent.state_dict()[k],v) for k,v in parent_state.items())
    photos=[];font=ImageFont.load_default(size=15)
    selected=val+['102.png','27.png']
    for index,p in enumerate(selected):
        if p not in images:continue
        pred,linear=predict_jpeg(ROOT/records[p]['jpeg_path']);params={k:pred['experimental_absolute'][k] for k in ('Exposure','HighlightRecovery')}
        variants={'identity':images[p],'existing-v3':encode_png(apply_tone(linear,params)[0])[1],'frozen-parent':bases[p]}
        variants.update({label:values[p] for label,values in outputs.items()})
        entry={'id':p,'partition':'validation' if p in val else 'old training regression','alignment':records[p]['alignment'],
               'metrics':{label:native_metrics(image,targets[p],images[p]) for label,image in variants.items()}}
        photos.append(entry)
        if index<8 or p in ('102.png','27.png'):
            shown=[('Existing v3',variants['existing-v3']),('Frozen shared base',bases[p]),('Spatial seed42',outputs['seed42'][p]),('Reference',targets[p])]
            sheet=Image.new('RGB',(1200,270),'white');draw=ImageDraw.Draw(sheet);crop=Image.new('RGB',(1200,330),'white');cd=ImageDraw.Draw(crop)
            for col,(name,image) in enumerate(shown):
                im=ImageOps.contain(Image.fromarray(image),(290,200));sheet.paste(im,(col*300,35));draw.text((col*300+5,7),name,font=font,fill='black')
                h,w=image.shape[:2];tile=Image.fromarray(image).crop((w//2-96,h//2-96,w//2+96,h//2+96)).resize((300,300),Image.Resampling.NEAREST)
                crop.paste(tile,(col*300,30));cd.text((col*300+5,7),name,font=font,fill='black')
            draw.text((5,245),p+' | '+entry['partition'],font=font,fill='black');sheet.save(OUT/(Path(p).stem+'-comparison.jpg'),quality=94);crop.save(OUT/(Path(p).stem+'-crop.jpg'),quality=94)
    other={label:average([p['metrics'][label] for p in photos if p['partition']=='validation']) for label in ('identity','existing-v3','frozen-parent')}
    write_json(OUT/'report.json',{'split':split,'runs':runs,'summary':dict(other,**summaries),'photos':photos,
                                 'parent_unchanged':True,'elapsed_seconds':time.perf_counter()-started,
                                 'scope':'Small development pilot; final CPU native metrics; uncertain alignment retained and recorded; no final test or production promotion.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);prepare() if args.stage=='prepare' else train()
