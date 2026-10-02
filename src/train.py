"""Validation-only frozen-embedding experiments and explicit final evaluation."""
import argparse
import copy
import hashlib
import io
import json
import random
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision
import PIL
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.dataset import ShotSenseDataset,ParameterNormalizer,load_or_create_splits
from src.model import ROOT,ShotSenseModel,WEIGHTS_NAME
from src.parameters import PARAMS,PARAM_MIN,PARAM_MAX
from src.preprocess import atomic_bytes,atomic_npz,sha256_file
from src.evaluation import diagnostics


def metrics(prediction, target):
    raw_pred=ParameterNormalizer.denormalize(np.clip(prediction,-1,1))
    raw_true=ParameterNormalizer.denormalize(target)
    error=raw_pred-raw_true
    return {'normalized_mae':np.abs(prediction-target).mean(axis=0).tolist(),
            'macro_normalized_mae':float(np.abs(prediction-target).mean()),
            'p2':float(np.abs(prediction-target)[:,[0,5]].mean()/2),
            'mae':np.abs(error).mean(axis=0).tolist(),
            'rmse':np.sqrt((error**2).mean(axis=0)).tolist(),
            'temperature_mired_mae':float(np.abs(1e6/raw_pred[:,3]-1e6/raw_true[:,3]).mean())}


def semantic_features(model,dataset,cache_path,batch_size=32):
    cache_path=Path(cache_path)
    signature={'metadata_sha256':dataset.data_hash,'decoder_sha256':sha256_file(ROOT/'src/image_io.py'),
               'weights_sha256':sha256_file(ROOT/'artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'),
               'torchvision':str(torchvision.__version__),'pillow':PIL.__version__,
               'weights':WEIGHTS_NAME,'image_sha256':hashlib.sha256(''.join(sha256_file(dataset.images_dir/(Path(i).stem+'.jpg')) for i in dataset.ids).encode()).hexdigest()}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    if cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as data:
            if str(data['fingerprint'])==fingerprint and np.array_equal(data['ids'],dataset.ids):
                features=data['features'].copy()
                if features.shape==(len(dataset),1280) and features.dtype==np.float32 and np.isfinite(features).all():
                    return features
    rows=[]; model.eval()
    with torch.inference_mode():
        for step,(image,_,_) in enumerate(DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)):
            rows.append(model.backbone(image).numpy())
            if step%20==0: print(f'Embedding images {min((step+1)*batch_size,len(dataset))}/{len(dataset)}',flush=True)
    features=np.concatenate(rows).astype(np.float32)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(cache_path,features=features,ids=dataset.ids,fingerprint=np.asarray(fingerprint))
    return features


def fit_head(model,features,physical,targets,train_indices,val_indices,config,mode='dual'):
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    model.train()
    z=model.fused_features(torch.from_numpy(features),torch.from_numpy(physical)).detach()
    if mode=='semantic': z[:,1280:]=0
    target=torch.from_numpy(targets)
    train_ids=torch.as_tensor(train_indices,dtype=torch.int64); val_ids=torch.as_tensor(val_indices,dtype=torch.int64)
    with torch.no_grad():
        model.head[-2].bias.copy_(torch.atanh(target[train_ids].mean(dim=0).clamp(-.99,.99)))
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best_score=float('inf'); best=None; history=[]; stale=0
    generator=torch.Generator().manual_seed(config['seed'])
    started=time.perf_counter()
    for epoch in range(config['epochs']):
        order=train_ids[torch.randperm(len(train_ids),generator=generator)]
        losses=[]
        for batch in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            prediction=model.head(z[batch])
            loss=F.smooth_l1_loss(prediction,target[batch],beta=.1)
            if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        with torch.inference_mode():
            prediction=model.head(z[val_ids]); error=(prediction-target[val_ids]).abs()
            score=float(error[:,[0,5]].mean()/2)
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),
                        'val_p2':score,'val_macro_normalized_mae':float(error.mean())})
        if score < best_score-1e-6:
            best_score=score; best=copy.deepcopy(model.head.state_dict()); stale=0
        else: stale+=1
        if epoch%10==0: print(f'{mode} epoch {epoch+1}: validation P2={score:.5f}',flush=True)
        if stale>=config['patience']: break
    model.head.load_state_dict(best); model.eval()
    prediction=np.full((len(targets),6),np.nan,dtype=np.float32)
    with torch.inference_mode(): prediction[val_indices]=model.head(z[val_ids]).numpy()
    return prediction,{'best_val_p2':best_score,'epochs_run':len(history),'elapsed_seconds':time.perf_counter()-started,'history':history}


def train(metadata_path,output_dir,epochs=120,seed=42,batch_size=128,pilot=False,*,
          split_seed=42,split_path=None,groups_path=None,learning_rate=.001,weight_decay=.01):
    """Never overwrite a run; record failures as well as completed experiments."""
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    status={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
            'requested_config':{'metadata':str(metadata_path),'epochs':epochs,'seed':seed,
                                'batch_size':batch_size,'pilot':pilot,'split_seed':split_seed,
                                'split_path':str(split_path) if split_path else None,
                                'groups_path':str(groups_path) if groups_path else None,
                                'learning_rate':learning_rate,'weight_decay':weight_decay}}
    atomic_bytes(output_dir/'run_status.json',json.dumps(status,indent=2).encode())
    try:
        report=_train(metadata_path,output_dir,epochs,seed,batch_size,pilot,
                      split_seed,split_path,groups_path,learning_rate,weight_decay)
    except Exception as error:
        status.update(status='failed',error_type=type(error).__name__,reason=str(error))
        raise
    else:
        status['status']='complete'
        return report
    finally:
        status['finished_utc']=datetime.now(timezone.utc).isoformat()
        atomic_bytes(output_dir/'run_status.json',json.dumps(status,indent=2).encode())


def _train(metadata_path,output_dir,epochs,seed,batch_size,pilot,
           split_seed,split_path,groups_path,learning_rate,weight_decay):
    if epochs<1 or batch_size<1 or not np.isfinite(learning_rate) or learning_rate<=0 or not np.isfinite(weight_decay) or weight_decay<0:
        raise ValueError('Invalid epochs, batch size, learning rate or weight decay')
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset=ShotSenseDataset(metadata_path)
    if dataset.scope not in ('full','pilot'): raise ValueError('Training requires an audited preprocessing manifest')
    photo_groups=json.loads(Path(groups_path).read_text()) if groups_path else None
    split_path=Path(split_path or output_dir/('splits.pilot.json' if pilot else 'splits.json'))
    splits=load_or_create_splits(dataset,split_path,pilot=pilot,seed=split_seed,groups=photo_groups)
    atomic_bytes(output_dir/'split_manifest.json',split_path.read_bytes())
    index={photo_id:i for i,photo_id in enumerate(dataset.ids.tolist())}
    groups={name:np.array([index[photo_id] for photo_id in ids]) for name,ids in splits.items()}
    train_ids,val_ids=[groups[name] for name in ('train','val')]
    mean=dataset.X[train_ids].mean(axis=0); scale=dataset.X[train_ids].std(axis=0); scale[scale==0]=1
    model=ShotSenseModel(mean,scale)
    frozen_state={name:value.clone() for name,value in model.backbone.state_dict().items()}
    physical=dataset.X; target=dataset.labels.numpy()
    config={'seed':seed,'split_seed':split_seed,'epochs':epochs,'patience':20,'learning_rate':learning_rate,'weight_decay':weight_decay,'batch_size':batch_size,
            'protocol':'validation-only-v1','checkpoint_selection':'validation_p2',
            'split_manifest_sha256':sha256_file(output_dir/'split_manifest.json'),
            'groups_sha256':sha256_file(groups_path) if groups_path else None,
            'device':'cpu','threads':4,'loss':'smooth_l1, beta=0.1, equal parameter weights','weights':WEIGHTS_NAME,
            'metadata_sha256':dataset.data_hash,'config_hash':dataset.data_version,'pilot':pilot}
    config.update({'decoder_sha256':sha256_file(ROOT/'src/image_io.py'),'pillow':PIL.__version__,
                   'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                   'weights_sha256':sha256_file(ROOT/'artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'),
                   'split_sha256':hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
                   'implementation_sha256':{name:sha256_file(ROOT/'src'/name) for name in ('train.py','model.py','dataset.py','evaluation.py')}})
    atomic_bytes(output_dir/'config.json',json.dumps(config,indent=2,allow_nan=False).encode())
    features=semantic_features(model,dataset,output_dir/'semantic_features.npz')
    config['semantic_features_sha256']=hashlib.sha256(features.tobytes()).hexdigest()
    atomic_bytes(output_dir/'config.json',json.dumps(config,indent=2,allow_nan=False).encode())
    constant_value=target[train_ids].mean(axis=0)
    constant=np.full((len(dataset),6),np.nan,dtype=np.float32); constant[val_ids]=constant_value
    design=np.column_stack(((physical-mean)/scale,np.ones(len(dataset))))
    coefficients=np.linalg.lstsq(design[train_ids],target[train_ids],rcond=None)[0]
    linear=np.full((len(dataset),6),np.nan,dtype=np.float32)
    linear[val_ids]=np.clip(design[val_ids]@coefficients,-1,1).astype(np.float32)
    predictions={'constant':constant,'physical_linear':linear}
    predictions['dual'],training=fit_head(model,features,physical,target,train_ids,val_ids,config)
    chosen_state=copy.deepcopy(model.state_dict())
    # Same initial seed and backbone, but no physical information reaches this ablation.
    torch.manual_seed(seed)
    semantic=ShotSenseModel(mean,scale,pretrained=False)
    semantic.backbone.load_state_dict(model.backbone.state_dict())
    predictions['semantic'],semantic_training=fit_head(semantic,features,physical,target,train_ids,val_ids,config,mode='semantic')
    model.load_state_dict(chosen_state)
    if any(not torch.equal(value,frozen_state[name]) for name,value in model.backbone.state_dict().items()):
        raise ValueError('Frozen backbone parameters or BatchNorm buffers changed during training')
    report={'config':config,'parameter_order':list(PARAMS),'split_sizes':{n:len(v) for n,v in groups.items()},
            'validation':{name:metrics(values[val_ids],target[val_ids]) for name,values in predictions.items()},
            'training':training,'semantic_training':semantic_training,'backbone_unchanged':True,
            'expert_disagreement_mean':dataset.Y_std[train_ids].mean(axis=0).tolist(),
            'evaluation_scope':'validation_only','constant_baseline':'training_mean',
            'limitations':[('Provided photo groups; completeness depends on the recorded grouping audit.' if photo_groups else 'Photo-ID split; burst/near-duplicate groups have not been supplied.'),
                           'Absolute Temperature/Tint are experimental without online white-balance context.',
                           'Legacy PV2003 fields; six parameters do not reproduce full expert renderings.']}
    base=np.array(report['validation']['constant']['mae']); actual=np.array(report['validation']['dual']['mae'])
    supported=(actual<base).tolist()
    report['validation_baseline_passed']=bool(report['validation']['dual']['p2']<report['validation']['constant']['p2'])
    report['promotion_status']='not_assessed; validation-only experiment'
    report['candidate_parameters']=[name for i,name in enumerate(PARAMS) if supported[i] and name not in ('Temperature','Tint')]
    report['validated_parameters']=[]  # Experiment scores do not authorize release promotion.
    report['experimental_parameters']=[name for name in PARAMS if name not in report['validated_parameters']]
    if not pilot:
        report['stratified_diagnostics']=diagnostics(dataset,predictions,{'val':val_ids},train_ids,metrics)
    checkpoint={'state_dict':model.state_dict(),'config':config,'parameter_order':list(PARAMS),
                'parameter_min':PARAM_MIN.tolist(),'parameter_max':PARAM_MAX.tolist(),
                'validated_parameters':report['validated_parameters'],'experimental_parameters':report['experimental_parameters'],
                'pipeline_config':json.loads((Path(metadata_path).parent/'manifest.json').read_text())['config']}
    buffer=io.BytesIO(); torch.save(checkpoint,buffer); atomic_bytes(output_dir/'best.pt',buffer.getvalue())
    controls={'semantic_head':semantic.head.state_dict(),'constant':torch.from_numpy(constant_value),
              'physical_coefficients':torch.from_numpy(coefficients)}
    buffer=io.BytesIO(); torch.save(controls,buffer); atomic_bytes(output_dir/'controls.pt',buffer.getvalue())
    atomic_npz(output_dir/'validation_predictions.npz',ids=dataset.ids[val_ids],
               **{name:values[val_ids] for name,values in predictions.items()})
    atomic_bytes(output_dir/'evaluation.json',json.dumps(report,indent=2,allow_nan=False).encode())
    frozen={'schema_version':1,'files':{name:sha256_file(output_dir/name) for name in
            ('best.pt','controls.pt','config.json','split_manifest.json','evaluation.json','validation_predictions.npz')},
            'note':'Final evaluation is explicit; validation does not grant promotion.'}
    atomic_bytes(output_dir/'frozen_run.json',json.dumps(frozen,indent=2).encode())
    print(json.dumps({'validation_baseline_passed':report['validation_baseline_passed'],'validated_parameters':report['validated_parameters'],
                      'validation_dual_mae':actual.tolist(),'output_dir':str(output_dir)}),flush=True)
    return report


def final_evaluate(metadata_path,run_dir):
    """Evaluate frozen controls/checkpoints once; log access before test prediction."""
    run_dir=Path(run_dir)
    frozen=json.loads((run_dir/'frozen_run.json').read_text())
    for name,expected in frozen['files'].items():
        if sha256_file(run_dir/name)!=expected:
            raise ValueError('Frozen run artifact changed: '+name)
    checkpoint=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
    config=checkpoint['config']
    if config.get('protocol')!='validation-only-v1': raise ValueError('Only frozen validation-only runs can use this evaluator')
    if config!=json.loads((run_dir/'config.json').read_text()): raise ValueError('Checkpoint configuration mismatch')
    for name,expected in config['implementation_sha256'].items():
        if sha256_file(ROOT/'src'/name)!=expected: raise ValueError('Evaluation implementation changed: '+name)
    for name,expected in checkpoint['pipeline_config'].get('implementation_sha256',{}).items():
        if sha256_file(ROOT/'src'/name)!=expected: raise ValueError('Preprocessing implementation changed: '+name)
    if sha256_file(ROOT/'src/image_io.py')!=config['decoder_sha256'] or PIL.__version__!=config['pillow']:
        raise ValueError('Semantic decoder differs from the frozen run')
    if str(torch.__version__)!=config['torch'] or str(torchvision.__version__)!=config['torchvision']:
        raise ValueError('Training framework differs from the frozen run')
    dataset=ShotSenseDataset(metadata_path)
    if dataset.data_hash!=config['metadata_sha256'] or dataset.data_version!=config['config_hash']:
        raise ValueError('Final evaluation data differs from frozen training data')
    split_manifest=json.loads((run_dir/'split_manifest.json').read_text())
    from src.dataset import _validate_splits
    _validate_splits(split_manifest['splits'],dataset.ids.tolist())
    index={photo_id:i for i,photo_id in enumerate(dataset.ids.tolist())}
    groups={name:np.array([index[i] for i in ids]) for name,ids in split_manifest['splits'].items()}
    event={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
           'frozen_run_sha256':sha256_file(run_dir/'frozen_run.json'),'checkpoint_sha256':sha256_file(run_dir/'best.pt'),
           'test_count':len(groups['test']),'warning':'Test has been accessed; do not retune against it.'}
    # Exclusive creation also prevents concurrent/repeated final evaluation of this run.
    with (run_dir/'final_test_access.json').open('x') as handle: json.dump(event,handle,indent=2)
    try:
        torch.set_num_threads(4)
        model=ShotSenseModel(pretrained=False); model.load_state_dict(checkpoint['state_dict']); model.eval()
        features=semantic_features(model,dataset,run_dir/'semantic_features.npz')
        if hashlib.sha256(features.tobytes()).hexdigest()!=config['semantic_features_sha256']:
            raise ValueError('Semantic inputs/features differ from the frozen run')
        controls=torch.load(run_dir/'controls.pt',map_location='cpu',weights_only=True)
        selected=groups['test']; physical=dataset.X
        predictions={name:np.full((len(dataset),6),np.nan,dtype=np.float32)
                     for name in ('dual','semantic','constant','physical_linear')}
        with torch.inference_mode():
            fused=model.fused_features(torch.from_numpy(features[selected]),torch.from_numpy(physical[selected]))
            predictions['dual'][selected]=model.head(fused).numpy()
            model.head.load_state_dict(controls['semantic_head']); fused[:,1280:]=0
            predictions['semantic'][selected]=model.head(fused).numpy()
        predictions['constant'][selected]=controls['constant'].numpy()
        design=np.column_stack(((physical[selected]-model.physical_mean.numpy())/model.physical_scale.numpy(),np.ones(len(selected))))
        predictions['physical_linear'][selected]=np.clip(design@controls['physical_coefficients'].numpy(),-1,1)
        report={'config':config,'test':{name:metrics(values[selected],dataset.labels.numpy()[selected]) for name,values in predictions.items()},
                'promotion_status':'not_assessed; compare candidates, uncertainty and deployment gates separately'}
        if not config['pilot']:
            report['stratified_diagnostics']=diagnostics(dataset,predictions,{'test':selected},groups['train'],metrics)
        atomic_bytes(run_dir/'final_evaluation.json',json.dumps(report,indent=2,allow_nan=False).encode())
        event['status']='complete'
        return report
    except Exception as error:
        event.update(status='failed',reason=str(error)); raise
    finally:
        event['finished_utc']=datetime.now(timezone.utc).isoformat()
        atomic_bytes(run_dir/'final_test_access.json',json.dumps(event,indent=2).encode())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--output-dir',type=Path,required=True,help='New experiment directory; existing directories are never overwritten')
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--split-path',type=Path)
    parser.add_argument('--groups',type=Path,help='JSON photo-ID to group-ID mapping')
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--final-evaluate',action='store_true',help='Evaluate an existing frozen run once; does not train')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('epochs and batch-size must be positive')
    if args.final_evaluate:
        final_evaluate(args.metadata,args.output_dir)
    else:
        train(args.metadata,args.output_dir,args.epochs,args.seed,args.batch_size,args.pilot,
              split_seed=args.split_seed,split_path=args.split_path,groups_path=args.groups,
              learning_rate=args.learning_rate,weight_decay=args.weight_decay)


if __name__=='__main__': main()
