"""Train on frozen embeddings; select on validation before one final test pass."""
import argparse
import copy
import hashlib
import io
import json
import random
from pathlib import Path
import time

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
            prediction=model.head(z[val_ids]); score=float((prediction-target[val_ids]).abs().mean())
        history.append({'epoch':epoch+1,'training_loss':float(np.mean(losses)),'val_macro_normalized_mae':score})
        if score < best_score-1e-6:
            best_score=score; best=copy.deepcopy(model.head.state_dict()); stale=0
        else: stale+=1
        if epoch%10==0: print(f'{mode} epoch {epoch+1}: validation MAE={score:.5f}',flush=True)
        if stale>=config['patience']: break
    model.head.load_state_dict(best); model.eval()
    with torch.inference_mode(): prediction=model.head(z).numpy()
    return prediction,{'best_val_macro_normalized_mae':best_score,'epochs_run':len(history),'elapsed_seconds':time.perf_counter()-started,'history':history}


def train(metadata_path,output_dir,epochs=120,seed=42,batch_size=128,pilot=False):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset=ShotSenseDataset(metadata_path)
    if dataset.scope not in ('full','pilot'): raise ValueError('Training requires an audited preprocessing manifest')
    splits=load_or_create_splits(dataset,pilot=pilot,seed=seed)
    index={photo_id:i for i,photo_id in enumerate(dataset.ids.tolist())}
    groups={name:np.array([index[photo_id] for photo_id in ids]) for name,ids in splits.items()}
    train_ids,val_ids,test_ids=[groups[name] for name in ('train','val','test')]
    mean=dataset.X[train_ids].mean(axis=0); scale=dataset.X[train_ids].std(axis=0); scale[scale==0]=1
    model=ShotSenseModel(mean,scale)
    frozen_state={name:value.clone() for name,value in model.backbone.state_dict().items()}
    physical=dataset.X; target=dataset.labels.numpy()
    features=semantic_features(model,dataset,output_dir/'semantic_features.npz')
    config={'seed':seed,'epochs':epochs,'patience':20,'learning_rate':.001,'weight_decay':.01,'batch_size':batch_size,
            'device':'cpu','threads':4,'loss':'smooth_l1, beta=0.1, equal parameter weights','weights':WEIGHTS_NAME,
            'metadata_sha256':dataset.data_hash,'config_hash':dataset.data_version,'pilot':pilot}
    config.update({'decoder_sha256':sha256_file(ROOT/'src/image_io.py'),'pillow':PIL.__version__,
                   'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                   'weights_sha256':sha256_file(ROOT/'artifacts/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'),
                   'split_sha256':hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
                   'implementation_sha256':{name:sha256_file(ROOT/'src'/name) for name in ('train.py','model.py','dataset.py','evaluation.py')}})
    constant=np.broadcast_to(np.median(target[train_ids],axis=0),(len(dataset),6)).copy()
    design=np.column_stack(((physical-mean)/scale,np.ones(len(dataset))))
    coefficients=np.linalg.lstsq(design[train_ids],target[train_ids],rcond=None)[0]
    linear=np.clip(design@coefficients,-1,1).astype(np.float32)
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
            'limitations':['Photo-ID split; burst/near-duplicate groups have not been supplied.',
                           'Absolute Temperature/Tint are experimental without online white-balance context.',
                           'Legacy PV2003 fields; six parameters do not reproduce full expert renderings.']}
    base=np.array(report['validation']['constant']['mae']); actual=np.array(report['validation']['dual']['mae'])
    supported=(actual<base).tolist()
    report['model_gate_passed']=bool(report['validation']['dual']['macro_normalized_mae']<report['validation']['constant']['macro_normalized_mae'])
    report['validated_parameters']=[name for i,name in enumerate(PARAMS) if supported[i] and name not in ('Temperature','Tint')] if not pilot and report['model_gate_passed'] else []
    report['experimental_parameters']=[name for name in PARAMS if name not in report['validated_parameters']]
    # Test is opened only after architecture, hyperparameters and checkpoint selection are final.
    report['test']={name:metrics(values[test_ids],target[test_ids]) for name,values in predictions.items()}
    if not pilot:
        report['stratified_diagnostics']=diagnostics(dataset,predictions,groups,train_ids,metrics)
    checkpoint={'state_dict':model.state_dict(),'config':config,'parameter_order':list(PARAMS),
                'parameter_min':PARAM_MIN.tolist(),'parameter_max':PARAM_MAX.tolist(),
                'validated_parameters':report['validated_parameters'],'experimental_parameters':report['experimental_parameters'],
                'pipeline_config':json.loads((Path(metadata_path).parent/'manifest.json').read_text())['config']}
    buffer=io.BytesIO(); torch.save(checkpoint,buffer); atomic_bytes(output_dir/'best.pt',buffer.getvalue())
    atomic_bytes(output_dir/'evaluation.json',json.dumps(report,indent=2,allow_nan=False).encode())
    print(json.dumps({'model_gate_passed':report['model_gate_passed'],'validated_parameters':report['validated_parameters'],
                      'validation_dual_mae':actual.tolist(),'output_dir':str(output_dir)}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/model')
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('epochs and batch-size must be positive')
    train(args.metadata,args.output_dir,args.epochs,args.seed,args.batch_size,args.pilot)


if __name__=='__main__': main()
