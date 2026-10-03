"""Validation-only last-stage pilot with a version-checked frozen-prefix cache."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.dataset import ShotSenseDataset
from src.evaluation import diagnostics
from src.finetune_candidate import FineTuneCandidate
from src.model import ROOT
from src.preprocess import atomic_bytes, atomic_npz, sha256_file
from src.train import metrics

BASE=ROOT/'artifacts/experiments/model_vnext'


def write_json(path,value):
    atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def prefix_cache(model,dataset,ids,indices):
    prefix=model.backbone.features[:7].eval()
    signature={'metadata_sha256':dataset.data_hash,'ids':ids,
               'image_bytes_sha256':hashlib.sha256(''.join(sha256_file(dataset.images_dir/(Path(photo).stem+'.jpg')) for photo in ids).encode()).hexdigest(),
               'prefix_weights_sha256':hashlib.sha256(b''.join(value.numpy().tobytes() for value in prefix.state_dict().values())).hexdigest(),
               'cache_builder_sha256':hashlib.sha256(inspect.getsource(prefix_cache).encode()).hexdigest(),
               'torch':str(torch.__version__),'source_sha256':{name:sha256_file(ROOT/name) for name in ('src/model.py','src/dataset.py','src/image_io.py')}}
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    path=ROOT/'data/processed/model_vnext/frozen-prefix.npz'
    if path.exists():
        recorded=json.loads(path.with_suffix('.json').read_text())
        if sha256_file(path)!=recorded['sha256'] or recorded['fingerprint']!=fingerprint: raise ValueError('Changed frozen-prefix cache artifact')
        with np.load(path,allow_pickle=False) as cache:
            if str(cache['fingerprint'])!=fingerprint or cache['ids'].tolist()!=ids: raise ValueError('Frozen-prefix cache version mismatch')
            features=cache['features'].copy()
    else:
        batches=[]; start=time.perf_counter()
        with torch.inference_mode():
            for step,(image,_,_) in enumerate(DataLoader(Subset(dataset,indices),batch_size=32,shuffle=False,generator=torch.Generator().manual_seed(0))):
                batches.append(prefix(image).numpy())
                if step%20==0: print(f'Frozen prefix {min((step+1)*32,len(ids))}/{len(ids)}',flush=True)
        features=np.concatenate(batches).astype(np.float32)
        atomic_npz(path,features=features,ids=np.array(ids),fingerprint=np.asarray(fingerprint))
        write_json(path.with_suffix('.json'),{'signature':signature,'fingerprint':fingerprint,'build_seconds':time.perf_counter()-start,'sha256':sha256_file(path)})
    if features.shape!=(len(ids),192,7,7) or features.dtype!=np.float32 or not np.isfinite(features).all(): raise ValueError('Invalid frozen-prefix activations')
    return torch.from_numpy(features),fingerprint


def parity(model,dataset,indices,features):
    selected=np.linspace(0,len(indices)-1,8,dtype=int)
    model.eval()
    with torch.inference_mode():
        rows=[dataset[indices[i]] for i in selected]
        image=torch.stack([row[0] for row in rows]); physical=torch.stack([row[1] for row in rows])
        device=next(model.parameters()).device
        online=model(image.to(device),physical.to(device)); cached=model.forward_tail(features[selected].to(device),physical.to(device))
    delta=float((online-cached).abs().max())
    if delta>1e-5: raise ValueError('Frozen-prefix/full-image parity failed')
    return delta


def train(seed=42,device='cpu'):
    if device=='mps':
        gate=json.loads((BASE/'mps_compatibility.json').read_text())
        if not torch.backends.mps.is_available() or gate['status']!='passed' or gate['torch']!=str(torch.__version__): raise ValueError('MPS compatibility gate missing')
    parent=BASE/('control-seed42' if seed==42 else f'head-h1-seed{seed}')
    for name,expected in json.loads((parent/'frozen_run.json').read_text())['files'].items():
        if sha256_file(parent/name)!=expected: raise ValueError('Changed parent artifact: '+name)
    parent_config=json.loads((parent/'config.json').read_text())
    if parent_config['seed']!=seed: raise ValueError('Parent seed mismatch')
    dataset=ShotSenseDataset(ROOT/'data/processed/metadata.npz')
    if dataset.data_hash!=parent_config['metadata_sha256']: raise ValueError('Dataset mismatch')
    split=json.loads((parent/'split_manifest.json').read_text())['splits']
    index={photo:i for i,photo in enumerate(dataset.ids.tolist())}
    ids=split['train']+split['val']; indices=[index[photo] for photo in ids]; n=len(split['train'])
    output=BASE/f'finetune-seed{seed}'; output.mkdir(exist_ok=False)
    status={'status':'running'}; started=time.perf_counter()
    try:
        torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        checkpoint=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
        model=FineTuneCandidate(); model.load_state_dict(checkpoint['state_dict'])
        initial={name:value.clone() for name,value in model.state_dict().items()}
        allowed={name for name,param in model.named_parameters() if param.requires_grad}
        features,fingerprint=prefix_cache(model,dataset,ids,indices)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        physical=torch.from_numpy(dataset.X[indices]); targets=dataset.labels[indices]
        model.to(device)
        config={'seed':seed,'split_seed':42,'epochs':30,'patience':5,'batch_size':16,'stage_lr':1e-5,'head_lr':1e-4,'weight_decay':.01,
                'loss':'smooth_l1 beta=.1 equal six-field weights','checkpoint_selection':'validation_p2 including parent epoch 0',
                'scope':'validation_only','stage':'backbone.features.7','trainable_parameters':sorted(allowed),
                'metadata_sha256':dataset.data_hash,'parent_frozen_sha256':sha256_file(parent/'frozen_run.json'),
                'prefix_cache_fingerprint':fingerprint,'source_sha256':{name:sha256_file(ROOT/name) for name in ('scripts/train_finetuning.py','src/finetune_candidate.py','src/model.py','src/dataset.py','src/image_io.py','src/evaluation.py')},
                'threads':4,'training_device':device,'final_evaluation_device':'cpu','torch':str(torch.__version__)}
        write_json(output/'config.json',config); atomic_bytes(output/'split_manifest.json',(parent/'split_manifest.json').read_bytes())
        before=parity(model,dataset,indices,features)
        train_loader=DataLoader(TensorDataset(features[:n],physical[:n],targets[:n]),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed))
        val_loader=DataLoader(TensorDataset(features[n:],physical[n:],targets[n:]),batch_size=16,generator=torch.Generator().manual_seed(seed))
        def predict():
            model.eval()
            with torch.inference_mode(): return np.concatenate([model.forward_tail(x.to(device),p.to(device)).cpu().numpy() for x,p,_ in val_loader])
        initial_prediction=predict(); parent_p2=metrics(initial_prediction,targets[n:].numpy())['p2']
        reference=json.loads((parent/'evaluation.json').read_text())['validation']['dual']['p2']
        if not np.isclose(parent_p2,reference,atol=1e-7): raise ValueError('Parent score mismatch')
        best=parent_p2; best_state=copy.deepcopy(model.state_dict()); best_epoch=0; stale=0; history=[]
        optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01)
        for epoch in range(1,31):
            epoch_start=time.perf_counter(); model.train(); losses=[]
            for x,p,y in train_loader:
                optimizer.zero_grad(set_to_none=True); prediction=model.forward_tail(x.to(device),p.to(device))
                loss=torch.nn.functional.smooth_l1_loss(prediction,y.to(device),beta=.1)
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                loss.backward()
                gradients=[]
                for name,param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is None: raise ValueError('Missing gradient: '+name)
                        gradients.append(torch.isfinite(param.grad).all())
                    elif param.grad is not None: raise ValueError('Frozen gradient: '+name)
                if not torch.stack(gradients).all(): raise ValueError('Non-finite gradients')
                optimizer.step(); losses.append(float(loss.detach()))
            for name,value in model.state_dict().items():
                if name not in allowed and not torch.equal(value.cpu(),initial[name]): raise ValueError('Frozen state changed: '+name)
            validation=metrics(predict(),targets[n:].numpy()); score=validation['p2']
            history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_p2':score,'seconds':time.perf_counter()-epoch_start})
            if score<best:
                best=score; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
            else: stale+=1
            write_json(output/'progress.json',{'history':history,'best_epoch':best_epoch,'best_validation_p2':best})
            print(f'Epoch {epoch}: P2 {score:.6f}; best {best:.6f}; {history[-1]["seconds"]:.2f}s',flush=True)
            if stale>=5: break
        model.load_state_dict(best_state)
        gpu_selected=predict()
        model.cpu(); device='cpu'
        selected=predict(); after=parity(model,dataset,indices,features)
        device_delta=float(np.max(np.abs(selected-gpu_selected)))
        if device_delta>1e-5: raise ValueError('Final CPU/device prediction parity failed')
        predictions={}; val_ids=np.array([index[photo] for photo in split['val']]); train_ids=np.array([index[photo] for photo in split['train']])
        with np.load(parent/'validation_predictions.npz',allow_pickle=False) as values:
            if values['ids'].tolist()!=split['val']: raise ValueError('Parent validation IDs mismatch')
            for name in ('constant','physical_linear'):
                predictions[name]=np.full((len(dataset),6),np.nan,dtype=np.float32); predictions[name][val_ids]=values[name]
        for name,values in (('dual',selected),('matched_control',initial_prediction)):
            predictions[name]=np.full((len(dataset),6),np.nan,dtype=np.float32); predictions[name][val_ids]=values
        report={'config':config,'evaluation_scope':'validation_only','validated_parameters':[],'production_replaced':False,'test_accessed':False,
                'validation':{name:metrics(value[val_ids],targets[n:].numpy()) for name,value in predictions.items()},
                'best_epoch':best_epoch,'epochs_run':len(history),'history':history,'frozen_state_and_batchnorm_unchanged':True,
                'input_parity_max_difference':{'before':before,'after':after,'selected_device_cpu_full_validation':device_delta},'elapsed_seconds':time.perf_counter()-started,
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'stratified_diagnostics':diagnostics(dataset,predictions,{'val':val_ids},train_ids,metrics),
                'limitations':['Validation-only selection; previously inspected FiveK and incomplete grouping.', 'No JPEG accuracy, test evaluation, export acceptance or production promotion.']}
        checkpoint.update(state_dict=model.state_dict(),finetuning_config=config,validated_parameters=[])
        torch.save(checkpoint,output/'best.pt')
        atomic_npz(output/'validation_predictions.npz',ids=np.array(split['val']),**{name:value[val_ids] for name,value in predictions.items()})
        write_json(output/'evaluation.json',report)
        write_json(output/'frozen_run.json',{'files':{name:sha256_file(output/name) for name in ('config.json','split_manifest.json','evaluation.json','validation_predictions.npz','best.pt')}})
        status={'status':'complete'}
        return report
    except BaseException as error:
        status={'status':'failed','reason':str(error)}; raise
    finally:
        write_json(output/'run_status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--device',choices=['cpu','mps'],default='cpu'); args=parser.parse_args()
    pilot=train(42,args.device)
    improved=pilot['validation']['dual']['p2']<pilot['validation']['matched_control']['p2']
    reports=[pilot]
    if improved:
        reports.extend(train(seed,args.device) for seed in (43,44))
    write_json(BASE/'phase_e_training_summary.json',{'scope':'validation_only','seed42_improved':improved,'confirmation_seeds':[r['config']['seed'] for r in reports],
        'runs':[{k:r[k] for k in ('validation','best_epoch','epochs_run','elapsed_seconds','input_parity_max_difference')} for r in reports],
        'test_accessed':False,'production_replaced':False})
