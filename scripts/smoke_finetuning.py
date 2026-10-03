"""CPU implementation/cost smoke; no fine-tuning quality or promotion claim."""
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import ShotSenseDataset
from src.finetune_candidate import FineTuneCandidate
from src.model import ROOT
from src.preprocess import sha256_file
from src.train import metrics


BASE=ROOT/'artifacts/experiments/model_vnext'


def smoke():
    parent=BASE/'control-seed42'
    frozen=json.loads((parent/'frozen_run.json').read_text())
    for name,expected in frozen['files'].items():
        if sha256_file(parent/name)!=expected: raise ValueError('Changed parent artifact: '+name)
    dataset=ShotSenseDataset(ROOT/'data/processed/metadata.npz')
    parent_config=json.loads((parent/'config.json').read_text())
    if dataset.data_hash!=parent_config['metadata_sha256']: raise ValueError('Dataset mismatch')
    split=json.loads((parent/'split_manifest.json').read_text())['splits']
    index={photo:i for i,photo in enumerate(dataset.ids.tolist())}
    selected={name:split[name][:count] for name,count in [('train',128),('val',16)]}
    train_loader=DataLoader(Subset(dataset,[index[p] for p in selected['train']]),batch_size=16,shuffle=False)
    val_loader=DataLoader(Subset(dataset,[index[p] for p in selected['val']]),batch_size=16,shuffle=False)
    output=BASE/'finetune-smoke-seed42'
    output.mkdir(exist_ok=False)
    config={'scope':'implementation_and_cost_smoke_only','seed':42,'batch_size':16,'training_steps':8,
            'selection':'First 128 training IDs and first 16 validation IDs in the fixed split; selected before candidate predictions',
            'selected_ids':selected,'trainable_stage':'backbone.features.7','backbone_lr':1e-5,'head_lr':1e-4,'weight_decay':.01,
            'batchnorm_running_buffers':'frozen','parent_frozen_sha256':sha256_file(parent/'frozen_run.json'),
            'metadata_sha256':dataset.data_hash,'source_sha256':{name:sha256_file(ROOT/name) for name in ('src/finetune_candidate.py','scripts/smoke_finetuning.py','src/model.py','src/dataset.py')},
            'torch':str(torch.__version__),'threads':4}
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    status={'status':'running'}
    try:
        torch.set_num_threads(4); random.seed(42); np.random.seed(42); torch.manual_seed(42)
        checkpoint=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)
        model=FineTuneCandidate(); model.load_state_dict(checkpoint['state_dict'])
        initial={name:value.clone() for name,value in model.state_dict().items()}
        allowed=[name for name,param in model.named_parameters() if param.requires_grad]
        if not all(name.startswith(('backbone.features.7.','head.')) for name in allowed): raise ValueError('Unexpected trainable parameter')
        if not any(name.startswith('backbone.features.7.') for name in allowed): raise ValueError('No trainable MBConv parameters')
        optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01)
        losses=[]; started=time.perf_counter(); model.train()
        for image,physical,target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction=model(image,physical)
            loss=torch.nn.functional.smooth_l1_loss(prediction,target,beta=.1)
            if not torch.isfinite(loss): raise ValueError('Non-finite smoke loss')
            loss.backward()
            for name,param in model.named_parameters():
                if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()): raise ValueError('Missing/non-finite gradient: '+name)
                if not param.requires_grad and param.grad is not None: raise ValueError('Frozen parameter has gradients: '+name)
            optimizer.step(); losses.append(float(loss.detach()))
        elapsed=time.perf_counter()-started
        changed=[name for name,value in model.state_dict().items() if not torch.equal(value,initial[name])]
        if set(changed)-set(allowed): raise ValueError('Frozen parameter or BatchNorm buffer changed')
        if not any(name.startswith('backbone.features.7.') for name in changed) or not any(name.startswith('head.') for name in changed): raise ValueError('Intended parameters did not update')
        model.eval(); predictions=[]; targets=[]
        with torch.inference_mode():
            for image,physical,target in val_loader:
                predictions.append(model(image,physical).numpy()); targets.append(target.numpy())
        report={'status':'passed','scope':config['scope'],'trainable_parameters':allowed,'changed_parameters':changed,
                'frozen_parameters_and_all_batchnorm_buffers_unchanged':True,'finite_gradients_and_losses':True,
                'losses':losses,'training_seconds':elapsed,'photos_per_second':128/elapsed,
                'estimated_3957_photo_training_epoch_seconds':elapsed/128*3957,
                'estimate_note':'Linear extrapolation of a short warmup-inclusive CPU smoke, excluding full validation/checkpoint overhead; not a measured full epoch.',
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'rss_units':'macOS ru_maxrss bytes',
                'subset_validation_diagnostic':metrics(np.concatenate(predictions),np.concatenate(targets)),
                'diagnostic_warning':'16 fixed validation photos after 8 training steps; no full validation comparison or gain claim.',
                'test_accessed':False,'production_replaced':False}
        (output/'smoke_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status={'status':'complete'}
        print(json.dumps(report,indent=2))
    except Exception as error:
        status={'status':'failed','reason':str(error)}
        raise
    finally:
        (output/'run_status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':
    smoke()
