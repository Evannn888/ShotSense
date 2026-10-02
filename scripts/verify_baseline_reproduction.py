"""Repeat seed-matched heads from ImageNet initialization, using validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.dataset import ShotSenseDataset
from src.model import ROOT, ShotSenseModel
from src.preprocess import sha256_file
from src.train import fit_head, semantic_features


def verify(run_dir, metadata_path):
    run_dir=Path(run_dir)
    config=json.loads((run_dir/'config.json').read_text())
    frozen=json.loads((run_dir/'frozen_run.json').read_text())
    for name,expected in frozen['files'].items():
        if sha256_file(run_dir/name)!=expected: raise ValueError('Changed frozen artifact: '+name)
    for name,expected in config['implementation_sha256'].items():
        if sha256_file(ROOT/'src'/name)!=expected: raise ValueError('Changed implementation: '+name)
    dataset=ShotSenseDataset(metadata_path)
    if dataset.data_hash!=config['metadata_sha256']: raise ValueError('Dataset differs from the run')
    split=json.loads((run_dir/'split_manifest.json').read_text())['splits']
    index={photo:i for i,photo in enumerate(dataset.ids.tolist())}
    train_ids=np.array([index[i] for i in split['train']]); val_ids=np.array([index[i] for i in split['val']])
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    mean=dataset.X[train_ids].mean(axis=0); scale=dataset.X[train_ids].std(axis=0); scale[scale==0]=1
    model=ShotSenseModel(mean,scale)
    features=semantic_features(model,dataset,run_dir/'semantic_features.npz')
    if hashlib.sha256(features.tobytes()).hexdigest()!=config['semantic_features_sha256']:
        raise ValueError('Semantic features differ from the run')
    dual,dual_training=fit_head(model,features,dataset.X,dataset.labels.numpy(),train_ids,val_ids,config)
    torch.manual_seed(config['seed'])
    semantic=ShotSenseModel(mean,scale,pretrained=False)
    semantic.backbone.load_state_dict(model.backbone.state_dict())
    semantic_predictions,semantic_training=fit_head(semantic,features,dataset.X,dataset.labels.numpy(),train_ids,val_ids,config,mode='semantic')
    original=json.loads((run_dir/'evaluation.json').read_text())
    with np.load(run_dir/'validation_predictions.npz',allow_pickle=False) as reference:
        if not np.array_equal(reference['ids'],dataset.ids[val_ids]): raise ValueError('Validation ID mismatch')
        differences={name:float(np.max(np.abs(values[val_ids]-reference[name])))
                     for name,values in (('dual',dual),('semantic',semantic_predictions))}
    for name,training in (('dual',dual_training),('semantic',semantic_training)):
        recorded=original['training' if name=='dual' else 'semantic_training']
        if training['history']!=recorded['history']: raise ValueError('Training history is not reproducible: '+name)
    if max(differences.values())>1e-6: raise ValueError('Validation predictions are not reproducible')
    report={'status':'passed','initialization':'Official ImageNet weights; no ShotSense checkpoint loaded',
            'evaluation_scope':'validation_only','seed':config['seed'],
            'validation_max_absolute_normalized_difference':differences,
            'training_histories_identical':True,
            'frozen_run_sha256':sha256_file(run_dir/'frozen_run.json'),
            'replay_elapsed_seconds':{'dual':dual_training['elapsed_seconds'],'semantic':semantic_training['elapsed_seconds']}}
    with (run_dir/'reproduction.json').open('x') as handle: json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    args=parser.parse_args(); verify(args.run_dir,args.metadata)
