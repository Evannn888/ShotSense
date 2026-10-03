"""Validate candidate caches and exact control memberships, then train unchanged heads."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.dataset import ShotSenseDataset, load_or_create_splits
from src.preprocess import ROOT
from src.semantic_candidate import validate_candidate_cache
from src.train import train


def run(candidate_dir, output_dir, control_metadata, groups_path, reference_split, seed=42):
    candidate_dir=Path(candidate_dir)
    dataset=validate_candidate_cache(candidate_dir)
    control=ShotSenseDataset(control_metadata)
    if any(not np.array_equal(getattr(dataset,key),getattr(control,key)) for key in ('X','Y','Y_std','ids')):
        raise ValueError('Geometry comparison must use exactly the same features, labels and IDs')
    groups=json.loads(Path(groups_path).read_text())
    reference=json.loads(Path(reference_split).read_text())
    if reference.get('data_hash')!=control.data_hash or reference.get('data_version')!=control.data_version:
        raise ValueError('Reference split differs from the control metadata')
    split_path=candidate_dir/'splits.json'
    splits=load_or_create_splits(dataset,split_path,groups=groups,seed=reference['seed'])
    if splits!=reference['splits']: raise ValueError('Candidate memberships differ from the control')
    return train(candidate_dir/'metadata.npz',output_dir,seed=seed,split_seed=reference['seed'],
                 split_path=split_path,groups_path=groups_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--control-metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--groups',type=Path,default=ROOT/'data/processed/model_vnext/groups.json')
    parser.add_argument('--reference-split',type=Path,default=ROOT/'data/processed/model_vnext/splits.json')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    run(args.candidate_dir,args.output_dir,args.control_metadata,args.groups,args.reference_split,args.seed)
