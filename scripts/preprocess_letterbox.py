"""Build isolated candidate caches from audited supported DNGs, with resume."""
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np

from src.preprocess import ROOT, atomic_bytes, atomic_npz, read_cached, sha256_file, source_signature
from src.semantic_candidate import SEMANTIC_GEOMETRY, extract_candidate_inputs
from src.color_pipeline import PIPELINE_CONFIG


def process_candidate(job):
    source_path, output_dir, config_hash, expected_source, x, y, y_std = job
    source_path=Path(source_path); output_dir=Path(output_dir); photo=source_path.name.lower()
    cache=output_dir/'features'/(source_path.stem.lower()+'.npz')
    image=output_dir/'images'/(source_path.stem.lower()+'.jpg')
    try:
        cv2.setNumThreads(1)
        if not np.array_equal(source_signature(source_path),expected_source):
            raise ValueError('RAW source changed since original preprocessing')
        cached=read_cached(cache,image,photo,config_hash,expected_source)
        if cached is not None and all(np.array_equal(cached[key],expected) for key,expected in (('X',x),('Y',y),('Y_std',y_std))):
            return {'id':photo,'status':'cached'}
        physical,jpeg,outside=extract_candidate_inputs(source_path)
        if not np.array_equal(physical,x): raise ValueError('Candidate physical features differ from the control')
        if not np.array_equal(source_signature(source_path),expected_source): raise ValueError('RAW changed during development')
        atomic_bytes(image,jpeg)
        atomic_npz(cache,id=np.array(photo),config_hash=np.array(config_hash),source=expected_source,
                   image_sha256=np.array(hashlib.sha256(jpeg).hexdigest()),X=x,Y=y,Y_std=y_std,lab_outside_fraction=outside)
        return {'id':photo,'status':'processed'}
    except Exception as error:
        return {'id':photo,'status':'failed','reason':type(error).__name__+': '+str(error)}


def run(metadata_path, raw_dir, output_dir, workers=1, limit=None):
    if workers<1 or (limit is not None and limit<1): raise ValueError('workers and limit must be positive')
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/'.preprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return build(metadata_path,raw_dir,output_dir,workers,limit)


def build(metadata_path,raw_dir,output_dir,workers,limit):
    # Spawned image workers do not need the training framework.
    from src.dataset import ShotSenseDataset
    dataset=ShotSenseDataset(metadata_path)
    if dataset.scope!='full': raise ValueError('Candidate requires complete audited original data')
    original_manifest=json.loads((Path(metadata_path).parent/'manifest.json').read_text())
    config=json.loads(json.dumps(original_manifest['config']))
    if config['pipeline']!=json.loads(json.dumps(PIPELINE_CONFIG)):
        raise ValueError('Current RAW development differs from the original contract')
    config.update(semantic_geometry=SEMANTIC_GEOMETRY,source_metadata_sha256=dataset.data_hash)
    config['implementation_sha256']['semantic_candidate.py']=sha256_file(ROOT/'src/semantic_candidate.py')
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    for name,expected in config['implementation_sha256'].items():
        if sha256_file(ROOT/'src'/name)!=expected: raise ValueError('Source preprocessing differs from original contract: '+name)
    config_path=output_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())['config_hash']!=config_hash:
        raise ValueError('Candidate configuration changed; use a new output directory')
    manifest_path=output_dir/'manifest.json'
    if limit and manifest_path.exists() and json.loads(manifest_path.read_text())['scope']=='full':
        raise ValueError('Pilot cannot replace a full candidate cache')
    files=[p for p in Path(raw_dir).iterdir() if p.is_file() and p.suffix.lower()=='.dng']
    sources={p.name.lower():p for p in files}
    if len(sources)!=len(files): raise ValueError('RAW filename collision')
    count=min(limit,len(dataset)) if limit else len(dataset)
    for directory in ('images','features'): (output_dir/directory).mkdir(exist_ok=True)
    atomic_bytes(config_path,json.dumps({'config_hash':config_hash,'config':config},indent=2).encode())
    jobs=[]
    for i,photo in enumerate(dataset.ids[:count]):
        with np.load(Path(metadata_path).parent/'features'/(Path(photo).stem+'.npz'),allow_pickle=False) as original:
            source=original['source'].copy()
        jobs.append((str(sources[str(photo)]),str(output_dir),config_hash,source,dataset.X[i],dataset.Y[i],dataset.Y_std[i]))
    report={'scope':'pilot' if limit else 'full','status':'running','config_hash':config_hash,'config':config,
            'input_count':count,'workers':workers,'source_manifest_sha256':sha256_file(Path(metadata_path).parent/'manifest.json')}
    atomic_bytes(manifest_path,json.dumps(report,indent=2).encode())
    started=time.perf_counter(); results=[]
    if workers==1:
        iterator=map(process_candidate,jobs)
        for i,result in enumerate(iterator,1):
            results.append(result)
            if i%50==0 or i==count: print('Candidate inputs %d/%d' % (i,count),flush=True)
    else:
        with mp.get_context('spawn').Pool(workers) as pool:
            for i,result in enumerate(pool.imap_unordered(process_candidate,jobs),1):
                results.append(result)
                if i%50==0 or i==count: print('Candidate inputs %d/%d' % (i,count),flush=True)
    failures={r['id']:r['reason'] for r in results if r['status']=='failed'}
    report.update(status='failed' if failures else 'complete',failed_images=failures,failed_count=len(failures),
                  success_count=count-len(failures),cached_count=sum(r['status']=='cached' for r in results),
                  elapsed_seconds=time.perf_counter()-started,physical_features_exactly_equal=not failures)
    metadata=output_dir/'metadata.npz'
    if not failures:
        atomic_npz(metadata,X=dataset.X[:count],Y=dataset.Y[:count],Y_std=dataset.Y_std[:count],
                   ids=dataset.ids[:count],config_hash=np.array(config_hash))
        report['metadata_sha256']=sha256_file(metadata)
        support=Path(metadata_path).parent/'input_support_audit.json'
        if support.exists():
            atomic_bytes(output_dir/support.name,support.read_bytes()); report['support_audit_sha256']=sha256_file(support)
    else: metadata.unlink(missing_ok=True)
    atomic_bytes(manifest_path,json.dumps(report,indent=2,allow_nan=False).encode())
    print(json.dumps({k:report[k] for k in ('status','input_count','success_count','failed_count','cached_count','elapsed_seconds')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw/dngs')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    report=run(args.metadata,args.raw_dir,args.output_dir,args.workers,args.limit)
    raise SystemExit(0 if report['status']=='complete' else 1)
