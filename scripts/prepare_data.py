"""Prepare data, then audit RAW inputs outside the explicit camera-WB contract."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import rawpy

from src.evaluation import catalog_context
from src.preprocess import ROOT, atomic_bytes, run_preprocessing, sha256_file

UNSUPPORTED_WB = 'ValueError: Missing or invalid camera white balance; automatic fallback is disabled'


def finalize_support(output_dir,raw_dir):
    output_dir,raw_dir=Path(output_dir),Path(raw_dir)
    manifest_path=output_dir/'manifest.json'
    report=json.loads(manifest_path.read_text())
    failures=report['failed_images']
    if not failures: return report
    if not report['success_count'] or any(reason!=UNSUPPORTED_WB for reason in failures.values()):
        raise ValueError('Unresolved preprocessing failures; inspect manifest.json')
    if sha256_file(output_dir/'metadata.npz')!=report['metadata_sha256']:
        raise ValueError('Metadata changed since preprocessing')
    sources={path.name.lower():path for path in raw_dir.iterdir() if path.suffix.lower()=='.dng'}
    cameras,_,_=catalog_context(ROOT/'data/raw/fivek_dataset/raw_photos/fivek.lrcat')
    unsupported={}
    for photo_id in sorted(failures):
        path=sources[photo_id]
        with rawpy.RawPy() as header:
            header.open_file(str(path))
            values=np.asarray(header.camera_whitebalance[:3],dtype=np.float64)
        if values.shape==(3,) and np.isfinite(values).all() and (values>.00001).all():
            raise ValueError('Rejected input now has valid camera WB: '+photo_id)
        unsupported[photo_id]={'reason':'missing_valid_camera_wb','camera':cameras.get(photo_id,'unknown'),
                              'camera_wb_rgb':[float(v) if np.isfinite(v) else None for v in values],
                              'source_size':path.stat().st_size,'source_mtime_ns':path.stat().st_mtime_ns}
    # Preserve the original failure report, then declare the narrower input scope explicitly.
    original=manifest_path.read_bytes()
    original_name='preprocessing_run.'+sha256_file(manifest_path)+'.json'
    atomic_bytes(output_dir/original_name,original)
    audit={'schema_version':1,'unsupported_count':len(unsupported),'unsupported_images':unsupported,
           'camera_counts':dict(Counter(value['camera'] for value in unsupported.values())),
           'selection_sha256':sha256_file(Path(__file__)),'original_report':original_name,
           'rawpy':rawpy.__version__,'libraw':list(rawpy.libraw_version)}
    atomic_bytes(output_dir/'input_support_audit.json',json.dumps(audit,indent=2,allow_nan=False).encode())
    report['unsupported_images']=unsupported
    report['unsupported_count']=len(unsupported)
    report['filtered_images'].update({photo:['unsupported input: missing valid camera WB'] for photo in unsupported})
    report['filtered_count']+=len(unsupported)
    report.update({'failed_images':{},'failed_count':0,'status':'complete',
                   'input_scope':'DNG with valid camera white balance; no automatic fallback',
                   'support_audit_sha256':sha256_file(output_dir/'input_support_audit.json')})
    assert report['input_count']==report['success_count']+report['filtered_count']
    atomic_bytes(manifest_path,json.dumps(report,indent=2,sort_keys=True,allow_nan=False).encode())
    print(json.dumps({key:report[key] for key in ('status','input_count','success_count','filtered_count','unsupported_count')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--audit-existing',action='store_true',help='Classify an existing verified failure report without rerunning development')
    args=parser.parse_args()
    output=ROOT/('data/processed/pilot' if args.limit else 'data/processed')
    raw=ROOT/'data/raw/dngs'
    if not args.audit_existing:
        run_preprocessing(raw,ROOT/'data/intermediate/expert_labels.json',output,limit=args.limit,workers=args.workers)
    finalize_support(output,raw)
