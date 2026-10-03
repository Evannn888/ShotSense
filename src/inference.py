"""Torch-free RAW-to-parameter inference using a verified ONNX bundle."""
import argparse
from functools import lru_cache
import io
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import cv2
import rawpy
import PIL

from src.image_io import decode_semantic_image
from src.parameters import PARAMS,PARAM_MIN,PARAM_MAX
from src.preprocess import ROOT,extract_inputs,sha256_file


@lru_cache(maxsize=2)
def _session(model_path,expected_hash,threads):
    if sha256_file(model_path)!=expected_hash: raise ValueError('ONNX model hash does not match bundle metadata')
    options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
    return ort.InferenceSession(model_path,sess_options=options,providers=['CPUExecutionProvider'])


def load_bundle(bundle_dir):
    bundle_dir=Path(bundle_dir)
    config=json.loads((bundle_dir/'model.json').read_text())
    if config.get('schema_version')!=1 or config.get('parameter_order')!=list(PARAMS):
        raise ValueError('Unsupported model/parameter contract')
    if config.get('parameter_min')!=PARAM_MIN.tolist() or config.get('parameter_max')!=PARAM_MAX.tolist():
        raise ValueError('Parameter encoding differs from current implementation')
    # Offline and online must share processing code/dependencies, not label availability.
    from src.color_pipeline import PIPELINE_CONFIG
    trained=config['pipeline_config']
    if 'semantic_geometry' in trained:
        from src.semantic_candidate import SEMANTIC_GEOMETRY
        if trained['semantic_geometry']!=SEMANTIC_GEOMETRY:
            raise ValueError('Unsupported semantic geometry contract')
        if trained['implementation_sha256'].get('semantic_candidate.py')!=sha256_file(ROOT/'src/semantic_candidate.py'):
            raise ValueError('Candidate semantic implementation differs from training')
    if trained['pipeline']!=json.loads(json.dumps(PIPELINE_CONFIG)):
        raise ValueError('RAW processing dependency/configuration differs from training')
    if trained['opencv']!=cv2.__version__ or trained['numpy']!=np.__version__:
        raise ValueError('Image processing dependency differs from training')
    if config['training_config']['decoder_sha256']!=sha256_file(ROOT/'src/image_io.py') or config['training_config']['pillow']!=PIL.__version__:
        raise ValueError('Semantic JPEG decoder differs from training')
    for filename,expected in trained['implementation_sha256'].items():
        if sha256_file(ROOT/'src'/filename)!=expected:
            raise ValueError('Preprocessing implementation differs from training: '+filename)
    return _session(str((bundle_dir/'model.onnx').resolve()),config['model_sha256'],config['threads']),config


def predict_arrays(session,image,physical):
    image=np.asarray(image); physical=np.asarray(physical)
    if image.dtype!=np.float32 or image.ndim!=4 or not len(image) or image.shape[1:]!=(3,224,224):
        raise ValueError('Image input must be float32 (B,3,224,224)')
    if physical.dtype!=np.float32 or physical.shape!=(len(image),132) or not np.isfinite(image).all() or not np.isfinite(physical).all():
        raise ValueError('Physical input must be finite float32 (B,132), aligned with image')
    output=session.run(None,{'image':image,'physical_raw':physical})[0]
    if output.shape!=(len(image),6) or not np.isfinite(output).all() or (np.abs(output)>1.000001).any():
        raise ValueError('Invalid model output')
    return np.clip(output,-1,1)


def predict_dng(dng_path,bundle_dir=ROOT/'artifacts/model'):
    started=time.perf_counter()
    dng_path=Path(dng_path)
    if dng_path.suffix.lower()!='.dng': raise ValueError('This model supports unedited DNG input')
    if not 0<dng_path.stat().st_size<=128*1024*1024: raise ValueError('DNG must be between 1 byte and 128 MB')
    # Identify the header before unpacking; this does not run a second imread.
    with rawpy.RawPy() as header:
        header.open_file(str(dng_path))
        if not 0<header.sizes.raw_width*header.sizes.raw_height<=40_000_000:
            raise ValueError('DNG pixel count exceeds the 40 megapixel limit')
    session,config=load_bundle(bundle_dir)
    preprocessing_started=time.perf_counter()
    if 'semantic_geometry' in config['pipeline_config']:
        from src.semantic_candidate import extract_candidate_inputs
        physical,jpeg,outside=extract_candidate_inputs(dng_path)
    else:
        physical,jpeg,outside=extract_inputs(dng_path)
    image=decode_semantic_image(io.BytesIO(jpeg))
    preprocess_seconds=time.perf_counter()-preprocessing_started
    inference_started=time.perf_counter()
    normalized=predict_arrays(session,image[None],physical[None])[0]
    inference_ms=(time.perf_counter()-inference_started)*1000
    absolute=(normalized.astype(np.float64)+1)/2*(PARAM_MAX-PARAM_MIN)+PARAM_MIN
    values=dict(zip(PARAMS,map(float,absolute)))
    validated=config['validated_parameters']
    result={'schema_version':1,'recommended_absolute':{k:v for k,v in values.items() if k in validated},
            'experimental_absolute':{k:v for k,v in values.items() if k not in validated},
            'units':dict(zip(PARAMS,['EV','slider','slider','K','slider','slider'])),
            'parameter_semantics':config['parameter_semantics'],'model_sha256':config['model_sha256'],
            'data_version':config['training_config']['config_hash'],
            'timing':{'preprocess_seconds':preprocess_seconds,'model_forward_ms':inference_ms,'end_to_end_seconds':time.perf_counter()-started},
            'lab_outside_fraction':outside.tolist()}
    return result,jpeg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dng',type=Path)
    parser.add_argument('--bundle',type=Path,default=ROOT/'artifacts/model')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preview',type=Path)
    parser.add_argument('--preview-source',type=Path,help='Independent aspect-preserving linear RAW display source, NPZ')
    args=parser.parse_args()
    result,jpeg=predict_dng(args.dng,args.bundle)
    if args.preview_source:
        from src.preview import prepare_preview_source
        from src.preprocess import atomic_npz
        preview_started=time.perf_counter()
        source=prepare_preview_source(args.dng)
        atomic_npz(args.preview_source,linear_rgb=source)
        result['timing']['preview_source_seconds']=time.perf_counter()-preview_started
    payload=json.dumps(result,indent=2,allow_nan=False)
    if args.output: args.output.write_text(payload+'\n')
    if args.preview: args.preview.write_bytes(jpeg)
    print(payload)
