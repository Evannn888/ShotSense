"""Experimental rendered-JPEG estimates; not validated RAW recommendations."""
import argparse
import io
import json
from pathlib import Path
import time

import colour
import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from src.color_pipeline import D50, LIBRAW_SRGB_TO_XYZ_D50
from src.image_io import decode_semantic_image
from src.inference import ROOT, load_bundle, predict_arrays
from src.parameters import PARAMS, PARAM_MIN, PARAM_MAX
from src.preprocess import atomic_npz, encode_semantic_image, extract_132d_features, sha256_file
from src.preview import MAX_EDGE, validate_source


_SRGB = np.arange(256,dtype=np.float32)/255
_LINEAR_LUT = np.where(_SRGB<=.04045,_SRGB/12.92,((_SRGB+.055)/1.055)**2.4)
MAX_JPEG_PIXELS = 64_000_000


def decode_jpeg(path):
    path=Path(path)
    if path.suffix.lower() not in ('.jpg','.jpeg'): raise ValueError('Expected a JPG or JPEG file')
    if not 0<path.stat().st_size<=128*1024*1024: raise ValueError('JPEG must be between 1 byte and 128 MB')
    with Image.open(path) as image:
        if image.format!='JPEG': raise ValueError('File content is not JPEG')
        if not 0<image.width*image.height<=MAX_JPEG_PIXELS: raise ValueError('JPEG exceeds the 64 megapixel limit')
        profile=image.info.get('icc_profile')
        if image.mode not in ('RGB','L','CMYK'): raise ValueError('Unsupported JPEG color mode')
        if image.mode=='CMYK' and not profile: raise ValueError('CMYK JPEG requires an embedded ICC profile')
        oriented=ImageOps.exif_transpose(image)
        if profile:
            try:
                oriented=ImageCms.profileToProfile(oriented,ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                                  ImageCms.createProfile('sRGB'),outputMode='RGB')
            except (ImageCms.PyCMSError,OSError,ValueError) as error:
                raise ValueError('Unable to interpret the JPEG ICC color profile') from error
        else: oriented=oriented.convert('RGB')
        return np.asarray(oriented,dtype=np.uint8).copy(),bool(profile)


def predict_jpeg(path,bundle_dir=ROOT/'artifacts/model'):
    started=time.perf_counter(); rgb,profile_applied=decode_jpeg(path)
    session,config=load_bundle(bundle_dir)
    preprocessing_started=time.perf_counter()
    linear=_LINEAR_LUT[rgb]
    small=cv2.resize(linear,(224,224),interpolation=cv2.INTER_AREA)
    lab=colour.XYZ_to_Lab(small @ LIBRAW_SRGB_TO_XYZ_D50.T,illuminant=D50).astype(np.float32)
    physical=extract_132d_features(lab)
    if 'semantic_geometry' in config['pipeline_config']:
        from src.semantic_candidate import letterbox_rgb
        semantic=letterbox_rgb(rgb)
    else: semantic=rgb
    jpeg=encode_semantic_image(semantic); image=decode_semantic_image(io.BytesIO(jpeg))
    h,w=rgb.shape[:2]; factor=min(1,MAX_EDGE/max(h,w))
    preview=cv2.resize(linear,(max(1,round(w*factor)),max(1,round(h*factor))),interpolation=cv2.INTER_AREA)
    # Area-resize float32 roundoff can put white a few ULPs above 1; keep the display buffer bounded.
    preview=validate_source(np.ascontiguousarray(np.clip(preview,0,1),dtype=np.float32))
    preprocessing_seconds=time.perf_counter()-preprocessing_started
    inference_started=time.perf_counter(); normalized=predict_arrays(session,image[None],physical[None])[0]
    inference_ms=(time.perf_counter()-inference_started)*1000
    absolute=(normalized.astype(np.float64)+1)/2*(PARAM_MAX-PARAM_MIN)+PARAM_MIN
    values={name:float(value) for name,value in zip(PARAMS,absolute) if name not in ('Temperature','Tint')}
    result={'schema_version':1,'input_format':'JPEG','input_status':'experimental_rendered_input',
            'jpeg_processing_version':'rendered-jpeg-srgb-v2','jpeg_processing_sha256':sha256_file(__file__),
            'recommended_absolute':{},'experimental_absolute':values,'unavailable_parameters':['Temperature','Tint'],
            'units':dict(zip(PARAMS,['EV','slider','slider','K','slider','slider'])),
            'parameter_semantics':'Experimental legacy PV2003 model outputs from an already rendered JPEG; not validated RAW settings or adjustment deltas.',
            'warnings':['The model was trained on unedited DNGs; JPEG estimates have not been validated.',
                        'JPEG camera/software processing cannot be undone; clipped detail cannot be recovered.',
                        'Absolute Temperature/Tint are unavailable for rendered JPEG input.'],
            'jpeg_color':{'embedded_icc_converted_to_srgb':profile_applied,'without_icc':'assumed sRGB','exif_orientation_applied':True},
            'input_size':[w,h],'model_sha256':config['model_sha256'],'data_version':config['training_config']['config_hash'],
            'timing':{'preprocess_seconds':preprocessing_seconds,'model_forward_ms':inference_ms,
                      'preview_source_seconds':0.0,'end_to_end_seconds':time.perf_counter()-started}}
    return result,preview


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jpeg',type=Path)
    parser.add_argument('--bundle',type=Path,default=ROOT/'artifacts/model')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preview-source',type=Path)
    args=parser.parse_args(); result,preview=predict_jpeg(args.jpeg,args.bundle)
    if args.preview_source: atomic_npz(args.preview_source,linear_rgb=preview)
    payload=json.dumps(result,indent=2,allow_nan=False)
    if args.output: args.output.write_text(payload+'\n')
    print(payload)
