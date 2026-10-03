"""Versioned aspect-preserving semantic input; original physical branch unchanged."""
import json
from pathlib import Path

import cv2
import numpy as np

from src.preprocess import (IMAGE_SIZE, encode_semantic_image, extract_132d_features,
                            read_cached, sha256_file)
from src.color_pipeline import PIPELINE_CONFIG, develop_dng_outputs, prophoto16_to_lab_d50

SEMANTIC_GEOMETRY = {
    'version': 'letterbox-imagenet-mean-v1', 'size': [224, 224],
    'padding_rgb': [124, 116, 104], 'alignment': 'center; odd remainder bottom/right',
    'interpolation': 'cv2.INTER_AREA', 'rounding': 'nearest integer, minimum one pixel',
}


def letterbox_rgb(srgb):
    if srgb.dtype != np.uint8 or srgb.ndim != 3 or srgb.shape[2] != 3 or min(srgb.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 RGB semantic output')
    height, width = srgb.shape[:2]
    scale = 224 / max(height, width)
    fitted_width = max(1, round(width * scale)); fitted_height = max(1, round(height * scale))
    small = cv2.resize(srgb, (fitted_width, fitted_height), interpolation=cv2.INTER_AREA)
    result = np.empty((224, 224, 3), dtype=np.uint8)
    result[:] = SEMANTIC_GEOMETRY['padding_rgb']
    left = (224 - fitted_width) // 2; top = (224 - fitted_height) // 2
    result[top:top+fitted_height, left:left+fitted_width] = small
    return result


def extract_candidate_inputs(dng_path):
    """Shared by candidate preprocessing and explicitly versioned inference."""
    prophoto, srgb = develop_dng_outputs(str(dng_path))
    linear = cv2.resize(prophoto.astype(np.float32)/65535.0, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    lab = prophoto16_to_lab_d50(linear)
    outside = np.mean((lab < [0,-128,-128]) | (lab > [100,128,128]), axis=(0,1))
    return extract_132d_features(lab), encode_semantic_image(letterbox_rgb(srgb)), outside.astype(np.float32)


def validate_candidate_cache(output_dir):
    """Reject mixed or damaged semantic caches before training reads them."""
    from src.dataset import ShotSenseDataset
    output_dir = Path(output_dir)
    dataset = ShotSenseDataset(output_dir/'metadata.npz')
    manifest = json.loads((output_dir/'manifest.json').read_text())
    config = manifest['config']
    if config.get('semantic_geometry') != SEMANTIC_GEOMETRY:
        raise ValueError('Unsupported candidate geometry contract')
    import hashlib
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != dataset.data_version:
        raise ValueError('Candidate configuration hash mismatch')
    for name, expected in config['implementation_sha256'].items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError('Candidate preprocessing implementation changed: '+name)
    if config['opencv'] != cv2.__version__ or config['numpy'] != np.__version__:
        raise ValueError('Candidate preprocessing dependencies changed')
    if config['pipeline'] != json.loads(json.dumps(PIPELINE_CONFIG)):
        raise ValueError('Candidate RAW dependencies/configuration changed')
    for i, photo in enumerate(dataset.ids):
        cache = output_dir/'features'/(Path(photo).stem+'.npz')
        image = output_dir/'images'/(Path(photo).stem+'.jpg')
        with np.load(cache, allow_pickle=False) as values: source = values['source'].copy()
        values = read_cached(cache, image, str(photo), dataset.data_version, source)
        if values is None or any(not np.array_equal(values[key], getattr(dataset,key)[i]) for key in ('X','Y','Y_std')):
            raise ValueError('Mixed or damaged candidate cache: '+str(photo))
    return dataset
