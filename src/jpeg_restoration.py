"""Local, frozen HVI-CIDNet JPEG enhancement; experimental, not RAW parameters."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/jpeg_restoration'
MAX_EDGE = 960
VERSION = 'hvi-jpeg-restoration-v1'


def restore_jpeg(path, bundle=BUNDLE):
    # Keep Torch out of the Streamlit process and the existing RAW/estimate workers.
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from src.vendor.hvi.CIDNet import CIDNet
    from src.jpeg_inference import decode_jpeg, sha256_file

    started = time.perf_counter()
    torch.set_num_threads(1)
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    weights = bundle / 'model.safetensors'
    if not weights.exists():
        raise ValueError('JPEG enhancement weights are missing. Run venv/bin/python scripts/prepare_jpeg_restoration.py.')
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest['weights_sha256']:
        raise ValueError('JPEG enhancement checkpoint checksum mismatch; prepare the weights again.')
    model = CIDNet(**manifest['architecture'])
    model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
    model.eval()
    model.trans.gated = model.trans.gated2 = True
    model.trans.alpha_s = model.trans.alpha = 1.0
    rgb, profile = decode_jpeg(path)
    height, width = rgb.shape[:2]
    scale = min(1., MAX_EDGE / max(height, width))
    if scale < 1:
        rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
    pad = ((-w) % 8, (-h) % 8)
    # Reflect padding needs an axis longer than its pad; replication handles tiny JPEGs.
    tensor = F.pad(tensor, (0, pad[0], 0, pad[1]), mode='reflect' if w > pad[0] and h > pad[1] else 'replicate')
    forward_started = time.perf_counter()
    with torch.inference_mode():
        restored = model(tensor)[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).numpy().copy()
    if restored.shape != rgb.shape or not np.isfinite(restored).all():
        raise ValueError('JPEG enhancement returned invalid pixels.')
    result = {'schema_version': 1, 'input_format': 'JPEG', 'input_status': 'experimental_restoration',
              'restoration': {'version': VERSION, 'model': manifest['model'],
                              'weights_revision': manifest['weights_revision'],
                              'weights_sha256': manifest['weights_sha256'],
                              'source_revision': manifest['source_revision'],
                              'device': 'cpu', 'gamma': 1., 'alpha_s': 1., 'alpha_i': 1.,
                              'training_overlap': 'Unknown; diagnostic images do not establish unseen accuracy'},
              'input_size': [width, height], 'output_size': [w, h], 'input_sha256': sha256_file(path),
              'jpeg_color': {'embedded_icc_converted_to_srgb': profile, 'without_icc': 'assumed sRGB',
                             'exif_orientation_applied': True},
              'recommended_absolute': {}, 'experimental_absolute': {},
              'warnings': ['Experimental low-light restoration may alter color or remove fine detail.',
                           'Clipped or missing image detail cannot be guaranteed to recover.',
                           'Output is resized to at most 960 pixels per edge; Lightroom parameters are unavailable.'],
              'timing': {'model_forward_ms': (time.perf_counter() - forward_started) * 1000,
                         'end_to_end_seconds': time.perf_counter() - started}}
    return result, {'original': rgb, 'restored': restored}


def render_restoration(source, strength=1.):
    original, restored = source['original'], source['restored']
    if (original.dtype != np.uint8 or original.ndim != 3 or original.shape[-1] != 3
            or restored.dtype != np.float32 or restored.shape != original.shape
            or min(original.shape[:2]) < 1 or max(original.shape[:2]) > MAX_EDGE
            or not np.isfinite(restored).all() or (restored < 0).any() or (restored > 1).any()
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('Invalid restoration pixels or strength.')
    blended = np.rint(original.astype(np.float32) * (1 - strength) + restored * (255 * strength)).clip(0, 255).astype(np.uint8)
    def png(rgb):
        buffer = io.BytesIO()
        info = PngImagePlugin.PngInfo()
        info.add(b'sRGB', b'\x00')
        Image.fromarray(rgb).save(buffer, format='PNG', pnginfo=info)
        return buffer.getvalue()
    metadata = {'renderer_version': VERSION, 'adjustment_mode': 'pretrained_restoration',
                'strength': float(strength), 'blend_space': 'display sRGB',
                'size': [original.shape[1], original.shape[0]], 'applied_parameters': {},
                'changed_pixel_fraction': float(np.any(blended != original, axis=-1).mean()),
                'semantics': 'Blend original JPEG with a frozen learned restoration; not Lightroom adjustments.'}
    before, after = png(original), png(blended)
    metadata['output_png_sha256'] = hashlib.sha256(after).hexdigest()
    return before, after, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jpeg', type=Path)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-source', type=Path, required=True)
    args = parser.parse_args()
    result, source = restore_jpeg(args.jpeg, args.bundle)
    np.savez_compressed(args.preview_source, **source)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
