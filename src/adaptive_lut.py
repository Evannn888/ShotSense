"""Frozen paired sRGB Image-Adaptive-3DLUT pilot; source-only, bounded CPU work.

Classifier architecture adapted from Hui Zeng et al., Image-Adaptive-3DLUT,
revision b491f6df64a588864739a157db271e5c848e1805 (Apache-2.0).
Changes: isolated inference, safe pinned loading, sampled bilinear predictor
input, current row-bounded Torch grid_sample instead of the legacy extension.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'data/external_models/image_adaptive_3dlut'
VERSION = 'image-adaptive-3dlut-paired-srgb-v1'
REVISION = 'b491f6df64a588864739a157db271e5c848e1805'
MAX_PIXELS = 64_000_000
WEIGHTS = {'classifier.pth': (1083261, '439f2ada316f45d1d28dd86a161790aa3b4e334b'),
           'LUTs.pth': (1294452, '0830dceee2256ea5d0965c029945c329fac82a26')}


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Upsample(size=(256, 256), mode='bilinear'),
                  nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                  nn.InstanceNorm2d(16, affine=True)]
        for a, b, normalize in ((16, 32, True), (32, 64, True), (64, 128, True), (128, 128, False)):
            layers.extend([nn.Conv2d(a, b, 3, stride=2, padding=1), nn.LeakyReLU(.2)])
            if normalize: layers.append(nn.InstanceNorm2d(b, affine=True))
        layers.extend([nn.Dropout(.5), nn.Conv2d(128, 3, 8, padding=0)])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def validate_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or rgb.shape[0] * rgb.shape[1] > MAX_PIXELS):
        raise ValueError('Expected uint8 sRGB pixels, at most64MP')
    return rgb


def load_model(bundle=BUNDLE):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['revision'] != REVISION or manifest['variant'] != 'paired-sRGB':
        raise ValueError('Wrong adaptive LUT model variant/revision')
    states = {}
    for name, (size, blob) in WEIGHTS.items():
        path = bundle / 'pretrained_models/sRGB' / name
        if path.stat().st_size != size: raise ValueError('LUT weight size mismatch: ' + name)
        data = path.read_bytes()
        if hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + data).hexdigest() != blob:
            raise ValueError('LUT weight checksum mismatch: ' + name)
        if hashlib.sha256(data).hexdigest() != manifest['files']['pretrained_models/sRGB/' + name]['sha256']:
            raise ValueError('LUT weight manifest mismatch: ' + name)
        states[name] = torch.load(path, map_location='cpu', weights_only=True)
    model = Classifier().eval().requires_grad_(False)
    model.load_state_dict(states['classifier.pth'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite LUT predictor')
    tables = states['LUTs.pth']
    if set(tables) != {'0', '1', '2'} or any(set(tables[str(i)]) != {'LUT'} for i in range(3)):
        raise ValueError('Unsupported basis LUT keys')
    bases = torch.stack([tables[str(i)]['LUT'] for i in range(3)])
    if bases.dtype != torch.float32 or bases.shape != (3, 3, 33, 33, 33) or not torch.isfinite(bases).all():
        raise ValueError('Unsupported basis LUT pixels')
    return model, bases, manifest


def predictor_input(rgb):
    """Sample the upstream non-antialiased half-pixel256resize without a full float image."""
    validate_rgb(rgb)
    if rgb.shape[0] * rgb.shape[1] <= 256 * 256:
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].contiguous().float() / 255
        return F.interpolate(tensor, (256, 256), mode='bilinear', align_corners=False)
    positions = []
    for length in rgb.shape[:2]:
        coords = np.maximum((np.arange(256, dtype=np.float64) + .5) * (length / 256) - .5, 0)
        low = np.floor(coords).astype(np.int64)
        positions.append((low, np.minimum(low + 1, length - 1), (coords - low).astype(np.float32)))
    (y0, y1, dy), (x0, x1, dx) = positions
    dx = dx[None, :, None]; dy = dy[:, None, None]
    top = rgb[y0[:, None], x0].astype(np.float32) / 255 * (1 - dx) + rgb[y0[:, None], x1].astype(np.float32) / 255 * dx
    bottom = rgb[y1[:, None], x0].astype(np.float32) / 255 * (1 - dx) + rgb[y1[:, None], x1].astype(np.float32) / 255 * dx
    return torch.from_numpy(top * (1 - dy) + bottom * dy).permute(2, 0, 1).unsqueeze(0).contiguous()


def interpolate(lut, rgb_float):
    # Upstream CPU interpolation indexes flattened R + G*dim + B*dim².
    grid = (rgb_float / 1.0001 * 2 - 1)[None, None]
    return F.grid_sample(lut[None], grid, mode='bilinear', padding_mode='border',
                         align_corners=True)[0, :, 0].permute(1, 2, 0)


def adjust_rgb(rgb, model=None):
    started = time.perf_counter(); validate_rgb(rgb)
    classifier, bases, manifest = load_model() if model is None else model
    with torch.inference_mode():
        weights = classifier(predictor_input(rgb)).flatten()
        if weights.shape != (3,) or not torch.isfinite(weights).all(): raise ValueError('Invalid predicted LUT weights')
        lut = weights[0] * bases[0] + weights[1] * bases[1] + weights[2] * bases[2]
        if not torch.isfinite(lut).all(): raise ValueError('Invalid predicted LUT')
        predicted = time.perf_counter()
        adjusted = np.empty_like(rgb)
        below = above = new_full = new_black = 0
        for row in range(0, len(rgb), 128):
            chunk = rgb[row:row+128]
            mapped = interpolate(lut, torch.from_numpy(np.ascontiguousarray(chunk)).float() / 255)
            if not torch.isfinite(mapped).all(): raise ValueError('Nonfinite LUT output')
            below += int(torch.any(mapped < 0, dim=-1).sum())
            above += int(torch.any(mapped > 1, dim=-1).sum())
            pixels = (mapped * 255 + .5).clamp(0, 255).to(torch.uint8).numpy()
            adjusted[row:row+128] = pixels
            new_full += int((np.any(pixels == 255, axis=-1) & ~np.any(chunk == 255, axis=-1)).sum())
            new_black += int((np.all(pixels == 0, axis=-1) & ~np.all(chunk == 0, axis=-1)).sum())
    count = rgb.shape[0] * rgb.shape[1]
    metadata = {'version': VERSION, 'engine': 'Image-Adaptive-3DLUT', 'model': manifest,
                'weights': weights.tolist(), 'lut_sha256': hashlib.sha256(lut.numpy().tobytes()).hexdigest(),
                'size': [rgb.shape[1], rgb.shape[0]], 'device': 'cpu', 'row_chunk': 128,
                'new_full_channel_fraction': new_full / count, 'new_black_pixel_fraction': new_black / count,
                'float_below_zero_pixel_fraction': below / count, 'float_above_one_pixel_fraction': above / count,
                'input_pixels_sha256': hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest(),
                'output_pixels_sha256': hashlib.sha256(np.ascontiguousarray(adjusted)).hexdigest(),
                'predictor_seconds': predicted - started, 'table_seconds': time.perf_counter() - predicted,
                'seconds': time.perf_counter() - started,
                'quality_status': 'isolated experimental pilot; personal-photo preference pending',
                'semantics': 'Learned global sRGB color/tone transform of original pixels; not camera Kelvin, semantic region edits, denoising or reconstructed detail.'}
    return metadata, adjusted


def process_jpeg(path):
    from src.jpeg_inference import decode_jpeg
    started = time.perf_counter(); path = Path(path)
    original, profile = decode_jpeg(path)
    metadata, adjusted = adjust_rgb(original)
    metadata.update(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_format='JPEG',
                    embedded_icc_converted_to_srgb=profile, end_to_end_seconds=time.perf_counter() - started)
    return metadata, {'original': original, 'adjusted': adjusted}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jpeg', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preview-source', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metadata, pixels = process_jpeg(args.jpeg)
    np.savez_compressed(args.preview_source, **pixels)
    args.output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
