import ast
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src import adaptive_lut as engine
from src.photo_engine import render_photo


def test_predictor_resize_and_weights_match_frozen_author():
    if not (engine.BUNDLE / 'manifest.json').exists(): pytest.skip('Prepare pinned LUT weights')
    torch.set_num_threads(1)
    model, bases, manifest = engine.load_model()
    source = (engine.BUNDLE / 'models.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == manifest['files']['models.py']['sha256']
    nodes = [n for n in ast.parse(source).body if getattr(n, 'name', '') in ('discriminator_block', 'Classifier')]
    scope = {'nn': nn}; exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen-author-classifier', 'exec'), scope)
    author = scope['Classifier']().eval()
    author.load_state_dict(model.state_dict(), strict=True)
    rng = np.random.default_rng(91)
    for shape in ((1, 1, 3), (13, 17, 3), (257, 256, 3), (1023, 801, 3)):
        rgb = rng.integers(0, 256, shape, dtype=np.uint8)
        full = torch.from_numpy(rgb).permute(2, 0, 1)[None].contiguous().float() / 255
        expected = F.interpolate(full, (256, 256), mode='bilinear', align_corners=False)
        actual = engine.predictor_input(rgb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=3e-7)
        with torch.inference_mode():
            assert torch.equal(model(expected), author(expected))
            torch.testing.assert_close(model(actual), author(full), rtol=0, atol=2e-6)


def test_trilinear_axes_endpoints_and_random_values_match_author_cpu():
    reference = engine.BUNDLE / 'trilinear-reference.dylib'
    if not reference.exists(): pytest.skip('Build isolated author CPU reference')
    torch.set_num_threads(1)
    function = ctypes.CDLL(str(reference)).TriLinearForwardCpu
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [pointer, pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                         ctypes.c_int, ctypes.c_int, ctypes.c_int]
    function.restype = None
    rng = np.random.default_rng(33)
    rgb = rng.random((41, 63, 3), dtype=np.float32)
    rgb[0, :8] = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.2, .7, .9], [.9999, .5, .01], [.5, .5, .5]]
    b, g, r = np.meshgrid(*([np.linspace(0, 1, 33, dtype=np.float32)] * 3), indexing='ij')
    tables = [np.stack([r, g, b]), np.stack([.1 + .7*b, .2 + .4*r, .9 - .6*g]),
              rng.uniform(-.2, 1.2, (3, 33, 33, 33)).astype(np.float32)]
    for table in tables:
        planar = np.ascontiguousarray(rgb.transpose(2, 0, 1)); expected = np.empty_like(planar)
        function(table.ctypes.data_as(pointer), planar.ctypes.data_as(pointer), expected.ctypes.data_as(pointer),
                 33, 33**3, 1.0001/32, rgb.shape[1], rgb.shape[0], 3)
        actual = engine.interpolate(torch.from_numpy(table), torch.from_numpy(rgb)).numpy()
        np.testing.assert_allclose(actual, expected.transpose(1, 2, 0), rtol=0, atol=1e-5)
    # An identity table preserves all8-bit values despite the author's endpoint epsilon.
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    mapped = engine.interpolate(torch.from_numpy(tables[0]), torch.from_numpy(ramp).float()/255)
    np.testing.assert_array_equal((mapped*255+.5).clamp(0,255).to(torch.uint8).numpy(), ramp)


def test_actual_lut_replay_identity_export_validation_and_pinned_loading(tmp_path):
    if not (engine.BUNDLE / 'manifest.json').exists(): pytest.skip('Prepare pinned LUT weights')
    torch.set_num_threads(1)
    rng = np.random.default_rng(20); rgb = rng.integers(20, 240, (279, 347, 3), dtype=np.uint8)
    path = tmp_path / 'color.jpg'; Image.fromarray(rgb).save(path, quality=97)
    original_bytes = path.read_bytes()
    metadata, arrays = engine.process_jpeg(path)
    model = engine.load_model(); replay, adjusted = engine.adjust_rgb(arrays['original'], model)
    np.testing.assert_array_equal(adjusted, arrays['adjusted'])
    assert replay['weights'] == metadata['weights'] and replay['lut_sha256'] == metadata['lut_sha256']
    assert path.read_bytes() == original_bytes and metadata['size'] == [347,279]
    before, after, export = render_photo(arrays)
    assert export['size'] == metadata['size'] and export['output_png_sha256'] == hashlib.sha256(after).hexdigest()
    zero_before, zero_after, zero = render_photo(arrays, 0)
    assert before == zero_before == zero_after and zero['changed_pixel_fraction'] == 0
    for bad in (rgb.astype(np.float32), rgb[:, :, :2], rgb[0:0], np.broadcast_to(np.zeros((1,1,3),np.uint8),(1,64_000_001,3))):
        with pytest.raises(ValueError): engine.adjust_rgb(bad, model)
    # Manifest tampering is rejected before any checkpoint is loaded.
    manifest = json.loads((engine.BUNDLE / 'manifest.json').read_text()); manifest['variant'] = 'XYZ'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='variant'): engine.load_model(tmp_path)
