import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageCms
import pytest
from streamlit.testing.v1 import AppTest

from src.jpeg_restoration import BUNDLE, ROOT, restore_jpeg, render_restoration


def test_restoration_worker_orientation_blend_exports_and_errors(tmp_path, monkeypatch):
    if not (BUNDLE / 'model.safetensors').exists():
        pytest.skip('Prepare the pinned author checkpoint before restoration acceptance')
    path = tmp_path / 'input.jpeg'
    image = Image.new('RGB', (65, 41), (15, 22, 30))
    exif = Image.Exif(); exif[274] = 6
    image.save(path, exif=exif, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
    result, source = restore_jpeg(path)
    repeat, repeated = restore_jpeg(path)
    np.testing.assert_array_equal(source['restored'], repeated['restored'])
    assert source['original'].shape == (65, 41, 3)
    assert result['jpeg_color']['embedded_icc_converted_to_srgb']
    assert result['recommended_absolute'] == result['experimental_absolute'] == {}
    before, after, metadata = render_restoration(source)
    assert before != after and metadata['changed_pixel_fraction'] > 0
    zero_before, zero_after, zero_metadata = render_restoration(source, 0)
    assert zero_before == zero_after and zero_metadata['changed_pixel_fraction'] == 0
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(zero_after))), source['original'])
    with pytest.raises(ValueError): render_restoration(source, float('nan'))
    with pytest.raises(ValueError): render_restoration(source, 1.01)
    bad = tmp_path / 'bundle'; bad.mkdir()
    (bad / 'manifest.json').write_bytes((BUNDLE / 'manifest.json').read_bytes())
    with pytest.raises(ValueError, match='missing'): restore_jpeg(path, bad)
    (bad / 'model.safetensors').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'): restore_jpeg(path, bad)
    tiny = tmp_path / 'tiny.jpg'; Image.new('RGB', (1, 2), (0, 0, 0)).save(tiny)
    _, small = restore_jpeg(tiny)
    assert small['restored'].shape == (2, 1, 3) and np.isfinite(small['restored']).all()

    # Exercise the real isolated CLI with network calls forbidden.
    output, cache = tmp_path / 'result.json', tmp_path / 'preview.npz'
    code = """
import sys,runpy,socket,urllib.request
socket.create_connection=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network'))
urllib.request.urlopen=socket.create_connection
sys.argv=['src.jpeg_restoration']+sys.argv[1:]
runpy.run_module('src.jpeg_restoration',run_name='__main__')
"""
    subprocess.run([sys.executable, '-c', code, str(path), '--output', str(output), '--preview-source', str(cache)],
                   cwd=ROOT, check=True, capture_output=True, timeout=60)
    with np.load(cache, allow_pickle=False) as worker:
        np.testing.assert_array_equal(worker['restored'], source['restored'])
    assert json.loads(output.read_text())['restoration'] == result['restoration']

    import streamlit as st
    exports = {}
    original_download = st.download_button
    def capture(label, data, *args, **kwargs):
        exports[label] = data
        return original_download(label, data, *args, **kwargs)
    monkeypatch.setattr(st, 'download_button', capture)
    from app.streamlit_app import run_job
    worker_result, worker_source = run_job(path.read_bytes(), '.jpeg', enhance=True)
    np.testing.assert_array_equal(worker_source['restored'], source['restored'])
    at = AppTest.from_file(str(ROOT / 'app/streamlit_app.py')).run(timeout=30)
    at.session_state['prediction'] = (worker_result, worker_source)
    at.run(timeout=30)
    assert not at.exception and not at.error and len(at.get('imgs')) == 1
    assert exports['Download original preview PNG'] == before
    assert 'Download enhanced PNG' not in exports
    assert not at.slider
    assert any('withdrawn' in warning.value for warning in at.warning)
    assert 'AI low-light enhancement (experimental)' not in at.radio[0].options

    # New default manual workflow must start exactly unchanged and export its real settings.
    from src.jpeg_inference import predict_jpeg
    from src.preview import render_linear_preview, apply_tone
    result, linear = predict_jpeg(path)
    result['jpeg_workflow'] = 'manual'
    result['timing']['local_worker_seconds'] = 0
    at.session_state['prediction'] = (result, linear)
    at.run(timeout=30)
    assert not at.exception and not at.error
    baseline, unchanged, _ = render_linear_preview(linear, {'Exposure':0, 'HighlightRecovery':0})
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    assert baseline == unchanged == exports['Download approximate preview PNG']
    assert not any(c.label == 'Apply experimental JPEG estimates to preview' for c in at.checkbox)
    next(s for s in at.slider if s.label == 'Manual exposure (EV)').set_value(1.0).run(timeout=30)
    _, adjusted, expected = render_linear_preview(linear, {'Exposure':1., 'HighlightRecovery':0})
    assert not at.get('download_button')
    next(b for b in at.button if b.label=='Prepare PNG download').click().run(timeout=30)
    assert exports['Download approximate preview PNG'] == adjusted
    payload = json.loads(exports['Download parameters JSON'])
    assert payload['preview']['adjustment_mode'] == 'manual'
    assert payload['preview']['applied_parameters'] == expected['applied_parameters']
    # Shared linear RGB gain preserves channel ratios, including dark eye-like pixels;
    # neutral black must remain black rather than receiving colored network offsets.
    pixels = np.array([[[.004,.003,.002],[.1,.09,.07],[0,0,0],[.7,.65,.6]]], dtype=np.float32)
    adjusted, _ = apply_tone(pixels, {'Exposure':1., 'HighlightRecovery':0})
    np.testing.assert_allclose(adjusted[0,:2] / pixels[0,:2], np.full((2,3),2.), rtol=1e-6)
    np.testing.assert_array_equal(adjusted[0,2], [0,0,0])
    assert np.isfinite(adjusted).all() and adjusted.max() <= 1
