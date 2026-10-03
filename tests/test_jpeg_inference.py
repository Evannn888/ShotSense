import io
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image,ImageCms
from streamlit.testing.v1 import AppTest

from src.inference import ROOT
from src.jpeg_inference import decode_jpeg,predict_jpeg


def jpeg_file(tmp_path,extension='.jpeg',profile=False):
    path=tmp_path/('input'+extension)
    image=Image.new('RGB',(90,50),(100,150,200))
    exif=Image.Exif(); exif[274]=6
    options={'exif':exif}
    if profile: options['icc_profile']=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    image.save(path,quality=95,**options)
    return path


def test_real_jpeg_estimates_orientation_and_torch_free_worker(tmp_path):
    path=jpeg_file(tmp_path,profile=True)
    result,source=predict_jpeg(path)
    assert source.shape==(90,50,3) and source.dtype==np.float32
    assert result['recommended_absolute']=={}
    assert set(result['experimental_absolute'])=={'Exposure','Contrast','Saturation','HighlightRecovery'}
    assert result['unavailable_parameters']==['Temperature','Tint']
    assert result['input_size']==[50,90] and result['jpeg_color']['embedded_icc_converted_to_srgb']
    script="""
import sys,socket
sys.modules['torch']=None; sys.modules['torchvision']=None
socket.create_connection=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('network'))
from src.jpeg_inference import predict_jpeg
result,source=predict_jpeg(sys.argv[1]); assert not result['recommended_absolute']
assert source.shape==(90,50,3)
"""
    subprocess.run([sys.executable,'-c',script,str(path)],cwd=ROOT,capture_output=True,text=True,check=True,timeout=60)
    from app.streamlit_app import run_job
    for suffix in ('.jpg','.jpeg'):
        worker_result,worker_source=run_job(path.read_bytes(),suffix)
        assert worker_result['experimental_absolute']==result['experimental_absolute']
        np.testing.assert_array_equal(worker_source,source)


def test_jpeg_rejects_wrong_content_corruption_and_unsupported_profile(tmp_path):
    wrong=tmp_path/'wrong.jpg'; Image.new('RGB',(10,10)).save(wrong,format='PNG')
    with pytest.raises(ValueError,match='not JPEG'): decode_jpeg(wrong)
    wrong.write_bytes(b'invalid jpeg')
    with pytest.raises(OSError): decode_jpeg(wrong)
    cmyk=tmp_path/'cmyk.jpg'; Image.new('CMYK',(10,10)).save(cmyk)
    with pytest.raises(ValueError,match='ICC'): decode_jpeg(cmyk)
    malformed=jpeg_file(tmp_path,'.jpg'); Image.new('RGB',(10,10)).save(malformed,icc_profile=b'bad')
    with pytest.raises(ValueError,match='ICC'): decode_jpeg(malformed)


def test_jpeg_page_experimental_preview_and_exports(tmp_path):
    result,source=predict_jpeg(jpeg_file(tmp_path,'.jpg'))
    result['timing']['local_worker_seconds']=result['timing']['end_to_end_seconds']
    result['input_name']='sample.jpg'
    at=AppTest.from_file(str(ROOT/'app/streamlit_app.py')).run(timeout=30)
    at.session_state['prediction']=(result,source); at.run(timeout=30)
    assert not at.exception and not at.error
    assert any('JPEG estimates are experimental' in warning.value for warning in at.warning)
    assert len(at.get('imgs'))==2
    assert any('Exposure 0.00 EV' in caption.value for caption in at.caption)
    toggle=next(item for item in at.checkbox if item.label=='Apply experimental JPEG estimates to preview')
    toggle.check().run(timeout=30)
    assert not at.exception and not at.error
    assert {item.proto.label for item in at.get('download_button')}=={'Download approximate preview PNG','Download parameters JSON'}
    assert any('Temperature and tint: unavailable' in caption.value for caption in at.caption)
    assert at.session_state['prediction'][0]['recommended_absolute']=={}
