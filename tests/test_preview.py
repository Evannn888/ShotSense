import io

import numpy as np
from PIL import Image
import pytest

from src.preview import render_preview,apply_tone,render_linear_preview,prepare_preview_source,validate_source


def test_preview_neutral_exposure_highlights_and_idempotence():
    rgb=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    source=io.BytesIO(); Image.fromarray(rgb).save(source,format='JPEG',quality=100)
    jpeg=source.getvalue(); original=np.array(Image.open(io.BytesIO(jpeg)))
    def pixels(parameters):
        return np.array(Image.open(io.BytesIO(render_preview(jpeg,parameters))))
    np.testing.assert_array_equal(pixels({}),original)
    brighter=pixels({'Exposure':1}); darker=pixels({'Exposure':-1})
    assert (brighter>=original).all() and (darker<=original).all()
    assert brighter[0,100,0]>original[0,100,0] and darker[0,100,0]<original[0,100,0]
    recovery=pixels({'HighlightRecovery':100})
    np.testing.assert_array_equal(recovery[:,:150],original[:,:150])
    assert recovery[0,240,0]<original[0,240,0] and np.diff(recovery[0,:,0].astype(int)).min()>=0
    combined={'Exposure':.8,'HighlightRecovery':10}
    assert render_preview(jpeg,combined)==render_preview(jpeg,combined)
    for invalid in ({'Temperature':5000},{'Exposure':np.nan},{'HighlightRecovery':101},{'Exposure':5}):
        with pytest.raises(ValueError): render_preview(jpeg,invalid)


def test_protection_monotonic_preserves_color_and_strong_exposure_headroom():
    ramp=np.linspace(0,1,1000,dtype=np.float32)[None,:,None]*np.array([1,.7,.3],dtype=np.float32)
    protected,_=apply_tone(ramp,{'Exposure':2})
    direct,_=apply_tone(ramp,{'Exposure':2},protect_highlights=False)
    assert np.diff(protected[0,:,0]).min()>0
    assert (protected[0,:-1,0]<1).all() and (direct[0,:,0]==1).sum()>700
    np.testing.assert_allclose(protected[...,1],protected[...,0]*.7,atol=1e-7)
    np.testing.assert_array_equal(apply_tone(ramp,{'Exposure':2,'HighlightRecovery':80},strength=0)[0],ramp)
    before,after,metadata=render_linear_preview(ramp,{'Exposure':2,'HighlightRecovery':20},.5)
    assert metadata['applied_parameters']=={'Exposure':1,'HighlightRecovery':10}
    assert metadata['size']==[1000,1] and Image.open(io.BytesIO(after)).size==(1000,1)
    assert metadata['new_full_channel_fraction']==0
    assert render_linear_preview(ramp,{'Exposure':2,'HighlightRecovery':20},.5)[1]==after
    for bad in (np.full((1,2,3),np.nan,dtype=np.float32),ramp.astype(np.float64),np.ones((1,1601,3),dtype=np.float32)):
        with pytest.raises(ValueError): validate_source(bad)


def test_raw_source_aspect_and_linear_conversion(monkeypatch):
    raw=np.full((120,240,3),12000,dtype=np.uint16)
    monkeypatch.setattr('src.color_pipeline.develop_dng_outputs',lambda _: (raw,np.zeros_like(raw,dtype=np.uint8)))
    source=prepare_preview_source('unused.dng')
    assert source.shape==(120,240,3) and source.dtype==np.float32
    assert source.max()-source.min()<.0001


def test_shoulder_linear_midtones_continuity_and_source_diagnostics():
    ramp=np.linspace(0,1,1001,dtype=np.float32)[None,:,None]*np.ones(3,dtype=np.float32)
    for ev in (.001,.8,2,4):
        adjusted,_=apply_tone(ramp,{'Exposure':ev})
        gain=2**ev; knee=.6/gain
        below=ramp[0,:,0]<=knee
        np.testing.assert_allclose(adjusted[0,below,0],ramp[0,below,0]*gain,atol=1e-7)
        assert np.diff(adjusted[0,:,0]).min()>0
        assert abs(float(adjusted[0,-1,0])-1)<2e-7
        side=np.array([knee-1e-4,knee,knee+1e-4],dtype=np.float32)[None,:,None]*np.ones(3,dtype=np.float32)
        mapped,_=apply_tone(side,{'Exposure':ev})
        np.testing.assert_allclose(np.diff(mapped[0,:,0])/np.diff(side[0,:,0]),gain,rtol=.01)
    values=np.array([[[1,1,1],[1,.4,.2],[.9,.9,.9]]],dtype=np.float32)
    _,_,meta=render_linear_preview(values,{'Exposure':1,'HighlightRecovery':50})
    assert meta['source_channel_ceiling_fraction']==2/3
    assert meta['source_white_ceiling_fraction']==1/3
    assert meta['after_full_channel_fraction']==0
    assert 'not sensor' in meta['source_ceiling_semantics']
