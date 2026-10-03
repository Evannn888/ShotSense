import numpy as np
import pytest
import torch
from src.jpeg_illumination_candidate import IlluminationCandidate,denoise_base,predict_illumination


def test_illumination_endpoints_gain_gradients_strength_and_denoiser():
    torch.set_num_threads(1)
    rgb=np.random.default_rng(2).integers(0,256,(32,40,3),dtype=np.uint8)
    clean=denoise_base(rgb)
    x=torch.tensor(clean.transpose(2,0,1)[None].astype(np.float32)/255)
    for constant in (False,True):
        model=IlluminationCandidate(constant)
        assert np.array_equal(predict_illumination(model,rgb,clean),clean)
        assert np.array_equal(predict_illumination(model,rgb,clean,0),rgb)
        model(x).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            if constant:model.logit.fill_(20)
            else:model.layers[-1].bias.fill_(20)
            assert model.gain(x).min()>=.25 and model.gain(x).max()<=4
            endpoints=torch.tensor([0.,1.])[None,None,None,:].expand(1,3,4,2)
            assert torch.equal(model(endpoints),endpoints)
            ramp=torch.linspace(0,1,256)[None,None,None,:].expand(1,3,4,256)
            assert torch.diff(model(ramp),dim=-1).min()>=0
        with pytest.raises(ValueError):predict_illumination(model,rgb,clean,float('nan'))
    # OpenCV's colored nonlocal means uses BGR internally; conversion must preserve RGB ordering.
    red=np.zeros((32,32,3),dtype=np.uint8);red[:,:,0]=180
    result=denoise_base(red)
    assert result[:,:,0].mean()>170 and result[:,:,2].mean()<10
