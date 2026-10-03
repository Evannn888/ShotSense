import numpy as np
import pytest
import torch

from src.jpeg_curve_candidate import CurveCandidate, apply_curve, source_features


def test_curve_monotonic_identity_strength_and_torch_parity():
    ramp=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    features=source_features(ramp)
    assert features.shape==(19,) and np.isfinite(features).all()
    model=CurveCandidate(np.zeros(19),np.ones(19),constant=True)
    with torch.no_grad():model.logits.copy_(torch.tensor([-.8,-.3,.4]))
    f=torch.tensor(features[None]);x=torch.tensor(ramp.transpose(2,0,1)[None].astype(np.float32)/255)
    gamma=model.gamma(f)[0].detach().numpy()
    output=apply_curve(ramp,gamma)
    assert np.all(np.diff(output.astype(int),axis=1)>=0)
    assert np.all(output[:,0]==0) and np.all(output[:,-1]==255)
    assert np.array_equal(apply_curve(ramp,gamma,0),ramp)
    assert np.array_equal(apply_curve(ramp,np.ones(3,dtype=np.float32)),ramp)
    half=apply_curve(ramp,gamma,.5)
    assert np.all(half>=np.minimum(ramp,output)) and np.all(half<=np.maximum(ramp,output))
    numpy_float=(ramp.astype(np.float32)/255)**gamma
    torch_float=model(x,f)[0].permute(1,2,0)
    assert np.max(np.abs(torch_float.detach().numpy()-numpy_float))<2e-7
    torch_float.sum().backward()
    assert torch.isfinite(model.logits.grad).all()
    with pytest.raises(ValueError):apply_curve(ramp,np.array([0.,1.,1.]))
    with pytest.raises(ValueError):apply_curve(ramp,gamma,float('nan'))
