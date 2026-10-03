import numpy as np
import pytest
import torch

from src.jpeg_noise_candidate import NoiseCandidate,render_noise_candidate,quality_loss


def test_shared_curve_gain_endpoints_identity_and_native_parity():
    rgb=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    model=NoiseCandidate(np.zeros(19),np.ones(19),constant=True)
    with torch.no_grad():model.logits.copy_(torch.tensor([-.8,.3,-.2,.1]))
    features=torch.zeros(1,19);g,a=model.curve_parameters(features)
    gamma=float(g[0,0].detach());color=a[0].detach().numpy()
    image=render_noise_candidate(rgb,gamma,color,smooth=False)
    assert np.all(np.diff(image.astype(int),axis=1)>=0)
    assert np.all(image[:,0]==0) and np.all(image[:,-1]==255)
    assert np.all(image[:,1]<=10)
    assert np.array_equal(render_noise_candidate(rgb,gamma,color,0),rgb)
    assert np.array_equal(render_noise_candidate(rgb,1.,np.zeros(3),smooth=False),rgb)
    x=torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)/255)
    value=model(x,features);encoded=np.rint(value.detach().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
    assert np.array_equal(encoded,image)
    half=render_noise_candidate(rgb,gamma,color,.5,smooth=False)
    assert np.all(half>=np.minimum(image,rgb)) and np.all(half<=np.maximum(image,rgb))
    flat=x.repeat(1,1,4,1)
    loss=quality_loss(model(flat,features),flat,flat,a)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(model.logits.grad).all()
    with pytest.raises(ValueError):render_noise_candidate(rgb,gamma,np.ones(3))
