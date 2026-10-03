import numpy as np
import pytest

from src.jpeg_shadow_guard import guarded_curve, MAX_GAIN


def test_guard_identity_endpoints_and_bounded_amplification():
    ramp=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
    gamma=np.array([.15,.4,6.666666],dtype=np.float32)
    result=guarded_curve(ramp,gamma)
    assert np.all(np.diff(result.astype(int),axis=1)>=0)
    assert np.all(result[:,0]==0) and np.all(result[:,-1]==255)
    assert np.all(result[:,1]<=MAX_GAIN)
    assert np.array_equal(guarded_curve(ramp,gamma,0),ramp)
    assert np.array_equal(guarded_curve(ramp,np.ones(3,dtype=np.float32)),ramp)
    half=guarded_curve(ramp,gamma,.5)
    assert np.all(half>=np.minimum(ramp,result)) and np.all(half<=np.maximum(ramp,result))
    x=np.linspace(0,1,100001,dtype=np.float64)
    for exponent in gamma:
        y=np.minimum(x**exponent,MAX_GAIN*x)
        assert np.max(np.diff(y)/np.diff(x))<MAX_GAIN+1e-7
        if exponent<1:
            join=MAX_GAIN**(1/(float(exponent)-1))
            assert abs(join**exponent-MAX_GAIN*join)<1e-10
    with pytest.raises(ValueError):guarded_curve(ramp,gamma,2)
    with pytest.raises(ValueError):guarded_curve(ramp,np.array([0.,1.,1.]))
