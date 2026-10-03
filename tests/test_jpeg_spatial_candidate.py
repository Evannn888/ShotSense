import numpy as np
import pytest
import torch

from scripts.train_spatial_candidate import alignment
from src.jpeg_spatial_candidate import SpatialCandidate,predict_spatial


def test_spatial_initial_parent_strength_and_alignment_translation():
    torch.manual_seed(1);rng=np.random.default_rng(1)
    source=rng.integers(0,256,(40,48,3),dtype=np.uint8);base=np.clip(source.astype(int)+15,0,255).astype(np.uint8)
    model=SpatialCandidate()
    assert np.array_equal(predict_spatial(model,source,base),base)
    assert np.array_equal(predict_spatial(model,source,base,0),source)
    half=predict_spatial(model,source,base,.5)
    assert np.all(half>=source) and np.all(half<=base)
    x=torch.tensor(np.concatenate([source,base],-1).transpose(2,0,1)[None].astype(np.float32)/255)
    loss=model(x).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):predict_spatial(model,source,base,2)
    texture=rng.integers(20,240,(128,160,3),dtype=np.uint8)
    moved=np.roll(texture,4,axis=1);check=alignment(texture,moved)
    assert abs(check['native_shift_xy'][0]-4)<.5 and abs(check['native_shift_xy'][1])<.5
    assert check['status']=='quarantine_shift'
