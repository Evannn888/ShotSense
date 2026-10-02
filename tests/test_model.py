import numpy as np
import torch
from src.model import ShotSenseModel


def test_backbone_frozen_deterministic_and_head_can_learn():
    torch.set_num_threads(2); torch.manual_seed(42)
    model=ShotSenseModel(pretrained=False)
    model.train()
    assert not model.backbone.training and not any(p.requires_grad for p in model.backbone.parameters())
    before={name:value.clone() for name,value in model.backbone.named_buffers()}
    image=torch.rand(2,3,224,224); physical=torch.rand(2,132)
    with torch.no_grad():
        first=model(image,physical); second=model(image,physical)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    assert first.shape==(2,6) and torch.isfinite(first).all() and (first.abs()<=1).all()
    for name,value in model.backbone.named_buffers(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    semantic=torch.randn(8,1280); raw=torch.randn(8,132)
    fused=model.fused_features(semantic,raw)
    targets=torch.linspace(-.6,.6,48).reshape(8,6)
    optimizer=torch.optim.Adam(model.head.parameters(),lr=.003)
    initial=float((model.head(fused)-targets).abs().mean().detach())
    for _ in range(100):
        optimizer.zero_grad(); loss=(model.head(fused)-targets).square().mean(); loss.backward(); optimizer.step()
    final=float((model.head(fused)-targets).abs().mean().detach())
    assert final<.03 and final<initial/5
