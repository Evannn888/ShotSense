"""Protect frozen-stage and BatchNorm boundaries during partial tuning."""
import torch
from torch import nn

from src.finetune_candidate import FineTuneCandidate


def test_only_last_mbconv_and_head_train_with_frozen_batchnorm():
    torch.set_num_threads(4)
    model=FineTuneCandidate()
    model.train()
    names=[name for name,param in model.named_parameters() if param.requires_grad]
    assert names and all(name.startswith(('backbone.features.7.','head.')) for name in names)
    assert model.backbone.features[7].training
    assert all(not module.training for module in model.backbone.modules() if isinstance(module,nn.modules.batchnorm._BatchNorm))
    assert all(not stage.training for index,stage in enumerate(model.backbone.features) if index!=7)
    buffers={name:value.clone() for name,value in model.named_buffers()}
    model(torch.rand(2,3,224,224),torch.zeros(2,132)).square().mean().backward()
    assert any(param.grad is not None and bool(param.grad.abs().sum()>0) for name,param in model.named_parameters() if name.startswith('backbone.features.7.'))
    assert all(param.grad is None for name,param in model.named_parameters() if not param.requires_grad)
    assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers())
    model.eval()
    assert not model.backbone.features[7].training


def test_cached_frozen_prefix_matches_full_image_after_tail_update():
    torch.set_num_threads(4)
    model=FineTuneCandidate(); model.eval()
    image=torch.rand(2,3,224,224); physical=torch.rand(2,132)
    with torch.no_grad():
        prefix=model.backbone.features[:7](image).detach()
        torch.testing.assert_close(model.forward_tail(prefix,physical),model(image,physical),atol=1e-6,rtol=1e-5)
    frozen={name:value.clone() for name,value in model.state_dict().items()
            if not name.startswith(('backbone.features.7.','head.'))}
    optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01)
    model.train(); optimizer.zero_grad()
    model.forward_tail(prefix,physical).square().mean().backward(); optimizer.step()
    assert all(torch.equal(model.state_dict()[name],value) for name,value in frozen.items())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.forward_tail(prefix,physical),model(image,physical),atol=1e-6,rtol=1e-5)
