"""Intermediate-cache reuse must preserve RNG and reject changed artifacts."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.train_finetuning as candidate


class TinyDataset:
    data_hash='fixture'

    def __init__(self,path):
        self.images_dir=path

    def __len__(self):
        return 2

    def __getitem__(self,index):
        return torch.full((3,224,224),float(index)),torch.zeros(132),torch.zeros(6)


def test_prefix_cache_preserves_rng_and_checks_artifact_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(candidate,'ROOT',tmp_path)
    (tmp_path/'data/processed/model_vnext').mkdir(parents=True)
    (tmp_path/'src').mkdir()
    for name in ('model.py','dataset.py','image_io.py'):
        (tmp_path/'src'/name).write_text('fixture')
    for name in ('a.jpg','b.jpg'):
        (tmp_path/name).write_bytes(b'fixture image hash')
    prefix=nn.Sequential(nn.AdaptiveAvgPool2d((7,7)),nn.Conv2d(3,192,1),*[nn.Identity() for _ in range(5)])
    model=SimpleNamespace(backbone=SimpleNamespace(features=prefix))
    dataset=TinyDataset(tmp_path)
    torch.manual_seed(123); before=torch.get_rng_state().clone()
    fresh,fingerprint=candidate.prefix_cache(model,dataset,['a.dng','b.dng'],[0,1])
    assert torch.equal(torch.get_rng_state(),before)
    cached,other=candidate.prefix_cache(model,dataset,['a.dng','b.dng'],[0,1])
    assert torch.equal(torch.get_rng_state(),before)
    assert fingerprint==other and torch.equal(fresh,cached)
    path=tmp_path/'data/processed/model_vnext/frozen-prefix.npz'
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='Changed frozen-prefix'):
        candidate.prefix_cache(model,dataset,['a.dng','b.dng'],[0,1])
