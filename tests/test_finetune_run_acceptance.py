"""Check actual complete pilot checkpoints against their matched parents."""
import hashlib
import json

import pytest
import torch

from src.model import ROOT


def test_completed_finetuning_keeps_frozen_state_and_cpu_parity():
    base=ROOT/'artifacts/experiments/model_vnext'
    if not (base/'finetune-seed42/best.pt').exists():
        pytest.skip('Local completed fine-tuning checkpoints are not bundled')
    for seed in (42,43,44):
        path=base/f'finetune-seed{seed}'
        parent=base/('control-seed42' if seed==42 else f'head-h1-seed{seed}')
        report=json.loads((path/'evaluation.json').read_text())
        assert report['evaluation_scope']=='validation_only' and report['validated_parameters']==[]
        assert not report['test_accessed'] and not report['production_replaced']
        assert max(report['input_parity_max_difference'].values())<=1e-5
        for name,expected in json.loads((path/'frozen_run.json').read_text())['files'].items():
            assert hashlib.sha256((path/name).read_bytes()).hexdigest()==expected
        old=torch.load(parent/'best.pt',map_location='cpu',weights_only=False)['state_dict']
        new=torch.load(path/'best.pt',map_location='cpu',weights_only=False)['state_dict']
        allowed=set(report['config']['trainable_parameters'])
        assert all(torch.equal(value,old[name]) for name,value in new.items() if name not in allowed)
        if report['best_epoch']==0:
            assert all(torch.equal(value,old[name]) for name,value in new.items())
    summary=json.loads((base/'phase_e_summary.json').read_text())
    assert summary['decision']=='retain_frozen_control'
    assert not summary['research_gates']['mean_p2_improvement_at_least_5_percent']
