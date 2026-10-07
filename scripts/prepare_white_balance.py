"""Prepare the pinned author AWB model offline; photo processing never downloads."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'data/external_models/deep_white_balance'
REVISION = 'd70afa227c94152fd8421db75b3ee6ac06a417dc'
FILES = ('LICENSE.md', 'README.md', 'PyTorch/arch/__init__.py',
         'PyTorch/arch/deep_wb_blocks.py', 'PyTorch/arch/deep_wb_single_task.py', 'PyTorch/arch/deep_wb_model.py',
         'PyTorch/utilities/deepWB.py', 'PyTorch/utilities/utils.py', 'PyTorch/models/net_awb.pth')


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    if (BUNDLE/'manifest.json').exists():
        raise ValueError('Preserve existing prepared bundle; do not overwrite')
    tree_url = 'https://api.github.com/repos/mahmoudnafifi/Deep_White_Balance/git/trees/'+REVISION+'?recursive=1'
    with urllib.request.urlopen(tree_url, timeout=30) as response:
        blobs = {v['path']:v['sha'] for v in json.load(response)['tree'] if v['type']=='blob'}
    hashes = {}
    for name in FILES:
        url = 'https://raw.githubusercontent.com/mahmoudnafifi/Deep_White_Balance/'+REVISION+'/'+name
        target = BUNDLE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(50_000_001)
        if len(data)>50_000_000:
            raise ValueError('Author asset exceeds bounded acquisition')
        assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest() == blobs[name], name
        if not target.exists():
            target.write_bytes(data)
        hashes[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'url': url}
        print('Prepared', name, len(data), flush=True)
    torch.set_num_threads(1)
    sys.path.insert(0, str(BUNDLE/'PyTorch'))
    net = importlib.import_module('arch.deep_wb_single_task').deepWBnet()
    net.load_state_dict(torch.load(BUNDLE/'PyTorch/models/net_awb.pth', map_location='cpu', weights_only=True))
    net.eval()
    model = BUNDLE/'awb_fp32.onnx'
    tensor = torch.rand(1, 3, 256, 384, generator=torch.Generator().manual_seed(73))
    torch.onnx.export(net, tensor, str(model), input_names=['image'], output_names=['awb'],
                      dynamic_axes={'image': {2:'height',3:'width'}, 'awb': {2:'height',3:'width'}},
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    checks = []
    with torch.inference_mode():
        for h,w in ((256,384),(384,256),(256,256)):
            rgb = torch.rand(1,3,h,w, generator=torch.Generator().manual_seed(h+w))
            expected = net(rgb).numpy()
            current = session.run(None, {'image':rgb.numpy()})[0]
            error = float(np.abs(current-expected).max())
            assert current.shape == expected.shape and np.isfinite(current).all() and error <= 2e-5
            checks.append({'size':[w,h], 'max_absolute_error':error})
    hashes[model.name] = {'sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'bytes':model.stat().st_size}
    manifest = {'revision':REVISION,'model':'Deep White-Balance Editing single-task AWB',
                'authors':'Mahmoud Afifi and Michael S. Brown, CVPR2020',
                'copyright':'Copyright (c)2019Samsung Electronics Co., Ltd.',
                'license':'CC BY-NC-SA4.0; retained LICENSE.md',
                'files':hashes,'cpu_onnx_parity':checks,'opset':17,
                'runtime':'CPU FP32; local assets only; dynamic dimensions divisible by16'}
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'model_sha256':hashes[model.name]['sha256'],'checks':checks}), flush=True)


if __name__=='__main__':
    main()
