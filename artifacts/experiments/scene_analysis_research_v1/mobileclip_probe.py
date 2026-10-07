"""Pinned source-only MobileCLIP2 research; no production dependency or default changes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).parent
LOCAL = ROOT / 'data/user_photo_diagnostics/scene-analysis-research-v1'
sys.path.insert(0, str(LOCAL / 'portable-deps'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from PIL import Image, ImageOps
import open_clip
from open_clip.convert import convert_state_dict
import timm
import torch

assert not (FOLDER / 'mobileclip_results.json').exists(), 'Preserve completed research'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
inputs = json.loads((FOLDER / 'inputs.json').read_text())
prompts = json.loads((FOLDER / 'mobileclip_prompts.json').read_text())
manifest = json.loads((FOLDER / 'mobileclip2-manifest.json').read_text())
weights = LOCAL / 'mobileclip2_s0.pt'
assert hashlib.sha256(weights.read_bytes()).hexdigest() == manifest['weights']['lfs']['sha256']
started = time.perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP2-S0', pretrained=None, image_mean=(0, 0, 0), image_std=(1, 1, 1))
author_state = torch.load(weights, map_location='cpu', weights_only=True)
model.load_state_dict(convert_state_dict(model, author_state), strict=True)
model.eval()
loaded_seconds = time.perf_counter()-started
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

with torch.no_grad():
    text_started = time.perf_counter()
    text_features = {}
    for group in ('subject', 'lighting'):
        for label, descriptions in prompts[group].items():
            encoded = model.encode_text(tokenizer(descriptions), normalize=True).mean(0)
            text_features[(group, label)] = torch.nn.functional.normalize(encoded, dim=0)
    text_seconds = time.perf_counter()-text_started
    first = preprocess(Image.open(inputs[0]['path']).convert('RGB')).unsqueeze(0)
    before = model.encode_image(first, normalize=True)
    model = timm.utils.reparameterize_model(model, inplace=True)
    after = model.encode_image(first, normalize=True)
    reparameterization_max_difference = float((before-after).abs().max())
    assert reparameterization_max_difference <= 1e-4
    rows = []
    for pass_index in range(2):
        for item in inputs:
            with Image.open(item['path']) as image:
                for view in ('stock_center_crop', 'full_frame_letterbox'):
                    begin = time.perf_counter()
                    rgb = image.convert('RGB')
                    if view == 'full_frame_letterbox':
                        rgb = ImageOps.pad(rgb, (256, 256), method=Image.Resampling.BICUBIC,
                                           color=(127, 127, 127), centering=(.5, .5))
                    tensor = preprocess(rgb).unsqueeze(0)
                    prepared = time.perf_counter()
                    embedding = model.encode_image(tensor, normalize=True)
                    inferred = time.perf_counter()
                    scores = {group: {label: float(embedding[0] @ text_features[(group, label)])
                                      for label in prompts[group]} for group in ('subject', 'lighting')}
                    assert all(np.isfinite(value) and -1.0001 <= value <= 1.0001
                               for group in scores.values() for value in group.values())
                    rows.append(dict(id=item['id'], pass_index=pass_index, view=view, scores=scores,
                                     preprocessing_ms=1000*(prepared-begin),
                                     image_forward_ms=1000*(inferred-prepared),
                                     preprocessing_and_forward_ms=1000*(inferred-begin)))

baseline = json.loads((FOLDER / 'baseline.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
           for name, expected in baseline.items())
report = dict(scope='Frozen17reused diagnostic sources; no classification/preference accuracy claim',
              model='apple/MobileCLIP2-S0', revision=manifest['revision'],
              weights_sha256=manifest['weights']['lfs']['sha256'],
              open_clip_version=open_clip.__version__, timm_version=timm.__version__,
              torch_version=torch.__version__, torch_threads=1, device='cpu',
              strict_safe_loading=True, offline_inference=True,
              load_seconds=loaded_seconds, text_cache_seconds=text_seconds,
              reparameterization_max_embedding_difference=reparameterization_max_difference,
              preprocessing=str(preprocess), model_config=open_clip.get_model_config('MobileCLIP2-S0'),
              image_parameter_count=sum(x.numel() for x in model.visual.parameters()),
              full_parameter_count=sum(x.numel() for x in model.parameters()),
              prompt_sha256=hashlib.sha256((FOLDER/'mobileclip_prompts.json').read_bytes()).hexdigest(),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              app_dependency_hashes_unchanged=True, rows=rows)
with (FOLDER/'mobileclip_results.json').open('x') as handle:
    json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
print('Completed pinned MobileCLIP2-S0:17sources,2views,2passes; app unchanged.')
