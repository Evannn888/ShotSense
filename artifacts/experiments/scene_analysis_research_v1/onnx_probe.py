"""Image-only portable deployment feasibility, not a promoted scene model."""
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
import onnx
import onnxruntime as ort

assert not (FOLDER / 'onnx_results.json').exists(), 'Preserve completed probe'
torch.set_num_threads(1); torch.set_num_interop_threads(1)
inputs = json.loads((FOLDER/'inputs.json').read_text())
prompts = json.loads((FOLDER/'mobileclip_prompts.json').read_text())
manifest = json.loads((FOLDER/'mobileclip2-manifest.json').read_text())
weights = LOCAL/'mobileclip2_s0.pt'
assert hashlib.sha256(weights.read_bytes()).hexdigest() == manifest['weights']['lfs']['sha256']
model, _, preprocess = open_clip.create_model_and_transforms(
    'MobileCLIP2-S0', pretrained=None, image_mean=(0,0,0), image_std=(1,1,1))
model.load_state_dict(convert_state_dict(model, torch.load(weights, map_location='cpu', weights_only=True)), strict=True)
model.eval()
model = timm.utils.reparameterize_model(model, inplace=True)


class ImageEncoder(torch.nn.Module):
    def __init__(self, visual):
        super().__init__(); self.visual = visual

    def forward(self, image):
        return torch.nn.functional.normalize(self.visual(image), dim=-1)


encoder = ImageEncoder(model.visual).eval()
labels = [(group,label) for group in ('subject','lighting') for label in prompts[group]]
tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')
with torch.no_grad():
    text = torch.stack([torch.nn.functional.normalize(
        model.encode_text(tokenizer(prompts[group][label]), normalize=True).mean(0),dim=0)
        for group,label in labels]).numpy()
embedding_path = LOCAL/'text_embeddings.npy'
assert not embedding_path.exists(); np.save(embedding_path, text, allow_pickle=False)
with (FOLDER/'text_embedding_manifest.json').open('x') as handle:
    json.dump(dict(labels=labels, shape=list(text.shape), dtype=str(text.dtype),
                   sha256=hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
                   prompt_sha256=hashlib.sha256((FOLDER/'mobileclip_prompts.json').read_bytes()).hexdigest()),
              handle,indent=2);handle.write('\n')
first = preprocess(Image.open(inputs[0]['path']).convert('RGB')).unsqueeze(0)
export = LOCAL/'mobileclip2_s0_image_fp32.onnx'
assert not export.exists()
before = time.perf_counter()
torch.onnx.export(encoder,first,str(export),dynamo=False,opset_version=17,
                  input_names=['image'],output_names=['image_embedding'])
export_seconds = time.perf_counter()-before
onnx.checker.check_model(str(export))
options = ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
before = time.perf_counter()
session = ort.InferenceSession(str(export), sess_options=options, providers=['CPUExecutionProvider'])
session_seconds = time.perf_counter()-before
rows = []
with torch.no_grad():
    for item in inputs:
        with Image.open(item['path']) as image:
            for view in ('stock_center_crop','full_frame_letterbox'):
                rgb = image.convert('RGB')
                if view == 'full_frame_letterbox':
                    rgb = ImageOps.pad(rgb,(256,256),method=Image.Resampling.BICUBIC,
                                       color=(127,127,127),centering=(.5,.5))
                tensor = preprocess(rgb).unsqueeze(0)
                expected = encoder(tensor).numpy()
                actual = session.run(None,{'image':tensor.numpy()})[0]
                repeated = session.run(None,{'image':tensor.numpy()})[0]
                difference = float(np.max(np.abs(actual-expected)))
                assert difference <= 1e-4 and np.array_equal(actual,repeated)
                a = expected@text.T; b = actual@text.T
                rows.append(dict(id=item['id'],view=view,max_embedding_difference=difference,
                                 max_cosine_difference=float(np.max(np.abs(a-b))),
                                 rankings_match=all(int(a[0,[i for i,x in enumerate(labels) if x[0]==group]].argmax()) ==
                                                    int(b[0,[i for i,x in enumerate(labels) if x[0]==group]].argmax())
                                                    for group in ('subject','lighting'))))

feed = {'image':first.numpy()}
for _ in range(10): session.run(None,feed)
times = []
for _ in range(50):
    before = time.perf_counter();session.run(None,feed)
    times.append((time.perf_counter()-before)*1000)
baseline = json.loads((FOLDER/'baseline.json').read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in baseline.items())
report = dict(scope='17reused sources×2views, fixed image-only FP32 feasibility, not scene accuracy',
              image_model_bytes=export.stat().st_size, image_model_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
              export_seconds=export_seconds, session_setup_seconds=session_seconds, onnx_version=onnx.__version__,
              ort_version=ort.__version__, providers=session.get_providers(), threads=1, batch=1,
              input_shape=[1,3,256,256], output_shape=[1,512], opset=17,
              warmups=10, repeats=50, model_only_p50_p95_ms=[float(x) for x in np.quantile(times,[.5,.95])],
              maximum_embedding_difference=max(x['max_embedding_difference'] for x in rows),
              all_rankings_match=all(x['rankings_match'] for x in rows), repeat_bitwise_identical=True,
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
with (FOLDER/'onnx_results.json').open('x') as handle:
    json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
print('Image-only FP32 ONNX passed34views and50timed calls; no production model change.')
