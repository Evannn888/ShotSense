"""Fresh Torch/OpenCLIP-free image-only research feasibility check."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

FOLDER = Path(__file__).parent
ROOT = FOLDER.parents[2]
LOCAL = ROOT/'data/user_photo_diagnostics/scene-analysis-research-v1'
inputs = json.loads((FOLDER/'inputs.json').read_text())
item = next(x for x in inputs if x['id']=='nir-himi-1P1yaGS_Gek-unsplash')
before = time.perf_counter()
with Image.open(item['path']) as image:
    letterbox = ImageOps.pad(image.convert('RGB'),(256,256),method=Image.Resampling.BICUBIC,
                            color=(127,127,127),centering=(.5,.5))
    tensor = np.ascontiguousarray(np.asarray(letterbox,dtype=np.float32).transpose(2,0,1)[None]/255)
prepared = time.perf_counter()
options = ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
session = ort.InferenceSession(str(LOCAL/'mobileclip2_s0_image_fp32.onnx'),
                              sess_options=options,providers=['CPUExecutionProvider'])
loaded = time.perf_counter()
embedding = session.run(None,{'image':tensor})[0]
finished = time.perf_counter()
assert embedding.shape == (1,512) and np.isfinite(embedding).all()
assert not any(name.split('.')[0] in ('torch','timm','open_clip') for name in sys.modules)
text = np.load(LOCAL/'text_embeddings.npy',allow_pickle=False)
print(json.dumps(dict(id=item['id'],view='full_frame_letterbox',torch_openclip_timm_not_imported=True,
                      preprocessing_ms=(prepared-before)*1000,session_setup_ms=(loaded-prepared)*1000,
                      first_forward_ms=(finished-loaded)*1000,total_after_import_ms=(finished-before)*1000,
                      embedding=embedding[0].tolist(),cosine_scores=(embedding@text.T)[0].tolist()),allow_nan=False))
