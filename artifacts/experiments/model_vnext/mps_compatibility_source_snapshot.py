import json,time,copy
from pathlib import Path
import numpy as np
import torch
from src.finetune_candidate import FineTuneCandidate
from src.dataset import ShotSenseDataset
from src.preprocess import sha256_file
r=Path.cwd(); base=r/'artifacts/experiments/model_vnext'; parent=base/'control-seed42'
torch.set_num_threads(4); torch.manual_seed(42)
a=FineTuneCandidate(); a.load_state_dict(torch.load(parent/'best.pt',map_location='cpu',weights_only=False)['state_dict']); b=copy.deepcopy(a).to('mps'); a.eval(); b.eval()
split=json.loads((parent/'split_manifest.json').read_text())['splits']; d=ShotSenseDataset(r/'data/processed/metadata.npz'); ids={v:i for i,v in enumerate(d.ids.tolist())}
rows=[d[ids[v]] for v in split['train'][:16]]; images=torch.stack([v[0] for v in rows]); physical=torch.stack([v[1] for v in rows]); target=torch.stack([v[2] for v in rows])
with np.load(r/'data/processed/model_vnext/frozen-prefix.npz',allow_pickle=False) as cache: prefix=torch.from_numpy(cache['features'][:16].copy())
pa=a.forward_tail(prefix,physical); pb=b.forward_tail(prefix.to('mps'),physical.to('mps'))
forward_delta=float((pa-pb.cpu()).abs().max()); assert forward_delta<=1e-5
la=torch.nn.functional.smooth_l1_loss(pa,target,beta=.1); lb=torch.nn.functional.smooth_l1_loss(pb,target.to('mps'),beta=.1); la.backward(); lb.backward()
maxgrad=0
for (name,x),(name2,y) in zip(a.named_parameters(),b.named_parameters()):
 assert name==name2
 if x.requires_grad:
  assert x.grad is not None and y.grad is not None
  torch.testing.assert_close(x.grad,y.grad.cpu(),atol=2e-5,rtol=1e-3); maxgrad=max(maxgrad,float((x.grad-y.grad.cpu()).abs().max()))
opt=torch.optim.AdamW(b.optimizer_groups(),weight_decay=.01); b.train(); start=time.perf_counter()
for i in range(8):
 opt.zero_grad(set_to_none=True); loss=torch.nn.functional.smooth_l1_loss(b.forward_tail(prefix.to('mps'),physical.to('mps')),target.to('mps'),beta=.1); loss.backward(); opt.step()
torch.mps.synchronize(); seconds=time.perf_counter()-start
b.eval()
with torch.no_grad():
 full=b(images.to('mps'),physical.to('mps')); cached=b.forward_tail(prefix.to('mps'),physical.to('mps')); delta=float((full-cached).abs().max().cpu()); assert delta<=1e-5
report={'status':'passed','scope':'MPS compatibility and 8 cached-tail training batches; no accuracy claim','torch':str(torch.__version__),'cpu_mps_forward_max_normalized_difference':forward_delta,'cpu_mps_gradient_max_difference':maxgrad,'post_update_full_image_cached_prefix_parity_max_difference':delta,'eight_batch_seconds':seconds,'training_photos_per_second':128/seconds,'test_accessed':False,'production_replaced':False}
(base/'mps_compatibility.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
