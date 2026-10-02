"""Export a self-describing FP32 bundle and verify framework parity."""
import argparse
import io
import json
from pathlib import Path
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.dataset import ShotSenseDataset
from src.model import ROOT,ShotSenseModel
from src.preprocess import atomic_bytes,sha256_file


def export(checkpoint_path,metadata_path,output_dir,benchmark_runs=100):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    model=ShotSenseModel(pretrained=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    dataset=ShotSenseDataset(metadata_path)
    if dataset.data_hash!=checkpoint['config']['metadata_sha256']:
        raise ValueError('Export data does not match checkpoint')
    # Representative real inputs, including each parameter's extrema.
    selected=sorted({0,*[int(np.argmin(dataset.Y[:,i])) for i in range(6)],
                       *[int(np.argmax(dataset.Y[:,i])) for i in range(6)]})
    image=torch.stack([dataset[i][0] for i in selected]); physical=torch.from_numpy(dataset.X[selected])
    temporary=output_dir/'model.tmp.onnx'
    torch.onnx.export(model,(image[:1],physical[:1]),str(temporary),opset_version=17,dynamo=False,
                      input_names=['image','physical_raw'],output_names=['normalized_absolute'],
                      dynamic_axes={'image':{0:'batch'},'physical_raw':{0:'batch'},'normalized_absolute':{0:'batch'}})
    onnx.checker.check_model(str(temporary))
    sessions={}
    parity=[]
    with torch.inference_mode(): expected=model(image,physical).numpy()
    for threads in (1,2,4):
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        sessions[threads]=ort.InferenceSession(str(temporary),sess_options=options,providers=['CPUExecutionProvider'])
        actual=sessions[threads].run(None,{'image':image.numpy(),'physical_raw':physical.numpy()})[0]
        difference=float(np.max(np.abs(actual-expected)))
        if difference>1e-4: raise ValueError(f'FP32 export parity failed: {difference}')
        parity.append({'threads':threads,'max_normalized_difference':difference})
    timing=[]; feed={'image':image[:1].numpy(),'physical_raw':physical[:1].numpy()}
    for threads,session in sessions.items():
        for _ in range(10): session.run(None,feed)
        samples=[]
        for _ in range(benchmark_runs):
            started=time.perf_counter(); session.run(None,feed); samples.append((time.perf_counter()-started)*1000)
        timing.append({'threads':threads,'p50_ms':float(np.percentile(samples,50)),'p95_ms':float(np.percentile(samples,95)),
                       'runs':benchmark_runs,'batch_size':1,'warmup_runs':10})
    best=min(timing,key=lambda r:r['p95_ms'])
    path=output_dir/'model.onnx'; temporary.replace(path)
    bundle={key:checkpoint[key] for key in ('parameter_order','parameter_min','parameter_max','validated_parameters','experimental_parameters','pipeline_config')}
    bundle.update({'schema_version':1,'model_sha256':sha256_file(path),'model_size_bytes':path.stat().st_size,
                   'checkpoint_sha256':sha256_file(checkpoint_path),'training_config':checkpoint['config'],
                   'parameter_semantics':'absolute legacy Camera Raw PV2003, HighlightRecovery is not modern Highlights',
                   'provider':'CPUExecutionProvider','threads':best['threads'],'opset':17,'precision':'FP32'})
    atomic_bytes(output_dir/'model.json',json.dumps(bundle,indent=2).encode())
    report={'framework_parity':parity,'model_forward_timings':timing,'selected_threads':best['threads'],
            'model_forward_p95_under_30ms':best['p95_ms']<30,'hardware':platform.platform(),'processor':platform.machine(),
            'onnx':onnx.__version__,'onnxruntime':ort.__version__,'model_size_bytes':path.stat().st_size,
            'quantization':'Not applied: FP32 is the verified reference; optimize only if a deployment measurement requires it.'}
    atomic_bytes(output_dir/'deployment_report.json',json.dumps(report,indent=2).encode())
    print(json.dumps({'max_parity_error':max(r['max_normalized_difference'] for r in parity),'best_timing':best,'bundle':str(output_dir)}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/model/best.pt')
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/processed/metadata.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/model')
    parser.add_argument('--benchmark-runs',type=int,default=100)
    args=parser.parse_args()
    if args.benchmark_runs<1: parser.error('benchmark-runs must be positive')
    export(args.checkpoint,args.metadata,args.output_dir,args.benchmark_runs)
