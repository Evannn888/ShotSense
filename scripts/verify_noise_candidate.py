"""Verify completed noise-aware pilot provenance, native rendering and source-only replay."""
import json

import numpy as np
import torch

from scripts.train_noise_candidate import OUT
from src.jpeg_curve_candidate import CurveCandidate,apply_curve,source_features
from src.jpeg_noise_candidate import NoiseCandidate,prefilter,render_noise_candidate
from src.jpeg_inference import decode_jpeg
from src.preprocess import ROOT,sha256_file


def verify():
    c=json.loads((OUT/'config.json').read_text());r=json.loads((OUT/'report.json').read_text())
    m=json.loads((OUT/'manifest.json').read_text());s=r['split'];records={p['id']:p for p in m['photos']}
    for path,expected in c['source_sha256'].items():assert sha256_file(ROOT/path)==expected
    for name in ('manifest','split'):assert sha256_file(OUT/(name+'.json'))==c[name+'_sha256']
    assert sha256_file(OUT/'group_review.json')==c['review_sha256']
    assert sha256_file(OUT/'PROTOCOL.md')==c['protocol_sha256']
    assert sha256_file(ROOT/'artifacts/model/model.onnx')==c['production_model_sha256']
    assert not set(s['train'])&set(s['validation'])
    assert set(s['train'])|set(s['validation'])==set(records)
    assert all(not(set(group)&set(s['train']) and set(group)&set(s['validation'])) for group in s['groups'].values())
    assert all(p['id'] in s['train'] for p in m['photos'] if p['previously_inspected'])
    assert len(s['validation_groups'])>=8
    assert not list((ROOT/'data/external/lol_pilot').rglob('*eval15*'))
    images={};hash_count=0
    for p in m['photos']:
        for a in p['sources'].values():assert sha256_file(ROOT/a['path'])==a['sha256'];hash_count+=1
        assert sha256_file(ROOT/p['jpeg_path'])==p['jpeg_sha256']
        images[p['id']]=decode_jpeg(ROOT/p['jpeg_path'])[0]
    values=np.stack([source_features(images[p]) for p in s['train']])
    assert np.max(np.abs(values.mean(0)-c['feature_mean']))<1e-6
    assert np.max(np.abs(np.maximum(values.std(0),1e-6)-c['feature_scale']))<1e-6
    torch.set_num_threads(1);max_code=0;different=0;total=0;max_parameter=0.
    for label,run in r['runs'].items():
        checkpoint=OUT/(label+'.pt');assert sha256_file(checkpoint)==run['checkpoint_sha256']
        model=CurveCandidate(c['feature_mean'],c['feature_scale']) if run['kind']=='gamma' else NoiseCandidate(c['feature_mean'],c['feature_scale'],run['kind']=='constant')
        model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
        for e in r['photos']:
            rgb=images[e['id']];f=torch.tensor(source_features(rgb)[None])
            with torch.no_grad():
                if run['kind']=='gamma':
                    gamma=model.gamma(f)[0].numpy()
                    max_parameter=max(max_parameter,float(np.max(np.abs(gamma-e['parameters'][label]['gamma']))))
                    encoded=apply_curve(rgb,gamma);source=rgb
                else:
                    gamma,color=model.curve_parameters(f);g=float(gamma[0,0]);a=color[0].numpy()
                    max_parameter=max(max_parameter,abs(g-e['parameters'][label]['gamma']),float(np.max(np.abs(a-e['parameters'][label]['color']))))
                    encoded=render_noise_candidate(rgb,g,a);source=prefilter(rgb)
                    assert np.array_equal(render_noise_candidate(rgb,g,a,0),rgb)
                    assert np.array_equal(render_noise_candidate(rgb,g,a),encoded)
                x=torch.tensor(source.transpose(2,0,1)[None].astype(np.float32)/255)
                expected=np.rint(model(x,f)[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
            delta=np.abs(encoded.astype(int)-expected.astype(int));max_code=max(max_code,int(delta.max()))
            different+=int(np.count_nonzero(delta));total+=delta.size
    assert max_parameter<1e-6 and max_code<=1 and different/total<1e-4
    record={'source_member_hashes_verified':hash_count,'groups_exclusive':True,'old_ids_training_only':True,
            'train_only_statistics_recomputed':True,'eval15_not_extracted':True,'model_unchanged':True,
            'max_parameter_replay_error':max_parameter,'native_numpy_torch_max_8bit_code_difference':max_code,
            'native_numpy_torch_different_code_fraction':different/total,'zero_strength_exact':True,
            'repeatable_render':True,'limit':'Checkpoint/calculation replay, not full-training replay or final quality acceptance.'}
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(record)


if __name__=='__main__':verify()
