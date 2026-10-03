"""Replay and audit the separate denoising/illumination pilot."""
import json
import cv2
import numpy as np
import torch
from PIL import Image
from scripts.train_illumination_candidate import OUT,PRIOR,PARENT,DATA,parent_model,numeric_gates
from scripts.train_noise_candidate import native_metrics
from scripts.train_jpeg_curve import write_json
from src.jpeg_curve_candidate import source_features
from src.jpeg_noise_candidate import render_noise_candidate
from src.jpeg_illumination_candidate import IlluminationCandidate,denoise_base,predict_illumination
from src.jpeg_inference import decode_jpeg
from src.preprocess import ROOT,sha256_file


def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    config=json.loads((OUT/'config.json').read_text());report=json.loads((OUT/'report.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text());split=json.loads((OUT/'split.json').read_text())
    for name in ('protocol','manifest','review','split'):
        path=OUT/({'protocol':'PROTOCOL.md','review':'group_review.json'}.get(name,name+'.json'))
        assert sha256_file(path)==config[name+'_sha256']
    for path,h in config['source_sha256'].items():assert sha256_file(ROOT/path)==h
    assert sha256_file(PRIOR/'manifest.json')==manifest['prior_manifest_sha256']
    assert sha256_file(DATA/'LOLdataset.zip')==manifest['source_archive_sha256']
    assert sha256_file(PARENT/'candidate-42.pt')==config['parent_checkpoint_sha256']
    assert sha256_file(PARENT/'config.json')==config['parent_config_sha256']
    assert sha256_file(ROOT/'artifacts/model/model.onnx')==config['production_model_sha256']
    records={r['id']:r for r in manifest['photos']};train=set(split['train']);val=set(split['validation'])
    assert len(records)==168 and not train&val
    assert train|val|set(split['excluded_alignment'])==set(records)
    assert all(r['id'] not in val for r in records.values() if r['previously_inspected'])
    for members in split['groups_all'].values():assert not (set(members)&train and set(members)&val)
    assert len(split['validation_groups'])>=8
    for r in records.values():
        for s in r['sources'].values():assert sha256_file(ROOT/s['path'])==s['sha256']
        assert sha256_file(ROOT/r['jpeg_path'])==r['jpeg_sha256']
    assert not any('eval15' in str(p) for p in DATA.rglob('*'))
    parent=parent_model();models={};max_replay=0.;max_zero=0;max_initial=0
    for label,run in report['runs'].items():
        path=OUT/(label+'.pt');assert sha256_file(path)==run['checkpoint_sha256']
        model=IlluminationCandidate(run['constant']);model.load_state_dict(torch.load(path,weights_only=True));model.eval();models[label]=model
        assert sum(p.numel() for p in model.parameters())==(1 if run['constant'] else 881)
        if not run['constant']:
            assert run['numeric_gates']==numeric_gates(report['summary'][label],report['summary']['frozen-base'],report['summary']['global42'])
    initial=IlluminationCandidate()
    for photo in report['photos']:
        r=records[photo['id']];rgb,_=decode_jpeg(ROOT/r['jpeg_path'])
        with Image.open(ROOT/r['sources']['high']['path']) as im:target=np.array(im)
        with torch.no_grad():g,a=parent.curve_parameters(torch.tensor(source_features(rgb)[None]))
        base=render_noise_candidate(rgb,float(g[0,0]),a[0].numpy());clean=denoise_base(base)
        variants={'frozen-base':base,'denoised-base':clean}
        for label,model in models.items():
            variants[label]=predict_illumination(model,rgb,clean)
            max_zero=max(max_zero,int(np.abs(predict_illumination(model,rgb,clean,0).astype(int)-rgb.astype(int)).max()))
        max_initial=max(max_initial,int(np.abs(predict_illumination(initial,rgb,clean).astype(int)-clean.astype(int)).max()))
        for label,image in variants.items():
            for k,v in native_metrics(image,target,rgb).items():
                if v is not None:max_replay=max(max_replay,abs(v-photo['metrics'][label][k]))
    assert max_replay==0 and max_zero==0 and max_initial==0
    result={'source_png_hashes_verified':336,'jpeg_hashes_verified':168,'scene_exclusive':True,
            'validation_pairs':len(val),'validation_groups':len(split['validation_groups']),
            'old_ids_training_only':True,'eval15_not_extracted':True,'production_model_unchanged':True,
            'parent_provenance_verified':True,'spatial_parameter_count':881,'global_parameter_count':1,
            'max_native_metric_replay_difference':max_replay,'max_zero_strength_codes':max_zero,
            'max_initial_identity_codes':max_initial,'verification_source_sha256':sha256_file(ROOT/'scripts/verify_illumination_candidate.py'),
            'scope':'Provenance and checkpoint calculation replay, not complete retraining or independent final acceptance.'}
    write_json(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
