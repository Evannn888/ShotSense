"""Verify spatial pilot provenance, grouping and checkpoint/native replay."""
import json
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scripts.train_spatial_candidate import OUT,PARENT,DATA,parent_model
from scripts.train_noise_candidate import native_metrics
from scripts.train_jpeg_curve import write_json
from src.jpeg_curve_candidate import source_features
from src.jpeg_noise_candidate import render_noise_candidate
from src.jpeg_spatial_candidate import SpatialCandidate,predict_spatial
from src.jpeg_inference import decode_jpeg
from src.preprocess import ROOT,sha256_file


def main():
    torch.set_num_threads(1)
    config=json.loads((OUT/'config.json').read_text());report=json.loads((OUT/'report.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text());split=json.loads((OUT/'split.json').read_text())
    for name in ('protocol','manifest','review','split'):
        path=OUT/({'protocol':'PROTOCOL.md','review':'group_review.json'}.get(name,name+'.json'))
        assert sha256_file(path)==config[name+'_sha256']
    for path,h in config['source_sha256'].items():assert sha256_file(ROOT/path)==h
    assert sha256_file(PARENT/'candidate-42.pt')==config['parent_checkpoint_sha256']
    assert sha256_file(PARENT/'config.json')==config['parent_config_sha256']
    assert sha256_file(ROOT/'artifacts/model/model.onnx')==config['production_model_sha256']
    records={p['id']:p for p in manifest['photos']};train=set(split['train']);val=set(split['validation'])
    assert len(records)==120 and not train&val
    assert train|val|set(split['excluded_alignment'])==set(records)
    assert all(p not in val for p,r in records.items() if r['previously_inspected'])
    for members in split['groups_all'].values():assert not (set(members)&train and set(members)&val)
    assert len(split['validation_groups'])>=8
    for r in records.values():
        for s in r['sources'].values():assert sha256_file(ROOT/s['path'])==s['sha256']
        assert sha256_file(ROOT/r['jpeg_path'])==r['jpeg_sha256']
    assert not any('eval15' in str(p) for p in DATA.rglob('*'))
    parent=parent_model();max_replay=0.;max_zero=0;parameter_count=0
    selected=split['validation'][:8]+['102.png','27.png'];font=ImageFont.load_default(size=14)
    for label,run in report['runs'].items():
        path=OUT/(label+'.pt');assert sha256_file(path)==run['checkpoint_sha256']
        model=SpatialCandidate();model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        parameter_count=sum(p.numel() for p in model.parameters());assert parameter_count==2295
        for photo in report['photos']:
            p=photo['id'];r=records[p];rgb,_=decode_jpeg(ROOT/r['jpeg_path'])
            with Image.open(ROOT/r['sources']['high']['path']) as im:target=np.array(im)
            with torch.no_grad():g,a=parent.curve_parameters(torch.tensor(source_features(rgb)[None]))
            base=render_noise_candidate(rgb,float(g[0,0]),a[0].numpy());output=predict_spatial(model,rgb,base)
            scores=native_metrics(output,target,rgb)
            for k,v in scores.items():
                if v is not None:max_replay=max(max_replay,abs(v-photo['metrics'][label][k]))
            max_zero=max(max_zero,int(np.abs(predict_spatial(model,rgb,base,0).astype(int)-rgb.astype(int)).max()))
            if label=='seed42' and p in selected:
                sheet=Image.new('RGB',(900,320),'white');d=ImageDraw.Draw(sheet)
                for i,(title,image) in enumerate([('Frozen base',base),('Spatial seed42',output),('Reference',target)]):
                    tile=Image.fromarray(image[:96,:96]).resize((300,300),Image.Resampling.NEAREST)
                    sheet.paste(tile,(i*300,20));d.text((i*300+4,2),title,font=font,fill='black')
                sheet.save(OUT/(p.replace('.png','')+'-edge.jpg'),quality=94)
    assert max_replay==0 and max_zero==0
    result={'source_png_hashes_verified':240,'jpeg_hashes_verified':120,'scene_exclusive':True,
            'validation_pairs':len(val),'validation_groups':len(split['validation_groups']),
            'old_ids_training_only':True,'eval15_not_extracted':True,'production_model_unchanged':True,
            'parent_provenance_verified':True,'parameter_count':parameter_count,
            'max_native_metric_replay_difference':max_replay,'max_zero_strength_codes':max_zero,
            'scope':'Checkpoint/metric replay, not full training reproduction or production acceptance.'}
    write_json(OUT/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
