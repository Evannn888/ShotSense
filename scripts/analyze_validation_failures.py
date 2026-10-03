"""Validation-only failure analysis; never score or inspect the new test partition."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.dataset import ShotSenseDataset, ParameterNormalizer
from src.model import ROOT
from src.preprocess import sha256_file


def analyze(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    for name, expected in json.loads((run_dir/'frozen_run.json').read_text())['files'].items():
        if sha256_file(run_dir/name)!=expected: raise ValueError('Changed frozen artifact: '+name)
    dataset=ShotSenseDataset(ROOT/'data/processed/metadata.npz')
    config=json.loads((run_dir/'config.json').read_text())
    if dataset.data_hash!=config['metadata_sha256']: raise ValueError('Dataset mismatch')
    split=json.loads((run_dir/'split_manifest.json').read_text())['splits']
    index={photo:i for i,photo in enumerate(dataset.ids.tolist())}
    train=np.array([index[photo] for photo in split['train']])
    val=np.array([index[photo] for photo in split['val']])
    with np.load(run_dir/'validation_predictions.npz',allow_pickle=False) as data:
        if data['ids'].tolist()!=split['val']: raise ValueError('Validation ID mismatch')
        predictions={name:ParameterNormalizer.denormalize(data[name]).astype(np.float64) for name in ('dual','semantic','physical_linear','constant')}
    target=dataset.Y[val].astype(np.float64)
    errors={name:np.abs(values-target) for name,values in predictions.items()}
    p2={name:(values[:,0]/8+values[:,5]/100)/2 for name,values in errors.items()}
    lightness=dataset.X[:,:32]@((np.arange(32)+.5)*100/32)
    lo,hi=np.quantile(lightness[train],[1/3,2/3])
    strata={name:np.flatnonzero(mask) for name,mask in [('dark',lightness[val]<=lo),('middle',(lightness[val]>lo)&(lightness[val]<=hi)),('bright',lightness[val]>hi)]}
    report={'scope':'validation_only','run':str(run_dir.relative_to(ROOT)),'validation_count':len(val),
            'run_frozen_sha256':sha256_file(run_dir/'frozen_run.json'),
            'brightness_thresholds_train_only':[float(lo),float(hi)],'strata':{},
            'limitations':['Descriptive post-selection analysis; it cannot establish a causal failure source.', 'Expert disagreement is not irreducible error or a universal aesthetic target.', 'Dark developed Lab statistics do not imply that every scene should be brightened.']}
    for name,rows in strata.items():
        report['strata'][name]={'count':len(rows),'models':{model:{'p2':float(p2[model][rows].mean()),'exposure_mae':float(error[rows,0].mean()),'recovery_mae':float(error[rows,5].mean()),'exposure_bias':float((predictions[model][rows,0]-target[rows,0]).mean())} for model,error in errors.items()}}
    ranked=np.argsort(-p2['dual'],kind='stable')
    report['worst_20_share_of_dual_error']=float(p2['dual'][ranked[:20]].sum()/p2['dual'].sum())
    report['dual_beats_physical_photo_fraction']=float((p2['dual']<p2['physical_linear']).mean())
    report['dual_beats_semantic_photo_fraction']=float((p2['dual']<p2['semantic']).mean())
    report['exposure_absolute_error_vs_expert_std_correlation']=float(np.corrcoef(errors['dual'][:,0],dataset.Y_std[val,0])[0,1])
    output_dir.mkdir(parents=True,exist_ok=False)
    fields=['photo','p2','exposure_target','exposure_prediction','exposure_error','recovery_target','recovery_prediction','recovery_error','expert_exposure_std','developed_lab_lightness']
    with (output_dir/'validation_residuals.csv').open('w') as handle:
        writer=csv.writer(handle,lineterminator='\n'); writer.writerow(fields)
        for i in ranked:
            writer.writerow([split['val'][i],p2['dual'][i],target[i,0],predictions['dual'][i,0],errors['dual'][i,0],target[i,5],predictions['dual'][i,5],errors['dual'][i,5],dataset.Y_std[val[i],0],lightness[val[i]]])
    sheet=Image.new('RGB',(5*224,4*266),'white'); draw=ImageDraw.Draw(sheet)
    for rank,i in enumerate(ranked[:20]):
        x,y=rank%5*224,rank//5*266
        photo=split['val'][i]
        with Image.open(dataset.images_dir/(Path(photo).stem+'.jpg')) as image: sheet.paste(image.convert('RGB'),(x,y))
        draw.text((x+2,y+226),photo,fill='black')
        draw.text((x+2,y+242),f'Exp {predictions["dual"][i,0]:.2f}/{target[i,0]:.2f}; HR {predictions["dual"][i,5]:.1f}/{target[i,5]:.1f}',fill='black')
    sheet.save(output_dir/'worst-validation-inputs.jpg',quality=95)
    (output_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    analyze(ROOT/'artifacts/experiments/model_vnext/control-seed42',ROOT/'artifacts/experiments/model_vnext/failure-analysis')
