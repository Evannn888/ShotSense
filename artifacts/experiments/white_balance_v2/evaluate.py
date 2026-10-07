"""Frozen same-input and fresh public actual-output comparison."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from artifacts.experiments.scene_policy_connection_v1 import evaluate as helper
from src import white_balance as wb
from src.lut_natural import process_jpeg

ROOT=helper.ROOT
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/external/white_balance_v2'
OLD=ROOT/'data/external/white_balance_v1'
NO_WB=ROOT/'data/external/scene_policy_connection_v2'


def frames(rows,destination):
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    canvas=Image.new('RGB',(1800,440*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    for row,(title,paths) in enumerate(rows):
        draw.text((12,row*440+8),title,font=font,fill='black')
        for col,(path,label) in enumerate(zip(paths,('Decoded input','WB v1','WB v2 candidate'))):
            with Image.open(path) as im:frame=ImageOps.contain(im,(580,370))
            canvas.paste(frame,(col*600+(600-frame.width)//2,row*440+38+(370-frame.height)//2))
            draw.text((col*600+12,row*440+412),label,font=font,fill='black')
    canvas.save(destination,icc_profile=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes())


def sheet(names,records,destination,public=False):
    rows=[]
    for identity in names:
        stem=Path(identity).stem;meta=records[identity]['result']['white_balance']
        source=(DATA/'public_original' if public else helper.PRIOR_DATA/'original')/(stem+'.png')
        rows.append((identity+' | '+meta.get('mode','skipped')+' / '+str(meta['reason']),
                     [source,OLD/'automatic'/(stem+'.png'),DATA/'automatic'/(stem+'.png')]))
    frames(rows,destination)


def main():
    helper.HERE=HERE;helper.DATA=DATA;helper.sheet=sheet
    helper.process_jpeg=lambda path,scene_aware=False:process_jpeg(path,scene_aware=scene_aware,white_balance=scene_aware)
    helper.main()
    records=[helper.read(p) for p in sorted((HERE/'records').glob('*.json'))]
    by_id={r['id']:r for r in records}
    accepted=[];changed=[];same_skipped=0;reasons={};modes={};reference=[];prior_reference=[]
    for record in records:
        identity=record['id'];stem=Path(identity).stem
        prior=helper.read(ROOT/'artifacts/experiments/white_balance_v1/records'/(stem+'.json'))
        assert record['jpeg_sha256']==prior['jpeg_sha256']
        assert record['result']['scene_analysis']['views']==prior['result']['scene_analysis']['views']
        with Image.open(OLD/'automatic'/(stem+'.png')) as a,Image.open(DATA/'automatic'/(stem+'.png')) as b:
            equal=np.array_equal(np.array(a),np.array(b))
        if not equal:changed.append(identity)
        meta=record['result']['white_balance'];reason=meta['reason']
        reasons[reason]=reasons.get(reason,0)+1
        if meta['applied']:
            accepted.append(identity);mode=meta['mode'];modes[mode]=modes.get(mode,0)+1
        else:
            with Image.open(NO_WB/'automatic'/(stem+'.png')) as a,Image.open(DATA/'automatic'/(stem+'.png')) as b:
                assert np.array_equal(np.array(a),np.array(b))
            same_skipped+=1
        if not record['public']:
            reference.append(record['scene']['pooled_rgb_reference_mae'])
            prior_reference.append(prior['scene']['pooled_rgb_reference_mae'])
    review=sorted(set(accepted)|set(helper.read(ROOT/'artifacts/experiments/white_balance_v1/wb_comparison.json')['applied_ids']))
    for page in range(0,len(review),4):
        sheet(review[page:page+4],by_id,DATA/'comparisons'/('wb-review-%02d.png'%(page//4+1)))
    sheet(['a0341-dgw_002.dng','a4485-_dsc0026.dng','a3746-_i2e5991.dng','a2916-wp_crw_7798.dng'],by_id,DATA/'comparisons/failure-controls.png')
    spec=importlib.util.spec_from_file_location('frozen_wb_v1',ROOT/'artifacts/experiments/white_balance_v1/source_snapshot/src/white_balance.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous);previous.BUNDLE=wb.BUNDLE
    fresh=[];fresh_rows=[]
    (DATA/'fresh').mkdir()
    icc=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes()
    for item in helper.read(HERE/'fresh_inputs.json'):
        jpeg=ROOT/item['jpeg_path'];assert helper.sha(jpeg)==item['jpeg_sha256']
        without,no_wb=process_jpeg(jpeg,scene_aware=True)
        with patch.object(wb,'correct_white_balance',previous.correct_white_balance),patch.object(wb,'PIPELINE_VERSION',previous.PIPELINE_VERSION):
            old,old_pixels=process_jpeg(jpeg,scene_aware=True,white_balance=True)
        result,pixels=process_jpeg(jpeg,scene_aware=True,white_balance=True)
        paths=[];hashes={}
        for label,rgb in [('original',pixels['original']),('v1',old_pixels['adjusted']),('v2',pixels['adjusted']),('no_wb',no_wb['adjusted'])]:
            path=DATA/'fresh'/(item['id']+'-'+label+'.png')
            Image.fromarray(rgb).save(path,icc_profile=icc)
            with Image.open(path) as im:assert np.array_equal(np.array(im),rgb) and im.info['icc_profile']==icc
            hashes[str(path.relative_to(ROOT))]=helper.sha(path)
            if label!='no_wb':paths.append(path)
        current=pixels['adjusted'];original=pixels['original']
        assert current.shape==original.shape and current.dtype==np.uint8
        assert not (np.all(current==0,axis=-1)&~np.all(original==0,axis=-1)).any()
        assert not (np.any(current==255,axis=-1)&~np.any(original==255,axis=-1)).any()
        if not result['white_balance']['applied']:assert np.array_equal(current,no_wb['adjusted'])
        record={'id':item['id'],'jpeg_sha256':item['jpeg_sha256'],'result':result,'v1_result':old,
                'no_wb_result':without,'exports_sha256':hashes,
                'different_from_v1':not np.array_equal(current,old_pixels['adjusted'])}
        fresh.append(record);helper.save(HERE/'records'/('fresh-'+item['id']+'.json'),record)
        fresh_rows.append((item['id']+' | '+result['white_balance'].get('mode','skipped')+' / '+str(result['white_balance']['reason']),paths))
    for page in range(0,len(fresh_rows),3):frames(fresh_rows[page:page+3],DATA/'comparisons'/('fresh-%d.png'%(page//3+1)))
    if accepted:
        first=next(p for p in helper.read(HERE/'input_baseline.json')['photos'] if p['id']==accepted[0])
        _,repeat=process_jpeg(ROOT/first['jpeg_path'],scene_aware=True,white_balance=True)
        with Image.open(DATA/'automatic'/(Path(first['id']).stem+'.png')) as im:assert np.array_equal(np.array(im),repeat['adjusted'])
    helper.save(HERE/'comparison.json',{'prior_count':len(records),'scene_scores_exact_count':len(records),
        'applied_ids':accepted,'applied_count':len(accepted),'mode_counts':modes,'changed_vs_v1_ids':changed,
        'changed_vs_v1_count':len(changed),'skipped_no_wb_exact_count':same_skipped,'reason_counts':reasons,
        'v1_reference_mae_mean':float(np.mean(prior_reference)),'v2_reference_mae_mean':float(np.mean(reference)),
        'reference_lower_vs_v1':sum(a<b for a,b in zip(reference,prior_reference)),
        'reference_higher_vs_v1':sum(a>b for a,b in zip(reference,prior_reference)),
        'fresh_count':len(fresh),'fresh_applied_ids':[r['id'] for r in fresh if r['result']['white_balance']['applied']],
        'fresh_changed_vs_v1_ids':[r['id'] for r in fresh if r['different_from_v1']],
        'repeat_accepted_exact':bool(accepted),'scope':'Failure-driven development and fresh diagnostics, no independent preference acceptance.'})
    print(json.dumps(helper.read(HERE/'comparison.json'),indent=2),flush=True)


if __name__=='__main__':main()
