"""Frozen same-input actual bounded-WB comparison; no after-result tuning."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from artifacts.experiments.scene_policy_connection_v1 import evaluate as helper
from src.lut_natural import process_jpeg

ROOT=helper.ROOT
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/external/white_balance_v1'
V2=ROOT/'data/external/scene_policy_connection_v2'


def sheet(names, records, destination, public=False):
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    canvas=Image.new('RGB',(1800,440*len(names)),'white');draw=ImageDraw.Draw(canvas)
    for row,identity in enumerate(names):
        stem=Path(identity).stem
        wb=records[identity]['result']['white_balance']
        draw.text((12,row*440+8),identity+' | WB: '+wb['status']+' / '+str(wb['reason']),font=font,fill='black')
        source=(DATA/'public_original' if public else helper.PRIOR_DATA/'original')/(stem+'.png')
        paths=(source,V2/'automatic'/(stem+'.png'),DATA/'automatic'/(stem+'.png'))
        for col,(path,label) in enumerate(zip(paths,('Decoded input','Scene v2 (no WB)','Bounded WB candidate'))):
            with Image.open(path) as im:frame=ImageOps.contain(im,(580,370))
            canvas.paste(frame,(col*600+(600-frame.width)//2,row*440+38+(370-frame.height)//2))
            draw.text((col*600+12,row*440+412),label,font=font,fill='black')
    canvas.save(destination,icc_profile=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes())


def main():
    helper.HERE=HERE;helper.DATA=DATA;helper.sheet=sheet
    helper.process_jpeg=lambda path,scene_aware=False:process_jpeg(path,scene_aware=scene_aware,white_balance=scene_aware)
    helper.main()
    records=[helper.read(p) for p in sorted((HERE/'records').glob('*.json'))]
    by_id={r['id']:r for r in records}
    changed=[];accepted=[];same_skipped=0;scores_exact=0;lower=higher=0
    old_error=[];new_error=[];reasons={}
    for record in records:
        identity=record['id'];stem=Path(identity).stem
        old=helper.read(ROOT/'artifacts/experiments/scene_policy_connection_v2/records'/(stem+'.json'))
        assert record['result']['scene_analysis']['views']==old['result']['scene_analysis']['views']
        scores_exact+=1
        assert record['jpeg_sha256']==old['jpeg_sha256']
        for path,expected in old['exports_sha256'].items():assert helper.sha(ROOT/path)==expected
        with Image.open(V2/'automatic'/(stem+'.png')) as a,Image.open(DATA/'automatic'/(stem+'.png')) as b:
            equal=np.array_equal(np.asarray(a),np.asarray(b))
        if not equal:changed.append(identity)
        wb=record['result']['white_balance'];reason=wb['reason']
        reasons[reason]=reasons.get(reason,0)+1
        if wb['applied']:accepted.append(identity)
        else:
            assert equal,identity+' non-applied WB must reproduce v2'
            same_skipped+=1
        if not record['public']:
            previous=old['scene']['pooled_rgb_reference_mae'];current=record['scene']['pooled_rgb_reference_mae']
            old_error.append(previous);new_error.append(current)
            lower+=current<previous;higher+=current>previous
    for page in range(0,len(accepted),4):
        # Accepting a rule is not accepting photographic preference; inspect every changed candidate.
        batch=accepted[page:page+4]
        sheet(batch,by_id,DATA/'comparisons'/('wb-applied-%02d.png'%(page//4+1)),public=False)
    sheet(['a3746-_i2e5991.dng','a2916-wp_crw_7798.dng'],by_id,DATA/'comparisons/candle-flash.png')
    if accepted:
        identity=accepted[0];stem=Path(identity).stem
        inputs=helper.read(HERE/'input_baseline.json')
        item=next((p for p in inputs['photos'] if p['id']==identity),None)
        path=ROOT/item['jpeg_path'] if item else DATA/'public_jpeg'/(stem+'.jpg')
        result,pixels=process_jpeg(path,scene_aware=True,white_balance=True)
        with Image.open(DATA/'automatic'/(stem+'.png')) as im:assert np.array_equal(np.array(im),pixels['adjusted'])
    helper.save(HERE/'wb_comparison.json',{'count':len(records),'same_scene_scores_count':scores_exact,
        'applied_count':len(accepted),'applied_ids':accepted,'changed_count':len(changed),'changed_ids':changed,
        'skipped_v2_exact_count':same_skipped,'reason_counts':reasons,
        'v2_fivek_reference_mae_mean':float(np.mean(old_error)),
        'wb_fivek_reference_mae_mean':float(np.mean(new_error)),
        'lower_reference_error_vs_v2_count':int(lower),'higher_reference_error_vs_v2_count':int(higher),
        'repeat_accepted_exact':bool(accepted),'scope':'Reused diagnostic development, not independent preference/skin accuracy.'})
    print(json.dumps(helper.read(HERE/'wb_comparison.json'),indent=2),flush=True)


if __name__=='__main__':main()
