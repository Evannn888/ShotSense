"""Frozen six-source acquisition; no retries/substitution or inference."""
import json
from pathlib import Path
import urllib.parse
import urllib.request

from PIL import Image
from artifacts.experiments.scene_analysis_public_v1 import prepare as helper
from src.jpeg_inference import decode_jpeg

HERE=Path(__file__).resolve().parent
ROOT=helper.ROOT
DATA=ROOT/'data/user_photo_diagnostics/white-balance-v2-fresh'
SELECTED=[
    ('window_portrait','Portrait at the window.jpg'),
    ('restaurant_window','Man in suit interacts in a restaurant.jpg'),
    ('restaurant_people','People in a restaurant (52378138372).jpg'),
    ('night_market','Night at street market ,people selling shoes and clothes.jpg'),
    ('gold_sunset','Golden Sunset Silhouette.jpg'),
    ('green_landscape','Rocky landscape with green trees and cloudy sky.jpg'),
]


def main():
    assert not (HERE/'fresh_inputs.json').exists(),'Preserve completed preparation.'
    DATA.mkdir(exist_ok=False)
    metadata=HERE/'commons_metadata.json'
    query={'action':'query','format':'json','prop':'imageinfo|revisions',
           'titles':'|'.join('File:'+title for _,title in SELECTED),
           'iiprop':'url|size|sha1|extmetadata','rvprop':'ids|timestamp'}
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(query)
    if not metadata.exists():
        request=urllib.request.Request(url,headers={'User-Agent':'ShotSenseLocalResearch/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=response.read(8*1024*1024+1)
        assert len(data)<=8*1024*1024
        metadata.write_bytes(data)
    helper.LOCAL=DATA
    helper.PAGES={p['title']:p for p in json.loads(metadata.read_text())['query']['pages'].values()}
    helper.PREVIEW_IDS=set()
    icc=(ROOT/'artifacts/experiments/fivek_paired_targets_v1/srgb.icc').read_bytes()
    inputs=[];failed=[]
    for selection in SELECTED:
        try:
            item=helper.download(selection)
            rgb,_=decode_jpeg(item['downloaded_path'])
            image=Image.fromarray(rgb);image.thumbnail((960,960),Image.Resampling.LANCZOS)
            jpeg=DATA/(item['id']+'.diagnostic.jpg')
            image.save(jpeg,quality=100,subsampling=0,icc_profile=icc)
            item.update(jpeg_path=str(jpeg.relative_to(ROOT)),jpeg_sha256=helper.sha(jpeg),
                        diagnostic_size=list(image.size),diagnostic_recipe='<=960px tagged sRGB JPEG quality100/subsampling0')
            inputs.append(item)
            print('Prepared '+item['id'],flush=True)
        except Exception as error:
            failed.append({'id':selection[0],'title':selection[1],'error':str(error),
                           'http_status':getattr(error,'code',None),
                           'retry_after':getattr(error,'headers',{}).get('Retry-After')})
            print('Unavailable '+selection[0]+': '+str(error),flush=True)
    (HERE/'fresh_inputs.json').write_text(json.dumps(inputs,indent=2,ensure_ascii=False)+'\n')
    (HERE/'fresh_unavailable.json').write_text(json.dumps(failed,indent=2)+'\n')
    if inputs:helper.contact_sheet(inputs,DATA/'source-contact.png')
    print(json.dumps({'prepared':len(inputs),'unavailable':len(failed)}),flush=True)


if __name__=='__main__':main()
