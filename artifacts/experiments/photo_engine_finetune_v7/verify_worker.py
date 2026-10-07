"""Actual sequential 50 MP fine-tuning workers with native export and sampled RSS."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from app.streamlit_app import run_fine_tuning
from src.photo_engine import render_photo

name=sys.argv[1]
cached=ROOT/'data/user_photo_diagnostics/highres-v6'/name
previous=json.loads((cached/'report.json').read_text())
folder=ROOT/'data/user_photo_diagnostics/finetune-v7'/('worker-'+name);folder.mkdir(exist_ok=True)
pixels={}
for key,file in [('original','before.png'),('adjusted','after.png')]:
    with Image.open(cached/file) as image:pixels[key]=np.asarray(image,dtype=np.uint8).copy()
stop=threading.Event();rss={'tree_peak_bytes':0,'root_peak_bytes':0,'samples':0}
def monitor():
    while not stop.is_set():
        output=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True)
        rows=[tuple(map(int,line.split())) for line in output.stdout.splitlines()];ids={os.getpid()}
        while True:
            expanded=ids|{pid for pid,parent,_ in rows if parent in ids}
            if ids==expanded:break
            ids=expanded
        rss['tree_peak_bytes']=max(rss['tree_peak_bytes'],sum(size*1024 for pid,_,size in rows if pid in ids))
        rss['root_peak_bytes']=max(rss['root_peak_bytes'],sum(size*1024 for pid,_,size in rows if pid==os.getpid()))
        rss['samples']+=1;stop.wait(.25)
thread=threading.Thread(target=monitor,daemon=True);thread.start();started=time.perf_counter()
report={'input':previous['input'],'input_sha256':previous['input_sha256_before'],'settings':{'vibrance':-4,'local_contrast':6},
        'memory_semantics':'250ms sampled process-tree RSS sum including sampling command; shared pages may count twice, short peaks may be missed. macOS rusage high-water bytes are separately recorded.'}
try:
    baseline_hash=hashlib.sha256(memoryview(pixels['adjusted'])).hexdigest()
    metadata,adjusted=run_fine_tuning(pixels['adjusted'],report['settings'])
    report['worker_call_seconds']=time.perf_counter()-started
    assert metadata['input_pixels_sha256']==baseline_hash
    assert hashlib.sha256(memoryview(pixels['adjusted'])).hexdigest()==baseline_hash
    tuned={'original':pixels['original'],'adjusted':adjusted}
    before,after,export=render_photo(tuned)
    (folder/'result.png').write_bytes(after)
    input_name=Path(previous['input']).stem
    frozen=ROOT/'data/user_photo_diagnostics/finetune-v7'/input_name/'restrained.png'
    assert hashlib.sha256(after).hexdigest()==hashlib.sha256(frozen.read_bytes()).hexdigest()
    with Image.open(folder/'result.png') as image:
        assert image.format=='PNG' and image.mode=='RGB' and list(image.size)==previous['payload']['output_size']
        assert 'srgb' in image.info
        np.testing.assert_array_equal(np.asarray(image),adjusted)
    del before,after
    before,after,zero=render_photo(tuned,0.)
    assert before==after and zero['changed_pixel_fraction']==0
    del before,after
    assert hashlib.sha256(Path(previous['input']).read_bytes()).hexdigest()==report['input_sha256']
    report.update(status='passed',fine_tuning=metadata,export=export,source_preserved=True,
                  matches_frozen_native_candidate=True,zero_strength_identity=True)
except Exception as error:
    report.update(status='failed',error=type(error).__name__+': '+str(error))
finally:
    stop.set();thread.join(timeout=2);report.update(total_seconds=time.perf_counter()-started,rss=rss,
        root_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        children_high_water_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','worker_call_seconds','total_seconds','rss','error')}),flush=True)
sys.exit(0 if report['status']=='passed' else 1)
