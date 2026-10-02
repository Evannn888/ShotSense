"""Compare the same 20 RAWs and record throughput and sampled process-tree RSS."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def tree_rss(pid):
    rows=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True,check=True).stdout.splitlines()
    processes=[tuple(map(int,row.split())) for row in rows]
    family={pid}
    while True:
        descendants={p for p,parent,_ in processes if parent in family}
        if descendants<=family: break
        family|=descendants
    return sum(rss for p,_,rss in processes if p in family)*1024


def main():
    measurements=[]
    reference=None
    import numpy as np
    for workers in (1,2,6):
        output=ROOT/f'data/processed/benchmark_{workers}'
        log_path=ROOT/f'data/intermediate/benchmark_{workers}.log'
        with log_path.open('w') as log:
            command=[sys.executable,'-m','src.preprocess','--limit','20','--workers',str(workers),'--output-dir',str(output),'--rebuild']
            started=time.perf_counter()
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log)
            peak=0
            try:
                while process.poll() is None:
                    peak=max(peak,tree_rss(process.pid)); time.sleep(.25)
            except BaseException:
                process.terminate()
                process.wait(timeout=10)
                raise
            elapsed=time.perf_counter()-started
        if process.returncode: raise RuntimeError(log_path.read_text())
        with np.load(output/'metadata.npz',allow_pickle=False) as data:
            current={k:data[k].copy() for k in ('X','Y','Y_std','ids')}
        if reference is None: reference=current
        for key in current:
            if not np.array_equal(current[key],reference[key]): raise ValueError(f'Parallel mismatch: {key}, workers={workers}')
        record={'workers':workers,'input_count':20,'wall_seconds':elapsed,'images_per_second':20/elapsed,
                'sampled_peak_process_tree_rss_bytes':peak,'identical_metadata':True}
        measurements.append(record)
        (ROOT/'data/intermediate/preprocessing_benchmark.json').write_text(json.dumps({'measurements':measurements,'rss_sampling_seconds':.25},indent=2))
        print(json.dumps(record),flush=True)


if __name__=='__main__': main()
