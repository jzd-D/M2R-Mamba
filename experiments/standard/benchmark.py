"""Measure latency only if there are no other GPU compute users."""
import json, subprocess, sys, time
import numpy as np
import torch
from prepare import HERE, OUT, write_json
sys.path.insert(0,str(HERE.parents[1]/'src'))
from m2r_mamba.models.factory import build_any_model

def main():
    probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    if probe.stdout.strip():
        write_json(OUT/'benchmark_status.json',dict(status='deferred_gpu_busy',reason='Timing while another compute job runs would be misleading. Run benchmark.py later on an idle GPU.'))
        return
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.15)
    torch.empty(1,device='cuda')
    def pids():
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
        return set(r.stdout.split())
    own=pids()
    if len(own)!=1:
        write_json(OUT/'benchmark_status.json',dict(status='deferred_gpu_busy',reason='Another job started before timing.'))
        return
    rows=[]
    manifest=json.loads((HERE/'manifest.json').read_text())
    for cfg in manifest['jobs']:
        if cfg['seed']!=1 or cfg['dataset']!='ptbxl': continue
        model=build_any_model(cfg).cuda().eval()
        ckpt=torch.load(OUT/'runs'/cfg['id']/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model']); del ckpt
        for batch in (1,64):
            if pids()!=own:
                write_json(OUT/'benchmark_status.json',dict(status='deferred_gpu_busy',reason='Another job started; all provisional timings discarded.'))
                return
            x=torch.zeros(batch,cfg['model']['n_leads'],1000 if cfg['dataset']=='ptbxl' else 900,device='cuda')
            with torch.inference_mode():
                for _ in range(20): model(x)
                torch.cuda.synchronize(); times=[]
                for _ in range(100):
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); model(x); end.record(); torch.cuda.synchronize(); times.append(start.elapsed_time(end))
            rows.append(dict(dataset=cfg['dataset'],method=cfg['method'],batch=batch,
                median_batch_ms=float(np.median(times)),p95_batch_ms=float(np.quantile(times,.95)),
                samples_per_second=float(batch*1000/np.median(times)),precision='float32',
                hardware=torch.cuda.get_device_name(),scope='GPU forward only; excludes host preprocessing and transfers'))
        del model,x; torch.cuda.empty_cache()
    if pids()!=own:
        write_json(OUT/'benchmark_status.json',dict(status='deferred_gpu_busy',reason='Another job started; all provisional timings discarded.'))
        return
    write_json(OUT/'benchmarks.json',rows)
    write_json(OUT/'benchmark_status.json',dict(status='complete',note='No competing compute process observed at benchmark boundaries.'))

if __name__=='__main__': main()
