"""One frozen, independent training run; no test-driven model selection."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'src'))
from m2r_mamba.models.factory import build_any_model
from m2r_mamba.metrics.multilabel import multilabel_metrics, compute_fmax
from m2r_mamba.metrics.multiclass import multiclass_metrics
from m2r_mamba.utils.experiment import setup_logger
from prepare import OUT, RAW, write_json

def state_hash(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():
        h.update(name.encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def forward_loss(model,criterion,x,y,use_amp):
    """Retry an invalid AMP forward in FP32 without double-updating BN or RNG.

    The same rule applies to every architecture. No sample or optimizer step is
    discarded here; a failure in FP32 remains a hard failure.
    """
    if use_amp:
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state()
        buffers={name:b.clone() for name,b in model.named_buffers()}
    with torch.autocast('cuda',enabled=use_amp):
        logits=model(x); loss=criterion(logits,y)
    fallback=False
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        if not use_amp: raise RuntimeError('Nonfinite training loss/logits in FP32')
        with torch.no_grad():
            for name,b in model.named_buffers(): b.copy_(buffers[name])
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng)
        del logits,loss
        with torch.autocast('cuda',enabled=False):
            logits=model(x.float()); loss=criterion(logits,y)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite training loss/logits persists in FP32 retry')
        fallback=True
    return logits,loss,fallback

class FrozenDataset(Dataset):
    def __init__(self,root,split,channels):
        self.x=np.load(root/f'{split}_signals.npy',mmap_mode='r')
        self.y=np.load(root/f'{split}_labels.npy',mmap_mode='r')
        self.ids=np.load(root/f'{split}_ids.npy')
        self.groups=np.load(root/f'{split}_groups.npy')
        z=np.load(root/'normalization.npz'); self.channels=channels
        self.mean=z['mean'][:channels,None]; self.std=z['std'][:channels,None]
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        x=(self.x[i,:self.channels].copy()-self.mean)/(self.std+1e-6)
        return torch.from_numpy(x),torch.from_numpy(np.array(self.y[i],copy=True))

@torch.no_grad()
def evaluate(model,loader,criterion,dataset,classes):
    model.eval(); logits=[]; ys=[]; total=0.; count=0
    for x,y in loader:
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        z=model(x)
        if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite evaluation logits')
        loss=criterion(z,y)
        total+=float(loss)*len(y); count+=len(y)
        logits.append(z.float().cpu().numpy()); ys.append(y.cpu().numpy())
    z=np.concatenate(logits); y=np.concatenate(ys)
    m=multilabel_metrics(z,y) if dataset=='ptbxl' else multiclass_metrics(z,y,classes)
    m['loss']=total/count
    return m,z,y

def run(cfg,smoke=False):
    assert cfg.get('init_checkpoint') is None
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    memory_fraction=float(cfg['gpu_memory_fraction'])
    memory_cap=os.environ.get('M2R_MAX_RESERVED_GIB')
    if memory_cap is not None:
        memory_cap=float(memory_cap)
        if not np.isfinite(memory_cap) or memory_cap<=0: raise ValueError('Invalid GPU memory cap')
        total_memory=torch.cuda.get_device_properties(0).total_memory
        memory_fraction=min(memory_fraction,memory_cap*2**30/total_memory)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    dataset=cfg['dataset']; tc=cfg['train']; droot=RAW
    meta=json.loads((droot/'ready.json').read_text()); classes=meta['classes']
    root=OUT/('smoke' if smoke else 'runs')/cfg['id']
    if (root/'metrics.json').exists(): raise RuntimeError('Completed run already exists')
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'config.json',cfg)
    log=setup_logger(root/'train.log',name=cfg['id'])
    loaders=[]
    for split in ('train','val'):
        ds=FrozenDataset(droot,split,cfg['model']['n_leads'])
        gen=torch.Generator().manual_seed(seed)
        loaders.append(DataLoader(ds,batch_size=tc['batch_size'],shuffle=split=='train',
            num_workers=tc['num_workers'],pin_memory=True,persistent_workers=tc['num_workers']>0,generator=gen))
    tr,va=loaders
    model=build_any_model(cfg).cuda()
    initial_hash=state_hash(model)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json(root/'initialization.json',dict(seed=seed,initial_state_sha256=initial_hash,from_scratch=True,trainable_params=params))
    if dataset=='ptbxl':
        pos=np.asarray(tr.dataset.y).sum(0); weight=(len(tr.dataset)-pos)/np.maximum(pos,1)
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32,device='cuda'))
    else:
        counts=np.bincount(tr.dataset.y,minlength=len(classes))
        weight=np.minimum(len(tr.dataset)/(len(classes)*np.maximum(counts,1)),tc['class_weight_cap'])
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device='cuda'))
    log.info('start %s params=%d initial_hash=%s weights=%s allocator_cap_gib=%s',cfg['id'],params,initial_hash,weight.tolist(),memory_cap)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=tc['epochs'],eta_min=tc['min_lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=tc['amp'])
    best=-float('inf'); best_epoch=-1; bad=0; history=[]; start=time.monotonic(); total_fallbacks=0
    epochs=1 if smoke else tc['epochs']
    for epoch in range(1,epochs+1):
        model.train(); total=0.; n=0; skipped=0; fallbacks=0
        for bi,(x,y) in enumerate(tr):
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z,loss,fallback=forward_loss(model,criterion,x,y,tc['amp'])
            if fallback:
                fallbacks+=1; total_fallbacks+=1
                log.warning('FP32 retry succeeded epoch=%d batch=%d loss=%.6f',epoch,bi,float(loss.detach()))
                with (root/'numerical_fallbacks.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch,batch_index=bi,fp32_loss=float(loss.detach())))+'\n')
            scale0=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); skipped+=int(scaler.get_scale()<scale0)
            total+=float(loss.detach())*len(y); n+=len(y)
            if not smoke and (bi+1)%50==0:
                write_json(root/'progress.json',dict(phase='training',epoch=epoch,batch=bi+1,
                    batches=len(tr),best_epoch=best_epoch,elapsed_seconds=time.monotonic()-start))
            # Leave gaps for the pre-existing GPU workload; identical for every method.
            if tc.get('batch_pause_seconds',0): time.sleep(tc['batch_pause_seconds'])
            if smoke and bi>=1: break
        sched.step()
        if smoke:
            model.eval()
            x,y=next(iter(va))
            with torch.no_grad(): z=model(x.cuda())
            assert torch.isfinite(z).all() and z.shape==(len(y),len(classes))
            result=dict(status='ok',training_steps=bi+1,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,trainable_params=params)
            write_json(root/'smoke.json',result); print(result,flush=True); return
        write_json(root/'progress.json',dict(phase='validation',epoch=epoch,batch=len(tr),
            batches=len(tr),best_epoch=best_epoch,elapsed_seconds=time.monotonic()-start))
        val,_,_=evaluate(model,va,criterion,dataset,classes)
        score=float(val[tc['monitor']]); assert np.isfinite(score)
        row=dict(epoch=epoch,train_loss=total/n,val=val,amp_skipped_steps=skipped,fp32_fallback_batches=fallbacks,lr=opt.param_groups[0]['lr'])
        history.append(row)
        write_json(root/'history.json',history)
        log.info('epoch=%d loss=%.5f val_%s=%.5f skipped=%d elapsed=%.1fs',epoch,total/n,tc['monitor'],score,skipped,time.monotonic()-start)
        write_json(root/'progress.json',dict(phase='epoch_complete',epoch=epoch,best_epoch=epoch if score>best else best_epoch,validation_score=score,elapsed_seconds=time.monotonic()-start))
        if score>best:
            best=score; best_epoch=epoch; bad=0
            tmp=root/'best.tmp'; torch.save(dict(model=model.state_dict(),config=cfg,epoch=epoch),tmp); tmp.replace(root/'best.pt')
        else:
            bad+=1
            if bad>=tc['patience']: break
    assert best_epoch>0
    ckpt=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'],strict=True); del ckpt
    val,vz,vy=evaluate(model,va,criterion,dataset,classes)
    threshold=None
    if dataset=='ptbxl':
        probs=1/(1+np.exp(-np.clip(vz,-80,80)))
        threshold=compute_fmax(probs,vy)['fmax_thr']
    np.savez_compressed(root/'val_predictions.npz',logits=vz,labels=vy,ids=va.dataset.ids,groups=va.dataset.groups)
    result=dict(status='ok',id=cfg['id'],dataset=dataset,method=cfg['method'],seed=seed,
        independent_training=True,initial_state_sha256=initial_hash,final_state_sha256=state_hash(model),
        best_epoch=best_epoch,validation=val,validation_threshold=threshold,test_evaluated=False,trainable_params=params,
        seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,history=history,
        numerics_revision='amp-with-fp32-forward-retry-v1',fp32_fallback_batches=total_fallbacks,
        allocator_cap_gib=memory_cap,actual_gpu_memory_fraction=memory_fraction)
    write_json(root/'metrics.json',result)
    log.info('completed training best_epoch=%d validation=%s; test deferred',best_epoch,val)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    cfg=next(x for x in manifest['jobs'] if x['id']==a.job)
    run(cfg,a.smoke)
