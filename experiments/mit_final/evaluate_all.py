"""Evaluate all five prespecified seeds after their checkpoints are frozen."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import FrozenDataset,evaluate,build_any_model,state_hash
from prepare import HERE,OUT,RAW,write_json
from imbalance import loss_weights

torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('M2R_MAX_RESERVED_GIB','4'))*2**30/torch.cuda.get_device_properties(0).total_memory)
jobs=json.loads((HERE/'manifest.json').read_text())['jobs']
frozen=json.loads((OUT/'checkpoints_frozen.json').read_text())
assert set(frozen)=={c['id'] for c in jobs} and len(jobs)==5
for cfg in jobs:
 root=OUT/'runs'/cfg['id'];m=json.loads((root/'metrics.json').read_text())
 assert m['status']=='ok' and m['id']==cfg['id'] and 'test' not in m
 assert json.loads((root/'config.json').read_text())==cfg
 assert hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()==frozen[cfg['id']]

droot=RAW;classes=json.loads((droot/'ready.json').read_text())['classes']
ds=FrozenDataset(droot,'test',10)
assert len(ds)==49603 and classes==['N','S','V','F']
loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
weights=loss_weights(jobs[0]['train'])
criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device='cuda'))
for cfg in jobs:
 root=OUT/'runs'/cfg['id'];target=root/'test_metrics.json'
 if target.exists():
  prior=json.loads(target.read_text());assert prior['checkpoint_sha256']==frozen[cfg['id']]
  assert (root/'test_predictions.npz').exists();continue
 model=build_any_model(cfg).cuda();ckpt=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
 assert ckpt['config']==cfg;model.load_state_dict(ckpt['model'],strict=True);del ckpt
 assert state_hash(model)==json.loads((root/'metrics.json').read_text())['final_state_sha256']
 scores,z,y=evaluate(model,loader,criterion,'mitbih',classes)
 assert np.isfinite(z).all()
 np.savez_compressed(root/'test_predictions.npz',logits=z,labels=y,ids=ds.ids,groups=ds.groups)
 write_json(target,dict(status='ok',id=cfg['id'],seed=cfg['seed'],checkpoint_sha256=frozen[cfg['id']],test=scores,
   protocol='Fixed argmax; all five validation-selected checkpoints frozen before any test evaluation in this stage'))
 print('EVALUATED',cfg['id'],scores['macro_f1'],flush=True)
 del model;torch.cuda.empty_cache()
print('ALL FIVE FROZEN CHECKPOINTS EVALUATED',flush=True)
