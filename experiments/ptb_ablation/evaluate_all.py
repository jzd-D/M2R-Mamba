"""Evaluate the prespecified models only after every new checkpoint is frozen."""
import gc,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from prepare import HERE,OUT,RAW,write_json
from train import FrozenDataset,evaluate,build_any_model,multilabel_metrics,state_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 torch.cuda.set_per_process_memory_fraction(min(.15,4.5*2**30/torch.cuda.get_device_properties(0).total_memory))
 manifest=json.loads((HERE/'manifest.json').read_text())
 frozen=json.loads((OUT/'checkpoints_frozen.json').read_text())
 assert len(frozen)==len(manifest['jobs'])==15
 for cfg in manifest['jobs']:
  root=OUT/'runs'/cfg['id']
  assert sha(root/'best.pt')==frozen[cfg['id']]
 for cfg in manifest['jobs']:
  root=OUT/'runs'/cfg['id'];target=root/'test_metrics.json'
  if target.exists():
   assert json.loads(target.read_text())['checkpoint_sha256']==frozen[cfg['id']]
   continue
  train=json.loads((root/'metrics.json').read_text())
  classes=json.loads((RAW/'ready.json').read_text())['classes']
  y=np.load(RAW/'train_labels.npy');pos=y.sum(0);weight=(len(y)-pos)/np.maximum(pos,1)
  criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32,device='cuda'))
  ds=FrozenDataset(RAW,'test',12)
  loader=DataLoader(ds,batch_size=cfg['train']['batch_size'],shuffle=False,num_workers=2,pin_memory=True)
  checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
  assert checkpoint['config']==cfg
  model=build_any_model(cfg).cuda();model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
  assert state_hash(model)==train['final_state_sha256']
  test,z,y=evaluate(model,loader,criterion,'ptbxl',classes)
  test['fmax_oracle']=test.pop('fmax');test['fmax_oracle_thr']=test.pop('fmax_thr')
  threshold=train['validation_threshold']
  test['macro_f1_val_threshold']=multilabel_metrics(z,y,thr=threshold)['macro_f1']
  test['validation_threshold']=threshold
  np.savez_compressed(root/'test_predictions.npz',logits=z,labels=y,ids=ds.ids,groups=ds.groups)
  write_json(target,dict(status='ok',id=cfg['id'],seed=cfg['seed'],test=test,checkpoint_sha256=frozen[cfg['id']],all_checkpoints_frozen=True))
  print('EVALUATED',cfg['id'],flush=True)
  del model,loader,ds;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
