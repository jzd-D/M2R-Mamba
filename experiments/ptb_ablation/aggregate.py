"""Keep original and additional runs traceable; never rank or filter runs by outcome."""
import csv,itertools,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t
from prepare import HERE,OUT,OUTPUT_ROOT,write_json

METRICS=['macro_auroc','macro_auprc','macro_f1','macro_f1_val_threshold']
def main():
 m=json.loads((HERE/'manifest.json').read_text());runs=[];trained=0;tested=0
 for ref in m['references']:
  r=json.loads((OUTPUT_ROOT/'standard/runs'/ref['id']/'metrics.json').read_text())
  r['source']='reused_original';runs.append(r)
 for cfg in m['jobs']:
  root=OUT/'runs'/cfg['id'];p=root/'metrics.json'
  if not p.exists():continue
  r=json.loads(p.read_text());assert r['status']=='ok';trained+=1
  r['source']='new_supplement'
  if (root/'test_metrics.json').exists():
   e=json.loads((root/'test_metrics.json').read_text());r['test']=e['test'];tested+=1
  runs.append(r)
 groups=defaultdict(list)
 for r in runs:groups[r['method']].append(r)
 rows=[]
 for method,items in groups.items():
  assert len(set(r['seed'] for r in items))==len(items)
  for split in ['validation','test']:
   selected=[r for r in items if split in r]
   for metric in METRICS:
    subset=[r for r in selected if metric in r[split]]
    if not subset:continue
    values=np.asarray([r[split][metric] for r in subset]);n=len(values)
    rows.append(dict(method=method,split=split,metric=metric,n=n,mean=float(values.mean()),sample_sd=float(values.std(ddof=1)) if n>1 else None,
                     seeds=[r['seed'] for r in subset],trainable_params=subset[0]['trainable_params'],source=subset[0]['source']))
 comparisons=[]
 for newer,older in m['contrasts']:
  a={r['seed']:r for r in groups[newer] if 'test' in r};b={r['seed']:r for r in groups[older] if 'test' in r}
  if set(a)!=set(range(1,6)) or set(b)!=set(range(1,6)):continue
  d=np.array([a[s]['test']['macro_auroc']-b[s]['test']['macro_auroc'] for s in range(1,6)])
  null=[abs(np.mean(d*np.asarray(signs))) for signs in itertools.product([-1,1],repeat=5)]
  half=float(t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5))
  comparisons.append(dict(newer=newer,older=older,metric='macro_auroc',seed_differences=d.tolist(),mean_difference=float(d.mean()),
    descriptive_seed_t95=[float(d.mean()-half),float(d.mean()+half)],p_raw=float(np.mean(np.asarray(null)>=abs(d.mean())-1e-12))))
 if len(comparisons)==len(m['contrasts']):
  previous=0
  for rank,i in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p_raw'])):
   previous=max(previous,min(1,comparisons[i]['p_raw']*(len(comparisons)-rank)));comparisons[i]['p_holm']=previous
 result=dict(status='complete' if tested==15 else 'in_progress',new_trained=trained,new_tested=tested,new_expected=15,
   reused_runs=len(m['references']),progression=m['progression'],rows=rows,paired_primary_contrasts=comparisons,
   caveat='All five seeds retained. Parameter counts differ. Five-pair two-sided exact minimum p=.0625. Prior test exposure remains. No monotonic-improvement guarantee.')
 write_json(OUT/'summary.json',result)
 if rows:
  with (OUT/'summary.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 if tested==15:
  write_json(OUT/'per_seed_results.json',[{k:r[k] for k in ['id','method','seed','source','validation','test','trainable_params','best_epoch']} for r in runs])
  order=m['progression']+['m2r_ss_residual','m2r_no_lead','m2r_no_multiscale','m2r_unidirectional']
  with (OUT/'ablation_report.txt').open('w') as f:
   f.write('PTB-XL supplementary ablation: five seeds, mean +/- sample SD\nProgression: SS-CNN -> MS-CNN -> + lead -> + BiMamba\n\n')
   for name in order:
    f.write(name+'\n')
    for row in rows:
     if row['method']==name and row['split']=='test':f.write(f"  {row['metric']}: {row['mean']:.6f} +/- {row['sample_sd']:.6f}; n={row['n']}\n")
   f.write('\n'+result['caveat']+'\n')
 print(json.dumps({k:result[k] for k in ['status','new_trained','new_tested','reused_runs']}),flush=True)
if __name__=='__main__':main()
