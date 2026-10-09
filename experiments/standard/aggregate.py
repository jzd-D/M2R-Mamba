"""Transparent aggregates; never pick seeds or experiments using test performance."""
import argparse, csv, itertools, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from prepare import HERE, OUT, write_json

def exact_paired_test(d):
    d=np.asarray(d,dtype=float)
    null=np.asarray([np.mean(d*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=len(d))])
    return float(np.mean(np.abs(null)>=abs(d.mean())-1e-12))

def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_raw'])
    previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,rows[i]['p_raw']*(len(rows)-rank)))
        rows[i]['p_holm']=previous

def collect():
    rows=[]
    for cfg in json.loads((HERE/'manifest.json').read_text())['jobs']:
        p=OUT/'runs'/cfg['id']/'metrics.json'
        if p.exists(): rows.append(json.loads(p.read_text()))
    return rows

def aggregate():
    rows=collect(); grouped=defaultdict(list)
    for r in rows: grouped[(r['dataset'],r['method'])].append(r)
    summary=[]
    for (ds,method),runs in sorted(grouped.items()):
        hashes=[r['initial_state_sha256'] for r in runs]
        assert len(hashes)==len(set(hashes)), f'Reused initialization in {ds}/{method}'
        keys=['macro_auroc','macro_auprc','macro_f1','macro_f1_val_threshold','fmax_oracle'] if ds=='ptbxl' else ['accuracy','macro_f1','n_sensitivity','s_sensitivity','v_sensitivity','f_sensitivity']
        for metric in keys:
            vals=np.asarray([r['test'][metric] for r in runs]); n=len(vals)
            summary.append(dict(dataset=ds,method=method,metric=metric,n=n,
                mean=float(vals.mean()),sample_sd=float(vals.std(ddof=1)) if n>1 else None,
                seeds=[r['seed'] for r in runs],trainable_params=runs[0]['trainable_params']))
    write_json(OUT/'summary.json',dict(rows=summary,completed=len(rows),expected=len(json.loads((HERE/'manifest.json').read_text())['jobs']),
        note='SD describes training-seed variation on the same test cohort; it is not population uncertainty. Historical runs are not pooled.'))
    if summary:
        with (OUT/'summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    comparisons=[]
    for ds,metric in [('ptbxl','macro_auroc'),('mitbih','macro_f1')]:
        full={x['seed']:x for x in grouped.get((ds,'m2r_full'),[])}
        family=[]
        for (dataset,method),runs in grouped.items():
            if dataset!=ds or method=='m2r_full': continue
            other={x['seed']:x for x in runs}; seeds=sorted(set(full)&set(other))
            if len(seeds)!=5: continue
            d=np.array([full[s]['test'][metric]-other[s]['test'][metric] for s in seeds])
            se=d.std(ddof=1)/np.sqrt(len(d)); half=stats.t.ppf(.975,len(d)-1)*se
            family.append(dict(dataset=ds,reference='m2r_full',comparator=method,metric=metric,
                n=5,seeds=seeds,mean_difference=float(d.mean()),seed_difference_sd=float(d.std(ddof=1)),
                seed_mean_difference_t95=[float(d.mean()-half),float(d.mean()+half)],p_raw=exact_paired_test(d)))
        holm(family); comparisons.extend(family)
    write_json(OUT/'paired_seed_comparisons.json',dict(comparisons=comparisons,
        caveat='Five seed pairs give a minimum two-sided exact sign-flip p of 0.0625; no p<0.05 claim can be made from this test. Seed intervals condition on the fixed test cohort.'))
    return rows

def patient_bootstrap(rows,n_boot=1000):
    results=[]
    for ds in ['ptbxl']:
        methods=['fcn_wang','resnet1d_wang','inception1d','lstm','lstm_bidir','xresnet1d101','ecg_mamba']
        metric='macro_auroc' if ds=='ptbxl' else 'macro_f1'
        data={}
        for method in ['m2r_full']+methods:
            items=[]
            for seed in range(1,6):
                p=OUT/'runs'/f'{ds}_{method}_s{seed}'/'test_predictions.npz'
                z=np.load(p); items.append({k:z[k] for k in z.files})
            data[method]=items
        first=data['m2r_full'][0]; y=first['labels']; groups=first['groups']
        unique,inv=np.unique(groups,return_inverse=True)
        for items in data.values():
            for x in items:
                assert np.array_equal(x['ids'],first['ids']) and np.array_equal(x['labels'],y)
        def scores(method,w):
            values=[]
            for x in data[method]:
                if ds=='ptbxl':
                    # AUROC is invariant to monotonic sigmoid; sample_weight resamples whole patients.
                    val=roc_auc_score(y,x['logits'],average='macro',sample_weight=w)
                else:
                    pred=x['logits'].argmax(1)
                    cm=np.bincount(4*y+pred,weights=w,minlength=16).reshape(4,4)
                    den=cm.sum(0)+cm.sum(1)
                    val=np.mean(np.divide(2*np.diag(cm),den,out=np.zeros(4),where=den>0))
                values.append(val)
            return np.mean(values)
        rng=np.random.default_rng(20260930)
        diffs={m:[] for m in methods}
        for _ in range(n_boot):
            multiplicity=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
            w=multiplicity[inv]
            # Reject resamples lacking any class needed by the metric, count them explicitly.
            if ds=='ptbxl':
                if any(np.sum(w*y[:,i])==0 or np.sum(w*(1-y[:,i]))==0 for i in range(y.shape[1])): continue
            elif any(np.sum(w[y==i])==0 for i in range(4)): continue
            full=scores('m2r_full',w)
            for method in methods: diffs[method].append(full-scores(method,w))
        for method,vals in diffs.items():
            assert len(vals)>n_boot*.8
            point=scores('m2r_full',np.ones(len(y)))-scores(method,np.ones(len(y)))
            results.append(dict(dataset=ds,metric=metric,reference='m2r_full',comparator=method,
                mean_seed_metric_difference=float(point),cluster_unit='patient',n_clusters=len(unique),
                requested_resamples=n_boot,valid_resamples=len(vals),
                percentile_95=np.quantile(vals,[.025,.975]).tolist(),
                caveat='Conditional on five trained seeds and the available test patients; exploratory unadjusted intervals, no external-validation claim.'))
        print('patient bootstrap completed',ds,flush=True)
    write_json(OUT/'patient_bootstrap.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--bootstrap',action='store_true'); a=p.parse_args()
    rows=aggregate()
    if a.bootstrap:
        assert len(rows)==len(json.loads((HERE/'manifest.json').read_text())['jobs'])
        patient_bootstrap(rows)
