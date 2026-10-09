"""Bounded validation-only decision calibration; freeze once before test evaluation."""
from pathlib import Path
import argparse, hashlib, json, statistics, os
from datetime import datetime, timezone
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get('M2R_OUTPUT_ROOT', str(BASE/'reproduced'))).resolve()
SOURCE = RESULTS / 'mit_final'
OUT = RESULTS / 'mit_calibration'
CLASSES = ['N', 'S', 'V', 'F']
KEYS = ['accuracy', 'macro_f1', 'n_sensitivity', 's_sensitivity',
        'v_sensitivity', 'f_sensitivity', 'f_precision', 'f_f1', 'normal_to_f']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def predpath(seed, split):
    return SOURCE / 'runs' / f'mitbih_center_early_sboost_s{seed}' / f'{split}_predictions.npz'

def metrics(y, pred):
    assert y.shape == pred.shape and np.isin(y, range(4)).all()
    cm = np.bincount(y.astype(int)*4 + pred, minlength=16).reshape(4, 4)
    tp = cm.diagonal(); support = cm.sum(1); predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support>0)
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted>0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(4), where=support+predicted>0)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_f1=float(f1.mean()),
                **dict(zip(KEYS[2:6], recall.tolist())),
                f_precision=float(precision[3]), f_f1=float(f1[3]), normal_to_f=int(cm[0, 3]),
                confusion_matrix=cm.tolist(), per_class={c:dict(precision=float(precision[i]),
                  recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])) for i,c in enumerate(CLASSES)})

def aggregate(rows):
    return {k:dict(values=[r[k] for r in rows], mean=statistics.mean(r[k] for r in rows),
                   sample_sd=statistics.stdev(r[k] for r in rows)) for k in KEYS}

def load_split(split):
    arrays=[]
    for seed in range(1,6):
        with np.load(predpath(seed, split), allow_pickle=False) as z:
            row={k:z[k].copy() for k in z.files}
        assert row['logits'].shape == (len(row['labels']), 4)
        assert np.isfinite(row['logits']).all()
        if arrays:
            for k in ('labels','ids','groups'):
                assert np.array_equal(row[k], arrays[0][k])
        arrays.append(row)
    return arrays

def scored(arrays, penalty):
    result=[]
    for z in arrays:
        adjusted=z['logits'].astype(np.float64).copy()
        adjusted[:,3] -= penalty
        result.append(metrics(z['labels'], adjusted.argmax(1)))
    return result

def init():
    OUT.mkdir(parents=True, exist_ok=False)
    save_new(OUT/'protocol.json', dict(created_utc=stamp(), source=str(SOURCE),
      script_sha256=sha(Path(__file__)), classes=CLASSES, seeds=list(range(1,6)),
      candidate_f_logit_penalties=[0,0.25,0.5,0.75,1,1.5,2,3,4],
      prediction='argmax(logits - [0,0,0,penalty]); common penalty across all five seeds; no ensemble',
      selection='Maximize five-seed mean validation accuracy among eligible candidates; ties prefer macro-F1 then smaller penalty.',
      gates=dict(mean_macro_f1_max_drop=0.005, per_seed_macro_f1_max_drop=0.02,
                 mean_f_recall_max_drop=0.10, mean_f_f1_max_drop=0.0,
                 mean_s_recall_max_drop=0.0, mean_v_recall_max_drop=0.0),
      val_sha256={str(seed):sha(predpath(seed,'val')) for seed in range(1,6)},
      limitations=['Historical test results and confusion matrices have been inspected; this is exploratory follow-up.',
        'Calibration uses the same validation set as checkpoint selection; S support 73 and F support 14 make it unstable.',
        'Original checkpoints and baseline results remain unchanged; tuning budgets differ.',
        'No automatic further search after this single frozen candidate is evaluated on test.']))
    print('PROTOCOL_SAVED', OUT/'protocol.json')

def select():
    protocol=read(OUT/'protocol.json')
    assert protocol['script_sha256']==sha(Path(__file__))
    assert not (OUT/'frozen_selection.json').exists()
    for seed in range(1,6):
        assert sha(predpath(seed,'val')) == protocol['val_sha256'][str(seed)]
    arrays=load_split('val'); baseline=scored(arrays,0); b=aggregate(baseline); g=protocol['gates']
    candidates=[]
    for penalty in protocol['candidate_f_logit_penalties']:
        rows=scored(arrays,penalty); a=aggregate(rows)
        checks={
          'mean_macro_f1':a['macro_f1']['mean'] >= b['macro_f1']['mean']-g['mean_macro_f1_max_drop']-1e-12,
          'per_seed_macro_f1':all(r['macro_f1']>=base['macro_f1']-g['per_seed_macro_f1_max_drop']-1e-12 for r,base in zip(rows,baseline)),
          'f_recall':a['f_sensitivity']['mean']>=b['f_sensitivity']['mean']-g['mean_f_recall_max_drop']-1e-12,
          'f_f1':a['f_f1']['mean']>=b['f_f1']['mean']-g['mean_f_f1_max_drop']-1e-12,
          's_recall':a['s_sensitivity']['mean']>=b['s_sensitivity']['mean']-g['mean_s_recall_max_drop']-1e-12,
          'v_recall':a['v_sensitivity']['mean']>=b['v_sensitivity']['mean']-g['mean_v_recall_max_drop']-1e-12}
        candidates.append(dict(penalty=penalty,eligible=all(checks.values()),checks=checks,summary=a,per_seed=rows))
    chosen=max((c for c in candidates if c['eligible']),key=lambda c:(c['summary']['accuracy']['mean'],c['summary']['macro_f1']['mean'],-c['penalty']))
    save_new(OUT/'validation_search.json',dict(baseline=b,candidates=candidates))
    save_new(OUT/'frozen_selection.json',dict(frozen_utc=stamp(),protocol_sha256=sha(OUT/'protocol.json'),
       validation_search_sha256=sha(OUT/'validation_search.json'),selected_penalty=chosen['penalty'],
       changed=chosen['penalty']>0,validation=chosen['summary'],selected_using='validation only'))
    print(json.dumps(dict(selected_penalty=chosen['penalty'],candidates=[dict(penalty=c['penalty'],eligible=c['eligible'],
       accuracy=c['summary']['accuracy']['mean'],macro_f1=c['summary']['macro_f1']['mean'],
       f_recall=c['summary']['f_sensitivity']['mean'],failed=[k for k,v in c['checks'].items() if not v]) for c in candidates]),indent=2))

def evaluate():
    protocol=read(OUT/'protocol.json'); selection=read(OUT/'frozen_selection.json')
    assert not (OUT/'test_evaluation.json').exists()
    assert protocol['script_sha256']==sha(Path(__file__))
    assert selection['protocol_sha256']==sha(OUT/'protocol.json')
    assert selection['validation_search_sha256']==sha(OUT/'validation_search.json')
    arrays=load_split('test'); penalty=selection['selected_penalty']
    baseline=scored(arrays,0); adjusted=scored(arrays,penalty)
    for seed,(before,after,z) in enumerate(zip(baseline,adjusted,arrays),1):
        original=read(predpath(seed,'test').parent/'test_metrics.json')['test']
        for k in KEYS[:6]:
            assert abs(before[k]-original[k])<1e-12
        assert np.array_equal(np.sum(after['confusion_matrix'],axis=1),[44165,1834,3216,388])
        logits=z['logits'].astype(np.float64).copy();logits[:,3]-=penalty
        np.savez_compressed(OUT/f'calibrated_test_predictions_s{seed}.npz',predictions=logits.argmax(1),
                            labels=z['labels'],ids=z['ids'],groups=z['groups'])
    b=aggregate(baseline);a=aggregate(adjusted)
    result=dict(evaluated_utc=stamp(),frozen_selection_sha256=sha(OUT/'frozen_selection.json'),
      selected_penalty=penalty,training_reused=True,ensemble=False,all_five_seeds_retained=True,
      source_test_sha256={str(s):sha(predpath(s,'test')) for s in range(1,6)},
      baseline=b,calibrated=a,changes={k:a[k]['mean']-b[k]['mean'] for k in KEYS},
      per_seed=[dict(seed=s,before=before,after=after) for s,(before,after) in enumerate(zip(baseline,adjusted),1)],
      limitations=protocol['limitations'])
    save_new(OUT/'test_evaluation.json',result)
    print(json.dumps({k:result[k] for k in ['selected_penalty','calibrated','changes']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['init','select','evaluate'])
    globals()[parser.parse_args().phase]()
