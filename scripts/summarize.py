"""Recompute paper comparison means and sample SDs (all five seeds), without ML dependencies."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PTB_METRICS = ('macro_auroc', 'macro_auprc', 'macro_f1', 'macro_f1_val_threshold')
MIT_METRICS = ('accuracy', 'macro_f1', 'n_sensitivity', 's_sensitivity',
               'v_sensitivity', 'f_sensitivity')


def collect(root):
    runs = []
    for stage in ('standard', 'ptb_ablation', 'mit_final'):
        jobs = json.loads((ROOT/'experiments'/stage/'manifest.json').read_text())['jobs']
        for cfg in jobs:
            folder = root/stage/'runs'/cfg['id']
            row = json.loads((folder/'metrics.json').read_text())
            if stage != 'standard':
                row['test'] = json.loads((folder/'test_metrics.json').read_text())['test']
            runs.append(row)
    calibration = json.loads((root/'mit_calibration/test_evaluation.json').read_text())
    return runs, calibration


def summarize(runs, calibration):
    grouped = defaultdict(list)
    for r in runs:
        grouped[r['dataset'], r['method']].append(r)
    result = []
    for (dataset, method), records in sorted(grouped.items()):
        records.sort(key=lambda r:r['seed'])
        if [r['seed'] for r in records] != [1, 2, 3, 4, 5]:
            raise ValueError(f'Missing or duplicate seeds: {dataset}/{method}')
        for metric in PTB_METRICS if dataset == 'ptbxl' else MIT_METRICS:
            values = [r['test'][metric] for r in records]
            result.append(dict(dataset=dataset, method=method, metric=metric, n=5,
                               mean=statistics.mean(values), sample_sd=statistics.stdev(values)))
    for metric in MIT_METRICS:
        rows = sorted(calibration['per_seed'], key=lambda r:r['seed'])
        assert [r['seed'] for r in rows] == [1,2,3,4,5]
        values = [r['after'][metric] for r in rows]
        result.append(dict(dataset='mitbih',method='m2r_calibrated',metric=metric,n=5,
                           mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-runs', type=Path, help='New output root after all training/evaluation/calibration phases')
    p.add_argument('--output', type=Path, default=ROOT/'reproduced/summary')
    a = p.parse_args()
    if a.from_runs:
        runs, calibration = collect(a.from_runs)
    else:
        runs = json.loads((ROOT/'reference_results/per_seed.json').read_text())
        calibration = json.loads((ROOT/'reference_results/mit_calibration.json').read_text())
    result = summarize(runs, calibration)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (a.output/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print(f'{len(runs)} training runs; {len(result)} metric summaries; mean and sample SD.')


if __name__ == '__main__':
    main()
