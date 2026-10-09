"""Offline checks: integrity, Python syntax, manifests, paper assets, all-seed numerical consistency."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(rel): return json.loads((ROOT/rel).read_text(encoding='utf-8'))


def main():
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p.relative_to(ROOT)))
    jobs=[]
    for stage,count in [('standard',95),('ptb_ablation',15),('mit_final',5)]:
        m=read(f'experiments/{stage}/manifest.json')
        assert len(m['jobs'])==count==m['n_jobs']
        jobs+=m['jobs']
    assert len({j['id'] for j in jobs})==115
    assert Counter(j['dataset'] for j in jobs)=={'ptbxl':75,'mitbih':40}
    rows=read('reference_results/per_seed.json')
    assert {r['id'] for r in rows}=={j['id'] for j in jobs}
    by_id={r['id']:r for r in rows}
    for j in jobs:
        r=by_id[j['id']]
        assert all(j[k]==r[k] for k in ['id','dataset','method','seed'])
    for ds,method in {(j['dataset'],j['method']) for j in jobs}:
        assert sorted(j['seed'] for j in jobs if (j['dataset'],j['method'])==(ds,method))==[1,2,3,4,5]
    # Cross-check independently prepared plot input against compact raw per-seed records.
    figures=read('thesis/data/analysis_figures_data.json')
    checks=0
    for method in figures['ptb']:
        for metric,summary in method['overall'].items():
            values=[by_id[f"ptbxl_{method['method']}_s{s}"]['test'][metric] for s in range(1,6)]
            assert values==summary['values']
            assert math.isclose(statistics.mean(values),summary['mean'],abs_tol=1e-12)
            assert math.isclose(statistics.stdev(values),summary['sample_sd'],abs_tol=1e-12)
            checks+=1
    for r in figures['ablation']:
        assert r['test']==by_id[r['id']]['test']
        checks+=1
    calibration=read('reference_results/mit_calibration.json')
    assert calibration['selected_penalty']==1.5
    for r,figure in zip(calibration['per_seed'],figures['mit_final']):
        assert r['seed']==figure['seed']
        assert all(figure[k]==v for k,v in r['after'].items())
        cm=r['after']['confusion_matrix']
        assert list(map(sum,cm))==[44165,1834,3216,388]
        checks+=1
    tex=(ROOT/'thesis/main.tex').read_text(encoding='utf-8')
    names=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex)
    assert len(names)==6 and all((ROOT/'thesis'/n).is_file() for n in names)
    assert len(re.findall(r'\\begin\{table\}',tex))==7
    if (ROOT/'SNAPSHOT.sha256').exists():
        for line in (ROOT/'SNAPSHOT.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(f'PASS: 115 frozen configurations, 115 seed records, {checks} numerical cross-checks, 7 tables, 6 figure PDFs; Python syntax and file hashes.')


if __name__=='__main__':main()
