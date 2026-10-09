"""Sequential reproduction of the frozen study; --dry-run requires no ML packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('standard', 'ptb_ablation', 'mit_final')
BASELINES = ('fcn_wang', 'resnet1d_wang', 'inception1d', 'lstm',
             'lstm_bidir', 'xresnet1d101', 'ecg_mamba')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(stage, output):
    manifest = json.loads((ROOT / 'experiments' / stage / 'manifest.json').read_text())
    hashes = {}
    for cfg in manifest['jobs']:
        run = output / stage / 'runs' / cfg['id']
        metrics = json.loads((run / 'metrics.json').read_text())
        if metrics['status'] != 'ok' or metrics['id'] != cfg['id']:
            raise RuntimeError(f'Incomplete training: {cfg["id"]}')
        if json.loads((run / 'config.json').read_text()) != cfg:
            raise RuntimeError(f'Configuration changed: {cfg["id"]}')
        hashes[cfg['id']] = sha(run / 'best.pt')
    target = output / stage / 'checkpoints_frozen.json'
    if target.exists():
        if json.loads(target.read_text()) != hashes:
            raise RuntimeError('Frozen checkpoints changed; use a new output directory.')
    else:
        with target.open('x', encoding='utf-8') as f:
            json.dump(hashes, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', choices=(*STAGES, 'ptb_main', 'mit_baselines', 'all'))
    p.add_argument('--phase', choices=('train', 'evaluate'), default='train')
    p.add_argument('--job', help='One exact job ID from a frozen manifest')
    p.add_argument('--seeds', type=int, nargs='+', choices=range(1, 6))
    p.add_argument('--output', type=Path, default=ROOT / 'reproduced')
    p.add_argument('--data', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Two training batches and one validation batch')
    a = p.parse_args()
    output, data = a.output.resolve(), a.data.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT / x) for x in
                            ['src', 'experiments', 'scripts', 'thesis', 'reference_results']):
        p.error('Output must not overwrite repository sources or frozen references.')
    if a.phase == 'evaluate' and (a.job or a.seeds or a.smoke):
        p.error('Evaluation freezes the entire stage; filters and smoke are training-only.')
    stages = list(STAGES) if a.suite == 'all' else [
        'standard' if a.suite in ('ptb_main', 'mit_baselines') else a.suite]
    env = dict(os.environ, M2R_OUTPUT_ROOT=str(output), M2R_DATA_ROOT=str(data),
               PYTHONDONTWRITEBYTECODE='1')
    selected = 0
    for stage in stages:
        folder = ROOT / 'experiments' / stage
        jobs = json.loads((folder / 'manifest.json').read_text())['jobs']
        if a.phase == 'evaluate':
            if stage == 'standard':
                print('standard: each trainer already evaluates its validation-selected checkpoint.')
                continue
            command = [sys.executable, str(folder / 'evaluate_all.py')]
            print(stage, 'freeze all checkpoints, then evaluate', flush=True)
            if not a.dry_run:
                freeze(stage, output)
                subprocess.run(command, env=env, cwd=ROOT, check=True)
            selected += 1
            continue
        for cfg in jobs:
            if a.suite == 'ptb_main' and not (
                cfg['dataset'] == 'ptbxl' and cfg['method'] in (*BASELINES, 'm2r_full')):
                continue
            if a.suite == 'mit_baselines' and cfg['dataset'] != 'mitbih':
                continue
            if a.job and cfg['id'] != a.job:
                continue
            if a.seeds and cfg['seed'] not in a.seeds:
                continue
            selected += 1
            print(stage, cfg['id'], 'smoke' if a.smoke else 'train', flush=True)
            if not a.dry_run:
                # Refuse both completed and interrupted runs. Never resume by silently overwriting.
                run = output / stage / ('smoke' if a.smoke else 'runs') / cfg['id']
                if run.exists():
                    raise FileExistsError(f'Run already exists; choose a fresh --output: {run}')
                command = [sys.executable, str(folder / 'train.py'), '--job', cfg['id']]
                if a.smoke:
                    command.append('--smoke')
                subprocess.run(command, env=env, cwd=ROOT, check=True)
    if not selected and a.phase == 'train':
        p.error('No matching job was selected.')
    print(f'Selected {selected}; dry_run={a.dry_run}; output={output}')


if __name__ == '__main__':
    main()
