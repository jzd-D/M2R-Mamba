"""Check installed training dependencies and instantiate each unique model on CPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',action='store_true',help='Also verify all unique configurations and parameter counts on CPU')
    a=p.parse_args()
    expected=json.loads((ROOT/'reference_results/runtime.json').read_text())['runtime']
    versions={}
    for name in expected['packages']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='MISSING'
    print(json.dumps({'expected_python':expected['python'],'actual_python':sys.version.split()[0],
                      'versions':versions,'matches_frozen_image':versions==expected['packages']},indent=2))
    if a.models:
        sys.path.insert(0,str(ROOT/'src'))
        import torch
        from m2r_mamba.models.factory import build_any_model
        from m2r_mamba.models.m2r_mamba_mit import build_model as build_mit
        torch.set_num_threads(2)
        references={r['id']:r for r in json.loads((ROOT/'reference_results/per_seed.json').read_text())}
        count=0
        for stage in ['standard','ptb_ablation','mit_final']:
            m=json.loads((ROOT/'experiments'/stage/'manifest.json').read_text())
            for cfg in m['jobs']:
                if cfg['seed']!=1:continue
                torch.manual_seed(1)
                model=(build_mit if stage=='mit_final' else build_any_model)(cfg)
                params=sum(p.numel() for p in model.parameters() if p.requires_grad)
                assert params==references[cfg['id']]['trainable_params'],cfg['id']
                # The plain CNN/RNN baselines also support a CPU forward.
                if cfg['model']['arch'] not in ['m2r_mamba','ecg_mamba']:
                    model.eval()
                    length=1000 if cfg['dataset']=='ptbxl' else 900
                    with torch.inference_mode():
                        logits=model(torch.zeros(2,cfg['model']['n_leads'],length))
                    assert tuple(logits.shape)==(2,cfg['model']['n_classes'])
                    assert torch.isfinite(logits).all()
                print(cfg['id'],params,'PASS',flush=True)
                del model
                count+=1
        print(f'{count} unique model configurations passed parameter-count checks; CNN/RNN CPU forwards passed. No GPU used.')


if __name__=='__main__':main()
