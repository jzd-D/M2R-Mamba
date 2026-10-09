"""Portable paths; all generated data/results are outside the frozen references."""
import json, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT_ROOT=Path(os.environ.get('M2R_OUTPUT_ROOT',str(ROOT/'reproduced'))).resolve()
OUT=OUTPUT_ROOT/'ptb_ablation'
DATA_ROOT=Path(os.environ.get('M2R_DATA_ROOT',str(ROOT/'data/processed'))).resolve()
RAW=DATA_ROOT/'ptbxl'

def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(p)
