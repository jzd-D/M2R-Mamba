"""Regenerate frozen train/validation/test caches from public WFDB records."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2),encoding="utf-8")

def mit_ids(root, records):
    import wfdb
    from m2r_mamba.datasets.mitbih import AAMI_MAP
    ids, groups = [], []
    for rid in records:
        ann = wfdb.rdann(str(root / rid), 'atr')
        length = wfdb.rdheader(str(root / rid)).sig_len
        peaks = [int(p) for p,s in zip(ann.sample,ann.symbol)
                 if AAMI_MAP.get(s) in ('N','S','V','F') and p >= 90 and p + 90 <= length]
        for peak in peaks[2:-2]:
            ids.append(f'{rid}:{peak}'); groups.append(rid)
    return np.asarray(ids), np.asarray(groups)

def prepare_dataset(name):
    dest = PROCESSED / name
    if dest.exists():
        raise FileExistsError(f'Use a new processed-data directory: {dest}')
    if name == 'ptbxl':
        from m2r_mamba.datasets.ptbxl import build_ptbxl_dataloaders, SUPERCLASSES
        tr,va,te,meta=build_ptbxl_dataloaders(str(RAW_ROOT),batch_size=64,num_workers=0,
            stats_cache=str(dest/'stats.npz'))
        classes=list(SUPERCLASSES)
        memberships=[set(x.dataset.db.patient_id.astype(str)) for x in (tr,va,te)]
        assert all(not memberships[i] & memberships[j] for i,j in ((0,1),(0,2),(1,2)))
        audit={'patient_disjoint':True, 'patients_per_split':[len(x) for x in memberships]}
    else:
        import m2r_mamba.datasets.mitbih as mit
        mit.DS1=tuple(x for x in mit.DS1 if x != '201')
        tr,va,te,meta=mit.build_mitbih_dataloaders(str(RAW_ROOT),batch_size=64,num_workers=0,
            cache_path=str(dest/'beats.npz'),sampler_mode='none',weight_cap=8.0,
            f_weight_cap=8.0,s_weight_mult=1.0)
        classes=list(meta['classes'])
        records=[meta[k] for k in ('train_recs','val_recs','test_recs')]
        patient=lambda x:'201_202' if x in ('201','202') else x
        memberships=[{patient(x) for x in r} for r in records]
        assert all(not memberships[i] & memberships[j] for i,j in ((0,1),(0,2),(1,2)))
        audit={'patient_disjoint':True,'train_records':records[0], 'val_records':records[1],
               'test_records':records[2],'excluded_training_record':'201'}
    dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'normalization.npz',mean=meta['mean'],std=meta['std'])
    for split,loader in zip(('train','val','test'),(tr,va,te)):
        ds=loader.dataset
        np.save(dest/f'{split}_signals.npy',ds.signals)
        np.save(dest/f'{split}_labels.npy',ds.labels)
        if name=='ptbxl':
            ids=ds.db.ecg_id.astype(str).to_numpy(dtype=str)
            groups=ds.db.patient_id.astype(str).to_numpy(dtype=str)
        else:
            records=meta[{'train':'train_recs','val':'val_recs','test':'test_recs'}[split]]
            ids,groups=mit_ids(Path(meta['root']),records)
        assert len(ids)==len(ds)==len(groups)
        np.save(dest/f'{split}_ids.npy',ids); np.save(dest/f'{split}_groups.npy',groups)
        labels=np.asarray(ds.labels)
        counts=labels.sum(0).tolist() if labels.ndim==2 else np.bincount(labels,minlength=len(classes)).tolist()
        audit[split]={'n':len(ds),'class_counts':counts,'ids_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
    audit['classes']=classes
    audit['normalization_source']='training samples only'
    audit['signal_shape']=list(tr.dataset.signals.shape[1:])
    write_json(dest/'ready.json',audit)
    print(name,audit,flush=True)


def main():
    global RAW_ROOT,PROCESSED
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['ptbxl','mitbih'])
    p.add_argument('--raw',required=True,type=Path)
    p.add_argument('--processed',type=Path,default=ROOT/'data/processed')
    a=p.parse_args();RAW_ROOT=a.raw.resolve();PROCESSED=a.processed.resolve()
    if not RAW_ROOT.is_dir():raise FileNotFoundError(RAW_ROOT)
    if PROCESSED==RAW_ROOT or PROCESSED.is_relative_to(RAW_ROOT):raise ValueError("Processed output must be outside raw data")
    prepare_dataset(a.dataset)
    actual=json.loads((PROCESSED/a.dataset/'ready.json').read_text())
    expected=json.loads((ROOT/'reference_results/data_audits.json').read_text())[a.dataset]
    if actual!=expected:raise RuntimeError("Dataset audit differs from frozen study; do not mix these results with the manuscript")
    print("Dataset split, class counts and ID hashes match the frozen study.")
if __name__=="__main__":main()
