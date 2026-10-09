"""Training-only class and record balancing; no model selection here."""
import numpy as np

def loss_weights(tc):
    weights=np.asarray(tc['class_weights'],dtype=np.float64)
    assert weights.shape==(4,) and np.isfinite(weights).all() and (weights>0).all()
    return weights

def sample_weights(y,groups,clip=20.):
    y=np.asarray(y);groups=np.asarray(groups);assert len(y)==len(groups) and len(y)>0
    result=np.zeros(len(y),dtype=np.float64)
    for label in range(4):
        indices=np.flatnonzero(y==label);assert len(indices)>0
        _,inverse,counts=np.unique(groups[indices],return_inverse=True,return_counts=True)
        within=1/np.sqrt(counts[inverse]);within/=within.mean()
        result[indices]=np.sqrt(len(y)/len(indices))*within
    result/=result.mean();result=np.minimum(result,float(clip))
    assert np.isfinite(result).all() and (result>0).all() and result.max()<=clip
    return result

def sampler_audit(y,groups,w):
    y=np.asarray(y);groups=np.asarray(groups);p=w/w.sum();records={}
    for label in range(4):
        mask=y==label;mass=p[mask].sum()
        records[str(label)]={str(g):float(p[mask&(groups==g)].sum()/mass) for g in sorted(set(groups[mask]))}
    return dict(class_counts=np.bincount(y,minlength=4).tolist(),
      expected_class_proportions=[float(p[y==i].sum()) for i in range(4)],
      record_proportions_within_class=records,min_weight=float(w.min()),max_weight=float(w.max()),
      n_draws_per_epoch=len(y),replacement=True,source='Current training fold only')
