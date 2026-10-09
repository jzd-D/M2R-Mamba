"""Multilabel metrics for PTB-XL (fixed-threshold + Fmax)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from m2r_mamba.datasets.ptbxl import SUPERCLASSES


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["label"].numpy()
        logits = model(x).detach().cpu().numpy()
        logits_all.append(logits)
        labels_all.append(y)
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def _macro_f1_at_thr(probs: np.ndarray, labels: np.ndarray, thr: float) -> float:
    preds = (probs >= thr).astype(np.float32)
    f1s = []
    for i in range(labels.shape[1]):
        f1s.append(f1_score(labels[:, i], preds[:, i], zero_division=0))
    return float(np.mean(f1s))


def compute_fmax(probs: np.ndarray, labels: np.ndarray, n_steps: int = 100) -> Dict:
    """Threshold-swept macro-F (Strodthoff-style Fmax)."""
    best_f, best_t = -1.0, 0.5
    for thr in np.linspace(0.0, 1.0, n_steps + 1):
        f = _macro_f1_at_thr(probs, labels, float(thr))
        if f > best_f:
            best_f, best_t = f, float(thr)
    return {"fmax": best_f, "fmax_thr": best_t}


def multilabel_metrics(logits: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> Dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= thr).astype(np.float32)

    per_class = {}
    aurocs, auprcs, f1s = [], [], []
    for i, name in enumerate(SUPERCLASSES):
        y = labels[:, i]
        p = probs[:, i]
        pr = preds[:, i]
        if y.min() == y.max():
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(y, p))
            auprc = float(average_precision_score(y, p))
        f1 = float(f1_score(y, pr, zero_division=0))
        per_class[name] = {"auroc": auroc, "auprc": auprc, "f1": f1}
        aurocs.append(auroc)
        auprcs.append(auprc)
        f1s.append(f1)

    def _nanmean(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return float(np.nanmean(arr))

    fmax_info = compute_fmax(probs, labels)
    return {
        "macro_auroc": _nanmean(aurocs),
        "macro_auprc": _nanmean(auprcs),
        "macro_f1": _nanmean(f1s),
        "fmax": fmax_info["fmax"],
        "fmax_thr": fmax_info["fmax_thr"],
        "per_class": per_class,
    }
