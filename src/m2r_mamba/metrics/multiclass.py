"""Multi-class metrics for MIT-BIH AAMI beat classification."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@torch.no_grad()
def collect_logits_labels(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        x = batch["signal"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(y.cpu().numpy())
    return np.concatenate(logits_all, 0), np.concatenate(labels_all, 0)


def multiclass_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str] = ("N", "S", "V", "F", "Q"),
) -> Dict:
    preds = logits.argmax(axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    # sensitivity = recall
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    name_to_i = {n: i for i, n in enumerate(class_names)}

    def _se(name: str) -> float:
        return float(r[name_to_i[name]]) if name in name_to_i else float("nan")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_sensitivity": _se("N"),
        "s_sensitivity": _se("S"),
        "v_sensitivity": _se("V"),
        "f_sensitivity": _se("F"),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
