"""Model factory: M2R-Mamba, ECG-Mamba-style, or Strodthoff-style baselines."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from m2r_mamba.models.baselines import build_baseline
from m2r_mamba.models.ecg_mamba import build_ecg_mamba
from m2r_mamba.models.m2r_mamba import build_model as build_m2r


def build_any_model(cfg: dict) -> nn.Module:
    m: dict[str, Any] = cfg.get("model", cfg)
    arch = str(m.get("arch", "m2r_mamba")).lower()
    if arch in ("m2r", "m2r_mamba", "proposed"):
        return build_m2r(cfg)
    if arch in ("ecg_mamba", "ecgmamba", "ecg-mamba"):
        return build_ecg_mamba(cfg)
    return build_baseline(
        arch,
        n_leads=int(m.get("n_leads", 12)),
        n_classes=int(m.get("n_classes", 5)),
    )
