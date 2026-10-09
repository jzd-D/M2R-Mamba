"""ECG-Mamba-style baseline (Jiang et al., JTEHM 2025) for fair same-pipeline repro.

Architecture (paper Fig.1 / Table 2), adapted to variable-length 1D ECG:
  Conv1d-BN-ReLU stem (2 layers) → bidirectional Vim/Mamba encoder → pool + FC.

Fair-comparison notes:
- Same train/data pipeline as other baselines (no Non-Uniform-Mix / Noam).
- Default depth = 5 Vim blocks (paper's lighter variant that still beat CinC ResNet);
  set n_mamba_layers=24 to match the full paper config.
- Stem strides/paddings follow Table 2; sequence length adapts to input T
  (paper used 8192@500Hz → 729 tokens; PTB-XL 1000@100Hz → ~130 tokens).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from m2r_mamba.models.m2r_mamba import BiMambaBlock


class ConcatPool1dBTD(nn.Module):
    """GAP || GMP over time: (B, T, D) -> (B, 2D)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)


class VimEncoderBlock(nn.Module):
    """Vim-style block: bidirectional SSM residual + MLP residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        mlp_ratio: int = 4,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.ssm = BiMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=True,
            drop=drop,
        )
        self.norm = nn.LayerNorm(d_model)
        hidden = d_model * int(mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssm(x)
        return x + self.mlp(self.norm(x))


class ECGMamba(nn.Module):
    """ECG-Mamba: CNN stem (Table 2) + bidirectional Vim encoder + FC head."""

    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        d_model: int = 384,
        n_mamba_layers: int = 5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        # Table 2 (Jiang et al. 2025); first in_ch adapted to n_leads for MIT-BIH.
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, d_model, kernel_size=15, stride=4, padding=100, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.ModuleList(
            [
                VimEncoderBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mlp_ratio=mlp_ratio,
                    drop=dropout,
                )
                for _ in range(int(n_mamba_layers))
            ]
        )
        self.pool = ConcatPool1dBTD()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        feat = self.stem(x)  # (B, D, T')
        h = feat.transpose(1, 2)  # (B, T', D)
        for block in self.encoder:
            h = block(h)
        return self.head(self.pool(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_ecg_mamba(cfg: dict) -> ECGMamba:
    m = cfg.get("model", cfg)
    return ECGMamba(
        n_leads=int(m.get("n_leads", 12)),
        n_classes=int(m.get("n_classes", 5)),
        d_model=int(m.get("d_model", 384)),
        n_mamba_layers=int(m.get("n_mamba_layers", 5)),
        d_state=int(m.get("d_state", 16)),
        d_conv=int(m.get("d_conv", 4)),
        expand=int(m.get("expand", 2)),
        dropout=float(m.get("dropout", 0.2)),
        mlp_ratio=int(m.get("mlp_ratio", 4)),
    )
