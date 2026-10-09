"""M2R-Mamba: Multi-scale CNN + Lead-aware residual + Bidirectional Mamba.

v3 design notes (from fair + tune rounds):
- Concat pool (GAP||GMP) beats attention-only for AUPRC.
- temporal_stride=2 + wider capacity helps F1; wider MS kernels help AUPRC/Fmax.
- Multi-resolution fusion keeps fine morphology (local pool) while Mamba sees
  downsampled rhythm — expected to lift AUROC/AUPRC together.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover
    raise ImportError("mamba_ssm is required inside the experiment container") from exc


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.net(x).unsqueeze(-1)
        return x * w


class MultiScaleCNN(nn.Module):
    def __init__(self, d_model: int, kernels=(7, 15, 31)) -> None:
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model * len(kernels), d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.fuse(torch.cat(feats, dim=1))


class SingleScaleCNN(nn.Module):
    def __init__(self, d_model: int, kernel: int = 15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMSBlock(nn.Module):
    def __init__(self, d_model: int, kernels=(7, 15, 31)) -> None:
        super().__init__()
        self.block = MultiScaleCNN(d_model, kernels=kernels)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(x + self.block(x)))


class ChannelStem(nn.Module):
    """Cross-lead morphology stem with optional residual multi-scale depth."""

    def __init__(
        self,
        n_leads: int,
        d_model: int,
        use_multiscale: bool = True,
        ms_kernels=(7, 15, 31),
        n_stem_blocks: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(n_leads, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        n_stem_blocks = max(int(n_stem_blocks), 1)
        if use_multiscale:
            blocks = [ResidualMSBlock(d_model, kernels=ms_kernels) for _ in range(n_stem_blocks)]
            self.morph = nn.Sequential(*blocks)
        else:
            self.morph = SingleScaleCNN(d_model)
        self.out = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.se = SEBlock1d(d_model) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.out(self.morph(self.proj(x))))


class LeadResidualBranch(nn.Module):
    """Per-lead morphology + attention, added as residual onto cross-lead stem."""

    def __init__(self, n_leads: int, d_model: int) -> None:
        super().__init__()
        self.n_leads = n_leads
        self.d_model = d_model
        self.lead_proj = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h = x.reshape(b * c, 1, t)
        h = self.lead_proj(h).reshape(b, c, self.d_model, t)
        scores = self.attn(h.mean(dim=-1)).squeeze(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        fused = (h * w).sum(dim=1)
        return self.gate * fused


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.bwd = None
            self.out_proj = nn.Identity()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y_f = self.fwd(h)
        if self.bidirectional:
            y_b = self.bwd(h.flip(1)).flip(1)
            y = self.out_proj(torch.cat([y_f, y_b], dim=-1))
        else:
            y = y_f
        return x + self.drop(y)


class TemporalAttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


class ConcatPool(nn.Module):
    """GAP || GMP over time: (B, T, D) -> (B, 2D)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)


class AttnConcatPool(nn.Module):
    """Attention pool || GAP || GMP -> (B, 3D)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = TemporalAttentionPool(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.attn(x), x.mean(dim=1), x.amax(dim=1)], dim=-1)


def _make_pool(pool_mode: str, d_model: int) -> tuple[nn.Module, int]:
    mode = str(pool_mode).lower()
    if mode == "concat":
        return ConcatPool(), d_model * 2
    if mode in ("attn_concat", "attn+concat"):
        return AttnConcatPool(d_model), d_model * 3
    return TemporalAttentionPool(d_model), d_model


def center_beat_features(h):
    """Five concatenated beats; read the central fifth after context encoding."""
    if h.shape[1] % 5:
        raise ValueError('Center readout requires five equal-length beat segments')
    width=h.shape[1]//5
    return h[:,2*width:3*width]


class RRFusion(nn.Module):
    """Project constant RR channels into d_model and add to morphology features."""

    def __init__(self, rr_channels: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(rr_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        # rr: (B, R, T) — values are nearly constant over T
        return self.gate * self.net(rr)


class M2RMamba(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        d_model: int = 128,
        n_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        use_multiscale: bool = True,
        use_lead_aware: bool = True,
        use_mamba: bool = True,
        bidirectional: bool = True,
        temporal_stride: int = 4,
        pool_mode: str = "attn",
        ms_kernels=(7, 15, 31),
        use_multires: bool = False,
        down_mode: str = "avg",
        n_stem_blocks: int = 1,
        use_se: bool = False,
        morph_channels: int = 0,
        use_cascade: bool = False,
        center_readout: bool = False,
        rr_fusion_mode: str = "early",
    ) -> None:
        super().__init__()
        self.use_lead_aware = use_lead_aware
        self.center_readout=bool(center_readout)
        self.rr_fusion_mode=str(rr_fusion_mode)
        if self.rr_fusion_mode not in ("early","late"): raise ValueError("Invalid RR fusion mode")
        self.use_mamba = use_mamba
        self.use_multires = bool(use_multires)
        self.temporal_stride = max(int(temporal_stride), 1)
        self.pool_mode = str(pool_mode).lower()
        self.down_mode = str(down_mode).lower()
        # morph_channels>0: first M channels → stem; rest → RR fusion (MIT dual-path)
        self.morph_channels = int(morph_channels) if morph_channels else 0
        stem_in = self.morph_channels if self.morph_channels > 0 else n_leads
        rr_ch = max(n_leads - stem_in, 0) if self.morph_channels > 0 else 0

        self.stem = ChannelStem(
            stem_in,
            d_model,
            use_multiscale=use_multiscale,
            ms_kernels=ms_kernels,
            n_stem_blocks=n_stem_blocks,
            use_se=use_se,
        )
        lead_in = stem_in if self.morph_channels > 0 else n_leads
        self.lead_branch = LeadResidualBranch(lead_in, d_model) if use_lead_aware else None
        self.rr_fusion = RRFusion(rr_ch, d_model) if rr_ch > 0 and self.rr_fusion_mode == 'early' else None
        self.rr_late = None
        if rr_ch > 0 and self.rr_fusion_mode == 'late':
            self.rr_late = nn.Sequential(nn.Linear(rr_ch,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(dropout))

        if self.temporal_stride > 1:
            if self.down_mode == "conv":
                self.down = nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=self.temporal_stride * 2 - 1,
                        stride=self.temporal_stride,
                        padding=self.temporal_stride - 1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(d_model),
                    nn.GELU(),
                )
            else:
                self.down = nn.AvgPool1d(
                    kernel_size=self.temporal_stride, stride=self.temporal_stride
                )
        else:
            self.down = nn.Identity()

        if use_mamba:
            self.temporal = nn.ModuleList(
                [
                    BiMambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        bidirectional=bidirectional,
                        drop=dropout,
                    )
                    for _ in range(n_mamba_layers)
                ]
            )
        else:
            self.temporal = nn.ModuleList()

        self.pool, pool_dim = _make_pool(self.pool_mode, d_model)
        if self.use_multires:
            self.local_pool, local_dim = _make_pool("concat", d_model)
            pool_dim = pool_dim + local_dim

        if self.rr_late is not None: pool_dim += 64

        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        # Optional cascade: binary N-vs-abnormal + 3-class S/V/F, fused to 4 logits
        self.use_cascade = bool(use_cascade)
        if self.use_cascade:
            self.head_bin = nn.Sequential(
                nn.LayerNorm(pool_dim),
                nn.Dropout(dropout),
                nn.Linear(pool_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            self.head_svf = nn.Sequential(
                nn.LayerNorm(pool_dim),
                nn.Dropout(dropout),
                nn.Linear(pool_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, max(n_classes - 1, 1)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.morph_channels > 0:
            morph = x[:, : self.morph_channels]
            rr = x[:, self.morph_channels :]
            feat = self.stem(morph)
            if self.lead_branch is not None:
                feat = feat + self.lead_branch(morph)
            if self.rr_fusion is not None and rr.shape[1] > 0:
                feat = feat + self.rr_fusion(rr)
        else:
            feat = self.stem(x)
            if self.lead_branch is not None:
                feat = feat + self.lead_branch(x)

        local_vec = None
        if self.use_multires:
            local_vec = self.local_pool(feat.transpose(1, 2))

        feat = self.down(feat)
        h = feat.transpose(1, 2)
        for block in self.temporal:
            h = block(h)
        global_vec = self.pool(center_beat_features(h) if self.center_readout else h)
        if local_vec is not None:
            global_vec = torch.cat([local_vec, global_vec], dim=-1)

        if self.rr_late is not None:
            global_vec = torch.cat([global_vec,self.rr_late(rr.mean(dim=-1))],dim=-1)

        if not self.use_cascade:
            return self.head(global_vec)

        # Cascade fuse → 4-class logits (N,S,V,F)
        # P(N)=σ(-z), P(abn)=σ(z); P(S/V/F)=P(abn)*softmax(svf)
        z = self.head_bin(global_vec).squeeze(-1)  # (B,)
        svf = self.head_svf(global_vec)  # (B,3)
        log_p_n = F.logsigmoid(-z)
        log_p_abn = F.logsigmoid(z)
        log_svf = F.log_softmax(svf, dim=-1)
        log_p_svf = log_p_abn.unsqueeze(-1) + log_svf
        # convert to logits (up to additive const): stack log-probs
        return torch.cat([log_p_n.unsqueeze(-1), log_p_svf], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> M2RMamba:
    m = cfg.get("model", cfg)
    ms_kernels = m.get("ms_kernels", (7, 15, 31))
    if isinstance(ms_kernels, (list, tuple)):
        ms_kernels = tuple(int(k) for k in ms_kernels)
    else:
        ms_kernels = (7, 15, 31)
    return M2RMamba(
        n_leads=int(m.get("n_leads", 12)),
        n_classes=int(m.get("n_classes", 5)),
        d_model=int(m.get("d_model", 128)),
        n_mamba_layers=int(m.get("n_mamba_layers", 2)),
        d_state=int(m.get("d_state", 16)),
        d_conv=int(m.get("d_conv", 4)),
        expand=int(m.get("expand", 2)),
        dropout=float(m.get("dropout", 0.2)),
        use_multiscale=bool(m.get("use_multiscale", True)),
        use_lead_aware=bool(m.get("use_lead_aware", True)),
        use_mamba=bool(m.get("use_mamba", True)),
        bidirectional=bool(m.get("bidirectional", True)),
        temporal_stride=int(m.get("temporal_stride", 4)),
        pool_mode=str(m.get("pool_mode", "attn")),
        ms_kernels=ms_kernels,
        use_multires=bool(m.get("use_multires", False)),
        down_mode=str(m.get("down_mode", "avg")),
        n_stem_blocks=int(m.get("n_stem_blocks", 1)),
        use_se=bool(m.get("use_se", False)),
        morph_channels=int(m.get("morph_channels", 0)),
        use_cascade=bool(m.get("use_cascade", False)),
        center_readout=bool(m.get("center_readout", False)),
        rr_fusion_mode=str(m.get("rr_fusion_mode", "early")),
    )
