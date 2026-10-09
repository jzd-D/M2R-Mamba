"""PTB-XL baselines adapted from Strodthoff et al. / Wang TSC architectures.

Same I/O as M2R-Mamba: input (B, C, T), output logits (B, n_classes).
Uses concat-pooling (GAP || GMP) + MLP head, matching the benchmark description.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1d(in_ch, out_ch, k, stride=1, groups=1):
    pad = k // 2
    return nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, groups=groups, bias=False)


class ConcatPool1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) -> (B, 2D)
        return torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, drop1: float = 0.25, drop2: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop1),
            nn.Linear(hidden, n_classes),
            nn.Dropout(drop2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCNWang(nn.Module):
    """Fully Convolutional Network (Wang et al. style)."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5, base: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            _conv1d(n_leads, base, 8),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            _conv1d(base, base * 2, 5),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            _conv1d(base * 2, base, 3),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )
        self.pool = ConcatPool1d()
        self.head = MLPHead(base * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 5):
        super().__init__()
        self.conv1 = _conv1d(in_ch, out_ch, k, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = _conv1d(out_ch, out_ch, k)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(_conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
        else:
            self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.down(x))


class ResNet1dWang(nn.Module):
    """1D ResNet (Wang / Strodthoff resnet1d_wang style, kernel=5)."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            _conv1d(n_leads, base, 7, stride=1),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(_ResBlock1d(base, base), _ResBlock1d(base, base))
        self.layer2 = nn.Sequential(_ResBlock1d(base, base * 2, stride=2), _ResBlock1d(base * 2, base * 2))
        self.layer3 = nn.Sequential(_ResBlock1d(base * 2, base * 4, stride=2), _ResBlock1d(base * 4, base * 4))
        self.pool = ConcatPool1d()
        self.head = MLPHead(base * 8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(self.pool(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _InceptionBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bottleneck: int = 32):
        super().__init__()
        self.use_bottleneck = in_ch > bottleneck
        bot = bottleneck if self.use_bottleneck else in_ch
        self.bottleneck = nn.Conv1d(in_ch, bot, 1, bias=False) if self.use_bottleneck else nn.Identity()
        self.b1 = _conv1d(bot, out_ch, 9)
        self.b2 = _conv1d(bot, out_ch, 19)
        self.b3 = _conv1d(bot, out_ch, 39)
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        self.residual = (
            nn.Sequential(_conv1d(in_ch, out_ch * 4, 1), nn.BatchNorm1d(out_ch * 4))
            if in_ch != out_ch * 4
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        y = torch.cat([self.b1(z), self.b2(z), self.b3(z), self.maxpool(x)], dim=1)
        return self.act(self.bn(y) + self.residual(x))


class Inception1d(nn.Module):
    """InceptionTime-style 1D network (Strodthoff inception1d)."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5, nf: int = 32, n_blocks: int = 6):
        super().__init__()
        blocks = []
        in_ch = n_leads
        for _ in range(n_blocks):
            blocks.append(_InceptionBlock1d(in_ch, nf))
            in_ch = nf * 4
        self.features = nn.Sequential(*blocks)
        self.pool = ConcatPool1d()
        self.head = MLPHead(in_ch * 2, n_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTM1d(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        hidden: int = 256,
        n_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_leads,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden * (2 if bidirectional else 1)
        self.pool = ConcatPool1d()
        # LSTM output (B,T,D) -> transpose to (B,D,T) for concat pool
        self.head = MLPHead(d * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C)
        h, _ = self.lstm(x.transpose(1, 2))
        h = h.transpose(1, 2)  # (B, D, T)
        return self.head(self.pool(h))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _XResBlock1d(nn.Module):
    """Simplified xResNet residual block with optional expansion."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, k: int = 5):
        super().__init__()
        self.conv1 = _conv1d(in_ch, out_ch, k, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = _conv1d(out_ch, out_ch, k)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = (
            nn.Sequential(_conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + self.down(x))


def _make_xlayer(in_ch, out_ch, n_blocks, stride):
    layers = [_XResBlock1d(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(_XResBlock1d(out_ch, out_ch))
    return nn.Sequential(*layers)


class XResNet1d101(nn.Module):
    """xResNet1d-101 style depth (approximate block counts 3-4-23-3)."""

    def __init__(self, n_leads: int = 12, n_classes: int = 5, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            _conv1d(n_leads, base, 5, stride=1),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            _conv1d(base, base, 5),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            _conv1d(base, base * 2, 5),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
        )
        # ResNet-101-ish: 3, 4, 23, 3
        self.layer1 = _make_xlayer(base * 2, base * 2, 3, 1)
        self.layer2 = _make_xlayer(base * 2, base * 4, 4, 2)
        self.layer3 = _make_xlayer(base * 4, base * 8, 23, 2)
        self.layer4 = _make_xlayer(base * 8, base * 16, 3, 2)
        self.pool = ConcatPool1d()
        self.head = MLPHead(base * 32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline(name: str, n_leads: int = 12, n_classes: int = 5, **kwargs) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in ("fcn", "fcn_wang"):
        return FCNWang(n_leads, n_classes)
    if key in ("resnet1d", "resnet1d_wang", "resnet"):
        return ResNet1dWang(n_leads, n_classes)
    if key in ("inception1d", "inception", "inceptiontime"):
        return Inception1d(n_leads, n_classes)
    if key in ("lstm",):
        return LSTM1d(n_leads, n_classes, bidirectional=False)
    if key in ("bilstm", "lstm_bidir", "bi_lstm"):
        return LSTM1d(n_leads, n_classes, bidirectional=True)
    if key in ("xresnet1d101", "xresnet", "xresnet1d"):
        return XResNet1d101(n_leads, n_classes)
    raise ValueError(f"Unknown baseline: {name}")
