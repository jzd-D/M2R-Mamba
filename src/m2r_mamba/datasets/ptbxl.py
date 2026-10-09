"""PTB-XL dataset utilities: diagnostic superclass labels, folds, loaders."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

SUPERCLASSES: Tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")


def resolve_ptbxl_root(data_root: str | Path) -> Path:
    """Accept either the versioned folder or its parent."""
    root = Path(data_root)
    if (root / "ptbxl_database.csv").is_file():
        return root
    matches = sorted(root.glob("ptb-xl-a-large-publicly-available-electrocardiography-dataset-*"))
    for cand in matches:
        if (cand / "ptbxl_database.csv").is_file():
            return cand
    raise FileNotFoundError(f"Cannot find PTB-XL database under {data_root}")


def compute_superclass_labels(db: pd.DataFrame, scp: pd.DataFrame) -> np.ndarray:
    """Return multi-hot labels (N, 5) for diagnostic superclasses."""
    diag = scp[scp["diagnostic"] == 1]
    scp_to_super = diag["diagnostic_class"].to_dict()
    labels = np.zeros((len(db), len(SUPERCLASSES)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(SUPERCLASSES)}
    for i, raw in enumerate(db["scp_codes"].tolist()):
        codes = ast.literal_eval(raw) if isinstance(raw, str) else raw
        supers = {scp_to_super[k] for k in codes if k in scp_to_super}
        for s in supers:
            if s in class_to_idx:
                labels[i, class_to_idx[s]] = 1.0
    return labels


class PTBXLDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        fold_ids: Sequence[int],
        sampling_rate: int = 100,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        signals: Optional[np.ndarray] = None,
        preload: bool = True,
    ) -> None:
        self.root = resolve_ptbxl_root(root)
        self.sampling_rate = sampling_rate
        self.mean = mean
        self.std = std

        db = pd.read_csv(self.root / "ptbxl_database.csv")
        scp = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        mask = db["strat_fold"].isin(list(fold_ids))
        self.db = db.loc[mask].reset_index(drop=True)
        all_labels = compute_superclass_labels(db, scp)
        self.labels = all_labels[mask.to_numpy()]

        fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        self.paths = [self.root / p for p in self.db[fname_col].tolist()]

        if signals is not None:
            self.signals = signals
        elif preload:
            self.signals = self._preload_all()
        else:
            self.signals = None

    def _load_raw(self, idx: int) -> np.ndarray:
        sig, _ = wfdb.rdsamp(str(self.paths[idx]))
        return sig.T.astype(np.float32)

    def _preload_all(self) -> np.ndarray:
        xs = [self._load_raw(i) for i in range(len(self.paths))]
        return np.stack(xs, axis=0)  # (N, C, T)

    def __len__(self) -> int:
        return len(self.db)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.signals is not None:
            x = self.signals[idx].astype(np.float32, copy=True)
        else:
            x = self._load_raw(idx)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None]) / (self.std[:, None] + 1e-6)
        y = self.labels[idx]
        return {
            "signal": torch.from_numpy(x),
            "label": torch.from_numpy(y),
            "ecg_id": torch.tensor(int(self.db.iloc[idx]["ecg_id"]), dtype=torch.long),
        }


def compute_train_stats_array(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """signals: (N, C, T) -> per-lead mean/std."""
    c = signals.shape[1]
    flat = signals.transpose(1, 0, 2).reshape(c, -1).astype(np.float64)
    mean = flat.mean(axis=1).astype(np.float32)
    std = flat.std(axis=1).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def multilabel_sample_weights(labels: np.ndarray, power: float = 1.0) -> np.ndarray:
    """Per-sample weights: sum of inverse-frequency of active positive classes."""
    freq = labels.mean(axis=0).astype(np.float64)
    class_w = 1.0 / np.maximum(freq, 1e-4)
    class_w = class_w ** float(power)
    w = (labels.astype(np.float64) * class_w[None, :]).sum(axis=1)
    # samples with no positive (should be rare) get minimum weight
    w = np.maximum(w, class_w.min())
    return w.astype(np.float64)


def build_ptbxl_dataloaders(
    data_root: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    sampling_rate: int = 100,
    train_folds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    val_folds: Sequence[int] = (9,),
    test_folds: Sequence[int] = (10,),
    cache_signals: bool = True,
    stats_cache: Optional[str | Path] = None,
    sampler_mode: str = "none",
    sampler_power: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    root = resolve_ptbxl_root(data_root)
    preload = bool(cache_signals)

    train_ds = PTBXLDataset(root, train_folds, sampling_rate, preload=preload)
    mean, std = None, None
    cache_path = Path(stats_cache) if stats_cache else None
    if cache_path is not None and cache_path.is_file():
        z = np.load(cache_path, allow_pickle=True)
        if (
            int(z["sampling_rate"]) == sampling_rate
            and np.array_equal(z["train_folds"], np.asarray(train_folds))
            and str(z["root"]) == str(root)
        ):
            mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    if mean is None or std is None:
        if train_ds.signals is None:
            train_ds.signals = train_ds._preload_all()
        mean, std = compute_train_stats_array(train_ds.signals)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                mean=mean,
                std=std,
                sampling_rate=np.asarray(sampling_rate),
                train_folds=np.asarray(train_folds),
                root=np.asarray(str(root)),
            )
    train_ds.mean, train_ds.std = mean, std

    val_ds = PTBXLDataset(root, val_folds, sampling_rate, mean=mean, std=std, preload=preload)
    test_ds = PTBXLDataset(root, test_folds, sampling_rate, mean=mean, std=std, preload=preload)

    pos = train_ds.labels.sum(axis=0)
    neg = len(train_ds) - pos
    pos_weight = (neg / np.maximum(pos, 1.0)).astype(np.float32)

    def _loader(ds: Dataset, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    train_sampler = None
    mode = str(sampler_mode).lower()
    if mode in ("rare", "weighted", "class_balanced"):
        from torch.utils.data import WeightedRandomSampler

        weights = multilabel_sample_weights(train_ds.labels, power=sampler_power)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    meta = {
        "mean": mean,
        "std": std,
        "pos_weight": pos_weight,
        "root": str(root),
        "sampler_mode": mode,
        "class_freq": train_ds.labels.mean(axis=0).astype(np.float32),
    }
    return (
        _loader(train_ds, True, sampler=train_sampler),
        _loader(val_ds, False),
        _loader(test_ds, False),
        meta,
    )