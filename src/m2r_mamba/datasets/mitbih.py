"""MIT-BIH Arrhythmia: AAMI beat classification, de Chazal DS1/DS2.

v2: drop Q (4-class), multi-beat temporal context, RR channels, rare-class sampler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

AAMI_MAP: Dict[str, str] = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}

CLASSES_5: Tuple[str, ...] = ("N", "S", "V", "F", "Q")
CLASSES_4: Tuple[str, ...] = ("N", "S", "V", "F")

DS1: Tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2: Tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)
DS1_VAL: Tuple[str, ...] = ("106", "119", "223", "230")


def resolve_mitbih_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if (root / "100.hea").is_file():
        return root
    matches = sorted(root.glob("mit-bih-arrhythmia-database-*"))
    for cand in matches:
        if (cand / "100.hea").is_file():
            return cand
    raise FileNotFoundError(f"Cannot find MIT-BIH records under {data_root}")


def _classes(drop_q: bool) -> Tuple[str, ...]:
    return CLASSES_4 if drop_q else CLASSES_5


def _extract_record_beats(
    rec_path: Path,
    before: int,
    after: int,
    drop_q: bool,
    context_k: int,
    use_rr: bool,
    fs: float = 360.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build samples for one record.

    Returns:
      x: (N, C, T) where C = 2 (+4 RR channels if use_rr), T = (before+after)*(2*k+1)
      y: (N,) class indices into CLASSES_4/5
    """
    classes = _classes(drop_q)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rec = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(rec_path), "atr")
    sig = rec.p_signal.astype(np.float32).T
    if sig.shape[0] != 2:
        sig = sig[:2]
    t_full = sig.shape[1]
    beat_len = before + after
    k = max(int(context_k), 0)

    # valid single-beat crops first
    peaks: List[int] = []
    labels: List[str] = []
    waves: List[np.ndarray] = []
    for samp, sym in zip(ann.sample.tolist(), ann.symbol):
        aami = AAMI_MAP.get(sym)
        if aami is None:
            continue
        if drop_q and aami == "Q":
            continue
        if aami not in class_to_idx:
            continue
        start = int(samp) - before
        end = int(samp) + after
        if start < 0 or end > t_full:
            continue
        peaks.append(int(samp))
        labels.append(aami)
        waves.append(sig[:, start:end])

    n = len(peaks)
    if n == 0:
        c = 2 + (8 if use_rr else 0)
        return np.zeros((0, c, beat_len * (2 * k + 1)), np.float32), np.zeros((0,), np.int64)

    peaks_arr = np.asarray(peaks, dtype=np.int64)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(n):
        if i - k < 0 or i + k >= n:
            continue
        # temporal concat of neighboring beat morphologies
        parts = [waves[j] for j in range(i - k, i + k + 1)]
        morph = np.concatenate(parts, axis=1)  # (2, T_ctx)

        if use_rr:
            # RR features of center beat (seconds), broadcast as constant channels.
            # v3: 8 channels — prematurity / local rhythm cues critical for SVEB.
            prev_i = max(i - 1, 0)
            next_i = min(i + 1, n - 1)
            rr_pre = (peaks_arr[i] - peaks_arr[prev_i]) / fs
            rr_post = (peaks_arr[next_i] - peaks_arr[i]) / fs
            lo, hi = max(i - 2, 0), min(i + 2, n - 1)
            segs = np.diff(peaks_arr[lo : hi + 1]) / fs
            rr_local = float(segs.mean()) if len(segs) else rr_pre
            rr_std = float(segs.std()) if len(segs) > 1 else 0.0
            rr_ratio = rr_pre / (rr_post + 1e-6)
            rr_pre_norm = rr_pre / (rr_local + 1e-6)
            rr_post_norm = rr_post / (rr_local + 1e-6)
            # compensatory pause proxy after premature beat
            rr_comp = rr_post / (rr_pre + 1e-6)
            rr_vec = np.asarray(
                [
                    rr_pre,
                    rr_post,
                    rr_local,
                    rr_ratio,
                    rr_pre_norm,
                    rr_post_norm,
                    rr_comp,
                    rr_std,
                ],
                dtype=np.float32,
            )[:, None]
            rr_ch = np.repeat(rr_vec, morph.shape[1], axis=1)  # (8, T)
            morph = np.concatenate([morph, rr_ch], axis=0)

        xs.append(morph.astype(np.float32))
        ys.append(class_to_idx[labels[i]])

    if not xs:
        c = 2 + (8 if use_rr else 0)
        return np.zeros((0, c, beat_len * (2 * k + 1)), np.float32), np.zeros((0,), np.int64)
    return np.stack(xs, 0), np.asarray(ys, dtype=np.int64)


def build_beat_arrays(
    root: Path,
    records: Sequence[str],
    before: int,
    after: int,
    drop_q: bool,
    context_k: int,
    use_rr: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    all_x, all_y = [], []
    for rid in records:
        x, y = _extract_record_beats(root / rid, before, after, drop_q, context_k, use_rr)
        if len(y) == 0:
            continue
        all_x.append(x)
        all_y.append(y)
    if not all_x:
        c = 2 + (8 if use_rr else 0)
        t = (before + after) * (2 * max(context_k, 0) + 1)
        return np.zeros((0, c, t), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(all_x, 0), np.concatenate(all_y, 0)


class MITBIHBeatDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.signals[idx].copy()
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None]) / (self.std[:, None] + 1e-6)
        return {
            "signal": torch.from_numpy(x),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _lead_stats(signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = signals.shape[1]
    flat = signals.transpose(1, 0, 2).reshape(c, -1).astype(np.float64)
    mean = flat.mean(axis=1).astype(np.float32)
    std = np.maximum(flat.std(axis=1).astype(np.float32), 1e-6)
    return mean, std


def _sample_weights(labels: np.ndarray, n_classes: int, s_boost: float = 3.0) -> np.ndarray:
    """Inverse-freq weights with extra boost on S (class idx 1)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_w = 1.0 / np.maximum(counts, 1.0)
    if n_classes > 1:
        class_w[1] *= float(s_boost)  # S
    class_w = class_w / class_w.mean()
    return class_w[labels].astype(np.float64)


def build_mitbih_dataloaders(
    data_root: str | Path,
    batch_size: int = 256,
    num_workers: int = 4,
    before: int = 90,
    after: int = 90,
    cache_path: Optional[str | Path] = None,
    val_records: Sequence[str] = DS1_VAL,
    drop_q: bool = True,
    context_k: int = 2,
    use_rr: bool = True,
    sampler_mode: str = "rare",
    s_boost: float = 2.0,
    weight_cap: float = 20.0,
    f_weight_cap: float = 6.0,
    s_weight_mult: float = 1.5,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    root = resolve_mitbih_root(data_root)
    classes = _classes(drop_q)
    n_classes = len(classes)
    val_set = tuple(str(r) for r in val_records)
    train_recs = tuple(r for r in DS1 if r not in val_set)
    test_recs = DS2

    cache = Path(cache_path) if cache_path else None
    loaded = False
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if (
            int(z["before"]) == before
            and int(z["after"]) == after
            and str(z["root"]) == str(root)
            and bool(z["drop_q"]) == drop_q
            and int(z["context_k"]) == context_k
            and bool(z["use_rr"]) == use_rr
            and int(z.get("rr_dim", 8 if use_rr else 0)) == (8 if use_rr else 0)
            and list(z["train_recs"]) == list(train_recs)
            and list(z["val_recs"]) == list(val_set)
            and list(z["test_recs"]) == list(test_recs)
        ):
            x_tr, y_tr = z["x_tr"], z["y_tr"]
            x_va, y_va = z["x_va"], z["y_va"]
            x_te, y_te = z["x_te"], z["y_te"]
            mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
            loaded = True

    if not loaded:
        x_tr, y_tr = build_beat_arrays(root, train_recs, before, after, drop_q, context_k, use_rr)
        x_va, y_va = build_beat_arrays(root, val_set, before, after, drop_q, context_k, use_rr)
        x_te, y_te = build_beat_arrays(root, test_recs, before, after, drop_q, context_k, use_rr)
        mean, std = _lead_stats(x_tr)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache,
                x_tr=x_tr,
                y_tr=y_tr,
                x_va=x_va,
                y_va=y_va,
                x_te=x_te,
                y_te=y_te,
                mean=mean,
                std=std,
                before=np.asarray(before),
                after=np.asarray(after),
                drop_q=np.asarray(drop_q),
                context_k=np.asarray(context_k),
                use_rr=np.asarray(use_rr),
                rr_dim=np.asarray(8 if use_rr else 0),
                root=np.asarray(str(root)),
                train_recs=np.asarray(train_recs),
                val_recs=np.asarray(val_set),
                test_recs=np.asarray(test_recs),
            )

    train_ds = MITBIHBeatDataset(x_tr, y_tr, mean, std)
    val_ds = MITBIHBeatDataset(x_va, y_va, mean, std)
    test_ds = MITBIHBeatDataset(x_te, y_te, mean, std)

    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    class_weight = (counts.sum() / (n_classes * np.maximum(counts, 1.0))).astype(np.float32)
    # Prefer Acc: cap F (idx 3) hard — prior N→F flood killed Acc.
    if n_classes >= 4:
        class_weight[3] = min(float(class_weight[3]), float(f_weight_cap))
    if n_classes > 1:
        class_weight[1] = float(class_weight[1]) * float(s_weight_mult)
    class_weight = np.minimum(class_weight, float(weight_cap)).astype(np.float32)

    train_sampler = None
    mode = str(sampler_mode).lower()
    if mode in ("rare", "weighted", "s_boost"):
        w = _sample_weights(y_tr, n_classes, s_boost=s_boost)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=len(w),
            replacement=True,
        )

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

    n_leads = int(x_tr.shape[1]) if len(x_tr) else (2 + (8 if use_rr else 0))
    morph_channels = 2
    rr_channels = max(n_leads - morph_channels, 0)
    meta = {
        "mean": mean,
        "std": std,
        "class_weight": class_weight,
        "class_counts_train": counts.astype(np.int64),
        "root": str(root),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "classes": list(classes),
        "n_classes": n_classes,
        "n_leads": n_leads,
        "morph_channels": morph_channels,
        "rr_channels": rr_channels,
        "train_recs": list(train_recs),
        "val_recs": list(val_set),
        "test_recs": list(test_recs),
        "before": before,
        "after": after,
        "drop_q": drop_q,
        "context_k": context_k,
        "use_rr": use_rr,
        "sampler_mode": mode,
        "weight_cap": float(weight_cap),
        "f_weight_cap": float(f_weight_cap),
        "s_weight_mult": float(s_weight_mult),
    }
    return (
        _loader(train_ds, True, sampler=train_sampler),
        _loader(val_ds, False),
        _loader(test_ds, False),
        meta,
    )


# backward-compatible alias
CLASSES = CLASSES_4
