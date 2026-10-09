"""Experiment directory helpers (Shanghai timezone)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")


def shanghai_now_tag() -> str:
    return datetime.now(tz=SHANGHAI).strftime("%Y%m%d_%H%M%S")


def create_main_run_dir(experiments_root: str | Path, task_brief: str) -> Path:
    root = Path(experiments_root)
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in task_brief).strip("-")
    # date-first: YYYYMMDD_HHMMSS_<taskBrief>
    run_dir = root / f"{shanghai_now_tag()}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_subtask_dir(main_dir: Path, subtask_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in subtask_name).strip("-")
    sub = main_dir / safe
    sub.mkdir(parents=True, exist_ok=False)
    (sub / "checkpoints").mkdir(exist_ok=True)
    return sub


def setup_logger(log_file: Path, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary_md(main_dir: Path, rows: list[dict], title: str = "PTB-XL Results") -> None:
    lines = [
        f"# {title}",
        "",
        f"- main_dir: `{main_dir}`",
        f"- time: {datetime.now(tz=SHANGHAI).isoformat()}",
        "",
        "| subtask | arch | val_auroc | test_auroc | test_auprc | test_f1 | test_fmax | best_epoch | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {subtask} | {arch} | {val:.4f} | {test_auroc:.4f} | {test_auprc:.4f} | {test_f1:.4f} | {fmax:.4f} | {best_epoch} | {status} |".format(
                subtask=r.get("subtask", ""),
                arch=r.get("arch", "-"),
                val=float(r.get("val_macro_auroc", float("nan"))),
                test_auroc=float(r.get("test_macro_auroc", float("nan"))),
                test_auprc=float(r.get("test_macro_auprc", float("nan"))),
                test_f1=float(r.get("test_macro_f1", float("nan"))),
                fmax=float(r.get("test_fmax", float("nan"))),
                best_epoch=r.get("best_epoch", "-"),
                status=r.get("status", ""),
            )
        )
    (main_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mitbih_summary_md(main_dir: Path, rows: list[dict], title: str = "MIT-BIH Results") -> None:
    lines = [
        f"# {title}",
        "",
        f"- main_dir: `{main_dir}`",
        f"- time: {datetime.now(tz=SHANGHAI).isoformat()}",
        "",
        "| subtask | arch | val_macro_f1 | test_acc | test_macro_f1 | N_Se | S_Se | V_Se | F_Se | best_epoch | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {subtask} | {arch} | {val:.4f} | {acc:.4f} | {f1:.4f} | {nse:.4f} | {sse:.4f} | {vse:.4f} | {fse:.4f} | {best_epoch} | {status} |".format(
                subtask=r.get("subtask", ""),
                arch=r.get("arch", "-"),
                val=float(r.get("val_macro_f1", float("nan"))),
                acc=float(r.get("test_accuracy", float("nan"))),
                f1=float(r.get("test_macro_f1", float("nan"))),
                nse=float(r.get("test_n_sensitivity", float("nan"))),
                sse=float(r.get("test_s_sensitivity", float("nan"))),
                vse=float(r.get("test_v_sensitivity", float("nan"))),
                fse=float(r.get("test_f_sensitivity", float("nan"))),
                best_epoch=r.get("best_epoch", "-"),
                status=r.get("status", ""),
            )
        )
    (main_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
