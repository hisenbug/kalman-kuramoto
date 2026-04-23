"""Multi-seed driver + manifest writer. Plots never call this; they read .npz."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PhysicsParams, PredictiveParams
from .kuramoto import run_baseline
from .predictive import run_predictive


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def make_manifest(
    experiment: str,
    phys: PhysicsParams,
    extra: dict[str, Any] | None = None,
) -> dict:
    return {
        "experiment": experiment,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "physics": asdict(phys),
        **(extra or {}),
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, default=str))


def multi_seed_predictive(
    phys: PhysicsParams,
    pred: PredictiveParams,
    seeds: list[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the predictive model across seeds. Returns stacked arrays."""
    r_ts = []
    inter = []
    eras = []
    r_finals = []
    for s in seeds:
        res = run_predictive(phys, pred, seed=s, device=device)
        r_ts.append(res.r_t)
        inter.append(res.interaction_cost_total)
        eras.append(res.erasure_cost_total)
        r_finals.append(res.r_final)
    return {
        "r_t": np.stack(r_ts, axis=0),                 # (n_seeds, T)
        "interaction_total": np.asarray(inter),        # (n_seeds,)
        "erasure_total": np.asarray(eras),             # (n_seeds,)
        "r_final": np.asarray(r_finals),               # (n_seeds,)
        "seeds": np.asarray(seeds),
    }


def multi_seed_baseline(
    phys: PhysicsParams,
    seeds: list[int],
    device: torch.device,
) -> dict[str, np.ndarray]:
    r_ts = []
    inter = []
    r_finals = []
    for s in seeds:
        res = run_baseline(phys, seed=s, device=device)
        r_ts.append(res.r_t)
        inter.append(res.interaction_cost_total)
        r_finals.append(res.r_final)
    return {
        "r_t": np.stack(r_ts, axis=0),
        "interaction_total": np.asarray(inter),
        "erasure_total": np.zeros(len(seeds)),
        "r_final": np.asarray(r_finals),
        "seeds": np.asarray(seeds),
    }
