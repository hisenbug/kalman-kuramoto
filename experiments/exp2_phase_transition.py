"""Experiment 2: fine epsilon sweep, 5 seeds per point, K in {5, 20, 50}.

Shows the sharp transition at epsilon_c ~ 0.0035 and that K controls timescale
(via alpha_freq / Kalman gain) but not the threshold location.
"""
from __future__ import annotations

import time
import numpy as np

from _common import DATA_DIR
from tct.config import PhysicsParams, PredictiveParams
from tct.device import get_device
from tct.predictive import run_predictive
from tct.runner import make_manifest, save_manifest


def _epsilon_grid() -> np.ndarray:
    """Widely-spaced sweep: deep subcritical to well above any plausible
    saturation, with dense resolution near the expected threshold (~0.003).
    """
    deep_sub  = np.array([0.0005, 0.0008])               # deep subcritical edge check
    coarse_lo = np.logspace(np.log10(0.0012), np.log10(0.0025), 3, endpoint=False)
    dense     = np.linspace(0.0025, 0.0055, 11)          # around expected eps_c
    mid       = np.array([0.007, 0.01])
    high      = np.array([0.02, 0.035, 0.05])            # supercritical saturation check
    return np.unique(np.concatenate([deep_sub, coarse_lo, dense, mid, high]))


def run(
    phys: PhysicsParams = PhysicsParams(),
    K_values: tuple[float, ...] = (5.0, 20.0, 50.0),
    epsilons: np.ndarray | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    out_stem: str = "exp2_phase_transition",
) -> dict:
    device = get_device()
    if epsilons is None:
        epsilons = _epsilon_grid()

    nK, nE, nS = len(K_values), len(epsilons), len(seeds)
    print(f"[exp2] device={device}, grid: {nK} K x {nE} eps x {nS} seeds = {nK*nE*nS} runs")
    print(f"[exp2] epsilons: {np.round(epsilons, 4).tolist()}")

    r_final = np.zeros((nK, nE, nS))
    t0 = time.time()
    for j, K in enumerate(K_values):
        for i, eps in enumerate(epsilons):
            for k, s in enumerate(seeds):
                pred = PredictiveParams(epsilon=float(eps), K=float(K))
                res = run_predictive(phys, pred, seed=s, device=device)
                r_final[j, i, k] = res.r_final
            elapsed = time.time() - t0
            mu = r_final[j, i].mean()
            print(f"[exp2]  K={K:5.1f}  eps={eps:.4f}  r_final={mu:.3f}  ({elapsed/60:.1f} min elapsed)")

    out_npz = DATA_DIR / f"{out_stem}.npz"
    np.savez_compressed(
        out_npz,
        r_final=r_final,
        epsilons=epsilons,
        K_values=np.asarray(K_values),
        seeds=np.asarray(seeds),
        dt=np.asarray(phys.dt),
    )
    manifest = make_manifest(
        "exp2_phase_transition",
        phys,
        extra={
            "K_values": list(K_values),
            "epsilons": epsilons.tolist(),
            "seeds": list(seeds),
            "wall_time_s": time.time() - t0,
        },
    )
    save_manifest(DATA_DIR / f"{out_stem}_manifest.json", manifest)
    print(f"[exp2] done in {(time.time() - t0)/60:.1f} min")
    return {"npz": out_npz}


if __name__ == "__main__":
    run()
