"""Experiment 3: r(t) at fixed supercritical epsilon, varying K.

Above eps_c, higher K converges slower (more inertia) but to the same
steady-state r. This visualizes the timescale separation.
"""
from __future__ import annotations

import time
import numpy as np

from _common import DATA_DIR
from tct.config import PhysicsParams, PredictiveParams
from tct.device import get_device
from tct.predictive import run_predictive
from tct.runner import make_manifest, save_manifest


def run(
    phys: PhysicsParams = PhysicsParams(T=6000),   # longer to show slow convergence at high K
    epsilon: float = 0.01,
    K_values: tuple[float, ...] = (5.0, 20.0, 50.0, 100.0),
    seeds: tuple[int, ...] = (1, 2, 3),
    out_stem: str = "exp3_convergence_vs_K",
) -> dict:
    device = get_device()
    nK, nS = len(K_values), len(seeds)
    print(f"[exp3] device={device}, N={phys.N}, T={phys.T}, eps={epsilon}, K={K_values}")

    r_t = np.zeros((nK, nS, phys.T))
    t0 = time.time()
    for j, K in enumerate(K_values):
        for k, s in enumerate(seeds):
            pred = PredictiveParams(epsilon=float(epsilon), K=float(K))
            res = run_predictive(phys, pred, seed=s, device=device)
            r_t[j, k] = res.r_t
        print(f"[exp3]  K={K:5.1f}  r_final(mean)={r_t[j].mean(axis=0)[-1]:.3f}  ({(time.time()-t0)/60:.1f} min)")

    out_npz = DATA_DIR / f"{out_stem}.npz"
    np.savez_compressed(
        out_npz,
        r_t=r_t,
        K_values=np.asarray(K_values),
        epsilon=np.asarray(epsilon),
        dt=np.asarray(phys.dt),
        seeds=np.asarray(seeds),
    )
    save_manifest(
        DATA_DIR / f"{out_stem}_manifest.json",
        make_manifest("exp3_convergence_vs_K", phys,
                      extra={"epsilon": epsilon, "K_values": list(K_values),
                             "seeds": list(seeds), "wall_time_s": time.time() - t0}),
    )
    print(f"[exp3] done in {(time.time()-t0)/60:.1f} min")
    return {"npz": out_npz}


if __name__ == "__main__":
    run()
