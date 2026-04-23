"""Experiment 4b (companion to exp4): dense K sweep at fixed supercritical
epsilon, records r(t) traces so t_90(K) can be extracted for the Pareto panel.

Provides the 'what is K good for / what does K cost in time?' axis that the
heatmap (exp4) doesn't expose directly.
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
    phys: PhysicsParams = PhysicsParams(T=4000),
    epsilon: float = 0.005,       # safely supercritical at N=2000
    K_values: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0),
    seeds: tuple[int, ...] = (1, 2, 3),
    out_stem: str = "exp4b_tconv_vs_K",
) -> dict:
    device = get_device()
    nK, nS = len(K_values), len(seeds)
    print(f"[exp4b] device={device}, N={phys.N}, T={phys.T}, eps={epsilon}, K={K_values}")

    r_t = np.zeros((nK, nS, phys.T))
    t0 = time.time()
    for j, K in enumerate(K_values):
        for k, s in enumerate(seeds):
            pred = PredictiveParams(epsilon=float(epsilon), K=float(K))
            res = run_predictive(phys, pred, seed=s, device=device)
            r_t[j, k] = res.r_t
        print(f"[exp4b] K={K:6.1f}  r_final(mean)={r_t[j].mean(axis=0)[-1]:.3f}  "
              f"({(time.time()-t0)/60:.1f} min)")

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
        make_manifest("exp4b_tconv_vs_K", phys,
                      extra={"epsilon": epsilon, "K_values": list(K_values),
                             "seeds": list(seeds), "wall_time_s": time.time() - t0}),
    )
    print(f"[exp4b] done in {(time.time()-t0)/60:.1f} min")
    return {"npz": out_npz}


if __name__ == "__main__":
    run()
