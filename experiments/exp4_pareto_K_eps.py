"""Experiment 4: 2D sweep over (K, epsilon). Heatmap of r_final with a
K * epsilon = const contour overlay from the information-balance argument.
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
    phys: PhysicsParams = PhysicsParams(T=2500),   # sweep doesn't need max T
    K_values: np.ndarray | None = None,
    epsilons: np.ndarray | None = None,
    seeds: tuple[int, ...] = (1, 2, 3),
    out_stem: str = "exp4_pareto_K_eps",
) -> dict:
    device = get_device()
    if K_values is None:
        K_values = np.logspace(np.log10(0.5), np.log10(100.0), 8)
    if epsilons is None:
        epsilons = np.logspace(np.log10(0.0005), np.log10(0.015), 8)

    nK, nE, nS = len(K_values), len(epsilons), len(seeds)
    print(f"[exp4] grid: {nK} K x {nE} eps x {nS} seeds = {nK*nE*nS} runs")

    r_final = np.zeros((nK, nE, nS))
    interact = np.zeros((nK, nE, nS))
    erasure = np.zeros((nK, nE, nS))

    t0 = time.time()
    for j, K in enumerate(K_values):
        for i, eps in enumerate(epsilons):
            for k, s in enumerate(seeds):
                pred = PredictiveParams(epsilon=float(eps), K=float(K))
                res = run_predictive(phys, pred, seed=s, device=device)
                r_final[j, i, k] = res.r_final
                interact[j, i, k] = res.interaction_cost_total
                erasure[j, i, k] = res.erasure_cost_total
        print(f"[exp4] row K={K:6.2f} done  ({(time.time()-t0)/60:.1f} min elapsed)")

    out_npz = DATA_DIR / f"{out_stem}.npz"
    np.savez_compressed(
        out_npz,
        r_final=r_final,
        r_final_mean=r_final.mean(axis=-1),
        interaction=interact,
        erasure=erasure,
        K_values=K_values,
        epsilons=epsilons,
        seeds=np.asarray(seeds),
    )
    save_manifest(
        DATA_DIR / f"{out_stem}_manifest.json",
        make_manifest("exp4_pareto_K_eps", phys,
                      extra={"K_values": K_values.tolist(),
                             "epsilons": epsilons.tolist(),
                             "seeds": list(seeds),
                             "wall_time_s": time.time() - t0}),
    )
    print(f"[exp4] done in {(time.time()-t0)/60:.1f} min")
    return {"npz": out_npz}


if __name__ == "__main__":
    run()
