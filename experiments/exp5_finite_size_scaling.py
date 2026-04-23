"""Experiment 5: finite-size scaling of the critical sparsity eps_c.

For each N in {500, 1000, 2000, 4000}, run a focused epsilon sweep across
the transition region with multiple seeds. Extract eps_c(N) from where the
seed-averaged r_final first crosses 0.5, and fit eps_c vs N on log-log.

The slope tells us whether the predictive model's cost advantage survives
asymptotically: if eps_c ~ N^{-alpha} with alpha > 0 (say alpha=1), total
bandwidth eps_c * N^2 scales as N^{2-alpha}, which is sub-N^2, i.e. the
'5% of baseline' claim is a fixed fraction, not a constant-factor artifact.
"""
from __future__ import annotations

import time
import numpy as np

from _common import DATA_DIR
from tct.config import PhysicsParams, PredictiveParams
from tct.device import get_device
from tct.predictive import run_predictive
from tct.runner import make_manifest, save_manifest


def _epsilon_grid(N: int) -> np.ndarray:
    """Grid concentrated around expected eps_c, which we don't know a priori
    for each N. Use a log span spanning ~0.5x .. 20x of 0.0035 so we bracket
    the transition for all N without presuming the value.
    """
    return np.logspace(np.log10(0.001), np.log10(0.03), 10)


def run(
    N_values: tuple[int, ...] = (500, 1000, 2000, 4000),
    T: int = 3000,
    K: float = 20.0,
    seeds: tuple[int, ...] = (1, 2, 3),
    out_stem: str = "exp5_finite_size_scaling",
) -> dict:
    device = get_device()

    per_N = {}
    t0 = time.time()
    for N in N_values:
        phys = PhysicsParams(N=N, T=T)
        eps_grid = _epsilon_grid(N)
        r_final = np.zeros((len(eps_grid), len(seeds)))
        for i, eps in enumerate(eps_grid):
            for k, s in enumerate(seeds):
                pred = PredictiveParams(epsilon=float(eps), K=float(K))
                res = run_predictive(phys, pred, seed=s, device=device)
                r_final[i, k] = res.r_final
        per_N[N] = {"epsilons": eps_grid, "r_final": r_final}
        mean_curve = r_final.mean(axis=-1)
        above = np.where(mean_curve >= 0.5)[0]
        eps_c = float(eps_grid[above[0]]) if len(above) else float("nan")
        print(f"[exp5] N={N:5d}  eps_c~{eps_c:.4f}  ({(time.time()-t0)/60:.1f} min)")

    # Pack into rectangular arrays (same eps grid for each N here).
    all_eps = np.stack([per_N[N]["epsilons"] for N in N_values])
    all_r = np.stack([per_N[N]["r_final"] for N in N_values])

    out_npz = DATA_DIR / f"{out_stem}.npz"
    np.savez_compressed(
        out_npz,
        N_values=np.asarray(N_values),
        epsilons=all_eps,                   # (n_N, n_eps)
        r_final=all_r,                      # (n_N, n_eps, n_seeds)
        r_final_mean=all_r.mean(axis=-1),   # (n_N, n_eps)
        K=np.asarray(K),
        T=np.asarray(T),
        seeds=np.asarray(seeds),
    )
    save_manifest(
        DATA_DIR / f"{out_stem}_manifest.json",
        make_manifest("exp5_finite_size_scaling",
                      PhysicsParams(N=max(N_values), T=T),
                      extra={"N_values": list(N_values),
                             "K": K,
                             "T": T,
                             "seeds": list(seeds),
                             "wall_time_s": time.time() - t0}),
    )
    print(f"[exp5] done in {(time.time()-t0)/60:.1f} min")
    return {"npz": out_npz}


if __name__ == "__main__":
    run()
