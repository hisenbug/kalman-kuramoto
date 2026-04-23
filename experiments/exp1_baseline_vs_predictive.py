"""Experiment 1: baseline (all-to-all Kuramoto) vs predictive at matched sync.

At supercritical epsilon (well above epsilon_c) and moderate K, the predictive
model reaches the same steady-state r as the all-to-all baseline while using
only ~5% of the interaction bandwidth. This experiment measures both.
"""
from __future__ import annotations

import numpy as np

from _common import DATA_DIR
from tct.config import PhysicsParams, PredictiveParams
from tct.device import get_device
from tct.runner import multi_seed_baseline, multi_seed_predictive, make_manifest, save_manifest


def run(
    phys: PhysicsParams = PhysicsParams(),
    epsilon: float = 0.05,
    K: float = 20.0,
    seeds: tuple[int, ...] = (1, 2, 3),
    out_stem: str = "exp1_baseline_vs_predictive",
) -> dict:
    device = get_device()
    pred = PredictiveParams(epsilon=epsilon, K=K)

    print(f"[exp1] device={device}, N={phys.N}, T={phys.T}, seeds={seeds}")
    print(f"[exp1] predictive: eps={epsilon}, K={K}")

    base = multi_seed_baseline(phys, list(seeds), device)
    pred_out = multi_seed_predictive(phys, pred, list(seeds), device)

    out_npz = DATA_DIR / f"{out_stem}.npz"
    np.savez_compressed(
        out_npz,
        baseline_r_t=base["r_t"],
        baseline_interaction=base["interaction_total"],
        baseline_r_final=base["r_final"],
        predictive_r_t=pred_out["r_t"],
        predictive_interaction=pred_out["interaction_total"],
        predictive_erasure=pred_out["erasure_total"],
        predictive_r_final=pred_out["r_final"],
        epsilon=np.asarray(epsilon),
        K=np.asarray(K),
        dt=np.asarray(phys.dt),
        seeds=np.asarray(seeds),
    )

    manifest = make_manifest(
        "exp1_baseline_vs_predictive",
        phys,
        extra={"epsilon": epsilon, "K": K, "seeds": list(seeds)},
    )
    save_manifest(DATA_DIR / f"{out_stem}_manifest.json", manifest)

    print(f"[exp1] baseline r_final: {base['r_final'].mean():.3f} +/- {base['r_final'].std():.3f}")
    print(f"[exp1] predictive r_final: {pred_out['r_final'].mean():.3f} +/- {pred_out['r_final'].std():.3f}")
    frac = (pred_out["interaction_total"].mean() + pred_out["erasure_total"].mean()) / base["interaction_total"].mean()
    print(f"[exp1] cost ratio (pred/baseline): {frac * 100:.1f}%")
    return {"npz": out_npz, "manifest_path": DATA_DIR / f"{out_stem}_manifest.json"}


if __name__ == "__main__":
    run()
