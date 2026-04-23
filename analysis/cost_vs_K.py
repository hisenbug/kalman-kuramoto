"""Cost-vs-K analysis (not a poster figure).

Uses the exp4 heatmap data (which stores per-cell interaction + erasure costs)
plus the exp4b trace data (which gives time-to-sync t_90) to ask:

  At matched steady-state r_infty, does higher K pay less per step (smaller
  Kalman gain => smaller innovations => less Landauer bill) enough to offset
  taking more steps?

Interaction cost per step = eps * N^2  (so K-independent)
Erasure cost per step     ~= 0.5 * N * log(1 + Var[innov]/R)
Total cost for sync       = (per-step cost) * t_90 / dt

Prints a table so we can decide whether to include it in future work.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _t90_from_trace(r_t: np.ndarray, dt: float) -> np.ndarray:
    """r_t: (nK, nS, T). Returns t_90 per (K, seed)."""
    nK, nS, T = r_t.shape
    r_inf = r_t[:, :, -100:].mean(axis=-1)       # (nK, nS)
    t90 = np.full((nK, nS), np.nan)
    for j in range(nK):
        for k in range(nS):
            above = np.where(r_t[j, k] >= 0.9 * r_inf[j, k])[0]
            if len(above):
                t90[j, k] = above[0] * dt
    return t90


def main() -> None:
    # --- 4b: t_90 vs K at fixed eps=0.005 ---
    b = np.load(DATA / "exp4b_tconv_vs_K.npz")
    r_t = b["r_t"]                       # (nK, nS, T)
    Ks_b = b["K_values"]
    dt = float(b["dt"])
    eps_b = float(b["epsilon"])
    t90 = _t90_from_trace(r_t, dt)       # (nK, nS) seconds
    t90_mu = np.nanmean(t90, axis=1)

    # --- exp4 heatmap: per-cell (K, eps) interaction + erasure costs ---
    # Note: these are *totals over the whole run* (length = phys.T * dt).
    # To compare agent-types at matched sync we rescale to cost-per-step.
    d = np.load(DATA / "exp4_pareto_K_eps.npz")
    interact = d["interaction"]          # (nK, nE, nS)  totals
    erasure = d["erasure"]               # (nK, nE, nS)  totals
    eps_4 = d["epsilons"]
    K_4 = d["K_values"]

    # Pick the eps column closest to eps_b from exp4.
    j_eps = int(np.argmin(np.abs(eps_4 - eps_b)))
    eps_pick = float(eps_4[j_eps])
    print(f"Using exp4 eps column closest to {eps_b}: eps={eps_pick:.5f}")

    # Per-step costs from exp4 (run length T in steps unknown here — infer from
    # interaction_per_step = eps * N^2 and the total).
    N = 2000
    per_step_interact = eps_pick * N * N               # scalar
    # Erasure is stored as total; divide by estimated number of steps.
    T_steps_exp4 = float(interact[0, j_eps, 0]) / per_step_interact  # should equal run T
    erasure_per_step = erasure[:, j_eps, :].mean(axis=-1) / T_steps_exp4  # (nK,)

    # Interpolate exp4's K grid onto exp4b's Ks for a fair comparison.
    erasure_per_step_at_K4b = np.interp(np.log(Ks_b), np.log(K_4), erasure_per_step)

    steps_to_sync = t90_mu / dt
    cost_interact_to_sync = per_step_interact * steps_to_sync
    cost_erasure_to_sync = erasure_per_step_at_K4b * steps_to_sync
    cost_total_to_sync = cost_interact_to_sync + cost_erasure_to_sync

    header = f"{'K':>6}  {'t_90 [s]':>10}  {'erasure/step':>14}  " \
             f"{'total-to-sync':>15}  {'erasure-to-sync':>15}  {'%erasure':>9}"
    print(header)
    print("-" * len(header))
    for j, K in enumerate(Ks_b):
        pct = 100 * cost_erasure_to_sync[j] / cost_total_to_sync[j]
        print(f"{K:6.1f}  {t90_mu[j]:10.2f}  {erasure_per_step_at_K4b[j]:14.3e}  "
              f"{cost_total_to_sync[j]:15.3e}  {cost_erasure_to_sync[j]:15.3e}  {pct:8.2f}%")

    # Is total cost flatter in K than steps_to_sync?
    print()
    slope_steps = np.polyfit(np.log(Ks_b), np.log(steps_to_sync), 1)[0]
    slope_total = np.polyfit(np.log(Ks_b), np.log(cost_total_to_sync), 1)[0]
    slope_erasure_ps = np.polyfit(np.log(Ks_b), np.log(np.clip(erasure_per_step_at_K4b, 1e-30, None)), 1)[0]
    print(f"t_90        ~ K^{slope_steps:+.2f}")
    print(f"erasure/step ~ K^{slope_erasure_ps:+.2f}")
    print(f"total-to-sync ~ K^{slope_total:+.2f}")
    print()
    print("Interpretation:")
    print("  * If slope_total ~ slope_steps, erasure/step is K-flat => high K just costs time.")
    print("  * If slope_total < slope_steps, high K saves erasure per step (partial offset).")
    print("  * If slope_total < 0, high K wins in total cost despite slower sync.")


if __name__ == "__main__":
    main()
