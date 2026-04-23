"""One build-function per experiment. Each returns a matplotlib Figure so
the save layer can size/style it twice (poster + slides).

All functions read from .npz only -- they never call the simulation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .plot_style import CONDITION_COLOR, OKABE_ITO


# =============================================================================
# Experiment 1: baseline vs predictive
# Three variants of the cost panel: "log", "linear", "fraction". Choose at call.
# =============================================================================
def build_exp1(npz_path: Path, cost_scale: str = "linear") -> plt.Figure:
    """cost_scale in {'log', 'linear', 'fraction'}:
      - log:      log y, total cost, stacked (erasure sliver on top)
      - linear:   linear y truncated at baseline, stacks clearly visible
      - fraction: normalized so baseline = 1.0; predictive height = cost ratio
    """
    assert cost_scale in {"log", "linear", "fraction"}

    d = np.load(npz_path, allow_pickle=True)
    r_base = d["baseline_r_t"]
    r_pred = d["predictive_r_t"]
    base_interact = d["baseline_interaction"]
    pred_interact = d["predictive_interaction"]
    pred_erasure = d["predictive_erasure"]
    epsilon = float(d["epsilon"])
    K = float(d["K"])
    dt = float(d["dt"])
    T = r_base.shape[1]
    t_axis = np.arange(T) * dt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # --- (a) r(t) traces, mean +/- std across seeds ---
    def _band(ax, r_mat, color, label):
        mu = r_mat.mean(axis=0)
        sd = r_mat.std(axis=0)
        ax.plot(t_axis, mu, color=color, linewidth=2.2, label=label)
        if r_mat.shape[0] > 1:
            ax.fill_between(t_axis, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)

    _band(axL, r_base, CONDITION_COLOR["baseline"], "Baseline (all-to-all)")
    _band(axL, r_pred, CONDITION_COLOR["predictive"], f"Predictive (\u03b5={epsilon:.3f}, K={K:g})")
    axL.set_xlabel("Time (s)")
    axL.set_ylabel("Order parameter $r(t)$")
    axL.set_ylim(0, 1.05)
    # Crop to the interesting transient; steady state is flat after ~25s.
    base_final = r_base.mean(axis=0)[-1]
    t_crop = min(t_axis[-1], 50.0)
    axL.set_xlim(0, t_crop)
    axL.set_title("Matched sync at ~5% bandwidth")
    axL.legend(loc="lower right")

    # --- (b) cost composition (scale chosen by caller) ---
    base_mu = float(base_interact.mean())
    pred_interact_mu = float(pred_interact.mean())
    pred_erasure_mu = float(pred_erasure.mean())
    pred_total_mu = pred_interact_mu + pred_erasure_mu
    frac_pct = pred_total_mu / base_mu * 100

    x = np.arange(2)
    width = 0.55

    if cost_scale == "fraction":
        # Normalize to fraction of baseline. Baseline = 1.0.
        base_h = 1.0
        pred_int_h = pred_interact_mu / base_mu
        pred_era_h = pred_erasure_mu / base_mu
        ylabel = "Cost / baseline cost"
        ylim_top = 1.1
    else:
        base_h = base_mu
        pred_int_h = pred_interact_mu
        pred_era_h = pred_erasure_mu
        ylabel = "Total cost (arb. units)"
        ylim_top = None

    axR.bar(x[0], base_h, width, color=CONDITION_COLOR["baseline"],
            label="Interaction", edgecolor="white", linewidth=0.5)
    axR.bar(x[1], pred_int_h, width, color=CONDITION_COLOR["predictive"],
            edgecolor="white", linewidth=0.5)
    axR.bar(x[1], pred_era_h, width, bottom=pred_int_h,
            color=OKABE_ITO["orange"], edgecolor="white", linewidth=0.5, label="Erasure (Landauer)")

    if cost_scale == "log":
        axR.set_yscale("log")
        y_annotate = pred_int_h + pred_era_h
        axR.annotate(
            f"{frac_pct:.1f}% of baseline",
            xy=(1, y_annotate),
            xytext=(1, base_h * 0.5),
            ha="center", fontsize=13,
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )
    elif cost_scale == "linear":
        axR.set_ylim(0, base_h * 1.15)
        axR.text(1, pred_int_h + pred_era_h + base_h * 0.02,
                 f"{frac_pct:.1f}%",
                 ha="center", va="bottom", fontsize=14, color=CONDITION_COLOR["predictive"],
                 fontweight="bold")
    else:  # fraction
        axR.set_ylim(0, ylim_top)
        axR.text(1, pred_int_h + pred_era_h + 0.02,
                 f"{frac_pct:.1f}%",
                 ha="center", va="bottom", fontsize=14, color=CONDITION_COLOR["predictive"],
                 fontweight="bold")

    axR.set_xticks(x)
    axR.set_xticklabels(["Baseline", "Predictive"])
    axR.set_ylabel(ylabel)
    axR.set_title(f"Cost composition ({cost_scale})")
    axR.legend(loc="upper right")

    fig.tight_layout()
    return fig


# =============================================================================
# Experiment 2: sharp phase transition at epsilon_c
# =============================================================================
def build_exp2(npz_path: Path, eps_c: float | None = None) -> plt.Figure:
    d = np.load(npz_path, allow_pickle=True)
    epsilons = d["epsilons"]           # (n_eps,)
    K_values = d["K_values"]           # (n_K,)
    r_final = d["r_final"]             # (n_K, n_eps, n_seeds)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Auto-fit eps_c: for each K, find eps at which the 0.5 crossing happens
    # via linear interpolation, then take the median over K.
    eps_c_per_K = []
    for j in range(len(K_values)):
        mu = r_final[j].mean(axis=-1)
        eps_c_per_K.append(_interp_crossing(epsilons, mu, 0.5))
    if eps_c is None:
        fit = [e for e in eps_c_per_K if np.isfinite(e)]
        eps_c = float(np.median(fit)) if fit else float("nan")

    for j, K in enumerate(K_values):
        key = f"K={int(K)}"
        color = CONDITION_COLOR.get(key, None)
        mu = r_final[j].mean(axis=-1)
        sd = r_final[j].std(axis=-1)
        ax.errorbar(
            epsilons, mu, yerr=sd,
            color=color, marker="o", markersize=6, linewidth=1.8, capsize=3,
            label=f"$K = {int(K)}$",
        )

    ax.axvline(eps_c, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(
        rf"$\varepsilon_c \approx {eps_c:.4f}$",
        xy=(eps_c, 0.5),
        xytext=(eps_c * 2.5, 0.35),
        fontsize=14,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Sparsity $\varepsilon$ (fraction sampled per step)")
    ax.set_ylabel(r"Steady-state order parameter $r_\infty$")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Sharp phase transition at $\varepsilon_c$; $K$ shifts timescale, not threshold")
    ax.legend(title="Model precision", loc="center right")
    fig.tight_layout()
    return fig


# =============================================================================
# Experiment 3: convergence timescale vs K
# =============================================================================
def build_exp3(npz_path: Path) -> plt.Figure:
    d = np.load(npz_path, allow_pickle=True)
    K_values = d["K_values"]           # (n_K,)
    r_t = d["r_t"]                     # (n_K, n_seeds, T)
    dt = float(d["dt"])
    epsilon = float(d["epsilon"])
    t_axis = np.arange(r_t.shape[-1]) * dt

    fig, ax = plt.subplots(figsize=(10, 6.0))

    for j, K in enumerate(K_values):
        key = f"K={int(K)}"
        color = CONDITION_COLOR.get(key, None)
        mu = r_t[j].mean(axis=0)
        sd = r_t[j].std(axis=0)
        ax.plot(t_axis, mu, color=color, linewidth=2.0, label=f"$K = {int(K)}$")
        if r_t.shape[1] > 1:
            ax.fill_between(t_axis, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)

    # Crop x-axis to the interesting transient. Find the time by which all K
    # curves have reached 95% of their steady-state value, then pad 20%.
    r_inf = r_t[:, :, -100:].mean()
    reach_times = []
    for j in range(r_t.shape[0]):
        mu_j = r_t[j].mean(axis=0)
        idx = np.where(mu_j >= 0.95 * r_inf)[0]
        reach_times.append(idx[0] * dt if len(idx) else t_axis[-1])
    t_crop = min(t_axis[-1], max(reach_times) * 1.3 + 2.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Order parameter $r(t)$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, t_crop)
    ax.set_title(rf"Higher $K$ integrates slower; same $r_\infty$ ($\varepsilon = {epsilon:.3f}$)")
    ax.legend(title="Model precision", loc="lower right")
    fig.tight_layout()
    return fig


# =============================================================================
# Experiment 4: Pareto view in (K, eps) -- left heatmap, right t_90 vs K
# =============================================================================
def build_exp4(npz_path: Path, npz_tconv_path: Path | None = None,
               eps_c: float | None = None) -> plt.Figure:
    d = np.load(npz_path, allow_pickle=True)
    epsilons = d["epsilons"]           # (n_eps,)
    K_values = d["K_values"]           # (n_K,)
    r_final_mean = d["r_final_mean"]   # (n_K, n_eps)

    # Auto-fit eps_c: for each K, find eps where the column crosses 0.5; median.
    if eps_c is None:
        crossings = []
        for j in range(len(K_values)):
            c = _interp_crossing(epsilons, r_final_mean[j], 0.5)
            if np.isfinite(c):
                crossings.append(c)
        eps_c = float(np.median(crossings)) if crossings else epsilons.min()

    if npz_tconv_path is not None:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={"width_ratios": [1.15, 1.0]})
    else:
        fig, axL = plt.subplots(figsize=(10, 7.5))
        axR = None

    # --- LEFT: heatmap ---
    K_edges = _edges(K_values, log=True)
    eps_edges = _edges(epsilons, log=True)
    KK, EE = np.meshgrid(K_edges, eps_edges, indexing="ij")

    pcm = axL.pcolormesh(KK, EE, r_final_mean, cmap="viridis",
                         vmin=0, vmax=1, shading="flat")
    cbar = fig.colorbar(pcm, ax=axL, pad=0.02)
    cbar.set_label(r"Steady-state $r_\infty$")

    axL.axhline(eps_c, color=OKABE_ITO["vermillion"], linestyle=":", linewidth=2, alpha=0.9)
    axL.fill_between(
        [K_values.min(), K_values.max()],
        epsilons.min(), eps_c,
        color=OKABE_ITO["vermillion"], alpha=0.12,
    )
    axL.text(
        K_values.max() * 0.7, eps_c * 0.55,
        "Death zone\n($\\varepsilon < \\varepsilon_c$, no sync)",
        color=OKABE_ITO["vermillion"], fontsize=12, ha="center", va="center",
    )

    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlim(K_edges[0], K_edges[-1])
    axL.set_ylim(eps_edges[0], eps_edges[-1])
    axL.set_xlabel(r"Model precision $K$")
    axL.set_ylabel(r"Sparsity $\varepsilon$")
    axL.set_title(r"$r_\infty(K, \varepsilon)$: $\varepsilon_c$ is $K$-independent")

    # --- RIGHT: t_90 vs K ---
    if axR is not None:
        dd = np.load(npz_tconv_path, allow_pickle=True)
        r_t = dd["r_t"]                        # (nK, nS, T)
        Ks = dd["K_values"]                    # (nK,)
        dt = float(dd["dt"])
        eps_b = float(dd["epsilon"])

        # Per-seed t_90, then mean +/- std across seeds.
        nK, nS, T = r_t.shape
        r_inf_per = r_t[:, :, -100:].mean(axis=-1)      # (nK, nS)
        t90 = np.full((nK, nS), np.nan)
        for j in range(nK):
            for k in range(nS):
                target = 0.9 * r_inf_per[j, k]
                above = np.where(r_t[j, k] >= target)[0]
                if len(above):
                    t90[j, k] = above[0] * dt

        t90_mu = np.nanmean(t90, axis=1)
        t90_sd = np.nanstd(t90, axis=1)

        axR.errorbar(Ks, t90_mu, yerr=t90_sd, color="k", marker="o", markersize=7,
                     linewidth=1.8, capsize=4, label=rf"$\varepsilon = {eps_b:.3f}$ (supercritical)")

        # Power-law fit t_90 ~ K^beta.
        valid = np.isfinite(t90_mu) & (t90_mu > 0)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(np.log(Ks[valid]), np.log(t90_mu[valid]), 1)
            K_line = np.logspace(np.log10(Ks.min() * 0.9), np.log10(Ks.max() * 1.1), 50)
            t_line = np.exp(intercept) * K_line ** slope
            axR.plot(K_line, t_line, color=OKABE_ITO["vermillion"], linestyle="--",
                     linewidth=2, label=rf"fit: $t_{{90}} \propto K^{{{slope:.2f}}}$")

        axR.set_xscale("log")
        axR.set_yscale("log")
        axR.set_xlabel(r"Model precision $K$")
        axR.set_ylabel(r"Time to 90% of $r_\infty$  [s]")
        axR.set_title(r"Higher $K$ $\Rightarrow$ slower sync (no sync-quality gain)")
        axR.legend(loc="lower right")

    fig.tight_layout()
    return fig


# =============================================================================
# Experiment 5: finite-size scaling of epsilon_c
# =============================================================================
def build_exp5(npz_path: Path) -> plt.Figure:
    d = np.load(npz_path, allow_pickle=True)
    N_values = d["N_values"]
    epsilons = d["epsilons"]            # (n_N, n_eps)
    r_final_mean = d["r_final_mean"]    # (n_N, n_eps)
    r_final_all = d["r_final"]          # (n_N, n_eps, n_seeds)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- left: r_final vs eps for each N ---
    from matplotlib import cm
    colors = cm.viridis(np.linspace(0.0, 0.85, len(N_values)))
    eps_c_list = []

    for i, N in enumerate(N_values):
        eps = epsilons[i]
        r = r_final_all[i]
        mu = r.mean(axis=-1)
        sd = r.std(axis=-1)
        axL.errorbar(eps, mu, yerr=sd, color=colors[i], marker="o", markersize=5,
                     linewidth=1.6, capsize=3, label=f"$N = {int(N)}$")
        above = np.where(mu >= 0.5)[0]
        eps_c_list.append(float(eps[above[0]]) if len(above) else float("nan"))

    axL.set_xscale("log")
    axL.set_xlabel(r"Sparsity $\varepsilon$")
    axL.set_ylabel(r"$r_\infty$ (seed mean)")
    axL.set_ylim(0, 1.05)
    axL.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    axL.set_title(r"Phase transition at each $N$")
    axL.legend(loc="lower right", fontsize=12)

    # --- right: eps_c vs N, log-log, power-law fit ---
    Ns = np.asarray(N_values, dtype=float)
    ec = np.asarray(eps_c_list, dtype=float)
    valid = np.isfinite(ec)
    axR.scatter(Ns[valid], ec[valid], s=60, color="k", zorder=3, label=r"$\varepsilon_c(N)$")
    if valid.sum() >= 2:
        log_fit = np.polyfit(np.log(Ns[valid]), np.log(ec[valid]), 1)
        slope, intercept = log_fit
        N_line = np.logspace(np.log10(Ns.min() * 0.8), np.log10(Ns.max() * 1.2), 50)
        ec_line = np.exp(intercept) * N_line ** slope
        axR.plot(N_line, ec_line, color=OKABE_ITO["vermillion"], linestyle="--",
                 linewidth=2, label=rf"fit: $\varepsilon_c \propto N^{{{slope:.2f}}}$")

    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel(r"System size $N$")
    axR.set_ylabel(r"Critical sparsity $\varepsilon_c$")
    axR.set_title(r"Finite-size scaling of $\varepsilon_c$")
    axR.legend(loc="upper right")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _interp_crossing(x: np.ndarray, y: np.ndarray, y_level: float) -> float:
    """Linear interpolation on the first upward crossing of y_level."""
    above = np.where(y >= y_level)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    t = (y_level - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def _edges(centers: np.ndarray, log: bool) -> np.ndarray:
    if log:
        lc = np.log(centers)
        le = np.concatenate([[lc[0] - (lc[1] - lc[0]) / 2],
                             (lc[:-1] + lc[1:]) / 2,
                             [lc[-1] + (lc[-1] - lc[-2]) / 2]])
        return np.exp(le)
    le = np.concatenate([[centers[0] - (centers[1] - centers[0]) / 2],
                         (centers[:-1] + centers[1:]) / 2,
                         [centers[-1] + (centers[-1] - centers[-2]) / 2]])
    return le


def _fit_K_eps_contour(K: np.ndarray, eps: np.ndarray, r: np.ndarray, r_level: float = 0.5) -> float:
    """For each K, find the smallest eps that yields r >= r_level; fit K*eps=C."""
    products = []
    for j, k in enumerate(K):
        row = r[j]
        above = np.where(row >= r_level)[0]
        if len(above) == 0:
            continue
        e = eps[above[0]]
        products.append(k * e)
    if not products:
        return float("nan")
    return float(np.median(products))
