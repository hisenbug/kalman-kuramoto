"""Build poster-sized + slide-sized figures from .npz data files.

Reads data/exp*.npz, writes figures/poster/ and figures/slides/ (PDF + PNG, 300 dpi).
Plots never call the simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from analysis.plot_style import apply_style, SLIDE_SIZE                   # noqa: E402
from analysis.plots import (                                              # noqa: E402
    build_exp1, build_exp2, build_exp3, build_exp4, build_exp5,
)


DATA = ROOT / "data"
POSTER = ROOT / "figures" / "poster"
SLIDES = ROOT / "figures" / "slides"
POSTER.mkdir(parents=True, exist_ok=True)
SLIDES.mkdir(parents=True, exist_ok=True)


# (stem, builder, poster_size, slide_size, extra_kwargs_to_builder)
FIGURES = [
    # exp1 -- three cost-scale variants.
    ("exp1_log",       lambda: build_exp1(DATA / "exp1_baseline_vs_predictive.npz", "log"),      (11.0, 4.5), SLIDE_SIZE),
    ("exp1_linear",    lambda: build_exp1(DATA / "exp1_baseline_vs_predictive.npz", "linear"),   (11.0, 4.5), SLIDE_SIZE),
    ("exp1_fraction",  lambda: build_exp1(DATA / "exp1_baseline_vs_predictive.npz", "fraction"), (11.0, 4.5), SLIDE_SIZE),
    # exp2
    ("exp2", lambda: build_exp2(DATA / "exp2_phase_transition.npz"), (10.0, 6.5), SLIDE_SIZE),
    # exp3
    ("exp3", lambda: build_exp3(DATA / "exp3_convergence_vs_K.npz"),  (10.0, 6.0), SLIDE_SIZE),
    # exp4 (heatmap + t_90 vs K companion panel)
    ("exp4", lambda: build_exp4(DATA / "exp4_pareto_K_eps.npz",
                                DATA / "exp4b_tconv_vs_K.npz"),       (13.0, 5.5), SLIDE_SIZE),
    # exp5
    ("exp5", lambda: build_exp5(DATA / "exp5_finite_size_scaling.npz"), (12.0, 5.0), SLIDE_SIZE),
]


def _save(fig: plt.Figure, out_dir: Path, stem: str, size: tuple[float, float]) -> None:
    fig.set_size_inches(*size)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def main() -> None:
    for stem, build, poster_size, slide_size in FIGURES:
        data_path = DATA / "exp1_baseline_vs_predictive.npz"   # placeholder check
        # Just build, save poster, close, re-build for slides.
        apply_style("poster")
        fig = build()
        _save(fig, POSTER, stem, poster_size)
        plt.close(fig)

        apply_style("slides")
        fig = build()
        _save(fig, SLIDES, stem, slide_size)
        plt.close(fig)

        print(f"  [ok] {stem}  poster={poster_size}  slide={slide_size}")


if __name__ == "__main__":
    main()
