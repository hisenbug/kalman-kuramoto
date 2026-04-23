"""Publication style. Two sizes: 'poster' (per-figure embed width) and 'slides'
(6x4.5"). Fonts are in matplotlib points, which are absolute on the saved
figure -- so specifying a figsize locks the relative type size at the embed.

Consistent condition -> color mapping across all plots, per user spec. Palette
is Okabe-Ito, colorblind-safe.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


# Okabe-Ito palette -- colorblind-safe 7 colors plus black.
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}


# Canonical condition -> color. Same on every figure.
CONDITION_COLOR = {
    "baseline":    OKABE_ITO["black"],
    "predictive":  OKABE_ITO["blue"],
    "K=5":         OKABE_ITO["orange"],
    "K=20":        OKABE_ITO["sky_blue"],
    "K=50":        OKABE_ITO["green"],
    "K=100":       OKABE_ITO["vermillion"],
    # For r(t) three-regime plot:
    "subcritical":    OKABE_ITO["vermillion"],
    "critical":       OKABE_ITO["orange"],
    "supercritical":  OKABE_ITO["green"],
}


# Font sizes requested for the POSTER embed. Keep this the source of truth.
POSTER_FONTS = {
    "axes.labelsize":  16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.titlesize":  18,
    "figure.titlesize": 18,
}

# Slides version -- 6x4.5" embed, fonts tuned so text remains legible.
SLIDE_FONTS = {
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize":  14,
    "figure.titlesize": 14,
}


_BASE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.grid": False,                 # user: no unnecessary gridlines
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "figure.facecolor": "white",        # user: no background color
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,                 # TrueType, editable in Illustrator
    "ps.fonttype": 42,
}


def apply_style(mode: str = "poster") -> None:
    """Call once before building any figure. mode in {'poster', 'slides'}."""
    rc = dict(_BASE_RC)
    rc.update(POSTER_FONTS if mode == "poster" else SLIDE_FONTS)
    mpl.rcParams.update(rc)


def figsize_for_embed(width_in: float, aspect: float = 0.65) -> tuple[float, float]:
    """figsize tuple for a given embed width. Aspect is height/width."""
    return (width_in, width_in * aspect)


# Canonical embed widths for the poster (inches).
POSTER_WIDTHS = {
    "full":    11.0,   # full-column figure on a 48x36 poster
    "half":    7.5,
    "square":  7.0,
}

# Slide size (single).
SLIDE_SIZE = (6.0, 4.5)


def savefig_pair(
    fig,
    out_stem: Path,
    poster_width_in: float,
    aspect: float,
    dpi: int = 300,
) -> dict[str, Path]:
    """Save fig twice: poster/<stem>.{pdf,png} and slides/<stem>.{pdf,png}.

    The current fig is resized in-place for poster; then a clone-by-resize is
    made for slides with the slide rcParams applied.
    """
    out_stem = Path(out_stem)
    poster_dir = out_stem.parent.parent / "poster"
    slides_dir = out_stem.parent.parent / "slides"
    poster_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)
    stem = out_stem.name

    # Poster save.
    fig.set_size_inches(poster_width_in, poster_width_in * aspect)
    written = {}
    for ext in ("pdf", "png"):
        p = poster_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi)
        written[f"poster_{ext}"] = p

    return written


def save_slide_version(
    build_fn,
    out_stem: Path,
    aspect: float,
    dpi: int = 300,
) -> dict[str, Path]:
    """Re-build the figure under slide rcParams and save at SLIDE_SIZE."""
    out_stem = Path(out_stem)
    slides_dir = out_stem.parent.parent / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    stem = out_stem.name

    apply_style("slides")
    fig = build_fn()
    w, h = SLIDE_SIZE
    fig.set_size_inches(w, w * aspect)
    written = {}
    for ext in ("pdf", "png"):
        p = slides_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi)
        written[f"slides_{ext}"] = p
    plt.close(fig)
    apply_style("poster")
    return written
