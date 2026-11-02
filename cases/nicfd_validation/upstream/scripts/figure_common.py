"""
figure_common.py — shared style, loading and panel drawing for the paper figures.

The two figure scripts in this folder, `plot_two_component.py` and
`plot_phase_change.py`, each render one column of the paper's side-by-side
comparison.  The two columns share their vertical geometry and only the left column
carries y-axis titles.  

The script reads the archived CSVs under `data/`.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nozzle_silhouette import add_nozzle

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGURES = REPO / "figures"

# ── Journal style ────────────────────────────────────────────────────────────
STYLE = {
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   10,
    "legend.fontsize":  9.0,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.8,
    "legend.frameon":   False,
    "figure.dpi":       120,
    "savefig.dpi":      300,
}

# Case colours
CMAP_NAME = "magma"
CMAP_LO, CMAP_HI = 0.20, 0.80
N_COLORS = 3

# ── Fixed panel geometry, in inches ──────────────────────────────────────────
FIG_W = 7.0
M_TOP = 0.09
M_BOT_LBL = 0.48        # bottom margin with x ticks, labels and the "x [m]" title
M_BOT_BARE = 0.09       # bottom margin with x ticks only
DATA_H = 2.05           # data-area height, identical for every panel

# Nozzle silhouette: False = schematic band
NOZZLE_TRUE_SCALE = False
SHOW_NOZZLE = True


@dataclass
class Column:
    """One column of the comparison figure."""
    prefix: str                       # output file-name prefix
    family: str                       # sub-directory under data/
    cases: list[tuple[str, str]]      # (regime directory, legend label)
    liquid_label: str                 # legend entry for u_l
    gas_label: str                    # legend entry for u_g
    alpha_ylim: tuple[float, float]
    quality_ylim: tuple[float, float]
    m_left: float
    m_right: float
    left_ylabels: bool                # y-axis titles on the left (left column)
    quality_right_label: bool         # "quality" title on the right (right column)
    void_legend_loc: str = "best"
    colors: list = field(default_factory=list)

    def __post_init__(self):
        cmap = plt.get_cmap(CMAP_NAME)
        self.colors = [cmap(t) for t in np.linspace(CMAP_LO, CMAP_HI, N_COLORS)]


def load_case(family: str, regime: str, label: str) -> dict:
    """Read one archived case.  Plain numpy: no solver, no property library."""
    path = DATA / family / regime / "solution.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing archived case: {path}")
    D = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "name": f"{family}/{regime}",
        "label": label,
        "x": np.asarray(D["x"]),
        "r": np.asarray(D["r"]),
        "p": np.asarray(D["p"]),
        "alpha_g": np.asarray(D["alpha_g"]),
        "u_g": np.asarray(D["u_g"]),
        "u_l": np.asarray(D["u_l"]),
        "quality": np.asarray(D["quality_flow"]),
        "Ma": np.asarray(D["Ma_mix"]),
    }


# ── Per-panel drawing
def draw_pressure(ax, col: Column, cases):
    for c, colour in zip(cases, col.colors):
        ax.plot(c["x"], c["p"] / 1e5, color=colour, ls="-", lw=1.9, zorder=2)
    if col.left_ylabels:
        ax.set_ylabel(r"pressure $p$ [bar]")
    ax.legend(handles=[Line2D([], [], color=colour, ls="-", lw=1.9,
                              label=c["label"])
                       for c, colour in zip(cases, col.colors)], loc="best")


def draw_velocity(ax, col: Column, cases):
    for c, colour in zip(cases, col.colors):
        ax.plot(c["x"], c["u_l"], color=colour, ls="--", lw=1.9, zorder=2)
        ax.plot(c["x"], c["u_g"], color=colour, ls="-", lw=1.4, zorder=2)
    if col.left_ylabels:
        ax.set_ylabel(r"phase velocity [m/s]")
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="--", lw=1.9, label=col.liquid_label),
        Line2D([], [], color="0.3", ls="-", lw=1.4, label=col.gas_label)],
        loc="best")


def draw_mach(ax, col: Column, cases):
    for c, colour in zip(cases, col.colors):
        ax.plot(c["x"], c["Ma"], color=colour, ls="-", lw=1.9, zorder=2)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)          # sonic line
    ax.text(0.012, 1.0, r"$M=1$", transform=ax.get_yaxis_transform(),
            color="0.4", fontsize=9, ha="left", va="bottom")
    if col.left_ylabels:
        ax.set_ylabel(r"mixture Mach [-]")


def draw_void_quality(ax, col: Column, cases):
    """Void fraction on the left axis, quality on a right twin.

    Both axes carry numbers so each column can be read on its own. Which side
    carries the title depends on where the column sits on the page.
    """
    axq = ax.twinx()
    for c, colour in zip(cases, col.colors):
        ax.plot(c["x"], c["alpha_g"], color=colour, ls="-", lw=1.9, zorder=2)
        axq.plot(c["x"], c["quality"], color=colour, ls="--", lw=1.4, zorder=2)
    ax.set_ylim(*col.alpha_ylim)
    axq.set_ylim(*col.quality_ylim)
    ax.tick_params(labelleft=True)
    axq.tick_params(labelright=True)
    if col.left_ylabels:
        ax.set_ylabel(r"void fraction $\alpha_g$ [-]")
    if col.quality_right_label:
        axq.set_ylabel(r"quality x [-]")
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="-", lw=1.9, label=r"$\alpha_g$ (void)"),
        Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$x$ (quality)")],
        loc=col.void_legend_loc)
    return axq


# Panel order, top to bottom.  The bottom one carries the x-axis label.
PANELS = [
    ("pressure",     draw_pressure,      False),
    ("velocity",     draw_velocity,      False),
    ("mach",         draw_mach,          False),
    ("void_quality", draw_void_quality,  True),
]


def render(col: Column, layout: str = "separate", out_dir: pathlib.Path = None):
    """Render one column.  Returns the list of files written."""
    plt.rcParams.update(STYLE)
    out_dir = pathlib.Path(out_dir) if out_dir else FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [load_case(col.family, regime, label) for regime, label in col.cases]
    for c in cases:
        interior = slice(1, -1)
        print(f"  {c['name']:34s} N={c['x'].size:4d}  "
              f"p_out={c['p'][-1]/1e5:7.3f} bar  "
              f"max M={c['Ma'][interior].max():.3f}  "
              f"alpha_g {c['alpha_g'][interior].min():.3f}.."
              f"{c['alpha_g'][interior].max():.3f}")

    nz = max(cases, key=lambda c: float(c["x"][-1]))        # widest domain
    xlim = (min(c["x"].min() for c in cases),
            max(c["x"].max() for c in cases))

    def finish_x(ax, bottom):
        ax.set_xlim(*xlim)
        if bottom:
            ax.set_xlabel(r"$x$ [m]")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    written = []
    if layout == "separate":
        for key, draw, bottom in PANELS:
            mb = M_BOT_LBL if bottom else M_BOT_BARE
            figh = DATA_H + M_TOP + mb
            fig = plt.figure(figsize=(FIG_W, figh))
            ax = fig.add_axes([col.m_left / FIG_W, mb / figh,
                               (FIG_W - col.m_left - col.m_right) / FIG_W,
                               DATA_H / figh])
            draw(ax, col, cases)
            if SHOW_NOZZLE:
                add_nozzle(ax, nz["x"], nz["r"], true_scale=NOZZLE_TRUE_SCALE)
            finish_x(ax, bottom)
            out = out_dir / f"{col.prefix}_{key}.png"
            fig.savefig(out, bbox_inches=None)   # keep the fixed geometry
            plt.close(fig)
            written.append(out)
            print(f"  wrote {out.relative_to(REPO)}")

    elif layout == "combined":
        fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.0, 10.6), sharex=True)
        for ax, (key, draw, bottom) in zip(axes, PANELS):
            draw(ax, col, cases)
            if SHOW_NOZZLE:
                add_nozzle(ax, nz["x"], nz["r"], true_scale=NOZZLE_TRUE_SCALE)
            finish_x(ax, bottom)
        fig.align_ylabels(axes)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.09)
        out = out_dir / f"{col.prefix}_stacked.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out.relative_to(REPO)}")

    else:
        raise SystemExit(f"unknown layout {layout!r} (use 'separate' or 'combined')")

    return written


def add_layout_arg(parser):
    """The one command-line option both figure scripts share."""
    parser.add_argument("--layout", choices=("separate", "combined"),
                        default="separate",
                        help="one image per panel (default, for LaTeX "
                             "subfigures) or all four in one figure")
    return parser
