"""
plot_two_component_stacked.py

Four quantity panels for the TWO-COMPONENT nozzle (nitrogen gas + liquid water,
no mass transfer), meant to sit in a single shared-x COLUMN:

  1. pressure  p
  2. phase velocities  u_l (dashed) and u_g (solid)
  3. mixture Mach number  M  (Stadtke), with the sonic line M = 1
  4. void fraction  alpha_g  (solid, left)  and  quality x  (dashed, right)

By default each panel is written as its OWN image (LAYOUT="separate"), all with
the SAME width, the SAME x-limits, and the SAME data-area position, so that when
they are stacked vertically (e.g. LaTeX subfigures) they line up as one column.
Only the bottom panel (void/quality) carries the x-axis tick labels and the
"x [m]" label; the upper three show the x ticks without labels.  Set
LAYOUT="combined" to instead get the four panels in a single stacked figure.

Derived from plot_two_component_journal.py — same loading, journal style, and
Stadtke mixture Mach  M = |u_hat| / a,  a = 1/2(lambda_3 - lambda_4).  Quality is
the vapour mass fraction  x = alpha_g rho_g / (alpha_g rho_g + alpha_l rho_l).

Run:  poetry run python plot_two_component_stacked.py
"""
import pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jaxprop as jxp

from case_study.nicfd.nicfd_figures.config_loader import load_config
from case_study.nicfd.nicfd_figures.nozzle_silhouette import add_nozzle
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    phase_props, common_quantities, eigenstructure)

# ── KNOBS: the cases to overlay (path relative to dev_scripts, label) ──────────
_REPO = "results_two_phase"
RUNS = [
    (f"{_REPO}/two_component_N500_muscl_van_albada_20260727_105958", "subsonic"),
    (f"results/nozzle_valid_gasdom_N500_muscl_van_albada_20260722_115901", "supersonic-shock"),
    (f"{_REPO}/two_component_N500_muscl_van_albada_20260726_115923", "supersonic-adapted"),
]
OUT_DIR = "nicfd_figures"    # where the figures are written
PREFIX  = "two_component"    # output file name prefix

# LAYOUT: "separate" = one image per panel (stack them into a column yourself);
#         "combined" = all four panels in a single shared-x figure.
LAYOUT = "separate"

# case colours are sampled from a colormap
CMAP_NAME = "magma"
N_COLORS  = 3
CMAP_LO, CMAP_HI = 0.20, 0.80
# Nozzle silhouette proportion (see nozzle_silhouette.py).  False = schematic
# 16% band; True = proportion-true (only for a long-and-thin real geometry).
NOZZLE_TRUE_SCALE = False
SHOW_NOZZLE = True           # grey nozzle silhouette behind every panel

# Quality definition.  True = FLOW quality from the void-fraction / slip relation
#   alpha_g = [1 + (1-x)/x · (rho_g/rho_l) · S]^-1 ,  S = u_g/u_l  (slip ratio),
#   inverted →  x = alpha_g rho_g u_g / (alpha_g rho_g u_g + alpha_l rho_l u_l)
#   (= vapour mass-FLUX fraction).  False = static mass fraction (sets S = 1).
QUALITY_SLIP = True

# Void-fraction and quality axis ranges for THIS column.  Set them EQUAL to the
# phase-change script's to SHARE one scale (then number only the outer edges);
# set them DIFFERENTLY to give each case its own scale (number both).
ALPHA_YLIM   = (0.9, 1.0)     # void fraction axis (this column)
QUALITY_YLIM = (0.0, 0.6)     # quality axis (this column)

# ── Fixed geometry for LAYOUT="separate" (inches).  FIG_W and the VERTICAL
# margins (M_TOP / M_BOT_* / DATA_H) must MATCH plot_phase_change_stacked.py so
# the two columns' rows line up; the L/R margins may differ per column (tuned to
# the ticks/labels each side actually uses, to keep the white space small).
FIG_W    = 7.0
M_LEFT   = 0.85             # left margin: y label + ticks (numbered)
M_RIGHT  = 0.55            # right margin: room for the quality (right-axis) numbers
M_TOP    = 0.09
M_BOT_LBL = 0.48           # bottom margin WITH x ticks+labels+"x [m]"
M_BOT_BARE = 0.09          # bottom margin with x ticks only (no labels)
DATA_H   = 2.05            # data-area height (same for every panel)
# ──────────────────────────────────────────────────────────────────────────────

# ── Journal style (matches plot_two_component_journal.py) ─────────────────────
plt.rcParams.update({
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
})
_cmap = plt.get_cmap(CMAP_NAME)
CASE_COLORS = [_cmap(t) for t in np.linspace(CMAP_LO, CMAP_HI, N_COLORS)]
_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")
_here = pathlib.Path(__file__).parent


def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"])
    return jxp.FluidBicubic(**lut)


def load_case(rel_path):
    """Return dict with x, p, alpha_g, u_g, u_l, quality, Mach (Stadtke), r."""
    run_dir = (_here / rel_path) if not pathlib.Path(rel_path).is_absolute() \
        else pathlib.Path(rel_path)
    if not (run_dir / "solution.csv").exists():
        print(f"  SKIP (no solution.csv): {run_dir}")
        return None
    yml = sorted(run_dir.glob("settings*.yaml"))[0]
    cfg = load_config(yml)
    fluid_g, fluid_l = _build_fluid(cfg.LUT_G), _build_fluid(cfg.LUT_L)
    k = float(cfg.K_VM)

    D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    cols = D.dtype.names
    x   = np.asarray(D["x"])
    p   = np.asarray(D["p"]);  ag = np.asarray(D["alpha_g"]);  al = 1.0 - ag
    u_g = np.asarray(D["u_g"]); u_l = np.asarray(D["u_l"])
    h_g = np.asarray(D["h_g"]); h_l = np.asarray(D["h_l"])
    r   = np.sqrt(np.asarray(D["A"], float) / np.pi)   # silhouette radius

    if "rho_g" in cols and "rho_l" in cols:
        rho_g, rho_l = np.asarray(D["rho_g"]), np.asarray(D["rho_l"])
    else:
        rho_g = np.asarray(fluid_g.get_state(jxp.HmassP_INPUTS,
                                             jnp.asarray(h_g), jnp.asarray(p))["rho"])
        rho_l = np.asarray(fluid_l.get_state(jxp.HmassP_INPUTS,
                                             jnp.asarray(h_l), jnp.asarray(p))["rho"])
    # vapour quality from the void-fraction / slip relation (see QUALITY_SLIP)
    if QUALITY_SLIP:                                    # FLOW quality, slip S = u_g/u_l
        g_flux  = ag * rho_g * u_g
        quality = g_flux / (g_flux + al * rho_l * u_l + 1e-300)
    else:                                               # static mass fraction (S = 1)
        quality = (ag * rho_g) / (ag * rho_g + al * rho_l)

    U_h = jnp.asarray(np.column_stack([p, u_g, u_l, ag, h_g, h_l]))

    def _mach_one(uh):
        eg = phase_props(fluid_g, uh[0], uh[4])
        el = phase_props(fluid_l, uh[0], uh[5])
        cq = common_quantities(uh, eg, el, k)
        ev = eigenstructure(uh, eg, el, cq)[0]
        return jnp.abs(0.5 * (ev[2] + ev[3])) / (0.5 * (ev[2] - ev[3]) + 1e-300)

    Ma = np.asarray(jax.vmap(_mach_one)(U_h))
    print(f"  {run_dir.name}:  N={x.size}  p_out={p[-1]/1e5:.2f} bar"
          f"  max M={Ma[1:-1].max():.3f}  "
          f"alpha_g {ag.min():.3f}..{ag.max():.3f}  quality {quality.min():.3e}..{quality.max():.3e}"
          f"  ({'supersonic' if Ma[1:-1].max() > 1 else 'subsonic'})")
    return dict(name=run_dir.name, x=x, p=p, alpha_g=ag, u_g=u_g, u_l=u_l,
                quality=quality, Ma=Ma, r=r)


# ── Per-panel drawing (each takes a primary Axes; caller owns the x-axis) ──────
def draw_pressure(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["p"] / 1e5, color=col, ls="-", lw=1.9, zorder=2)
    ax.set_ylabel(r"pressure $p$ [bar]")
    ax.legend(handles=[Line2D([], [], color=col, ls="-", lw=1.9, label=c["label"])
                       for c, col in zip(cases, CASE_COLORS)], loc="best")


def draw_velocity(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["u_l"], color=col, ls="--",  lw=1.9, zorder=2)
        ax.plot(c["x"], c["u_g"], color=col, ls="-", lw=1.4, zorder=2)
    ax.set_ylabel(r"phase velocity [m/s]")
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="--",  lw=1.9, label=r"$u_\ell$ (water)"),
        Line2D([], [], color="0.3", ls="-", lw=1.4, label=r"$u_g$ (nitrogen)")],
        loc="best")


def draw_mach(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["Ma"], color=col, ls="-", lw=1.9, zorder=2)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)          # sonic line
    ax.text(0.012, 1.0, r"$M=1$", transform=ax.get_yaxis_transform(),
            color="0.4", fontsize=9, ha="left", va="bottom")
    ax.set_ylabel(r"mixture Mach [-]")


def draw_void_quality(ax, cases):
    """void fraction (left, numbered + labelled) + quality (right twin, numbered
    so this column's quality scale is readable directly)."""
    axq = ax.twinx()
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"],  c["alpha_g"], color=col, ls="-",  lw=1.9, zorder=2)
        axq.plot(c["x"], c["quality"], color=col, ls="--", lw=1.4, zorder=2)
    ax.set_ylim(*ALPHA_YLIM)                       # void-fraction scale (this column)
    axq.set_ylim(*QUALITY_YLIM)                    # quality scale (this column)
    ax.set_ylabel(r"void fraction $\alpha_g$ [-]")  # LEFT: numbered + labelled
    axq.tick_params(labelright=True)              # RIGHT (quality): numbers shown
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$\alpha_g$ (void)"),
        Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$x$ (quality)")],
        loc="best")
    return axq


# panel registry: (key, draw-fn, is_bottom)  — order = top→bottom of the column
PANELS = [
    ("pressure",      draw_pressure,     False),
    ("velocity",      draw_velocity,     False),
    ("mach",          draw_mach,         False),
    ("void_quality",  draw_void_quality, True),    # bottom → carries the x label
]


# ── Load every case ───────────────────────────────────────────────────────────
print("Loading cases:")
cases = []
for rel, label in RUNS:
    c = load_case(rel)
    if c is not None:
        c["label"] = label
        cases.append(c)
if not cases:
    sys.exit("No cases loaded — check the RUNS paths.")
out_dir = (_here / OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

_nz  = max(cases, key=lambda c: float(c["x"][-1]))         # widest domain
XLIM = (min(c["x"].min() for c in cases),
        max(c["x"].max() for c in cases))                  # shared x-range


def _finish_x(ax, bottom):
    ax.set_xlim(*XLIM)
    if bottom:
        ax.set_xlabel(r"$x$ [m]")
    else:
        ax.tick_params(axis="x", labelbottom=False)        # ticks, no labels


if LAYOUT == "separate":
    # one image per panel, fixed geometry ⇒ they align when stacked in a column
    for key, draw, bottom in PANELS:
        mb   = M_BOT_LBL if bottom else M_BOT_BARE
        figh = DATA_H + M_TOP + mb
        fig  = plt.figure(figsize=(FIG_W, figh))
        ax   = fig.add_axes([M_LEFT / FIG_W, mb / figh,
                             (FIG_W - M_LEFT - M_RIGHT) / FIG_W, DATA_H / figh])
        draw(ax, cases)
        if SHOW_NOZZLE:
            add_nozzle(ax, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
        _finish_x(ax, bottom)
        out = out_dir / f"{PREFIX}_{key}.png"
        fig.savefig(out, bbox_inches=None)                 # keep the fixed geometry
        print(f"Plot saved: {out}")

elif LAYOUT == "combined":
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.0, 10.6), sharex=True)
    for ax, (key, draw, bottom) in zip(axes, PANELS):
        draw(ax, cases)
        if SHOW_NOZZLE:
            add_nozzle(ax, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
        _finish_x(ax, bottom)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.09)
    out = out_dir / f"{PREFIX}_stacked.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Plot saved: {out}")

else:
    sys.exit(f"unknown LAYOUT {LAYOUT!r} (use 'separate' or 'combined')")

plt.show()
