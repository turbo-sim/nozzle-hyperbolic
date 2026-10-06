"""
plot_phase_change_stacked.py

Four quantity panels for the ONE-COMPONENT flashing SMD nozzle (water: metastable
liquid + vapour, mass transfer ON), meant to sit as the RIGHT-HAND column of a
two-column comparison next to plot_two_component_stacked.py:

  1. pressure  p
  2. phase velocities  u_l (dashed) and u_g (solid)
  3. mixture Mach number  M  (Stadtke), with the sonic line M = 1
  4. void fraction  alpha_g  (solid, left)  and  quality x  (dashed, right)

SHOW_ONSET (below) puts ONE vertical line on every panel at the FLASHING ONSET —
the first x where the metastable liquid crosses the saturation line.  Same
criterion, and the same single-line reasoning, as plot_phase_change_journal.py.

Because this is the right column, it carries NO left-axis titles (the left,
two-component column already names each quantity on the same row); the ONLY
y-axis title is "quality" on the right of the bottom panel.  The x-axis label
appears only on that bottom (void/quality) panel.  The vertical geometry matches
plot_two_component_stacked.py exactly, so the rows line up side by side.

By default each panel is its own image (LAYOUT="separate"); set "combined" for a
single stacked figure.  Derived from plot_two_component_stacked.py — same Stadtke
mixture Mach and quality x = alpha_g rho_g / (alpha_g rho_g + alpha_l rho_l).

Run:  poetry run python plot_phase_change_stacked.py
"""
import pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import jaxprop as jxp

from case_study.nicfd.nicfd_figures.config_loader import load_config
from case_study.nicfd.nicfd_figures.nozzle_silhouette import add_nozzle
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    phase_props, common_quantities, eigenstructure)

# ── KNOBS: the cases to overlay (path relative to dev_scripts, label) ──────────
_REPO = "results_two_phase"
RUNS = [
    (f"{_REPO}/smd_supersonic_N600_muscl_van_albada_20260725_173528", "subsonic"),
    (f"results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260903_174651", "supersonic-shock"),
    (f"results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260913_125921", "supersonic-adapted"),
    # (f"{_REPO}/smd_supersonic_N420_muscl_van_albada_20260723_090657", "supersonic-shock"),
    # (f"{_REPO}/smd_supersonic_N420_muscl_van_albada_20260723_134102", "supersonic-adapted"),
]
OUT_DIR = "nicfd_figures"    # where the figures are written
PREFIX  = "phase_change"     # output file name prefix

# LAYOUT: "separate" = one image per panel; "combined" = single stacked figure.
LAYOUT = "separate"

# RIGHT-COLUMN labelling (this figure sits to the RIGHT of the two-component one):
LEFT_YLABELS        = True   # False = no left-axis titles (left column names them)
QUALITY_RIGHT_LABEL = True   # True  = label the quality axis on the right

# case colours (match plot_phase_change_journal.py: magma, 3 colours)
CMAP_NAME = "magma"
N_COLORS  = 3
CMAP_LO, CMAP_HI = 0.20, 0.80
# Nozzle silhouette proportion (see nozzle_silhouette.py).
NOZZLE_TRUE_SCALE = False
SHOW_NOZZLE = True           # grey nozzle silhouette behind every panel

# ── Phase-change (flashing) ONSET marker ────────────────────────────────
# ONE vertical line, on EVERY panel, at the first x where the LIQUID crosses the
# saturation line — where the metastable liquid turns superheated w.r.t. the
# local pressure and evaporation switches on.  The test mirrors the solver's own
# evaporation gate (sources.evaporation_weight):
#     T_l > T_sat(p) + dT_gate   AND   h_l > h_sat_l(p) + dh_gate
# with T_sat / h_sat_l read from the LIQUID table's saturation curve via the same
# fluid_l.get_state_saturation(p) call the solver makes — so the marker lands
# exactly where the scheme itself starts transferring mass.  Only the FIRST
# crossing is marked: downstream the liquid can re-subcool (shock recompression)
# and cross back, which is not an onset.
#
# ONE LINE, NOT ONE PER CASE: the cases differ only in back pressure, so they
# share the inlet state and are identical up to the onset (measured spread
# ~1e-4 m, a fraction of a cell).  Per-case lines would overdraw each other and
# invite the reader to look for a difference that is not there.  The line is
# drawn at the MEAN of the per-case onsets, in neutral grey; the spread is
# printed, and exceeding ONSET_SPREAD_TOL warns that one line has stopped being
# an honest summary (each case's own onset is printed above it).
SHOW_ONSET      = True
ONSET_CRITERION = "both"    # "T" = T_l > T_sat | "h" = h_l > h_sat_l | "both" (solver gate)
ONSET_DT_GATE   = 1.0e-3    # [K]    deadband, = sources.evaporation_weight dT_gate
ONSET_DH_GATE   = 1.0       # [J/kg] deadband, = sources.evaporation_weight dh_gate
ONSET_ANNOTATE  = False     # True = rotated "flashing onset" text at the line, on
                            # EVERY panel (loud in a 4-panel column; the legend
                            # entry below is the quieter cue)
ONSET_LEGEND_PANEL = "velocity"   # which panel's legend carries the onset entry:
                                  # "velocity" | "void_quality" | None.  The line
                                  # itself is drawn on all four either way.
ONSET_SPREAD_TOL = 5.0e-3   # [m] case-to-case scatter above which one line lies
ONSET_COLOR = "0.35"              # neutral grey: a shared feature, not a case
ONSET_LS    = (0, (5, 2, 1, 2))   # dash-dot: distinct from every data linestyle
ONSET_LW    = 1.2
ONSET_ALPHA = 0.95

# Quality definition.  True = FLOW quality from the void-fraction / slip relation
#   alpha_g = [1 + (1-x)/x · (rho_g/rho_l) · S]^-1 ,  S = u_g/u_l  (slip ratio),
#   inverted →  x = alpha_g rho_g u_g / (alpha_g rho_g u_g + alpha_l rho_l u_l)
#   (= vapour mass-FLUX fraction).  False = static mass fraction (sets S = 1).
QUALITY_SLIP = True

# Void-fraction and quality axis ranges for THIS column.  Set them EQUAL to the
# two-component script's to SHARE one scale (then number only the outer edges);
# set them DIFFERENTLY to give each case its own scale (number both).
ALPHA_YLIM   = (0.0, 1.0)     # void fraction axis (this column)
QUALITY_YLIM = (0.0, 0.6)     # quality axis (this column)

# ── Fixed geometry (inches).  FIG_W and the VERTICAL margins (M_TOP / M_BOT_* /
# DATA_H) MUST MATCH plot_two_component_stacked.py so the rows line up side by
# side; the L/R margins differ (no left title → small M_LEFT; quality label on
# the right → larger M_RIGHT).
FIG_W    = 7.0
M_LEFT   = 0.58            # left margin: y TICKS only (no left title)
M_RIGHT  = 0.95            # right margin: quality ticks + "quality" label
M_TOP    = 0.09
M_BOT_LBL = 0.48
M_BOT_BARE = 0.09
DATA_H   = 2.05
# ──────────────────────────────────────────────────────────────────────────────

# ── Journal style ─────────────────────────────────────────────────────────────
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
_here = _here.parent    # RUNS and OUT_DIR are relative to dev_scripts/, not to
                        # this nicfd_figures/ subfolder (same hop as
                        # plot_phase_change_journal.py)


def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"])
    return jxp.FluidBicubic(**lut)


def _first_crossing(x, g):
    """x where the margin g first turns positive — linear interp between cells.

    g < 0 upstream (subcooled / at saturation), g > 0 once the liquid is
    superheated.  Returns None if the margin never turns positive.
    """
    pos = np.flatnonzero(np.asarray(g) > 0.0)
    if pos.size == 0:
        return None                       # never flashes inside the domain
    i = int(pos[0])
    if i == 0:
        return float(x[0])                # already superheated at the inlet
    g0, g1 = float(g[i - 1]), float(g[i])
    t = g0 / (g0 - g1)                    # g0 <= 0 < g1  ->  t in [0, 1)
    return float(x[i - 1] + t * (x[i] - x[i - 1]))


def _onset_x(fluid_l, x, p, T_l, h_l):
    """x of the flashing onset (first saturation crossing), or None."""
    if not SHOW_ONSET:
        return None
    try:
        sat = fluid_l.get_state_saturation(jnp.asarray(p))
        T_sat = np.asarray(sat["temperature"], float)
        h_sat = np.asarray(sat["enthalpy"],    float)
    except (AttributeError, KeyError, TypeError):
        return None        # no saturation curve (e.g. a perfect-gas "liquid")
    g_T = np.asarray(T_l, float) - T_sat - ONSET_DT_GATE
    g_h = np.asarray(h_l, float) - h_sat - ONSET_DH_GATE
    if ONSET_CRITERION == "T":
        g = g_T
    elif ONSET_CRITERION == "h":
        g = g_h
    else:
        # smooth AND, as in the solver gate: the BINDING (least-activated)
        # criterion decides.  Each margin is normalised by its own span purely so
        # that K and J/kg are comparable — that leaves both zero crossings exactly
        # where they were, so the combined onset is the LATER of the two.
        s_T = float(np.max(np.abs(g_T))) or 1.0
        s_h = float(np.max(np.abs(g_h))) or 1.0
        g = np.minimum(g_T / s_T, g_h / s_h)
    return _first_crossing(np.asarray(x, float), g)


def onset_consensus(cases):
    """The column's single onset x = mean of the per-case onsets (or None).

    Prints the case-to-case spread, and warns past ONSET_SPREAD_TOL — the point
    at which collapsing the cases onto one line would misrepresent them.
    """
    xs = [c["x_onset"] for c in cases if c.get("x_onset") is not None]
    if not xs:
        return None
    x0 = float(np.mean(xs))
    spread = float(np.max(xs) - np.min(xs))
    print(f"Flashing onset: x = {x0:.4f} m   "
          f"(spread {spread:.1e} m over {len(xs)} case"
          f"{'s' if len(xs) > 1 else ''})")
    if spread > ONSET_SPREAD_TOL:
        print(f"  WARNING: the cases' onsets scatter by {spread:.3e} m "
              f"(> ONSET_SPREAD_TOL = {ONSET_SPREAD_TOL:g} m) — a single line no "
              f"longer represents them all; see the per-case values above.")
    return x0


def add_onset(ax, xo, annotate=ONSET_ANNOTATE):
    """The single vertical flashing-onset line, behind the data (zorder 1.5)."""
    if xo is None:
        return False
    ax.axvline(xo, color=ONSET_COLOR, ls=ONSET_LS, lw=ONSET_LW,
               alpha=ONSET_ALPHA, zorder=1.5)
    if annotate:
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate("flashing onset", xy=(xo, 0.98), xycoords=tr,
                    xytext=(-4, 0), textcoords="offset points",
                    rotation=90, rotation_mode="anchor", ha="right", va="bottom",
                    fontsize=8.5, color=ONSET_COLOR)
    return True


def _onset_handle(panel_key):
    """Legend handle for the onset line, on the ONE panel that carries it."""
    if not SHOW_ONSET or ONSET_LEGEND_PANEL != panel_key:
        return []
    return [Line2D([], [], color=ONSET_COLOR, ls=ONSET_LS, lw=ONSET_LW,
                   label="flashing onset")]


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
    T_l = np.asarray(D["T_l"])
    r   = np.sqrt(np.asarray(D["A"], float) / np.pi)

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
    x_on = _onset_x(fluid_l, x, p, T_l, h_l)
    print(f"  {run_dir.name}:  N={x.size}  p_out={p[-1]/1e5:.2f} bar"
          f"  max M={Ma[1:-1].max():.3f}  "
          f"alpha_g {ag.min():.3f}..{ag.max():.3f}  quality {quality.min():.3e}..{quality.max():.3e}"
          f"  ({'supersonic' if Ma[1:-1].max() > 1 else 'subsonic'})"
          + (f"  flashing onset x={x_on:.4f} m" if x_on is not None else
             ("  no saturation crossing" if SHOW_ONSET else "")))
    return dict(name=run_dir.name, x=x, p=p, alpha_g=ag, u_g=u_g, u_l=u_l,
                quality=quality, Ma=Ma, r=r, x_onset=x_on)


# ── Per-panel drawing (right-column labelling) ────────────────────────────────
# The onset line is a vertical near the left edge (x/L ~ 0.07), which runs
# straight through a left-placed legend or label — legend.frameon is False
# globally, so the line shows between the handles.  An OPAQUE WHITE patch with no
# edge masks it and is invisible everywhere else (white on white), so the
# frameless look is unchanged.  Same trick for the M=1 label's bbox below.
_LEGEND_KW = dict(frameon=True, facecolor="white", edgecolor="none",
                  framealpha=1.0)


def draw_pressure(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["p"] / 1e5, color=col, ls="-", lw=1.9, zorder=2)
    if LEFT_YLABELS:
        ax.set_ylabel(r"pressure $p$ [bar]")
    ax.legend(handles=[Line2D([], [], color=col, ls="-", lw=1.9, label=c["label"])
                       for c, col in zip(cases, CASE_COLORS)], loc="best",
              **_LEGEND_KW)


def draw_velocity(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["u_l"], color=col, ls="--", lw=1.9, zorder=2)
        ax.plot(c["x"], c["u_g"], color=col, ls="-",  lw=1.4, zorder=2)
    if LEFT_YLABELS:
        ax.set_ylabel(r"phase velocity [m/s]")
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="--",  lw=1.9, label=r"$u_\ell$ (liquid)"),
        Line2D([], [], color="0.3", ls="-", lw=1.4, label=r"$u_g$ (vapour)")]
        + _onset_handle("velocity"),
        loc="best", **_LEGEND_KW)


def draw_mach(ax, cases):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["Ma"], color=col, ls="-", lw=1.9, zorder=2)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.text(0.012, 1.0, r"$M=1$", transform=ax.get_yaxis_transform(),
            color="0.4", fontsize=9, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    if LEFT_YLABELS:
        ax.set_ylabel(r"mixture Mach [-]")


def draw_void_quality(ax, cases):
    """void fraction (left, numbered so this column's alpha scale is readable
    directly) + quality (right, numbered + labelled)."""
    axq = ax.twinx()
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"],  c["alpha_g"], color=col, ls="-",  lw=1.9, zorder=2)
        axq.plot(c["x"], c["quality"], color=col, ls="--", lw=1.4, zorder=2)
    ax.set_ylim(*ALPHA_YLIM)                       # void-fraction scale (this column)
    axq.set_ylim(*QUALITY_YLIM)                    # quality scale (this column)
    ax.tick_params(labelleft=True)                 # LEFT (alpha): numbers shown
    if LEFT_YLABELS:
        ax.set_ylabel(r"void fraction $\alpha_g$ [-]")
    if QUALITY_RIGHT_LABEL:
        axq.set_ylabel(r"quality x [-]")   # RIGHT: numbered + labelled
    ax.legend(handles=[
        Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$\alpha_g$ (void)"),
        Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$x$ (quality)")]
        + _onset_handle("void_quality"),
        loc="upper left", **_LEGEND_KW)
    return axq


# panel registry: (key, draw-fn, is_bottom)  — order = top→bottom of the column
PANELS = [
    ("pressure",      draw_pressure,     False),
    ("velocity",      draw_velocity,     False),
    ("mach",          draw_mach,         False),
    ("void_quality",  draw_void_quality, True),
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

_nz  = max(cases, key=lambda c: float(c["x"][-1]))
XLIM = (min(c["x"].min() for c in cases), max(c["x"].max() for c in cases))
X_ONSET = onset_consensus(cases)   # one flashing-onset x for the whole column


def _finish_x(ax, bottom):
    ax.set_xlim(*XLIM)
    if bottom:
        ax.set_xlabel(r"$x$ [m]")
    else:
        ax.tick_params(axis="x", labelbottom=False)


if LAYOUT == "separate":
    for key, draw, bottom in PANELS:
        mb   = M_BOT_LBL if bottom else M_BOT_BARE
        figh = DATA_H + M_TOP + mb
        fig  = plt.figure(figsize=(FIG_W, figh))
        ax   = fig.add_axes([M_LEFT / FIG_W, mb / figh,
                             (FIG_W - M_LEFT - M_RIGHT) / FIG_W, DATA_H / figh])
        draw(ax, cases)
        add_onset(ax, X_ONSET)
        if SHOW_NOZZLE:
            add_nozzle(ax, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
        _finish_x(ax, bottom)
        out = out_dir / f"{PREFIX}_{key}.png"
        fig.savefig(out, bbox_inches=None)
        print(f"Plot saved: {out}")

elif LAYOUT == "combined":
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.0, 10.6), sharex=True)
    for ax, (key, draw, bottom) in zip(axes, PANELS):
        draw(ax, cases)
        add_onset(ax, X_ONSET)
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
