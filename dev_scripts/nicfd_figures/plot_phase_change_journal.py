"""
plot_phase_change_journal.py

Journal-quality figures for the PHASE-CHANGE (flashing) SMD nozzle — the same
layout as plot_bernoulli_journal.py but with NO analytic reference (this is a
presentation of the solver solution, not a verification).  Overlays several
back-pressure regimes on shared axes:

  1. phase_change_p_mach.png  — pressure (solid, left axis) and MIXTURE Mach
                                number (dashed, right axis) vs position x, one
                                colour per case, with the sonic line M = 1.
  2. phase_change_velocity.png — phase velocities u_l (solid) and u_g (dashed)
                                vs position x, one colour per case.

SHOW_ONSET (below) adds ONE vertical line at the FLASHING ONSET — the first x
where the metastable liquid crosses the saturation line, i.e. where it becomes
superheated with respect to the local pressure.  All cases share the inlet state
and are identical upstream of it, so the onset is a single shared feature of the
figure, not a per-case one.

Mach here is the TWO-PHASE mixture Mach M = |u_mix| / a, using the solver's OWN
Städtke mixture sound speed a = ½(λ₃ − λ₄) from eigenstructure() — the speed the
scheme actually chokes on (CoolProp cannot give a two-phase sound speed, and the
single-phase liquid a ~ 1400 m/s would badly misclassify the regime).

Edit the RUNS list below (path, label).  Run:
    poetry run python plot_phase_change_journal.py
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
import yaml
import jaxprop as jxp

from config_loader import load_config
from nozzle_silhouette import add_nozzle
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    phase_props, common_quantities, eigenstructure)

# ── KNOBS: the cases to overlay (path relative to dev_scripts, label) ──────────
_REPO = "results_two_phase"
RUNS = [
    (f"{_REPO}/smd_supersonic_N600_muscl_van_albada_20260725_173528", "subsonic"), #  smd_supersonic_N200_muscl_van_albada_20260724_103127
    (f"results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260903_174651", "supersonic-shock"),  # smd_supersonic_N600_muscl_van_albada_20260903_174651 // smd_supersonic_N420_muscl_van_albada_20260723_090657
    # NOTE: the 'adapted' case was given as the SAME dir as 'shock' (copy-paste);
    # point this at the real supersonic-adapted run to separate the curves.
    (f"results_grid_sesitivity/smd_supersonic_N600_muscl_van_albada_20260913_125921", "supersonic (adapted)"),
]
OUT_DIR = _REPO          # where the overlay figures are written

# case colours are sampled from a colormap: pick the map and how many colours
CMAP_NAME = "magma"    # any matplotlib colormap ("plasma", "turbo", "cividis"…)
N_COLORS  = 3            # how many colours to draw (>= number of cases)
CMAP_LO, CMAP_HI = 0.20, 0.80   # sub-range of the map to sample (avoid washed-out end)
# Nozzle silhouette proportion.  False = schematic 16% band (radius NOT to scale).
# True = proportion-true (1 m in x == 1 m of radius) — sensible for the Elliot
# geometry (r_max/L ~ 0.08); nonsensical for the symmetric placeholder.
NOZZLE_TRUE_SCALE = False

# ── Phase-change (flashing) ONSET marker ────────────────────────────────
# ONE vertical line at the first x where the
# LIQUID crosses the saturation line — where the metastable liquid turns
# superheated w.r.t. the local pressure and evaporation switches on.  The test
# mirrors the solver's own evaporation gate (sources.evaporation_weight):
#     T_l > T_sat(p) + dT_gate   AND   h_l > h_sat_l(p) + dh_gate
# with T_sat / h_sat_l read from the LIQUID table's saturation curve via the same
# fluid_l.get_state_saturation(p) call the solver makes — so the marker lands
# exactly where the scheme itself starts transferring mass, not on a separately
# defined criterion.  Only the FIRST crossing is marked: downstream the liquid can
# re-subcool (shock recompression) and cross back, which is not an onset.
#
# ONE LINE, NOT ONE PER CASE: the cases differ only in back pressure, so they
# share the inlet state and are identical up to the onset (measured spread here
# ~2e-4 m, a fifth of a cell).  Coloured per-case lines would simply overdraw
# each other and invite the reader to look for a difference that is not there.
# The line is drawn at the MEAN of the per-case onsets, in neutral grey; the
# spread is printed, and exceeding ONSET_SPREAD_TOL warns that one line has
# stopped being an honest summary (each case's own onset is printed above it).
SHOW_ONSET      = True
ONSET_CRITERION = "both"    # "T" = T_l > T_sat | "h" = h_l > h_sat_l | "both" (solver gate)
ONSET_DT_GATE   = 1.0e-3    # [K]    deadband, = sources.evaporation_weight dT_gate
ONSET_DH_GATE   = 1.0       # [J/kg] deadband, = sources.evaporation_weight dh_gate
ONSET_ANNOTATE  = False     # True = rotated "flashing onset" text at the line too
                            # (the legend already carries the cue; the text is for
                            #  a stand-alone figure with no legend)
ONSET_SPREAD_TOL = 5.0e-3   # [m] case-to-case scatter above which one line lies
ONSET_COLOR = "0.35"              # neutral grey: a shared feature, not a case
ONSET_LS    = (0, (5, 2, 1, 2))   # dash-dot: distinct from pressure/Mach/velocity
ONSET_LW    = 1.2
ONSET_ALPHA = 0.95
# ──────────────────────────────────────────────────────────────────────────────

# ── Journal style (matches plot_bernoulli_journal.py) ─────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        11,
    "axes.labelsize":   12,
    "axes.titlesize":   10,
    "legend.fontsize":  9.5,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.top":        True,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.8,
    "legend.frameon":   False,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})
# one colour per case, sampled evenly from CMAP_NAME over [CMAP_LO, CMAP_HI]
_cmap = plt.get_cmap(CMAP_NAME)
CASE_COLORS = [_cmap(t) for t in np.linspace(CMAP_LO, CMAP_HI, N_COLORS)]
_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")

_here = pathlib.Path(__file__).parent
_here = _here.parent

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
    """The figure's single onset x = mean of the per-case onsets (or None).

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


def load_case(rel_path):
    """Return dict with x, p, u_g, u_l, Mach (Städtke mixture), plus meta."""
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
    x   = np.asarray(D["x"])
    p   = np.asarray(D["p"]);  ag = np.asarray(D["alpha_g"])
    u_g = np.asarray(D["u_g"]); u_l = np.asarray(D["u_l"])
    h_g = np.asarray(D["h_g"]); h_l = np.asarray(D["h_l"])
    T_l = np.asarray(D["T_l"])

    # nozzle radius r(x) = sqrt(A/pi) for the bottom silhouette
    geo = getattr(_geom, cfg.GEO_FUNCTION)(x, roughness=cfg.GEO_ROUGHNESS,
                                           **cfg.GEO_KWARGS)
    r = np.sqrt(np.asarray(geo.A, float) / np.pi)

    # Städtke mixture sound speed & Mach from the scheme's own eigenstructure.
    U_h = jnp.asarray(np.column_stack([p, u_g, u_l, ag, h_g, h_l]))

    def _mach_one(uh):
        eg = phase_props(fluid_g, uh[0], uh[4])
        el = phase_props(fluid_l, uh[0], uh[5])
        cq = common_quantities(uh, eg, el, k)
        ev = eigenstructure(uh, eg, el, cq)[0]
        a_s = 0.5 * (ev[2] - ev[3])              # mixture sound speed (>0)
        u_s = 0.5 * (ev[2] + ev[3])              # hat-weighted mixture velocity
        return jnp.abs(u_s) / (a_s + 1e-300)

    Ma = np.asarray(jax.vmap(_mach_one)(U_h))
    x_on = _onset_x(fluid_l, x, p, T_l, h_l)
    print(f"  {run_dir.name}:  N={x.size}  max M={Ma[1:-1].max():.3f}"
          f"  ({'supersonic' if Ma[1:-1].max() > 1 else 'subsonic'})"
          + (f"  flashing onset x={x_on:.4f} m" if x_on is not None else
             ("  no saturation crossing" if SHOW_ONSET else "")))
    return dict(name=run_dir.name, x=x, p=p, u_g=u_g, u_l=u_l, Ma=Ma, r=r,
                x_onset=x_on)


# add_nozzle now lives in nozzle_silhouette.py (shared by all journal scripts);
# NOZZLE_TRUE_SCALE (top of file) toggles schematic vs proportion-true.


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

# ── FIGURE 1: pressure (solid, left) + mixture Mach (dashed, right) ───────────
fig1, axp = plt.subplots(figsize=(7.0, 4.3))
axm = axp.twinx()
for c, col in zip(cases, CASE_COLORS):
    axp.plot(c["x"], c["p"] / 1e5, color=col, ls="-",  lw=1.9, zorder=2)
    axm.plot(c["x"], c["Ma"],      color=col, ls="--", lw=1.6, zorder=2)
axm.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)          # sonic line
# flashing onset (on axp: behind the data, in front of the nozzle silhouette)
X_ONSET = onset_consensus(cases)
_onset1 = add_onset(axp, X_ONSET)
# nozzle silhouette (widest-domain case → spans the full x-range)
_nz = max(cases, key=lambda c: float(c["x"][-1]))
add_nozzle(axp, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
axm.annotate(r"$M=1$", xy=(axp.get_xlim()[0], 1.0), xytext=(3, 3),
             textcoords="offset points", color="0.4", fontsize=9,
             ha="left", va="bottom")
axp.set_xlabel(r"$x$ [m]")
axp.set_ylabel(r"pressure $p$ [bar]")
axm.set_ylabel(r"mixture Mach number $M = |u_\mathrm{mix}| / a$")

# two-part legend: colour = case, linestyle = quantity
case_handles  = [Line2D([], [], color=col, ls="-", lw=1.9, label=c["label"])
                 for c, col in zip(cases, CASE_COLORS)]
style_handles = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label="pressure"),
                 Line2D([], [], color="0.3", ls="--", lw=1.6, label="Mach")]
if _onset1:
    style_handles.append(Line2D([], [], color=ONSET_COLOR, ls=ONSET_LS,
                                lw=ONSET_LW, label="flashing onset"))
# Case identity (colours) is shown once, on the velocity figure (Fig. 2).
# Here keep only the minimal quantity cue: solid = pressure, dashed = Mach.
axp.legend(handles=style_handles, loc="upper center")
fig1.tight_layout()
out1 = out_dir / "phase_change_p_mach.png"
fig1.savefig(out1)
print(f"Plot saved: {out1}")

# ── FIGURE 2: phase velocities u_l (solid) + u_g (dashed) ─────────────────────
fig2, axu = plt.subplots(figsize=(7.0, 4.3))

# nozzle radius on a SECONDARY RIGHT AXIS (true mm scale), drawn behind the data.
# The twin is placed below axu (zorder) and axu's patch is hidden so it shows
# through; ylim is set so the nozzle fills the bottom ~30% of the frame.
if NOZZLE_TRUE_SCALE:
    add_nozzle(axu, _nz["x"], _nz["r"], true_scale=True)   # proportion-true
else:
    axr = axu.twinx()
    axr.set_zorder(axu.get_zorder() - 1)
    axu.patch.set_visible(False)
    _rmm = _nz["r"] * 1e3                                   # radius in mm
    # same fill as add_nozzle() in Fig. 1 (colour 0.45 / alpha 0.22), no outline
    axr.fill_between(_nz["x"], 0.0, _rmm, color="0.45", alpha=0.22, lw=0, zorder=0)
    axr.set_ylim(0.0, float(_rmm.max()) / 0.16)            # same height as Fig. 1 (16%)
    axr.set_ylabel("nozzle radius [mm]", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    axr.spines["right"].set_color("0.35")

_onset2 = add_onset(axu, X_ONSET)
for c, col in zip(cases, CASE_COLORS):
    axu.plot(c["x"], c["u_l"], color=col, ls="-",  lw=1.9, zorder=2)
    axu.plot(c["x"], c["u_g"], color=col, ls="--", lw=1.4, zorder=2)
axu.set_xlabel(r"$x$ [m]")
axu.set_ylabel(r"phase velocity [m/s]")
vel_style = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$u_\ell$ (liquid)"),
             Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$u_g$ (vapour)")]
if _onset2:
    vel_style.append(Line2D([], [], color=ONSET_COLOR, ls=ONSET_LS,
                            lw=ONSET_LW, label="flashing onset"))
_leg2 = axu.legend(handles=case_handles, loc="upper left") #, title="case")
axu.add_artist(_leg2)
axu.legend(handles=vel_style, loc="upper center")
fig2.tight_layout()
out2 = out_dir / "phase_change_velocity.png"
fig2.savefig(out2)
print(f"Plot saved: {out2}")

plt.show()
