"""
plot_bubble_number_hem.py

BUBBLE-NUMBER SENSITIVITY of the flashing SMD nozzle (water, one component),
ADAPTED supersonic expansion only, against the HEM (homogeneous-equilibrium)
solution as the infinitely-fast-mass-transfer limit.

N_b sets the interfacial area (a_i ~ N_b^(1/3) alpha^(2/3)), hence the mass-
transfer rate: raising it drives the two-fluid solution toward equilibrium, so
the curves should march toward the HEM as N_b grows.  That march IS the figure.

Panels (same vertical geometry and style as plot_phase_change_stacked.py):

  1. pressure  p
  2. phase velocities  u_l (solid) / u_g (dashed); the HEM has ONE velocity
     (no slip) drawn in its own style
  3. mixture Mach number M, with the sonic line M = 1
  4. void fraction alpha_g (solid, left) and quality x (dashed, right)

CAVEAT on panel 3 — the two Machs are NOT the same quantity.  The two-fluid M
uses the solver's own Stadtke mixture sound speed a = 1/2(lambda_3 - lambda_4)
from eigenstructure(), which carries NO mass transfer in the characteristic
analysis; the HEM M is u/a with the full EQUILIBRIUM sound speed (a collapses to
~86 m/s once the quality lifts off zero).  Each curve is its own model's Mach and
each sonic crossing is that model's thermal throat; read the panel as "where does
each model choke", not as a pointwise comparison of one number.

RUNS / HEM_RUN at the top of the file are the only things to edit.  ANY entry may
be deleted or commented out, and any run directory may be missing: each one is
loaded independently, a missing solution.csv only prints a SKIP line, and the
figure is drawn from whatever survived (HEM alone, or two-fluid alone, both work).
The script only stops if NOTHING loaded.

Run (from dev_scripts/):
    poetry run python nicfd_figures/plot_bubble_number_hem.py
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

from case_study.nicfd.nicfd_figures.config_loader import load_config
from case_study.nicfd.nicfd_figures.nozzle_silhouette import add_nozzle
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    phase_props, common_quantities, eigenstructure)

# ── KNOBS: the runs (paths relative to dev_scripts/) ──────────────────────────
# TWO-FLUID runs — one per bubble number, ADAPTED supersonic expansion, N = 600.
# Label None = read N_b from the run's own settings*.yaml and label it
# "N_b = 10^k m^-3" (so a label can never drift from the data it names); pass a
# string instead to override.  Comment out / delete any line freely.
_BUB = "results_bubble_number_sens"
RUNS = [
    (f"{_BUB}/smd_supersonic_N600_muscl_van_albada_20260915_081202", None),  # N_b = 1e10
    (f"{_BUB}/smd_supersonic_N600_muscl_van_albada_20260913_125921", None),  # N_b = 1e11
    (f"{_BUB}/smd_supersonic_N600_muscl_van_albada_20260915_081252", None),  # N_b = 1e12
]

# HEM reference (single-fluid equilibrium Euler).  Set to None — or comment the
# line out — to drop it from the figure.
# HEM_RUN = ("results/get_nozzle_smd_divergent_smooth_hllc_N500", "HEM (equilibrium)")
HEM_RUN = None

OUT_DIR = "nicfd_figures"        # where the figures are written
PREFIX  = "bubble_number_hem"    # output file-name prefix

# LAYOUT: "combined" = one stacked figure (default — this is a single argument);
#         "separate" = one image per panel, for a journal's multi-part figure.
LAYOUT = "separate"

LEFT_YLABELS        = True    # stand-alone figure: name every quantity on the left
QUALITY_RIGHT_LABEL = True    # label the quality axis on the right

# case colours: magma, as in the sibling phase-change scripts.  Sampled over
# [CMAP_LO, CMAP_HI] in ORDER OF INCREASING N_b (the cases are sorted below), so
# the colour ramp reads as the physical ramp.
CMAP_NAME = "magma"
CMAP_LO, CMAP_HI = 0.20, 0.72

# HEM style: black, densely dotted — deliberately outside the case ramp, because
# it is a DIFFERENT MODEL and not another member of the N_b family.
HEM_COLOR = "k"
HEM_LS    = (0, (1, 1.2))
HEM_LW    = 1.7

NOZZLE_TRUE_SCALE = False
SHOW_NOZZLE = True

# Quality definition (matches plot_phase_change_stacked.py).  True = FLOW quality
#   x = alpha_g rho_g u_g / (alpha_g rho_g u_g + alpha_l rho_l u_l)
# i.e. the vapour mass-FLUX fraction.  False = static mass fraction (slip S = 1).
# The HEM has no slip, so its Qvap is both at once — comparable either way, but
# True is the like-for-like choice.
QUALITY_SLIP = True

ALPHA_YLIM   = (0.0, 1.0)
QUALITY_YLIM = (0.0, 0.4)

# ── Phase-change (flashing) ONSET marker ──────────────────────────────────────
# ONE vertical line at the first x where the TWO-FLUID liquid crosses the
# saturation line, by the solver's own evaporation gate (sources.evaporation_
# weight):  T_l > T_sat(p) + dT_gate  AND  h_l > h_sat_l(p) + dh_gate, with
# T_sat / h_sat_l from fluid_l.get_state_saturation(p).  See
# plot_phase_change_journal.py for the full rationale.
#
# The HEM cannot be gated this way — it is in equilibrium BY CONSTRUCTION, so its
# liquid never superheats (T - T_sat = 0 to machine precision once flashing
# starts) and the criterion above would report "no crossing".  Its onset is taken
# instead as the x where the equilibrium quality leaves zero (in equilibrium the
# two coincide), drawn in the HEM style when ONSET_INCLUDE_HEM.
#
# ONE LINE PER CASE HERE, unlike the sibling phase-change figures.  There the
# cases shared an inlet state and their onsets agreed to ~1e-4 m, so one line was
# honest; N_b MOVES THE ONSET (measured: 0.0740 / 0.0817 / 0.0868 m for
# 1e10 / 1e11 / 1e12, and 0.0890 m for the HEM), so each case gets its own line
# in its own colour.
#
# OFF BY DEFAULT: that whole 15 mm spread is ~1 % of the x-axis, so the lines
# stack into a single smudge at x ~ 0.08 and clutter a figure whose subject is
# what happens DOWNSTREAM.  The onsets are always PRINTED, drawn only on request.
SHOW_ONSET        = True
ONSET_INCLUDE_HEM = True    # also mark the HEM's own onset (quality leaving 0)
ONSET_CRITERION = "both"    # "T" = T_l > T_sat | "h" = h_l > h_sat_l | "both" (solver gate)
ONSET_DT_GATE   = 1.0e-3    # [K]    deadband, = sources.evaporation_weight dT_gate
ONSET_DH_GATE   = 1.0       # [J/kg] deadband, = sources.evaporation_weight dh_gate
ONSET_ANNOTATE  = False     # True = rotated "flashing onset" text at the line
ONSET_LEGEND_PANEL = "velocity"   # "pressure" | "velocity" | "void_quality" | None
ONSET_SPREAD_TOL = 5.0e-3   # [m] case-to-case scatter above which one line lies
ONSET_COLOR = "0.35"
ONSET_LS    = (0, (5, 2, 1, 2))
ONSET_LW    = 1.2
ONSET_ALPHA = 0.95

# ── Fixed geometry (inches) — same vertical rhythm as the sibling stacked script
FIG_W    = 7.0
M_LEFT   = 0.95            # left margin: ticks + y title (stand-alone figure)
M_RIGHT  = 0.95            # right margin: quality ticks + label
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
_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")
_here = pathlib.Path(__file__).parent.parent   # RUNS are relative to dev_scripts/


def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"])
    return jxp.FluidBicubic(**lut)


def _nb_label(nb):
    """1.0e11 -> '$N_b = 10^{11}$ m$^{-3}$' (mantissa kept only if it is not 1)."""
    e = int(np.floor(np.log10(nb)))
    m = nb / 10.0 ** e
    head = rf"10^{{{e}}}" if abs(m - 1.0) < 1e-9 else rf"{m:.1f}\times 10^{{{e}}}"
    return rf"$N_b = {head}$ m$^{{-3}}$"


# ── Flashing-onset helpers (shared logic with the sibling phase-change scripts) ─
def _first_crossing(x, g):
    """x where the margin g first turns positive — linear interp between cells."""
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
    """x of the flashing onset (first saturation crossing), or None.

    Computed unconditionally — SHOW_ONSET only decides whether it is DRAWN; the
    number is printed either way.
    """
    try:
        sat = fluid_l.get_state_saturation(jnp.asarray(p))
        T_sat = np.asarray(sat["temperature"], float)
        h_sat = np.asarray(sat["enthalpy"],    float)
    except (AttributeError, KeyError, TypeError):
        return None        # no saturation curve on this table
    g_T = np.asarray(T_l, float) - T_sat - ONSET_DT_GATE
    g_h = np.asarray(h_l, float) - h_sat - ONSET_DH_GATE
    if ONSET_CRITERION == "T":
        g = g_T
    elif ONSET_CRITERION == "h":
        g = g_h
    else:
        # smooth AND, as in the solver gate: the BINDING criterion decides.  Each
        # margin is normalised by its own span purely so that K and J/kg are
        # comparable — every zero crossing stays put, so the combined onset is
        # simply the LATER of the two.
        s_T = float(np.max(np.abs(g_T))) or 1.0
        s_h = float(np.max(np.abs(g_h))) or 1.0
        g = np.minimum(g_T / s_T, g_h / s_h)
    return _first_crossing(np.asarray(x, float), g)


def report_onsets(cases, hem):
    """Print where each model starts flashing, and how far apart they are.

    Always runs (SHOW_ONSET only governs the drawing).  Returns True if there is
    anything to draw.
    """
    xs = [c["x_onset"] for c in cases if c.get("x_onset") is not None]
    if hem is not None and hem.get("x_onset") is not None and ONSET_INCLUDE_HEM:
        xs.append(hem["x_onset"])
    if not xs:
        return False
    spread = float(np.max(xs) - np.min(xs))
    print(f"Flashing onset: {np.min(xs):.4f} .. {np.max(xs):.4f} m "
          f"(spread {spread*1e3:.1f} mm over {len(xs)} solution"
          f"{'s' if len(xs) > 1 else ''}) — per-case values above")
    if spread > ONSET_SPREAD_TOL:
        print(f"  N_b moves the onset by more than ONSET_SPREAD_TOL "
              f"({ONSET_SPREAD_TOL*1e3:.0f} mm), so the marker is drawn PER CASE, "
              f"not as one shared line.")
    return True


def add_onset(ax, cases, colors, hem, annotate=ONSET_ANNOTATE):
    """One vertical flashing-onset line PER case, behind the data (zorder 1.5).

    Each line takes its case's colour (the HEM's takes the HEM colour), because
    N_b moves the onset — see the knob block.
    """
    if not SHOW_ONSET:
        return False
    drawn = []
    for c, col in zip(cases, colors):
        if c.get("x_onset") is not None:
            ax.axvline(c["x_onset"], color=col, ls=ONSET_LS, lw=ONSET_LW,
                       alpha=ONSET_ALPHA, zorder=1.5)
            drawn.append(c["x_onset"])
    if hem is not None and hem.get("x_onset") is not None and ONSET_INCLUDE_HEM:
        ax.axvline(hem["x_onset"], color=HEM_COLOR, ls=ONSET_LS, lw=ONSET_LW,
                   alpha=ONSET_ALPHA, zorder=1.5)
        drawn.append(hem["x_onset"])
    if drawn and annotate:
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate("flashing onset", xy=(min(drawn), 0.98), xycoords=tr,
                    xytext=(-4, 0), textcoords="offset points",
                    rotation=90, rotation_mode="anchor", ha="right", va="bottom",
                    fontsize=8.5, color=ONSET_COLOR)
    return bool(drawn)


def _onset_handle(panel_key):
    """Legend handle for the onset lines, on the ONE panel that carries it."""
    if not SHOW_ONSET or ONSET_LEGEND_PANEL != panel_key or not HAS_ONSET:
        return []
    return [Line2D([], [], color=ONSET_COLOR, ls=ONSET_LS, lw=ONSET_LW,
                   label="flashing onset")]


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_case(rel_path, label=None):
    """One TWO-FLUID run → dict (x, p, alpha_g, u_g, u_l, quality, Ma, r, Nb)."""
    run_dir = (_here / rel_path) if not pathlib.Path(rel_path).is_absolute() \
        else pathlib.Path(rel_path)
    if not (run_dir / "solution.csv").exists():
        print(f"  SKIP (no solution.csv): {run_dir}")
        return None
    ymls = sorted(run_dir.glob("settings*.yaml"))
    if not ymls:
        print(f"  SKIP (no settings*.yaml): {run_dir}")
        return None
    cfg = load_config(ymls[0])
    fluid_g, fluid_l = _build_fluid(cfg.LUT_G), _build_fluid(cfg.LUT_L)
    k = float(cfg.K_VM)
    Nb = float(cfg.BUBBLE_NUMBER)

    D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    cols = D.dtype.names
    x   = np.asarray(D["x"])
    p   = np.asarray(D["p"]);   ag = np.asarray(D["alpha_g"]);  al = 1.0 - ag
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
    if QUALITY_SLIP:
        g_flux  = ag * rho_g * u_g
        quality = g_flux / (g_flux + al * rho_l * u_l + 1e-300)
    else:
        quality = (ag * rho_g) / (ag * rho_g + al * rho_l)

    # Stadtke mixture Mach from the scheme's own eigenstructure
    U_h = jnp.asarray(np.column_stack([p, u_g, u_l, ag, h_g, h_l]))

    def _mach_one(uh):
        eg = phase_props(fluid_g, uh[0], uh[4])
        el = phase_props(fluid_l, uh[0], uh[5])
        cq = common_quantities(uh, eg, el, k)
        ev = eigenstructure(uh, eg, el, cq)[0]
        return jnp.abs(0.5 * (ev[2] + ev[3])) / (0.5 * (ev[2] - ev[3]) + 1e-300)

    Ma = np.asarray(jax.vmap(_mach_one)(U_h))
    x_on = _onset_x(fluid_l, x, p, T_l, h_l)
    print(f"  {run_dir.name}:  N={x.size}  N_b={Nb:.1e}  p_out={p[-1]/1e5:.2f} bar"
          f"  max M={Ma[1:-1].max():.3f}  alpha_out={ag[-1]:.4f}"
          f"  x_out={quality[-1]:.4f}"
          + (f"  flashing onset x={x_on:.4f} m" if x_on is not None else ""))
    return dict(name=run_dir.name, label=label if label else _nb_label(Nb),
                x=x, p=p, alpha_g=ag, u_g=u_g, u_l=u_l, quality=quality,
                Ma=Ma, r=r, Nb=Nb, x_onset=x_on)


def load_hem(rel_path, label="HEM (equilibrium)"):
    """The HEM run → dict (x, p, u, Ma, quality, alpha_g, r).

    The HEM solution.csv carries x, rho, u, p, T, c, Ma, s, Qvap, Z — a quality
    but NO void fraction, so alpha is reconstructed from the mixture density:
        alpha_g = Qvap * rho_mix / rho_sat_g(p)
    with rho_sat_g from CoolProp.  The equilibrium table itself cannot supply it
    (its saturation_props block is empty, unlike the metastable two-fluid ones),
    so if CoolProp is unavailable alpha is simply dropped and the void/quality
    panel shows the HEM on the quality axis only.
    """
    run_dir = (_here / rel_path) if not pathlib.Path(rel_path).is_absolute() \
        else pathlib.Path(rel_path)
    if not (run_dir / "solution.csv").exists():
        print(f"  SKIP HEM (no solution.csv): {run_dir}")
        return None
    D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
    cols = D.dtype.names
    for need in ("x", "p", "u", "Ma"):
        if need not in cols:
            print(f"  SKIP HEM (no '{need}' column): {run_dir}")
            return None
    x  = np.asarray(D["x"]);  p = np.asarray(D["p"])
    u  = np.asarray(D["u"]);  Ma = np.asarray(D["Ma"])
    Q  = np.asarray(D["Qvap"]) if "Qvap" in cols else None
    rho = np.asarray(D["rho"]) if "rho" in cols else None

    # void fraction from quality + mixture density (needs rho_sat_g(p))
    ag = None
    if Q is not None and rho is not None:
        try:
            import CoolProp.CoolProp as CP
            fluid = "water"
            ymls = sorted(run_dir.glob("settings*.yaml"))
            if ymls:
                raw = yaml.safe_load(ymls[0].read_text(encoding="utf-8", errors="replace"))
                fluid = str(raw.get("fluid", {}).get("fluid_name", fluid))
            rho_g = np.array([CP.PropsSI("D", "P", float(pp), "Q", 1, fluid)
                              for pp in p])
            ag = np.clip(Q * rho / rho_g, 0.0, 1.0)
        except Exception as exc:                        # CoolProp missing / off-table
            print(f"  NOTE: HEM void fraction not reconstructed ({type(exc).__name__}"
                  f": {exc}); plotting HEM quality only.")
            ag = None

    # nozzle radius: the HEM csv has no area column, so rebuild it from the
    # geometry function named in its settings (guarded: a silhouette is optional)
    r = None
    try:
        ymls = sorted(run_dir.glob("settings*.yaml"))
        raw = yaml.safe_load(ymls[0].read_text(encoding="utf-8", errors="replace"))
        geo_cfg = raw.get("geometry", {})
        geo = getattr(_geom, geo_cfg["function"])(
            x, roughness=float(geo_cfg.get("roughness", 0.0)))
        r = np.sqrt(np.asarray(geo.A, float) / np.pi)
    except Exception as exc:
        print(f"  NOTE: HEM nozzle radius unavailable ({type(exc).__name__}: {exc})")

    # the HEM's own onset: equilibrium quality leaving zero (NOT the metastable
    # gate used for the two-fluid runs — printed for comparison, never drawn)
    x_on = None
    if Q is not None:
        i = np.flatnonzero(Q > 0.0)
        if i.size:
            x_on = float(x[i[0]])
    print(f"  {run_dir.name}:  N={x.size}  p_out={p[-1]/1e5:.2f} bar"
          f"  max M={Ma[1:-1].max():.3f}"
          + (f"  alpha_out={ag[-1]:.4f}" if ag is not None else "")
          + (f"  x_out={Q[-1]:.4f}" if Q is not None else "")
          + (f"  quality leaves 0 at x={x_on:.4f} m" if x_on is not None else ""))
    return dict(name=run_dir.name, label=label, x=x, p=p, u=u, Ma=Ma,
                quality=Q, alpha_g=ag, r=r, x_onset=x_on)


# ── Per-panel drawing ─────────────────────────────────────────────────────────
# A left-placed legend or label is crossed by the onset line (a vertical at
# x/L ~ 0.07); legend.frameon is False globally, so the line would show between
# the handles.  An opaque white, edgeless patch masks it and is invisible
# everywhere else (white on white), leaving the frameless look unchanged.
_LEGEND_KW = dict(frameon=True, facecolor="white", edgecolor="none",
                  framealpha=1.0)


def draw_pressure(ax, cases, hem):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["p"] / 1e5, color=col, ls="-", lw=1.9, zorder=2)
    if hem is not None:
        ax.plot(hem["x"], hem["p"] / 1e5, color=HEM_COLOR, ls=HEM_LS,
                lw=HEM_LW, zorder=3)
    if LEFT_YLABELS:
        ax.set_ylabel(r"pressure $p$ [bar]")
    # the CASE legend lives here (colour = N_b, black dotted = HEM)
    handles = [Line2D([], [], color=col, ls="-", lw=1.9, label=c["label"])
               for c, col in zip(cases, CASE_COLORS)]
    if hem is not None:
        handles.append(Line2D([], [], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW,
                              label=hem["label"]))
    handles += _onset_handle("pressure")
    if handles:
        ax.legend(handles=handles, loc="best", **_LEGEND_KW)


def draw_velocity(ax, cases, hem):
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["u_l"], color=col, ls="-",  lw=1.9, zorder=2)
        ax.plot(c["x"], c["u_g"], color=col, ls="--", lw=1.4, zorder=2)
    if hem is not None:
        ax.plot(hem["x"], hem["u"], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW,
                zorder=3)
    if LEFT_YLABELS:
        ax.set_ylabel(r"phase velocity [m/s]")
    handles = ([Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$u_\ell$ (liquid)"),
                Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$u_g$ (vapour)")]
               if cases else [])          # no two-fluid curve -> no slip legend
    if hem is not None:
        handles.append(Line2D([], [], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW,
                              label=r"$u$ (HEM, no slip)"))
    handles += _onset_handle("velocity")
    ax.legend(handles=handles, loc="best", **_LEGEND_KW)


def draw_mach(ax, cases, hem):
    # NOTE (see module docstring): two-fluid M uses the Stadtke mixture sound
    # speed, HEM M the equilibrium one.  Same symbol, different closure.
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"], c["Ma"], color=col, ls="-", lw=1.9, zorder=2)
    if hem is not None:
        ax.plot(hem["x"], hem["Ma"], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW,
                zorder=3)
    ax.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.text(0.012, 1.0, r"$M=1$", transform=ax.get_yaxis_transform(),
            color="0.4", fontsize=9, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    if LEFT_YLABELS:
        ax.set_ylabel(r"mixture Mach $M$ [-]")
    handles = _onset_handle("mach")
    if handles:
        ax.legend(handles=handles, loc="best", **_LEGEND_KW)


def draw_void_quality(ax, cases, hem):
    """void fraction (left) + quality (right)."""
    axq = ax.twinx()
    for c, col in zip(cases, CASE_COLORS):
        ax.plot(c["x"],  c["alpha_g"], color=col, ls="-",  lw=1.9, zorder=2)
        axq.plot(c["x"], c["quality"], color=col, ls="--", lw=1.4, zorder=2)
    if hem is not None:
        if hem["alpha_g"] is not None:
            ax.plot(hem["x"], hem["alpha_g"], color=HEM_COLOR, ls=HEM_LS,
                    lw=HEM_LW, zorder=3)
        if hem["quality"] is not None:
            axq.plot(hem["x"], hem["quality"], color=HEM_COLOR, ls=HEM_LS,
                     lw=HEM_LW * 0.75, alpha=0.75, zorder=3)
    ax.set_ylim(*ALPHA_YLIM)
    axq.set_ylim(*QUALITY_YLIM)
    ax.tick_params(labelleft=True)
    if LEFT_YLABELS:
        ax.set_ylabel(r"void fraction $\alpha_g$ [-]")
    if QUALITY_RIGHT_LABEL:
        axq.set_ylabel(r"quality $x$ [-]")
    # the swatches take whichever style is actually on the panel: the two-fluid
    # solid/dashed pair, or — with the HEM alone — its thick/thin dotted pair
    if cases:
        style = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$\alpha_g$ (void)"),
                 Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$x$ (quality)")]
    elif hem is not None:
        style = [Line2D([], [], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW,
                        label=r"$\alpha_g$ (void)"),
                 Line2D([], [], color=HEM_COLOR, ls=HEM_LS, lw=HEM_LW * 0.75,
                        alpha=0.75, label=r"$x$ (quality)")]
    else:
        style = []
    ax.legend(handles=style + _onset_handle("void_quality"),
              loc="upper left", **_LEGEND_KW)
    return axq


# panel registry: (key, draw-fn, is_bottom) — order = top→bottom
PANELS = [
    ("pressure",      draw_pressure,     False),
    ("velocity",      draw_velocity,     False),
    ("mach",          draw_mach,         False),
    ("void_quality",  draw_void_quality, True),
]


# ── Load everything (each entry independently; missing ones just drop out) ─────
print("Loading two-fluid cases:")
cases = []
for rel, lab in RUNS:
    c = load_case(rel, lab)
    if c is not None:
        cases.append(c)
# increasing N_b → the colour ramp is the physical ramp, whatever order RUNS is in
cases.sort(key=lambda c: c["Nb"])

print("Loading HEM reference:" if HEM_RUN else "HEM reference: (none configured)")
hem = load_hem(*HEM_RUN) if HEM_RUN else None

if not cases and hem is None:
    sys.exit("Nothing loaded — check the RUNS / HEM_RUN paths at the top of the file.")

_cmap = plt.get_cmap(CMAP_NAME)
CASE_COLORS = ([_cmap(t) for t in np.linspace(CMAP_LO, CMAP_HI, len(cases))]
               if len(cases) > 1 else [_cmap(CMAP_LO)] * max(len(cases), 1))

HAS_ONSET = report_onsets(cases, hem)

out_dir = (_here / OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

# x-range and nozzle silhouette from whatever loaded (two-fluid carries A in the
# csv; the HEM's radius is rebuilt from its geometry function — either may be the
# only one present)
_all = cases + ([hem] if hem is not None else [])
XLIM = (min(float(c["x"].min()) for c in _all),
        max(float(c["x"].max()) for c in _all))
_nz_pool = [c for c in _all if c.get("r") is not None]
_nz = max(_nz_pool, key=lambda c: float(c["x"][-1])) if _nz_pool else None


def _finish_x(ax, bottom):
    ax.set_xlim(*XLIM)
    if bottom:
        ax.set_xlabel(r"$x$ [m]")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _decorate(ax, bottom):
    add_onset(ax, cases, CASE_COLORS, hem)
    if SHOW_NOZZLE and _nz is not None:
        add_nozzle(ax, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
    _finish_x(ax, bottom)


if LAYOUT == "separate":
    for key, draw, bottom in PANELS:
        mb   = M_BOT_LBL if bottom else M_BOT_BARE
        figh = DATA_H + M_TOP + mb
        fig  = plt.figure(figsize=(FIG_W, figh))
        ax   = fig.add_axes([M_LEFT / FIG_W, mb / figh,
                             (FIG_W - M_LEFT - M_RIGHT) / FIG_W, DATA_H / figh])
        draw(ax, cases, hem)
        _decorate(ax, bottom)
        out = out_dir / f"{PREFIX}_{key}.png"
        fig.savefig(out, bbox_inches=None)
        print(f"Plot saved: {out}")

elif LAYOUT == "combined":
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(FIG_W, 10.6), sharex=True)
    for ax, (key, draw, bottom) in zip(axes, PANELS):
        draw(ax, cases, hem)
        _decorate(ax, bottom)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.09)
    out = out_dir / f"{PREFIX}_stacked.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Plot saved: {out}")

else:
    sys.exit(f"unknown LAYOUT {LAYOUT!r} (use 'separate' or 'combined')")

plt.show()
