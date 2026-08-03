"""
plot_two_component_journal.py

Journal-quality figures for the TWO-COMPONENT nozzle (nitrogen gas + liquid
water, NO mass transfer — only drag and interfacial heat transfer).  Same layout
and style as plot_phase_change_journal.py: no analytic reference, this is a
presentation of the solver solution.  Several back-pressure regimes are overlaid
on shared axes:

  1. two_component_p_mach.png   — pressure (solid, left axis) and MIXTURE Mach
                                  number (dashed, right axis) vs position x, one
                                  colour per case, with the sonic line M = 1.
  2. two_component_velocity.png — phase velocities u_l (solid) and u_g (dashed)
                                  vs position x, one colour per case.
  3. two_component_alpha_T.png  — void fraction alpha_g (top) and the phase
                                  temperatures T_g / T_l (bottom).  This is the
                                  two-component-specific panel: with no phase
                                  change, slip and thermal non-equilibrium are
                                  the only interphase physics.  Set
                                  MAKE_ALPHA_T_FIG = False to skip it.

Mach here is the TWO-PHASE mixture Mach M = |u_mix| / a, using the solver's OWN
Stadtke mixture sound speed a = 1/2(lambda_3 - lambda_4) from eigenstructure() —
the speed the scheme actually chokes on (a single-component sound speed would
badly misclassify the regime for a gas/liquid mixture).

Edit the RUNS list below (path, label).  Run:
    poetry run python plot_two_component_journal.py
"""
import pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
import jaxprop as jxp

from config_loader import load_config
from nozzle_silhouette import add_nozzle
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    phase_props, common_quantities, eigenstructure)

# ── KNOBS: the cases to overlay (path relative to dev_scripts, label) ──────────
_REPO = "results_two_phase"
RUNS = [
    # TODO: the SUBSONIC run does not exist yet — this entry is a copy of the
    # shock case as a placeholder so the script runs.  Re-run the two-component
    # case with a HIGH back-pressure (bc.p_out close to p0 = 20 bar, e.g.
    # ~18-19 bar, so the nozzle never chokes) and point this path at that run.
    (f"{_REPO}/two_component_N500_muscl_van_albada_20260727_105958", "subsonic"),
    # (f"{_REPO}/two_component_N500_muscl_van_albada_20260726_004729", "supersonic-shock"),
    (f"results/nozzle_valid_gasdom_N500_muscl_van_albada_20260722_115901", "supersonic-shock"),
    # (f"{_REPO}/two_component_N500_muscl_van_albada_20260727_105912", "supersonic-shock"),
    (f"{_REPO}/two_component_N500_muscl_van_albada_20260726_115923", "supersonic-adapted"),
]
OUT_DIR = _REPO          # where the overlay figures are written
PREFIX  = "two_component"  # output file name prefix
MAKE_ALPHA_T_FIG = True    # third figure: void fraction + phase temperatures

# case colours are sampled from a colormap: pick the map and how many colours
CMAP_NAME = "plasma"    # any matplotlib colormap ("plasma", "turbo", "cividis"…)
N_COLORS  = 4            # how many colours to draw (>= number of cases)
CMAP_LO, CMAP_HI = 0.0, 0.85   # sub-range of the map to sample (avoid washed-out end)
# Nozzle silhouette proportion.  False = schematic 16% band (radius NOT to scale).
# True = proportion-true (1 m in x == 1 m of radius) — sensible for the Elliot
# geometry (r_max/L ~ 0.08); nonsensical for the symmetric placeholder.
NOZZLE_TRUE_SCALE = False
# ──────────────────────────────────────────────────────────────────────────────

# ── Journal style (matches plot_phase_change_journal.py) ──────────────────────
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


def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"])
    return jxp.FluidBicubic(**lut)


def load_case(rel_path):
    """Return dict with x, p, alpha_g, u_g, u_l, T_g, T_l, Mach (Stadtke mixture)."""
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
    T_g = np.asarray(D["T_g"]); T_l = np.asarray(D["T_l"])

    # nozzle radius r(x) = sqrt(A/pi) for the bottom silhouette — taken straight
    # from the solver's own area column (no geometry re-evaluation needed).
    r = np.sqrt(np.asarray(D["A"], float) / np.pi)

    # Stadtke mixture sound speed & Mach from the scheme's own eigenstructure.
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
    print(f"  {run_dir.name}:  N={x.size}  p_out={p[-1]/1e5:.2f} bar"
          f"  max M={Ma[1:-1].max():.3f}"
          f"  ({'supersonic' if Ma[1:-1].max() > 1 else 'subsonic'})")
    return dict(name=run_dir.name, x=x, p=p, alpha_g=ag, u_g=u_g, u_l=u_l,
                T_g=T_g, T_l=T_l, Ma=Ma, r=r)


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

# widest-domain case → its silhouette spans the full x-range
_nz = max(cases, key=lambda c: float(c["x"][-1]))
# colour = case (legend drawn once, on the velocity figure)
case_handles = [Line2D([], [], color=col, ls="-", lw=1.9, label=c["label"])
                for c, col in zip(cases, CASE_COLORS)]

# ── FIGURE 1: pressure (solid, left) + mixture Mach (dashed, right) ───────────
fig1, axp = plt.subplots(figsize=(7.0, 4.3))
axm = axp.twinx()
for c, col in zip(cases, CASE_COLORS):
    axp.plot(c["x"], c["p"] / 1e5, color=col, ls="-",  lw=1.9, zorder=2)
    axm.plot(c["x"], c["Ma"],      color=col, ls="--", lw=1.6, zorder=2)
axm.axhline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)          # sonic line
add_nozzle(axp, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
axm.annotate(r"$M=1$", xy=(axp.get_xlim()[0], 1.0), xytext=(3, 3),
             textcoords="offset points", color="0.4", fontsize=9,
             ha="left", va="bottom")
axp.set_xlabel(r"$x$ [m]")
axp.set_ylabel(r"pressure $p$ [bar]")
axm.set_ylabel(r"mixture Mach number $M = |u_\mathrm{mix}| / a$")

# linestyle = quantity (case identity lives on Fig. 2)
style_handles = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label="pressure"),
                 Line2D([], [], color="0.3", ls="--", lw=1.6, label="Mach")]
axp.legend(handles=style_handles, loc="upper center")
fig1.tight_layout()
out1 = out_dir / f"{PREFIX}_p_mach.png"
fig1.savefig(out1)
print(f"Plot saved: {out1}")

# ── FIGURE 2: phase velocities u_l (solid) + u_g (dashed) ─────────────────────
fig2, axu = plt.subplots(figsize=(7.0, 4.3))

# nozzle radius on a SECONDARY RIGHT AXIS (true mm scale), drawn behind the data.
# The twin is placed below axu (zorder) and axu's patch is hidden so it shows
# through; ylim is set so the nozzle fills the bottom ~16% of the frame.
if NOZZLE_TRUE_SCALE:
    add_nozzle(axu, _nz["x"], _nz["r"], true_scale=True)   # proportion-true
else:
    axr = axu.twinx()
    axr.set_zorder(axu.get_zorder() - 1)
    axu.patch.set_visible(False)
    _rmm = _nz["r"] * 1e3                                   # radius in mm
    axr.fill_between(_nz["x"], 0.0, _rmm, color="0.45", alpha=0.22, lw=0, zorder=0)
    axr.set_ylim(0.0, float(_rmm.max()) / 0.16)            # same height as Fig. 1
    axr.set_ylabel("nozzle radius [mm]", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    axr.spines["right"].set_color("0.35")

for c, col in zip(cases, CASE_COLORS):
    axu.plot(c["x"], c["u_l"], color=col, ls="-",  lw=1.9, zorder=2)
    axu.plot(c["x"], c["u_g"], color=col, ls="--", lw=1.4, zorder=2)
axu.set_xlabel(r"$x$ [m]")
axu.set_ylabel(r"phase velocity [m/s]")
vel_style = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$u_\ell$ (water)"),
             Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$u_g$ (nitrogen)")]
_leg2 = axu.legend(handles=case_handles, loc="upper left")
axu.add_artist(_leg2)
axu.legend(handles=vel_style, loc="upper center")
fig2.tight_layout()
out2 = out_dir / f"{PREFIX}_velocity.png"
fig2.savefig(out2)
print(f"Plot saved: {out2}")

# ── FIGURE 3: void fraction (top) + phase temperatures (bottom) ───────────────
# Two-component specific: with mass transfer OFF, the interphase coupling shows
# up only as slip (Fig. 2), void-fraction redistribution and thermal lag.
if MAKE_ALPHA_T_FIG:
    fig3, (axa, axt) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for c, col in zip(cases, CASE_COLORS):
        axa.plot(c["x"], c["alpha_g"], color=col, ls="-", lw=1.9, zorder=2)
        axt.plot(c["x"], c["T_g"], color=col, ls="-",  lw=1.9, zorder=2)
        axt.plot(c["x"], c["T_l"], color=col, ls="--", lw=1.4, zorder=2)
    add_nozzle(axa, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
    add_nozzle(axt, _nz["x"], _nz["r"], true_scale=NOZZLE_TRUE_SCALE)
    axa.set_ylabel(r"void fraction $\alpha_g$")
    axt.set_ylabel(r"temperature [K]")
    axt.set_xlabel(r"$x$ [m]")
    axa.legend(handles=case_handles, loc="lower left")
    T_style = [Line2D([], [], color="0.3", ls="-",  lw=1.9, label=r"$T_g$ (nitrogen)"),
               Line2D([], [], color="0.3", ls="--", lw=1.4, label=r"$T_\ell$ (water)")]
    axt.legend(handles=T_style, loc="lower left")
    fig3.tight_layout()
    out3 = out_dir / f"{PREFIX}_alpha_T.png"
    fig3.savefig(out3)
    print(f"Plot saved: {out3}")

plt.show()
