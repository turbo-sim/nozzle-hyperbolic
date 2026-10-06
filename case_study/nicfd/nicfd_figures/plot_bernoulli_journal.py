"""
plot_bernoulli_journal.py

Journal-quality comparison of the two-fluid solver against the analytic
Bernoulli reference for the LIQUID-dominant low-speed case.  Produces TWO
standalone figures:

  1. bernoulli_p_mach.png  — pressure (left axis) and dominant-phase Mach number
                             (right axis) vs position x, with N analytic markers
                             on each curve.
  2. bernoulli_velocity.png — liquid velocity vs position x, with N analytic
                             markers.

The analytic reference is built exactly as in plot_bernoulli_compare.py /
check_bernoulli.py:
    u_ana(x) = mdot / (alpha_l rho_l A(x))          (continuity),
    p_ana(x) = p0 - 1/2 rho_l u_ana(x)^2            (Bernoulli),
with rho_l and the speed of sound a_l from CoolProp (independent of the solver
tables).  Mach = u / a_l  (a_l = dominant liquid phase; for this incompressible
limit a_l ~ 1400 m/s so Ma ~ 1e-2 — the plot certifies the O(Ma^2) assumption).

Run:  poetry run python plot_bernoulli_journal.py [run_dir]
"""
import pathlib, sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

# ── KNOBS (run_dir may also be given as the first CLI argument) ────────────────
# RUN_DIR = "auto"      # "auto" = latest results/*liquiddom*/ folder, or a path
# RUN_DIR = "results/nozzle_valid_liquiddom_N300_muscl_van_albada_20260718_164645"
# RUN_DIR = "results_single_phase_validation/bernoulli_N500_muscl_van_albada_20260726_003433"
# RUN_DIR = "results_single_phase_validation/bernoulli_N500_muscl_van_albada_20260722_163311"
# RUN_DIR = "results_single_phase_validation/bernoulli_N1000_muscl_van_albada_20260727_190052"
# RUN_DIR = "results_single_phase_validation/bernoulli_N500_muscl_van_albada_20260728_103535" 
# RUN_DIR = "results_single_phase_validation/bernoulli_N500_muscl_van_albada_20260726_233854"
RUN_DIR = "results_single_phase_validation/bernoulli_N500_muscl_van_albada_20260728_154326"

RUN_DIR = "results_single_phase_validation/bernoulli_N2000_muscl_van_albada_20260730_093109"
# RUN_DIR = "results_single_phase_validation/bernoulli_N2000_muscl_van_albada_20260730_234116"

NPOINTS = 30          # analytic markers, equispaced in x
ANCHOR  = "bc"        # 'bc'    = predictive reference from the run's BCs only
                      # 'inlet' = anchor at the settled inlet cell (shape-only)
MACH_SOS = "liquid"   # sound speed for Mach: 'liquid' = dominant-phase a_l
PRINT_TABLE = True    # also print the per-station error table (NPOINTS rows)
# Shared folder (relative to dev_scripts) collecting every paper figure next to
# the ones from plot_gasdom_journal.py — the figures are ALSO kept inside the run
# folder.  None = save only in the run folder.  Underscored, not "nicfd figures",
# so the path stays free of spaces for \includegraphics.
EXTRA_OUT_DIR = "nicfd_figures"
# Nozzle silhouette proportion.  False = schematic 16% band (radius NOT to scale,
# the historical look).  True = proportion-true (1 m in x == 1 m of radius) — use
# only for a long-and-thin nozzle; for the symmetric placeholder (r_max > L) it
# is nonsensical, so keep False for the Bernoulli/symmetric case.
NOZZLE_TRUE_SCALE = False
# ──────────────────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    RUN_DIR = sys.argv[1]

from case_study.nicfd.nicfd_figures.config_loader import load_config
from case_study.nicfd.nicfd_figures.nozzle_silhouette import add_nozzle
import nozzle_hyperbolic.fvm_solver.geometry as _geom
import CoolProp.CoolProp as CP

# ── Journal style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     10,
    "legend.fontsize":    10,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "axes.linewidth":     0.8,
    "lines.linewidth":    1.8,
    "legend.frameon":     False,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})
C_P = "#0b3d91"   # pressure  (blue)
C_M = "#a11a2a"   # Mach      (red)
C_U = "#0b3d91"   # velocity  (blue)
_SOLVER_KW = dict(ls="-",  lw=1.8, zorder=2)
_ANA_KW    = dict(ls="none", marker="o", mfc="white", ms=5.5, mew=1.1, zorder=3)


# add_nozzle now lives in nozzle_silhouette.py (shared by all journal scripts);
# NOZZLE_TRUE_SCALE (top of file) toggles schematic vs proportion-true.


def save_fig(fig, name):
    """Write the figure into the run folder and, if EXTRA_OUT_DIR is set, into
    the shared paper-figure folder as well."""
    out = run_dir / name
    fig.savefig(out)
    print(f"Plot saved: {out}")
    if EXTRA_OUT_DIR:
        shared = _here / EXTRA_OUT_DIR
        shared.mkdir(parents=True, exist_ok=True)
        fig.savefig(shared / name)
        print(f"Plot saved: {shared / name}")


# ── Load the run ──────────────────────────────────────────────────────────────
_here = pathlib.Path(__file__).parent
if RUN_DIR == "auto":
    _hits = glob.glob(str(_here / "results" / "*liquiddom*" / "solution.csv"))
    run_dir = pathlib.Path(sorted(_hits, key=os.path.getmtime)[-1]).parent
else:
    run_dir = (_here / RUN_DIR) if not pathlib.Path(RUN_DIR).is_absolute() \
        else pathlib.Path(RUN_DIR)
yml = sorted(run_dir.glob("settings*.yaml"))[0]
cfg = load_config(yml)
raw = yaml.safe_load(open(yml, encoding="utf-8"))
fl_name = raw["eos"]["liquid"]["fluid_name"]

D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
x = np.asarray(D["x"]); N = x.size
p, u_l, ag, h_l = D["p"], D["u_l"], D["alpha_g"], D["h_l"]
al = 1.0 - ag

geo = getattr(_geom, cfg.GEO_FUNCTION)(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A   = np.asarray(geo.A, float)
r   = np.sqrt(A / np.pi)                       # nozzle radius for the silhouette

# solver liquid density + sound speed (CoolProp, table-independent)
rho_l = np.array([CP.PropsSI("D", "P", pi, "Hmass", hi, fl_name)
                  for pi, hi in zip(p, h_l)])
a_l   = np.array([CP.PropsSI("A", "P", pi, "Hmass", hi, fl_name)
                  for pi, hi in zip(p, h_l)])
Ma_num = np.abs(u_l) / a_l

# ── Analytic Bernoulli reference ──────────────────────────────────────────────
if ANCHOR == "bc":
    p0    = float(cfg.BC_P0)
    p_out = float(cfg.P_OUT)
    T_l0  = float(cfg.BC_T_L0)
    al0   = 1.0 - float(cfg.BC_VOID)
    rho0  = CP.PropsSI("D", "P", p0, "T", T_l0, fl_name)
    u_out = np.sqrt(2.0 * (p0 - p_out) / rho0)
    mdot0 = al0 * rho0 * A[-1] * u_out
    u_ana = mdot0 / (al0 * rho0 * A)
    p_ana = p0 - 0.5 * rho0 * u_ana**2
    a_ana = np.array([CP.PropsSI("A", "P", pi, "T", T_l0, fl_name) for pi in p_ana])
    q_dyn = 0.5 * rho0 * np.max(np.abs(u_ana[1:-1]))**2
elif ANCHOR == "inlet":
    i0    = 1
    mdot0 = (al * rho_l * u_l * A)[i0]
    u_ana = mdot0 / (al * rho_l * A)
    p_ana = p[i0] + 0.5 * rho_l[i0] * (u_l[i0]**2 - u_ana**2)
    a_ana = a_l.copy()
    q_dyn = 0.5 * rho_l[i0] * np.max(np.abs(u_ana[1:-1]))**2
else:
    raise ValueError(f"ANCHOR must be 'bc' or 'inlet', got {ANCHOR!r}")
Ma_ana = np.abs(u_ana) / a_ana

# equispaced analytic marker stations (interior only)
idx = np.linspace(1, N - 2, NPOINTS).round().astype(int)

print(f"Run: {run_dir.name}   N={N}   NPOINTS={NPOINTS}   anchor={ANCHOR}")
print(f"max Mach: solver={Ma_num[1:-1].max():.4g}  analytic={Ma_ana[1:-1].max():.4g}"
      f"   (a_l ~ {a_l.mean():.1f} m/s)")

# ── Error report ──────────────────────────────────────────────────────────────
# SAME normalizations as check_bernoulli.py: the pressure error is divided by the
# DYNAMIC HEAD q = 1/2 rho u_max^2, NOT by p — the whole Bernoulli dip is only
# ~1e2..1e3 Pa out of ~2 MPa, so a %-of-p metric would hide a 100% dip error.
# Velocity is normalized by u_max (one constant scale, no blow-up where u -> 0).
s     = slice(1, -1)                       # interior cells (BC/ghost excluded)
u_max = float(np.max(np.abs(u_ana[s])))
e_p   = (p - p_ana) / q_dyn * 100.0        # [% of dynamic head]
e_u   = (u_l - u_ana) / u_max * 100.0      # [% of u_max]
mdot  = al * rho_l * u_l * A
i_p   = int(np.argmax(np.abs(e_p[s]))) + 1
i_u   = int(np.argmax(np.abs(e_u[s]))) + 1
i_th  = int(np.argmin(A))

print(f"dynamic head q = {q_dyn:.4g} Pa   u_max(analytic) = {u_max:.4f} m/s")
print("errors (interior cells; pressure normalized by q, velocity by u_max):")
print(f"  max |p - p_ana| / q      = {np.abs(e_p[s]).max():9.4f} %   at x = {x[i_p]:.4f} m")
print(f"  rms |p - p_ana| / q      = {np.sqrt(np.mean(e_p[s]**2)):9.4f} %")
print(f"  max |u - u_ana| / u_max  = {np.abs(e_u[s]).max():9.4f} %   at x = {x[i_u]:.4f} m")
print(f"  rms |u - u_ana| / u_max  = {np.sqrt(np.mean(e_u[s]**2)):9.4f} %")
print(f"  mass-flux max deviation  = {np.max(np.abs(mdot[s]/mdot[1] - 1.0))*100:9.4f} %")
print(f"  throat (x={x[i_th]:.4f}):  p-dip {p[1]-p[i_th]:.1f} vs analytic "
      f"{p_ana[1]-p_ana[i_th]:.1f} Pa   u_l {u_l[i_th]:.4f} vs {u_ana[i_th]:.4f} m/s")

if PRINT_TABLE:
    print(f"\n{'x [m]':>8} {'p_num [bar]':>12} {'p_ana [bar]':>12} {'dp/q [%]':>10} "
          f"{'u_num [m/s]':>12} {'u_ana [m/s]':>12} {'du/u_max [%]':>13}")
    for i in idx:
        print(f"{x[i]:8.4f} {p[i]/1e5:12.5f} {p_ana[i]/1e5:12.5f} {e_p[i]:10.4f} "
              f"{u_l[i]:12.4f} {u_ana[i]:12.4f} {e_u[i]:13.4f}")
    print()

# ── FIGURE 1: pressure + Mach (dual axis) vs x ────────────────────────────────
fig1, axp = plt.subplots(figsize=(6.6, 4.1))
axm = axp.twinx()

# pressure (left) — colour encodes the quantity, so no per-curve legend label
axp.plot(x, p / 1e5, color=C_P, **_SOLVER_KW)
axp.plot(x[idx], p_ana[idx] / 1e5, color=C_P, **_ANA_KW)
axp.set_xlabel(r"$x$ [m]")
axp.set_ylabel(r"pressure $p$ [bar]", color=C_P)
axp.tick_params(axis="y", colors=C_P)
axp.spines["left"].set_color(C_P)

# Mach (right)
axm.plot(x, Ma_num, color=C_M, **_SOLVER_KW)
axm.plot(x[idx], Ma_ana[idx], color=C_M, **_ANA_KW)
axm.set_ylabel(r"Mach number $M = u_\ell / a_\ell$", color=C_M)
axm.tick_params(axis="y", colors=C_M)
axm.spines["right"].set_color(C_M)
axm.spines["left"].set_color(C_P)

# nozzle silhouette in the bottom band (Mach axis already holds the right spine)
add_nozzle(axp, x, r, true_scale=NOZZLE_TRUE_SCALE)

# minimal neutral legend: solid line = solver, open marker = analytic
_neutral = [Line2D([], [], color="0.25", ls="-", lw=1.8, label="solver"),
            Line2D([], [], color="0.25", ls="none", marker="o", mfc="white",
                   ms=5.5, mew=1.1, label="analytic (Bernoulli)")]
axp.legend(handles=_neutral, loc="center right")
# axp.set_title(run_dir.name, fontsize=9, loc="left", pad=6)
fig1.tight_layout()
run_dit = ""
save_fig(fig1, "bernoulli_p_mach.png")

# ── FIGURE 2: velocity vs x ───────────────────────────────────────────────────
fig2, axu = plt.subplots(figsize=(6.6, 4.1))

# nozzle radius silhouette behind the data
if NOZZLE_TRUE_SCALE:
    add_nozzle(axu, x, r, true_scale=True)          # proportion-true (metres)
else:
    # SCHEMATIC: radius on a SECONDARY RIGHT AXIS (true mm ticks, 16% band)
    axr = axu.twinx()
    axr.set_zorder(axu.get_zorder() - 1)
    axu.patch.set_visible(False)
    _rmm = r * 1e3                                          # radius in mm
    axr.fill_between(x, 0.0, _rmm, color="0.45", alpha=0.22, lw=0, zorder=0)
    axr.set_ylim(0.0, float(_rmm.max()) / 0.16)            # nozzle → bottom 16%
    axr.set_ylabel("nozzle radius [mm]", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    axr.spines["right"].set_color("0.35")

axu.plot(x, u_l, color=C_U, **_SOLVER_KW, label=r"solver $u_\ell$")
axu.plot(x[idx], u_ana[idx], color=C_U, **_ANA_KW, label="analytic (Bernoulli)")
axu.set_xlabel(r"$x$ [m]")
axu.set_ylabel(r"liquid velocity $u_\ell$ [m/s]")
# axu.set_title(run_dir.name, fontsize=9, loc="left", pad=6)
axu.legend(loc="upper left")
fig2.tight_layout()
save_fig(fig2, "bernoulli_velocity.png")

plt.show()
