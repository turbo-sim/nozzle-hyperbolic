"""
Visual sanity check: solver solution vs analytic Bernoulli at a handful of
equispaced stations.

Edit the knobs below: RUN_DIR (which solution folder), NPOINTS (how many
analytic markers).  The analytic reference is built exactly as in
check_bernoulli.py: from the run's own settled inlet cell,
    u_ana(x) = mdot_in / (alpha_l rho_l A(x)),
    p_ana(x) = p_in + 1/2 rho_in (u_in^2 - u_ana^2),
with rho_l from CoolProp (independent of the solver's tables).

Run:  poetry run python plot_bernoulli_compare.py
"""
import pathlib, sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ── KNOBS (RUN_DIR can also be given as the first CLI argument) ───────────────
RUN_DIR = "auto"      # "auto" = latest results/*liquiddom*/ folder, or a path
# RUN_DIR = "results/nozzle_valid_liquiddom_N300_muscl_van_albada_20260717_173946"
RUN_DIR = "results/nozzle_valid_liquiddom_N300_muscl_van_albada_20260718_164645"
NPOINTS = 30          # analytic markers, equispaced in x
ANCHOR  = "bc"        # 'bc'    = PREDICTIVE reference from the run's boundary
                      #           conditions only (p0, T0_l -> rho; Bernoulli
                      #           down to p_out fixes mdot, u(x), p(x) before
                      #           looking at the solution -> tests BC respect)
                      # 'inlet' = legacy: anchor at the settled inlet cell
                      #           (shape-only test, any mdot passes)
# ──────────────────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    RUN_DIR = sys.argv[1]

from config_loader import load_config
import nozzle_hyperbolic.fvm_solver.geometry as _geom
import CoolProp.CoolProp as CP

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

rho_l = np.array([CP.PropsSI("D", "P", pi, "Hmass", hi, fl_name)
                  for pi, hi in zip(p, h_l)])

if ANCHOR == "bc":
    # PREDICTIVE reference from the boundary conditions alone.
    # Liquid static density from the stagnation state (incompressible to O(dp/K)):
    #   p0 = p + 1/2 rho u^2 everywhere  ->  at the outlet p = p_out fixes
    #   u_out = sqrt(2 (p0 - p_out) / rho),  mdot = alpha_l rho A_out u_out,
    #   u(x) = mdot / (alpha_l rho A(x)),   p(x) = p0 - 1/2 rho u(x)^2 .
    p0    = float(cfg.BC_P0)
    p_out = float(cfg.P_OUT)
    al0   = 1.0 - float(cfg.BC_VOID)
    rho0  = CP.PropsSI("D", "P", p0, "T", float(cfg.BC_T_L0), fl_name)
    u_out = np.sqrt(2.0 * (p0 - p_out) / rho0)
    mdot0 = al0 * rho0 * A[-1] * u_out
    u_ana = mdot0 / (al0 * rho0 * A)
    p_ana = p0 - 0.5 * rho0 * u_ana**2
    q_dyn = 0.5 * rho0 * np.max(np.abs(u_ana[1:-1]))**2
    print(f"BC-anchored reference: p0={p0/1e5:.4f} bar  p_out={p_out/1e5:.4f} bar"
          f"  rho={rho0:.2f} kg/m3  ->  u_in={u_ana[0]:.4f}  u_out={u_out:.4f} m/s"
          f"  mdot={mdot0:.4f} kg/s")
    print(f"solver settled:        u_in={u_l[1]:.4f} m/s  "
          f"mdot={( (1-ag)* rho_l * u_l * A)[1]:.4f} kg/s  "
          f"p[N-2]={p[-2]/1e5:.4f} bar")
elif ANCHOR == "inlet":
    # legacy: anchored at the settled inlet cell (as in check_bernoulli)
    i0    = 1
    mdot0 = (al * rho_l * u_l * A)[i0]
    u_ana = mdot0 / (al * rho_l * A)
    p_ana = p[i0] + 0.5 * rho_l[i0] * (u_l[i0]**2 - u_ana**2)
    q_dyn = 0.5 * rho_l[i0] * np.max(np.abs(u_ana[1:-1]))**2
else:
    raise ValueError(f"ANCHOR must be 'bc' or 'inlet', got {ANCHOR!r}")

# equispaced stations (interior only)
idx = np.linspace(1, N - 2, NPOINTS).round().astype(int)

print(f"Run: {run_dir.name}   N={N}   NPOINTS={NPOINTS}   q_dyn={q_dyn:.4g} Pa")
print(f"{'x [m]':>8} {'p_num [bar]':>12} {'p_ana [bar]':>12} {'dp/q [%]':>9} "
      f"{'u_num [m/s]':>12} {'u_ana [m/s]':>12} {'du/u [%]':>9}")
for i in idx:
    print(f"{x[i]:8.4f} {p[i]/1e5:12.5f} {p_ana[i]/1e5:12.5f} "
          f"{(p[i]-p_ana[i])/q_dyn*100:9.3f} {u_l[i]:12.4f} {u_ana[i]:12.4f} "
          f"{(u_l[i]-u_ana[i])/max(u_ana[i],1e-300)*100:9.3f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
(ax_p, ax_u), (ax_dp, ax_du) = axes

ax_p.plot(x, p / 1e5, "-", lw=1.2, label="solver")
ax_p.plot(x[idx], p_ana[idx] / 1e5, ".", mfc="none", ms=8, label="Bernoulli (analytic)")
ax_p.set_ylabel("p [bar]"); ax_p.legend(); ax_p.set_title("pressure")

ax_u.plot(x, u_l, "-", lw=1.2, label="solver $u_l$")
ax_u.plot(x[idx], u_ana[idx], ".", mfc="none", ms=8, label="continuity (analytic)")
ax_u.set_ylabel("$u_l$ [m/s]"); ax_u.legend(); ax_u.set_title("liquid velocity")

ax_dp.plot(x[1:-1], (p[1:-1] - p_ana[1:-1]) / q_dyn * 100, "-", lw=1.0)
ax_dp.axhline(0, color="k", lw=0.5)
ax_dp.set_ylabel(r"$(p - p_{ana})/q_{dyn}$ [%]"); ax_dp.set_xlabel("x [m]")
ax_dp.set_title("pressure error (dyn-head normalized, check-3 metric)")

ax_du.plot(x[1:-1], (u_l[1:-1] - u_ana[1:-1]) / np.max(u_ana) * 100, "-", lw=1.0)
ax_du.axhline(0, color="k", lw=0.5)
ax_du.set_ylabel(r"$(u_l - u_{ana})/u_{max}$ [%]"); ax_du.set_xlabel("x [m]")
ax_du.set_title("velocity error")

fig.suptitle(f"{run_dir.name} — solver vs analytic Bernoulli "
             f"({'BC-predicted' if ANCHOR == 'bc' else 'inlet-anchored'})",
             fontsize=10)
fig.tight_layout()
out = run_dir / "bernoulli_compare.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out}")
plt.show()
