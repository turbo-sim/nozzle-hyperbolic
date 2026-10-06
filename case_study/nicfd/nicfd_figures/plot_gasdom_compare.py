"""
Visual sanity check for the GAS-DOMINANT cases: solver solution vs the analytic
ideal-gas references (area-Mach perfect expansion, and Rankine-Hugoniot standing
shock).  Mirror of plot_bernoulli_compare.py.

Edit the knobs below: RUN_DIR (which solution folder), NPOINTS (analytic
markers), CASE ('auto' reads ic.type from the run's own settings yaml, or force
'isentropic' / 'shock').  For the shock case BOTH references are drawn: the
shocked one (rebuilt at the DETECTED shock cell, as in check_shock.py) and the
unshocked perfect expansion, so the departure from the isentrope is visible.

Run:  poetry run python plot_gasdom_compare.py [run_dir]
"""
import pathlib, sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ── KNOBS (RUN_DIR can also be given as the first CLI argument) ───────────────
RUN_DIR = "auto"      # "auto" = latest results/*gasdom*/ folder, or a path
RUN_DIR = "results/nozzle_valid_gasdom_N600_muscl_van_albada_20260717_165540"
RUN_DIR = "results/nozzle_valid_gasdom_N300_muscl_minmod_20260717_185347"
NPOINTS = 30          # analytic markers, equispaced in x
CASE    = "auto"      # 'auto' | 'isentropic' | 'shock'
# ──────────────────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    RUN_DIR = sys.argv[1]

import jax.numpy as jnp
import jaxprop as jxp
from case_study.nicfd.nicfd_figures.config_loader import load_config
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from case_study.nicfd.nicfd_figures.isentropic_ic import isentropic_ic, isentropic_shock_ic

_here = pathlib.Path(__file__).parent
if RUN_DIR == "auto":
    _hits = glob.glob(str(_here / "results" / "*gasdom*" / "solution.csv"))
    run_dir = pathlib.Path(sorted(_hits, key=os.path.getmtime)[-1]).parent
else:
    run_dir = (_here / RUN_DIR) if not pathlib.Path(RUN_DIR).is_absolute() \
        else pathlib.Path(RUN_DIR)

yml = sorted(run_dir.glob("settings*.yaml"))[0]
cfg = load_config(yml)
raw = yaml.safe_load(open(yml, encoding="utf-8"))
case = raw["ic"]["type"].replace("isentropic_shock", "shock") if CASE == "auto" else CASE
if case not in ("isentropic", "shock"):
    raise ValueError(f"unrecognized case '{case}' (ic.type={raw['ic']['type']})")

D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
x = np.asarray(D["x"]); N = x.size
p, u_g, h_g = D["p"], D["u_g"], D["h_g"]

geo = getattr(_geom, cfg.GEO_FUNCTION)(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A   = np.asarray(geo.A, float)

fluid_g = jxp.FluidPerfectGas(cfg.LUT_G["fluid_name"],
                              T_ref=cfg.LUT_G["T_ref"], p_ref=cfg.LUT_G["p_ref"])
fluid_l = jxp.FluidPerfectGas(cfg.LUT_L["fluid_name"],
                              T_ref=cfg.LUT_L["T_ref"], p_ref=cfg.LUT_L["p_ref"])
P0, T0 = float(cfg.BC_P0), float(cfg.BC_T_G0)
ic_kw = dict(p0=P0, T0_g=T0, T0_l=float(cfg.BC_T_L0),
             alpha_g=float(cfg.BC_VOID), slip=float(cfg.BC_SLIP),
             fluid_g=fluid_g, fluid_l=fluid_l)

# ── analytic references ───────────────────────────────────────────────────────
_, prof_isen = isentropic_ic(x, A, **ic_kw)     # always: the unshocked isentrope
i_th = int(np.argmin(A))
if case == "shock":
    # detect the discrete shock (as in check_shock) and rebuild the reference there
    dpx = np.abs(np.diff(p)); dpx[:i_th + 2] = 0.0
    i_sh = int(np.argmax(dpx))
    _, prof = isentropic_shock_ic(x, A, **ic_kw, x_shock=float(x[i_sh]))
else:
    i_sh, prof = None, prof_isen

gamma, R = prof["gamma"], prof["R"]
M_ana, p_ana = prof["M"], prof["p"]

# solver Mach from (h_g, p) via the gas EOS (independent of the reference)
T_g = np.asarray(fluid_g.get_state(jxp.HmassP_INPUTS,
                                   jnp.array(h_g), jnp.array(p))["temperature"])
M = np.abs(u_g) / np.sqrt(gamma * R * T_g)

# equispaced stations (interior only)
idx = np.linspace(1, N - 2, NPOINTS).round().astype(int)

# error mask: boundaries + smeared shock zone (check_shock convention)
ok = np.ones(N, dtype=bool)
ok[:2] = ok[-2:] = False
if i_sh is not None:
    ok[i_sh - 4: i_sh + 5] = False

print(f"Run: {run_dir.name}   N={N}   case={case}   NPOINTS={NPOINTS}")
if i_sh is not None:
    print(f"detected shock: cell {i_sh} (x={x[i_sh]:.4f})   "
          f"M1={prof['M1']:.4f}→M2={prof['M2']:.4f}   p2/p1={prof['p_ratio']:.4f}")
print(f"masked errors:  max|dp|/p0 = {np.max(np.abs(p[ok]-p_ana[ok]))/P0:.3e}   "
      f"max|dM| = {np.max(np.abs(M[ok]-M_ana[ok])):.3e}")
print(f"{'x [m]':>8} {'p_num [bar]':>12} {'p_ana [bar]':>12} {'dp/p0 [%]':>10} "
      f"{'M_num':>8} {'M_ana':>8} {'dM':>9}")
for i in idx:
    print(f"{x[i]:8.4f} {p[i]/1e5:12.5f} {p_ana[i]/1e5:12.5f} "
          f"{(p[i]-p_ana[i])/P0*100:10.4f} {M[i]:8.4f} {M_ana[i]:8.4f} "
          f"{M[i]-M_ana[i]:9.2e}")

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
(ax_p, ax_M), (ax_dp, ax_dM) = axes

ax_p.plot(x, p / 1e5, "-", lw=1.2, label="solver")
ax_p.plot(x[idx], p_ana[idx] / 1e5, ".", mfc="none", ms=8, label="analytic")
if case == "shock":
    ax_p.plot(x, prof_isen["p"] / 1e5, "--", lw=0.9, color="gray",
              label="unshocked isentrope")
ax_p.set_ylabel("p [bar]"); ax_p.legend(); ax_p.set_title("pressure")

ax_M.plot(x, M, "-", lw=1.2, label="solver")
ax_M.plot(x[idx], M_ana[idx], ".", mfc="none", ms=8, label="analytic")
if case == "shock":
    ax_M.plot(x, prof_isen["M"], "--", lw=0.9, color="gray",
              label="unshocked isentrope")
ax_M.axhline(1.0, color="k", lw=0.5, ls=":")
ax_M.set_ylabel("M"); ax_M.legend(); ax_M.set_title("Mach number")

xe, dpe = x[1:-1], (p[1:-1] - p_ana[1:-1]) / P0 * 100
ax_dp.plot(xe, dpe, "-", lw=1.0)
ax_dp.axhline(0, color="k", lw=0.5)
ax_dp.set_ylabel(r"$(p - p_{ana})/p_0$ [%]"); ax_dp.set_xlabel("x [m]")
ax_dp.set_title("pressure error (check metric)")

ax_dM.plot(xe, M[1:-1] - M_ana[1:-1], "-", lw=1.0)
ax_dM.axhline(0, color="k", lw=0.5)
ax_dM.set_ylabel(r"$M - M_{ana}$"); ax_dM.set_xlabel("x [m]")
ax_dM.set_title("Mach error")

if i_sh is not None:
    for ax in axes.flat:
        ax.axvspan(x[max(i_sh - 4, 0)], x[min(i_sh + 4, N - 1)],
                   color="orange", alpha=0.12)

fig.suptitle(f"{run_dir.name} — solver vs analytic ({case})", fontsize=10)
fig.tight_layout()
out = run_dir / "gasdom_compare.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {out}")
plt.show()
