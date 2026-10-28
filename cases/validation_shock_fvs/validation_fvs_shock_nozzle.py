"""
Single-phase FVS validation: normal shock held in the divergent section.

For each fraction f in `validation.shock_fractions` the shock sits at
    x_s = x_throat + f * (x_exit - x_throat).
The exact quasi-1D solution (nozzle_exact_real_gas, same EOS as the solver)
gives the back pressure p_out that holds the shock there; the Städtke FVS
Newton solver (fvm_euler_1d, time_discretization: fvs_newton) is run with that
p_out from a choked, fully subsonic start, and the converged profiles are
compared with the exact ones (rho/rho0, Mach, p/p0 vs x/L).

usage (cwd = dev_scripts):
    python validation_fvs_shock_nozzle.py [settings/settings_fvs_validation_lettieri_air.yaml]

Output: results/fvs_validation_<geometry>_<fluid>_N<Nx>_<timestamp>/
    shock_validation.png / .pdf   figure (Fig. 4 layout)
    case_<k>_shock_<pct>pct.csv   numerical + exact profiles at the cell centres
    summary.txt                   shock position, errors, mass flow, timings
    <settings>.yaml, run_info.yaml (+ git_diff.patch)
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import sys
import copy
import time
import shutil
import pathlib
import datetime
import types

import yaml
import numpy as np

_here = pathlib.Path(__file__).resolve().parent
os.chdir(_here)                      # relative fluid_tables / geometry paths
sys.path.insert(0, str(_here))

SETTINGS_YAML = sys.argv[1] if len(sys.argv) > 1 else \
    "settings_fvs_validation_lettieri_air.yaml"
case_data = yaml.safe_load(open(_here / SETTINGS_YAML))
ns  = case_data["numerical_settings"]
val = case_data.get("validation", {})

# if ns.get("cpu_affinity", None) not in (None, "", "none"):
#     from cpu_affinity import pin_cpus
#     pin_cpus(ns["cpu_affinity"])

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(_here.parent / ".jax_cache"))
import jaxprop as jxp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nozzle_hyperbolic as hyp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.models.euler_nozzle.analytical.nozzle_exact_real_gas import nozzle_exact_real_gas
# from run_info import write_run_info

hyp.set_plot_options(grid=False)

# ── Fluid, grid, geometry ───────────────────────────────────────────────────
lut = case_data["fluid"].get("lut", {})
fluid_name = case_data["fluid"]["fluid_name"]
fluid = jxp.FluidBicubic(
    fluid_name=fluid_name, backend=lut.get("backend", "HEOS"),
    h_min=float(lut["h_min"]), h_max=float(lut["h_max"]),
    p_min=float(lut["p_min"]), p_max=float(lut["p_max"]),
    N_h=int(lut["N_h"]), N_p=int(lut["N_p"]),
    table_dir=lut.get("table_dir", "fluid_tables"))

Nx = int(ns["Nx"])
x1, x2 = float(ns["x1"]), float(ns["x2"])
x = np.linspace(x1, x2, Nx)
dx = x[1] - x[0]
L = x2 - x1

geo_cfg = case_data["geometry"].copy()
geo_fn_name = geo_cfg.pop("function")
roughness = float(geo_cfg.pop("roughness", 0.0))
geo_fn = getattr(_geom, geo_fn_name)
geo = geo_fn(x, roughness=roughness, **geo_cfg)
A = np.asarray(geo.A)

n_exact = int(val.get("n_exact", 1201))
x_ex = np.linspace(x1, x2, n_exact)
A_ex = np.asarray(geo_fn(x_ex, roughness=roughness, **geo_cfg).A)
r_ex = np.sqrt(A_ex / np.pi)                 # equivalent radius (plot only)
i_thr = int(np.argmin(A_ex))
x_thr, A_thr = x_ex[i_thr], float(A_ex[i_thr])

p0 = float(case_data["boundary_conditions"]["p_stagnation"])
T0 = float(case_data["boundary_conditions"]["T_stagnation"])
fractions = [float(f) for f in val.get("shock_fractions", [0.25, 0.5, 0.75])]

# ── Output folder ───────────────────────────────────────────────────────────
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = (_here / "results" /
           f"fvs_validation_{geo_fn_name.replace('get_', '')}_{fluid_name}_N{Nx}_{stamp}")
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(_here / SETTINGS_YAML, out_dir / pathlib.Path(SETTINGS_YAML).name)
_cfg_stub = types.SimpleNamespace(_settings=types.SimpleNamespace(
    path=(_here / SETTINGS_YAML).resolve(), chain=[(_here / SETTINGS_YAML).resolve()],
    overrides={}))
_tables = sorted((_here / lut.get("table_dir", "fluid_tables")).glob(
    f"{fluid_name}_{lut['N_h']}x{lut['N_p']}.pkl"))
# write_run_info(out_dir, _cfg_stub,
#                input_files={"eos_table": _tables[0]} if _tables else None)
print(f"Output directory: {out_dir}")


def solver_parameters(p_out):
    """parameters dict for fvm_euler_1d — same keys and defaults as main.py."""
    g = lambda k, d: ns.get(k, d)
    return {
        "perfect_gas": False, "fluid_name": fluid_name, "fluid": fluid,
        "residuals": float(ns["residuals"]), "max_iterations": int(ns["max_iterations"]),
        "geometry": geo, "A_nodes": A, "dx": dx, "CFL": float(ns["CFL"]),
        "pT_stagnation": np.array([p0, T0]), "p_out": float(p_out), "N_eq": 3,
        "space_method": ns["flux_method"], "time_method": ns["time_discretization"],
        "reconstruction": g("reconstruction", "constant"),
        "initialization": g("initialization", "simple"),
        "x_shock_fraction": float(g("x_shock_fraction", 0.75)),
        "delta_factor": float(g("delta_factor", 0.1)),
        "newton_iter": int(g("newton_iter", 20)),
        "jacobian": g("jacobian", "dense"),
        "jacobian_validate": bool(g("jacobian_validate", False)),
        "adaptive_dt": bool(g("adaptive_dt", True)),
        "warmup_cfg": None,
        "nozzle_type": geo_fn_name.replace("get_", ""),
        "wall_friction": bool(g("wall_friction", True)),
        "eps_filter": float(g("eps_filter", 0.0)),
        "eps_filter_ucut": g("eps_filter_ucut", None),
        "eps_filter_uwidth": g("eps_filter_uwidth", None),
        "linear_solver": g("linear_solver", "dense"),
        "newton_tol": float(g("newton_tol", 0.0)),
        "flux_local_jac": bool(g("flux_local_jac", False)),
        "cell_local_jac": bool(g("cell_local_jac", False)),
        "fast_increments": bool(g("fast_increments", False)),
        "ser_beta": float(g("ser_beta", 0.0)),
        "ser_cfl_max": float(g("ser_cfl_max", 50.0)),
        # Q_o only fills the first output slice on the fvs_newton path
        "Q_o": lambda _x: np.zeros((3, len(_x))),
    }


def shock_position(xc, p, p1, p2):
    """x where p crosses (p1 + p2)/2 on the steepest rise (linear interpolation)."""
    pm = 0.5 * (p1 + p2)
    k = int(np.argmax(np.diff(p)))
    for j in list(range(k, len(p) - 1)) + list(range(k - 1, -1, -1)):
        if (p[j] - pm) * (p[j + 1] - pm) <= 0.0 and p[j + 1] != p[j]:
            return xc[j] + (pm - p[j]) / (p[j + 1] - p[j]) * (xc[j + 1] - xc[j])
    return np.nan


# ── Runs ────────────────────────────────────────────────────────────────────
results = []
ex_curves = None
for k, frac in enumerate(fractions, start=1):
    x_s = x_thr + frac * (x2 - x_thr)
    print(f"\n=== Case {k}: shock at {frac*100:.0f} % of the divergent section "
          f"(x_s = {x_s:.5f} m) ===", flush=True)
    t_ex = time.perf_counter()
    ex = nozzle_exact_real_gas(x_ex, A_ex, fluid, p0, T0, x_shock=x_s, A_throat=A_thr)
    ex_c = nozzle_exact_real_gas(x, A, fluid, p0, T0, x_shock=x_s, A_throat=A_thr)
    p_out = ex["p_out"]
    print(f"  exact ({time.perf_counter() - t_ex:.0f} s): M1 = {ex['pre']['M']:.4f} -> "
          f"M2 = {ex['post']['M']:.4f},  p_out = {p_out:.1f} Pa "
          f"(p_out/p0 = {p_out/p0:.5f}),  mdot = {ex['mdot']*1e3:.4f} g/s", flush=True)
    if ex_curves is None:
        ex_curves = ex

    P = solver_parameters(p_out)
    hyp.fvm_solver.fvm_euler_1d(x, np.array([float(ns["t1"]), float(ns["t2"])]), P)
    U = P["U_puh_final"]
    st = fluid.get_state(jxp.HmassP_INPUTS, U[:, 2], U[:, 0])
    rho, a = np.asarray(st["rho"]), np.asarray(st["a"])
    num = dict(p=U[:, 0], u=U[:, 1], h=U[:, 2], rho=rho, M=U[:, 1] / a)

    # errors at the interior cells, away from the shock (±3 cells)
    xs_num = shock_position(x, num["p"], ex["pre"]["p"], ex["post"]["p"])
    m = np.zeros(Nx, bool); m[1:-1] = True
    m &= np.abs(x - x_s) > 3.0 * dx
    err = {}
    for key, ref in (("p/p0", p0), ("rho/rho0", ex["rho0"]), ("M", 1.0)):
        q = key.split("/")[0] if "/" in key else key
        d = np.abs(num[q][m] - ex_c["shock"][q][m]) / ref
        err[key] = (float(np.mean(d)), float(np.max(d)),
                    float((x[m][np.argmax(d)] - x1) / L))       # x/L of the max
    # cell-centre rho*u*A is not the conserved flux (it smears at the shock):
    # median over the cells away from the shock
    mdot_num = rho * U[:, 1] * A
    mdot_err = float(np.median(mdot_num[m]) / ex["mdot"] - 1.0)
    rs = P["run_stats"]
    results.append(dict(k=k, frac=frac, x_s=x_s, xs_num=xs_num, p_out=p_out, ex=ex,
                        ex_c=ex_c, num=num, err=err, mdot_err=mdot_err, stats=rs))
    print(f"  shock position: exact {x_s:.5f} m, numerical {xs_num:.5f} m "
          f"(error {(xs_num - x_s)/dx:+.2f} dx)", flush=True)
    print("  errors away from the shock (mean / max, normalised): " +
          "  ".join(f"{kk} {v[0]:.2e}/{v[1]:.2e} (max at x/L={v[2]:.3f})" for kk, v in err.items()), flush=True)

    cols = [x, (x - x1) / L, A, num["p"], num["u"], num["h"], num["rho"], num["M"],
            ex_c["shock"]["p"], ex_c["shock"]["u"], ex_c["shock"]["rho"], ex_c["shock"]["M"]]
    np.savetxt(out_dir / f"case_{k}_shock_{frac*100:.0f}pct.csv", np.column_stack(cols),
               delimiter=",", comments="",
               header="x,x_over_L,A,p,u,h,rho,M,p_exact,u_exact,rho_exact,M_exact")

# ── Figure (layout of the reference Fig. 4) ─────────────────────────────────
colors = ["#7B1FA2", "#FFA000", "#2E7D32", "#1565C0", "#C62828"]
xl_ex = (x_ex - x1) / L
xl = (x - x1) / L
div = x_ex >= x_thr
rho0 = ex_curves["rho0"]
markevery = max(1, Nx // int(val.get("markers", 25)))
panels = [("rho", r"$\rho/\rho_0$ (-)", rho0), ("M", "Mach number (-)", 1.0),
          ("p", r"$P/P_0$ (-)", p0)]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for j, (ax, (q, ylab, ref)) in enumerate(zip(axes, panels)):
    ax2 = ax.twinx()
    ax2.fill_between(xl_ex, 0.0, r_ex / L, color="#D6E4F0", zorder=0)
    ax2.plot(xl_ex, r_ex / L, color="k", lw=0.9)
    ax2.set_ylim(0.0, 7.5 * r_ex.max() / L)
    ax2.set_ylabel("Normalized nozzle radius r/L (-)")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    for res, c in zip(results, colors):
        ax.plot(xl[1:-1], res["num"][q][1:-1] / ref, "-o", color=c, lw=1.0, ms=4.5,
                mfc="white", markevery=markevery,
                label=f"Shock {res['k']} numerical ({res['frac']*100:.0f} %)")
    for i_r, res in enumerate(results):
        s = res["ex"]["shock"]
        dn = x_ex >= res["x_s"]
        # exact shocked profile from just upstream of the shock (incl. the jump)
        i0 = max(int(np.argmax(dn)) - 1, 0)
        xx = np.concatenate([xl_ex[i0:i0 + 1], [(res["x_s"] - x1) / L] * 2, xl_ex[dn]])
        yy = np.concatenate([[s[q][i0]], [res["ex"]["pre"][q], res["ex"]["post"][q]],
                             s[q][dn]]) / ref
        ax.plot(xx, yy, "k-" if q == "rho" else "k--", lw=1.1, zorder=5,
                label="Shock wave exact" if i_r == 0 else None)
    ax.plot(xl_ex[div], ex_curves["sub"][q][div] / ref, color="r",
            ls="-" if q == "rho" else "--", lw=1.1, label="Subsonic exact")
    ax.plot(xl_ex, ex_curves["sup"][q] / ref, color="b",
            ls="-" if q == "rho" else "--", lw=1.1, label="Supersonic exact")
    ax.set_xlabel("Normalized axis position x/L (-)")
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.03, 1.03)
h_leg, l_leg = axes[0].get_legend_handles_labels()
fig.legend(h_leg, l_leg, loc="lower center", ncol=len(l_leg), fontsize=11,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Städtke FVS shock wave predictions at "
             f"{', '.join(f'{f*100:.0f}%' for f in fractions)} of the diverging "
             f"section length ({fluid_name} LUT, N = {Nx})", y=0.995)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(out_dir / "shock_validation.png", dpi=300)
fig.savefig(out_dir / "shock_validation.pdf")
plt.close(fig)

# ── Summary ─────────────────────────────────────────────────────────────────
lines = [f"FVS single-phase shock validation — {SETTINGS_YAML}",
         f"geometry {geo_fn_name}, {fluid_name} LUT {lut['N_h']}x{lut['N_p']}, N = {Nx}, "
         f"dx = {dx*1e3:.4f} mm, p0 = {p0:.0f} Pa, T0 = {T0:.2f} K",
         f"throat x = {x_thr:.5f} m (x/L = {(x_thr - x1)/L:.4f}), "
         f"A_exit/A* = {A_ex[-1]/A_thr:.4f}, mdot_exact = {ex_curves['mdot']*1e3:.5f} g/s",
         "",
         f"{'case':>4} {'frac':>5} {'p_out/p0':>9} {'M1':>6} {'x_s exact':>10} "
         f"{'x_s num':>10} {'err/dx':>7} {'p/p0 L1':>9} {'p/p0 max':>9} {'M L1':>9} "
         f"{'M max':>9} {'max at x/L':>10} {'mdot err':>9} {'steps':>6} {'wall s':>7} {'ms/step':>8} "
         f"{'SSR':>9}  stop"]
for r in results:
    s = r["stats"]
    lines.append(
        f"{r['k']:>4} {r['frac']:>5.2f} {r['p_out']/p0:>9.5f} {r['ex']['pre']['M']:>6.3f} "
        f"{r['x_s']:>10.5f} {r['xs_num']:>10.5f} {(r['xs_num']-r['x_s'])/dx:>+7.2f} "
        f"{r['err']['p/p0'][0]:>9.2e} {r['err']['p/p0'][1]:>9.2e} "
        f"{r['err']['M'][0]:>9.2e} {r['err']['M'][1]:>9.2e} {r['err']['p/p0'][2]:>10.4f} "
        f"{r['mdot_err']:>+9.2e} "
        f"{s['steps']:>6d} {s['wall_s']:>7.1f} {s['wall_s']/max(s['steps'],1)*1e3:>8.1f} "
        f"{s['ssr']:>9.2e}  {s['reason']}")
lines += ["", "Errors: interior cells farther than 3 dx from the exact shock; "
          "p normalised by p0, rho by rho0, M absolute.  'max at x/L' = location of "
          "the max p error (the throat cell = sonic point).",
          "mdot err = median(rho u A over those cells)/mdot_exact - 1.",
          f"Figure: {out_dir / 'shock_validation.png'}"]
(out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n" + "\n".join(lines))
