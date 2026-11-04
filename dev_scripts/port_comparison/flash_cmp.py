import glob, numpy as np, jax, matplotlib
jax.config.update("jax_enable_x64", True); matplotlib.use("Agg")
import matplotlib.pyplot as plt, jax.numpy as jnp, jaxprop as jxp
S = r"C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/port_comparison"
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
REF = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/single_component/supersonic_adapted/reference.csv"
names = ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")
eos = dict(fluid_name="water", backend="HEOS", p_min=1e4, p_max=1.3e7, N_h=90, N_p=90, N_p_sat=150, gradient_method="forward", table_dir=TAB)
fg = jxp.FluidBicubic(h_min=2.2e6, h_max=3.4e6, metastable_phase="vapor", **eos)
fl = jxp.FluidBicubic(h_min=1e5, h_max=1.8e6, metastable_phase="liquid", **eos)

f = sorted(glob.glob(S + "/flash/old_runs/flash_adapted_*/solution.csv"))[-1]
do = np.genfromtxt(f, delimiter=",", names=True); xo = do["x"]; Uo = np.column_stack([do[k] for k in names])
n = np.load(S + "/flash/new_cfl05.npz"); xn, Un = n["x"], n["U"]
dr = np.genfromtxt(REF, delimiter=",", names=True)

def derived(U):
    p, ug, ul, a, hg, hl = U.T
    sg = fg.get_state(jxp.HmassP_INPUTS, jnp.asarray(hg), jnp.asarray(p)); sl = fl.get_state(jxp.HmassP_INPUTS, jnp.asarray(hl), jnp.asarray(p))
    Tsat = np.asarray(fl.get_state_saturation(jnp.asarray(p))["temperature"])
    return dict(T_g=np.asarray(sg["T"]), T_l=np.asarray(sl["T"]), rho_g=np.asarray(sg["rho"]), rho_l=np.asarray(sl["rho"]), T_sat=Tsat)
from cases.NICFD2026.scripts.case import geometry_arrays
A = np.asarray(geometry_arrays(tuple(xo), "single_component")[0])
do_d, dn_d = derived(Uo), derived(Un)
def mdot(U, d, A): return A * (U[:, 3] * d["rho_g"] * U[:, 1] + (1 - U[:, 3]) * d["rho_l"] * U[:, 2])
mo, mn = mdot(Uo[1:-1], {k: v[1:-1] for k, v in do_d.items()}, A[1:-1]), mdot(Un[1:-1], {k: v[1:-1] for k, v in dn_d.items()}, A[1:-1])
a, b = Uo[1:-1], Un[1:-1]; e = np.abs(a - b) / (a.max(0) - a.min(0))
print("interior cells, max |old-new|/range:", "  ".join(f"{k}={e[:, i].max():.2e}@x={xo[1+e[:, i].argmax()]:.3f}" for i, k in enumerate(names)))
print("   excluding 3 cells each end:      ", "  ".join(f"{k}={e[3:-3, i].max():.2e}" for i, k in enumerate(names)))
for k in ("T_g", "T_l"):
    d = np.abs(do_d[k][1:-1] - dn_d[k][1:-1]); print(f"   {k}: max |old-new| = {d.max():.3f} K at x={xo[1+d.argmax()]:.3f}")
print(f"mass flow  old {mo.mean():.4f} kg/s (var {np.ptp(mo)/mo.mean()*100:.3f}%)   new {mn.mean():.4f} kg/s (var {np.ptp(mn)/mn.mean()*100:.3f}%)   diff {(mn.mean()-mo.mean())/mo.mean()*100:+.3f}%")
print(f"exit cell: p old/new {a[-1,0]/1e5:.4f}/{b[-1,0]/1e5:.4f} bar, alpha {a[-1,3]:.5f}/{b[-1,3]:.5f}, u_l {a[-1,2]:.2f}/{b[-1,2]:.2f} m/s")
on = lambda u: np.argmax(u[:, 3] > 2e-3)
print(f"flashing onset (alpha>2e-3): old x={xo[on(Uo)]:.4f}  new x={xn[on(Un)]:.4f} m")
er = np.max(np.abs(np.column_stack([np.interp(xo[1:-1], dr["x"], dr[k]) for k in names]) - a) / (a.max(0) - a.min(0)), axis=0)
print("old (faafff7, with p*Dalpha) vs Sept NICFD reference, max/range:", "  ".join(f"{k}={v:.2e}" for k, v in zip(names, er)))

fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
panels = [("p [bar]", lambda U, d: U[:, 0] / 1e5, "p"), ("void fraction [-]", lambda U, d: U[:, 3], "alpha_g"),
          ("velocity [m/s]", None, None), ("temperature [K]", None, None),
          ("liquid superheat T_l - T_sat [K]", lambda U, d: d["T_l"] - d["T_sat"], None), ("slip u_g/u_l [-]", lambda U, d: U[:, 1] / U[:, 2], None)]
for axi, (lab, fn, rk) in zip(ax.flat, panels):
    if lab.startswith("velocity"):
        for U, x, st, tag in ((Uo, xo, "-", "old"), (Un, xn, "--", "new")):
            axi.plot(x, U[:, 1], "C0" + st, label=f"u_g {tag}"); axi.plot(x, U[:, 2], "C1" + st, label=f"u_l {tag}")
    elif lab.startswith("temperature"):
        for d, x, st, tag in ((do_d, xo, "-", "old"), (dn_d, xn, "--", "new")):
            axi.plot(x, d["T_g"], "C0" + st, label=f"T_g {tag}"); axi.plot(x, d["T_l"], "C1" + st, label=f"T_l {tag}"); axi.plot(x, d["T_sat"], "C7" + st, lw=0.8, label=f"T_sat {tag}")
    else:
        if rk: axi.plot(dr["x"], dr[rk] / (1e5 if rk == "p" else 1), color="0.8", lw=3, label="NICFD ref (Sept, no p·Dα)")
        axi.plot(xo, fn(Uo, do_d), "k-", label="old (alberto_branch)"); axi.plot(xn, fn(Un, dn_d), "r--", label="new (refactor)")
    axi.set_ylabel(lab); axi.grid(alpha=0.3); axi.legend(fontsize=7)
for axi in ax[1]: axi.set_xlabel("x [m]")
fig.suptitle("Flashing water, NICFD supersonic_adapted (p0 120 bar, p_out 0.8 bar), 598 cells, same tables/settings")
fig.tight_layout(); fig.savefig(S + "/flash/flash_adapted_old_vs_new.png", dpi=130)
print("figure:", S + "/flash/flash_adapted_old_vs_new.png")
