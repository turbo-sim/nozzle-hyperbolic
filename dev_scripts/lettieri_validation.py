# %%
"""
Lettieri et al. validation of the JTurbo paper
"A Hyperbolic One-Dimensional Model for Two-Phase Flows in Converging-Diverging Nozzles"
(paper Fig. 8 and Table 3).

CO2 cases L1 and L5, HLL flux, 200 cells, with and without wall friction
(smooth wall), compared against the experiments of Lettieri et al. and the
HEM CFD of Romei and Persico. Rusanov is also run for the Table 3 mass flow.

Results are cached as .npz in results/paper_reproduction; set RERUN = True to
recompute them. Run from any directory with the hyper_env environment.
"""
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic as hyp

RERUN = False   # False: reuse cached .npz results when available

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "validation_cases" / "Lettieri2018"
OUT_DIR  = HERE / "results" / "paper_reproduction"

## Lettieri nozzle geometry (planar, y = half-height)
X1, X2 = -0.03201, 0.06636
WIDTH  = 6.48e-3
_, _, spline = hyp.fit_nozzle_shape(str(HERE / "Lettieri_nozzle.xlsx"), "Sheet1", degree=10)

CASES = {
    #     p0 [Pa]   T0 [K]  p_out [Pa] (low enough for a supersonic exit)
    "L1": (58.96e5, 314.7, 15e5),
    "L5": (84.74e5, 313.9, 25e5),
}
RUNS = {   # label: (flux, friction)
    "HLL - without friction": ("hll",     False),
    "HLL - smooth wall":      ("hll",     True),
    "Rusanov - smooth wall":  ("rusanov", True),
}
NX         = 200
EXP_FILES  = {"L1": "Case1", "L5": "Case5"}
X_ONSET_MM = {"L1": 3.748, "L5": -4.675}                       # experimental condensation onset
MDOT_ROMEI = {"L1": (29.43, 29.46), "L5": (48.94, 48.95)}      # CFD (mixture, barotropic)
P_LIMITS   = {"L1": (15, 60), "L5": (25, 88)}


def nozzle(Nx, friction=True):
    x  = np.linspace(X1, X2, Nx)
    h  = spline(x)
    A  = 2 * h * WIDTH
    # A huge hydraulic diameter makes the wall shear term 4*tau_w/D_h vanish
    Dh = 4 * A / (2 * (WIDTH + 2 * h)) if friction else np.full(Nx, 1e6)
    return x, h, A, Dh


def co2_fluid():
    return jxp.FluidBicubic(
        fluid_name="CO2", backend="HEOS",
        h_min=50e3, h_max=850e3, p_min=1e4, p_max=300e5, N_h=200, N_p=200,
        table_dir=str(HERE / "fluid_tables"),
    )


def solve(x, A, Dh, fluid, p0, T0, p_out, flux, CFL=0.5, time_method="euler",
          residuals=1e-5, max_iterations=100000, roughness=0.0):
    parameters = {
        "perfect_gas":    False,
        "fluid_name":     "CO2",
        "fluid":          fluid,
        "residuals":      residuals,
        "max_iterations": max_iterations,
        "A_nodes":        A,
        "D_hydraulic":    Dh,
        "roughness":      roughness,
        "dx":             x[1] - x[0],
        "CFL":            CFL,
        "pT_stagnation":  np.array([p0, T0]),
        "p_out":          p_out,
        "N_eq":           3,
        "space_method":   flux,
        "time_method":    time_method,
        "reconstruction": "constant",
        "warmup_cfg":     None,
    }
    Q_init = hyp.fvm_solver.isentropic_initialization(x, A, parameters)
    parameters["Q_o"] = lambda _: Q_init
    Q, _ = hyp.fvm_solver.fvm_euler_1d(x, np.array([0.0, 1e3]), parameters)
    return Q[:, :, -1]


def run_all():
    fluid   = co2_fluid()
    results = {}
    for case, (p0, T0, p_out) in CASES.items():
        for label, (flux, friction) in RUNS.items():
            tag  = f"lettieri_{case}_{flux}_{'smooth' if friction else 'nofric'}_N{NX}"
            file = OUT_DIR / f"{tag}.npz"
            if file.exists() and not RERUN:
                results[(case, label)] = dict(np.load(file))
                continue
            print(f"\n=== Lettieri {case} | {label} ===")
            x, h, A, Dh = nozzle(NX, friction)
            Q   = solve(x, A, Dh, fluid, p0, T0, p_out, flux)
            rho = Q[0]
            u   = Q[1] / rho
            e   = Q[2] / rho - 0.5 * u**2
            st  = hyp.fvm_solver.state_from_rho_u_batch(fluid, rho, e)
            q   = np.asarray(st.Q, dtype=float)
            res = {
                "x": x, "h": h, "rho": rho, "u": u,
                "p": np.asarray(st.pressure, dtype=float),
                "c": np.asarray(st.speed_sound, dtype=float),
                "y_liq": np.where((q >= 0) & (q <= 1), 1 - q, 0.0),
            }
            np.savez(file, **res)
            results[(case, label)] = res
    return results


def print_mass_flow(results):
    # Mass flow per unit depth of the half channel, as in the 2D CFD of Romei and Persico
    print("\nTable 3 - mass flow rate (kg/s, per unit depth, half channel)")
    print(f"{'Case':<6}{'CFD mix':>10}{'CFD baro':>10}{'Rusanov':>10}{'HLL':>10}")
    for case in CASES:
        m = {label: np.mean((r["rho"] * r["u"] * r["h"])[1:-1])
             for label in ("Rusanov - smooth wall", "HLL - smooth wall")
             for r in [results[(case, label)]]}
        m_mix, m_baro = MDOT_ROMEI[case]
        print(f"{case:<6}{m_mix:>10.2f}{m_baro:>10.2f}"
              f"{m['Rusanov - smooth wall']:>10.2f}{m['HLL - smooth wall']:>10.2f}")
        r = results[(case, "HLL - smooth wall")]
        print(f"      HLL exit Mach = {r['u'][-2] / r['c'][-2]:.2f}  (must be > 1 for a supersonic exit)")


def plot_fig8(results):
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1]})
    styles = {
        "HLL - without friction": dict(color="crimson", lw=1.8),
        "HLL - smooth wall":      dict(color="orange",  lw=1.8),
    }
    # [:-1] drops the outlet ghost cell, which carries the imposed p_out
    for ax, case in zip(axes[:2], CASES):
        exp = pd.read_excel(DATA_DIR / f"{EXP_FILES[case]}.xlsx")
        cfd = pd.read_excel(DATA_DIR / f"{EXP_FILES[case]}_CFD.xlsx")

        ax2 = ax.twinx()
        for label, kw in styles.items():
            r = results[(case, label)]
            ax.plot(r["x"][:-1] * 1e3, r["p"][:-1] / 1e5, label=f"1D HEM {label}", **kw)
        ax.plot(cfd["Position p"], cfd["Pressure"], "k--", lw=1.8, label="CFD - Romei et al. (2021)")
        ax.plot(exp["Position"], exp["Pressure"], "ks", ms=5, label="Exp. - Lettieri et al. (2017)")
        ax.axvline(X_ONSET_MM[case], ymax=0.04, color="darkorange", lw=3,
                   label="Lettieri et al. - condensation onset")

        r = results[(case, "HLL - smooth wall")]
        ax2.plot(r["x"][:-1] * 1e3, r["y_liq"][:-1], color="darkcyan", lw=1.8,
                 label="1D HEM - liquid mass fraction")
        ax2.plot(cfd["position frac"], cfd["liquid_fraction"], "--", color="blue", lw=1.8,
                 label="Romei et al. - liquid mass fraction")
        ax2.set_ylim(0, 0.4)
        ax2.set_ylabel("Liquid mass fraction (-)", color="darkcyan")
        ax2.tick_params(axis="y", colors="darkcyan")
        ax2.spines["right"].set_color("darkcyan")

        ax.set_ylim(*P_LIMITS[case])
        ax.set_ylabel("Static pressure (bar)")
        ax.text(0.05, 0.12, f"Case {case}", transform=ax.transAxes, fontsize=14,
                bbox=dict(facecolor="white", edgecolor="black"))
        if case == "L1":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    r = results[("L1", "HLL - smooth wall")]
    axes[2].plot(r["x"] * 1e3, r["h"] * 1e3, "k", lw=1)
    axes[2].fill_between(r["x"] * 1e3, 0, r["h"] * 1e3, color="tab:blue", alpha=0.15)
    axes[2].set_ylim(0, 7)
    axes[2].set_ylabel("Height (mm)")
    axes[2].set_xlabel("Position along nozzle axis (mm)")
    axes[2].set_xlim(X1 * 1e3, X2 * 1e3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_lettieri.png", dpi=300)


# %%
if __name__ == "__main__":
    hyp.set_plot_options(grid=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_all()
    print_mass_flow(results)
    plot_fig8(results)
    plt.show()
