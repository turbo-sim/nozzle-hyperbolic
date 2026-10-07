# %%
"""
Perfect-gas shock waves of the JTurbo paper
"A Hyperbolic One-Dimensional Model for Two-Phase Flows in Converging-Diverging Nozzles"
(paper Fig. 4 and Table 1).

Air (gamma = 1.4) in the Lettieri nozzle, Roe flux, 500 cells. The back
pressure is chosen so that the exact normal shock sits at 25 %, 50 % and 75 %
of the diverging section. Prints the Table 1 metrics (entropy jump, shock
position and critical mass flow errors).

Results are cached as .npz in results/paper_reproduction; set RERUN = True to
recompute them. Run from any directory with the hyper_env environment.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import nozzle_hyperbolic as hyp

RERUN = False   # False: reuse cached .npz results when available

HERE    = Path(__file__).resolve().parent
OUT_DIR = HERE / "results" / "paper_reproduction"

## Lettieri nozzle geometry (planar, y = half-height)
X1, X2 = -0.03201, 0.06636
WIDTH  = 6.48e-3
_, _, spline = hyp.fit_nozzle_shape(str(HERE / "Lettieri_nozzle.xlsx"), "Sheet1", degree=10)

P0, T0          = 7e5, 500.0
NX              = 500
FLUX            = "roe"
SHOCK_FRACTIONS = (0.25, 0.50, 0.75)
CP, CV          = hyp.fvm_solver.specific_heat2()
GAMMA, R_GAS    = CP / CV, CP - CV


def nozzle(Nx):
    x = np.linspace(X1, X2, Nx)
    h = spline(x)
    A = 2 * h * WIDTH
    return x, h, A, 4 * A / (2 * (WIDTH + 2 * h))


def throat():
    """Sonic area and throat position of the continuous geometry."""
    x_fine = np.linspace(X1, X2, 20001)
    A_fine = 2 * spline(x_fine) * WIDTH
    return A_fine.min(), x_fine[np.argmin(A_fine)]


def solve(x, A, Dh, p_out, flux, CFL=0.5, time_method="euler",
          residuals=1e-8, max_iterations=200000):
    parameters = {
        "perfect_gas":    True,
        "fluid_name":     "air",
        "fluid":          None,
        "residuals":      residuals,
        "max_iterations": max_iterations,
        "A_nodes":        A,
        "D_hydraulic":    Dh,
        "roughness":      0.0,
        "dx":             x[1] - x[0],
        "CFL":            CFL,
        "pT_stagnation":  np.array([P0, T0]),
        "p_out":          p_out,
        "N_eq":           3,
        "space_method":   flux,
        "time_method":    time_method,
        "reconstruction": "constant",
        "gamma":          GAMMA,
    }
    Q_init = hyp.fvm_solver.isentropic_initialization(x, A, parameters)
    parameters["Q_o"] = lambda _: Q_init
    Q, _ = hyp.fvm_solver.fvm_euler_1d(x, np.array([0.0, 1e3]), parameters)
    return Q[:, :, -1]


## Exact quasi-1D solution
def area_mach(M):
    g = GAMMA
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M**2)) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(area_ratio, supersonic):
    if area_ratio <= 1.0:
        return 1.0
    bracket = (1.0, 50.0) if supersonic else (1e-6, 1.0)
    return brentq(lambda M: area_mach(M) - area_ratio, *bracket)


def isentropic(M, p0):
    T = T0 / (1 + 0.5 * (GAMMA - 1) * M**2)
    p = p0 * (T / T0) ** (GAMMA / (GAMMA - 1))
    return p, p / (R_GAS * T)


def exact_branches(x, A, A_star, x_throat):
    """Fully subsonic and adapted supersonic isentropic solutions: (M, p, rho) each."""
    M_sub = np.array([mach_from_area(a / A_star, False) for a in A])
    M_sup = np.array([mach_from_area(a / A_star, xi > x_throat) for a, xi in zip(A, x)])
    return (M_sub, *isentropic(M_sub, P0)), (M_sup, *isentropic(M_sup, P0))


def exact_shock(x, A, A_star, x_throat, x_shock):
    """Exact Mach, pressure and density with a normal shock at x_shock."""
    A_s     = np.interp(x_shock, x, A)
    M1      = mach_from_area(A_s / A_star, True)
    M2      = np.sqrt((1 + 0.5 * (GAMMA - 1) * M1**2) / (GAMMA * M1**2 - 0.5 * (GAMMA - 1)))
    A2_star = A_s / area_mach(M2)          # sonic area of the post-shock isentropic flow
    p02     = P0 * A_star / A2_star        # stagnation pressure ratio = inverse sonic area ratio
    M = np.array([mach_from_area(a / A_star, xi > x_throat) if xi < x_shock
                  else mach_from_area(a / A2_star, False) for xi, a in zip(x, A)])
    p, rho = isentropic(M, np.where(x < x_shock, P0, p02))
    return M, p, rho, p02


def shock_position(x, p, x_throat):
    """Numerical shock position: steepest pressure rise in the diverging section."""
    dp = np.where(x[:-1] > x_throat, np.diff(p), -np.inf)
    i  = np.argmax(dp)
    return 0.5 * (x[i] + x[i + 1])


def critical_mass_flow(A_star):
    return (A_star * P0 / np.sqrt(T0) * np.sqrt(GAMMA / R_GAS)
            * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1))))


def run_all():
    x, h, A, Dh      = nozzle(NX)
    A_star, x_throat = throat()
    results = {"x": x, "h": h, "A": A, "A_star": A_star, "x_throat": x_throat}
    for frac in SHOCK_FRACTIONS:
        x_shock = x_throat + frac * (X2 - x_throat)
        M_ex, p_ex, rho_ex, p02 = exact_shock(x, A, A_star, x_throat, x_shock)
        p_out = p_ex[-1]
        file  = OUT_DIR / f"shock_{FLUX}_{int(frac * 100)}_N{NX}.npz"
        if file.exists() and not RERUN:
            Q = np.load(file)["Q"]
        else:
            print(f"\n=== Perfect gas, {FLUX}, shock at {frac:.0%} (p_out/p0 = {p_out / P0:.4f}) ===")
            Q = solve(x, A, Dh, p_out, FLUX)
            np.savez(file, Q=Q)
        rho = Q[0]
        u   = Q[1] / rho
        p   = (GAMMA - 1) * (Q[2] - 0.5 * rho * u**2)
        results[frac] = {
            "x_shock": x_shock, "p02": p02, "p_out": p_out,
            "M_ex": M_ex, "p_ex": p_ex, "rho_ex": rho_ex,
            "rho": rho, "u": u, "p": p, "M": u / np.sqrt(GAMMA * p / rho),
        }
    results["sub"], results["sup"] = exact_branches(x, A, A_star, x_throat)
    return results


def print_table1(results):
    x, A   = results["x"], results["A"]
    m_ex   = critical_mass_flow(results["A_star"])
    print(f"\nTable 1 - {FLUX}, deviation from the exact solution")
    print(f"{'Shock':<8}{'ds exact':>12}{'ds num':>12}{'ds err %':>10}"
          f"{'x_s err %L':>12}{'mdot err %':>12}")
    for frac in SHOCK_FRACTIONS:
        r        = results[frac]
        ds_exact = R_GAS * np.log(P0 / r["p02"])
        s        = CV * np.log(r["p"] / r["rho"] ** GAMMA)
        ds_num   = s[-2] - s[1]
        x_s_num  = shock_position(x, r["p"], results["x_throat"])
        m_num    = np.mean((r["rho"] * r["u"] * A)[1:-1])
        print(f"{frac:<8.0%}{ds_exact:>12.2f}{ds_num:>12.2f}"
              f"{100 * (ds_num - ds_exact) / ds_exact:>10.2f}"
              f"{100 * (x_s_num - r['x_shock']) / (X2 - X1):>12.2f}"
              f"{100 * (m_num - m_ex) / m_ex:>12.2f}")
    print(f"Exact critical mass flow: {m_ex:.4f} kg/s")


def plot_fig4(results):
    x, h  = results["x"], results["h"]
    L     = X2 - X1
    xn    = (x - X1) / L
    rho0  = P0 / (R_GAS * T0)
    i_thr = int(np.argmin(h))
    colors = {0.25: "tab:purple", 0.50: "orange", 0.75: "green"}
    M_sub, p_sub, rho_sub = results["sub"]
    M_sup, p_sup, rho_sup = results["sup"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        (r"$\rho/\rho_0$ (-)", "rho", rho0, rho_sub, rho_sup, "rho_ex"),
        ("Mach number (-)",    "M",   1.0,  M_sub,   M_sup,   "M_ex"),
        (r"$P/P_0$ (-)",       "p",   P0,   p_sub,   p_sup,   "p_ex"),
    ]
    for ax, (ylabel, key, ref, sub, sup, key_ex) in zip(axes, panels):
        for k, frac in enumerate(SHOCK_FRACTIONS):
            r = results[frac]
            ax.plot(xn, r[key] / ref, "-o", color=colors[frac], ms=3, mfc="white", lw=1,
                    markevery=8, label=f"Shock {k + 1} numerical")
            i_s = np.searchsorted(x, r["x_shock"])
            ax.plot(xn[i_s - 1:], r[key_ex][i_s - 1:] / ref, "k--", lw=1.2,
                    label="Shock wave exact" if k == 0 else None)
        ax.plot(xn, sub / ref, "r-", lw=1.2, label="Subsonic exact")
        ax.plot(xn[i_thr:], sup[i_thr:] / ref, "b-", lw=1.2, label="Supersonic exact")
        ax.set_xlabel("Normalized axis position x/L (-)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)

        ax_r = ax.twinx()
        ax_r.fill_between(xn, 0, h / L, color="tab:blue", alpha=0.15)
        ax_r.plot(xn, h / L, "k", lw=0.8)
        ax_r.set_ylim(0, 3 * (h / L).max())
        ax_r.set_ylabel("Normalized nozzle height h/L (-)")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig4_shocks_{FLUX}.png", dpi=300)


# %%
if __name__ == "__main__":
    hyp.set_plot_options(grid=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_all()
    print_table1(results)
    plot_fig4(results)
    plt.show()
