"""One steady perfect-gas nozzle case, FVS finite volumes versus exact flow.

Run this file directly, with no CLI arguments, from any working directory.
Configure the case in the neighboring YAML or the user settings below.
The exact shock location sets the outlet pressure; FVM must recover the profile.
The shared solver is imported from the installed nozzle_hyperbolic package.
"""

# %% User settings
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = CASE_DIR / "settings_steady_lettieri.yaml"
SOLVER_PRESET = "fast"  # None uses active_preset in YAML; or "fast", "no_ser", "baseline"
RESULTS_DIR = CASE_DIR / "results_steady"
SHOW_FIGURES = True  # Close the comparison window to finish the script.

# %% Imports and numerical precision
import json
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)  # Needed for tight steady residuals.
import jaxprop as jxp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from nozzle_hyperbolic.fvm_solver import fvm_euler_1d, geometry
from nozzle_hyperbolic.fvm_solver.nozzle_exact_real_gas import nozzle_exact_real_gas


def main():
    # -------------------------------------------------------------------------
    # 1. Read the configuration and choose a solver preset
    # -------------------------------------------------------------------------
    started = perf_counter()
    settings = SETTINGS_FILE.resolve()
    cfg = yaml.safe_load(settings.read_text(encoding="utf-8"))
    preset = SOLVER_PRESET or cfg["active_preset"]
    # The preset overrides common solver settings (same physics and mesh).
    solver = {**cfg["solver"], **cfg["presets"][preset]}

    # -------------------------------------------------------------------------
    # 2. Build the fluid and the nozzle mesh
    # -------------------------------------------------------------------------
    # Both solutions use the same perfect-gas fluid, without lookup tables.
    fluid = jxp.FluidPerfectGas(name=cfg["fluid_name"])
    mesh = cfg["mesh"]
    x = np.linspace(mesh["x1"], mesh["x2"], mesh["Nx"])
    dx = x[1] - x[0]
    geo_cfg = cfg["geometry"].copy()
    geo_fn = getattr(geometry, geo_cfg.pop("function"))
    if "file_path" in geo_cfg:
        # Resolve geometry data relative to the YAML, independent of working directory.
        geo_cfg["file_path"] = str((settings.parent / geo_cfg["file_path"]).resolve())
    geo = geo_fn(x, **geo_cfg)
    A = np.asarray(geo.A)

    # -------------------------------------------------------------------------
    # 3. Compute the exact reference and required outlet pressure
    # -------------------------------------------------------------------------
    # A fine geometry grid resolves the true throat independently of the FVM mesh.
    # xf/Af are the fine-grid coordinates/areas; xs is the prescribed shock position.
    xf = np.linspace(x[0], x[-1], cfg["case"]["n_exact"])
    Af = np.asarray(geo_fn(xf, **geo_cfg).A)
    it = int(np.argmin(Af))
    xs = xf[it] + cfg["case"]["shock_fraction"] * (x[-1] - xf[it])
    p0, T0 = cfg["case"]["p_stagnation"], cfg["case"]["T_stagnation"]
    exact_args = dict(fluid=fluid, p0=p0, T0=T0, x_shock=xs, A_throat=Af[it])
    # The general EOS routine also handles FluidPerfectGas, reducing to the
    # perfect-gas area-Mach and normal-shock relations. Evaluate it on both grids:
    # fine grid for plotting and FVM grid for pointwise error comparisons.
    exact = nozzle_exact_real_gas(xf, Af, **exact_args)
    reference = nozzle_exact_real_gas(x, A, **exact_args)["shock"]

    # -------------------------------------------------------------------------
    # 4. March the FVM equations to steady state
    # -------------------------------------------------------------------------
    # The solver gets the back pressure but not the shock position or exact profile.
    # It builds its own initial flow using the YAML initialization setting.
    parameters = dict(
        **solver, fluid=fluid, fluid_name=cfg["fluid_name"], geometry=geo,
        A_nodes=A, dx=dx, pT_stagnation=np.array([p0, T0]),
        p_out=exact["p_out"], N_eq=3, Q_o=lambda xx: np.zeros((3, len(xx))),
        # Q_o is only a placeholder output slice; FVS builds its own initial state.
        # This internal perfect_gas flag selects solver implementation, not EOS:
        # False keeps the general-EOS FVS Newton path, using FluidPerfectGas.
        perfect_gas=False, time_method="fvs_newton", space_method="fvs",
        wall_friction=False, nozzle_type="nozzle",
    )
    print(f"Preset: {preset}; N={len(x)}; shock x={xs:.6f} m; "
          f"outlet pressure={exact['p_out']:.1f} Pa", flush=True)
    setup_s = perf_counter() - started
    # Pseudo-time is a route to steady flow, not a physical transient simulation.
    # Infinite end time leaves stopping to the residual or maximum iteration count.
    fvm_euler_1d(x, np.array([0.0, np.inf]), parameters)
    U = parameters["U_puh_final"]  # Each row contains pressure, velocity, enthalpy.
    state = fluid.get_state(jxp.HmassP_INPUTS, U[:, 2], U[:, 0])
    numerical = dict(p=U[:, 0], rho=np.asarray(state["rho"]),
                     M=U[:, 1] / np.asarray(state["a"]))
    if not all(np.isfinite(v).all() for v in numerical.values()):
        raise RuntimeError("Non-finite numerical solution")

    # -------------------------------------------------------------------------
    # 5. Assess convergence and compare the steady profiles
    # -------------------------------------------------------------------------
    # Exclude the boundary ghost cells and the captured shock, which is spread
    # over a few cells. Normalize pressure/density errors by stagnation values.
    mask = (np.arange(len(x)) > 0) & (np.arange(len(x)) < len(x) - 1)
    mask &= np.abs(x - xs) > 3 * dx  # exclude the captured shock and ghost cells
    errors = {q: float(np.mean(np.abs(numerical[q][mask] - reference[q][mask])) / scale)
              for q, scale in (("p", p0), ("rho", exact["rho0"]), ("M", 1.0))}
    # Locate the strongest pressure rise in the divergent section.
    candidates = np.flatnonzero((x[:-1] > xf[it]) & (x[1:] < x[-1]))
    k = candidates[np.argmax(np.diff(numerical["p"])[candidates])]
    stats = parameters["run_stats"].copy()
    stats["pseudo_time"] = stats.pop("t_phys")
    # The driver also permits a plateau stop below 3*tolerance. Check the requested
    # tolerance itself so such a stop is not silently reported as convergence.
    summary = dict(preset=preset, nodes=len(x), setup_s=setup_s, **stats,
                   converged=bool(stats["ssr"] < solver["residuals"]),
                   p_out=exact["p_out"], shock_exact_m=float(xs),
                   shock_error_cells=float((0.5 * (x[k] + x[k+1]) - xs) / dx),
                   mean_errors_away_from_shock=errors)

    # -------------------------------------------------------------------------
    # 6. Save the case settings and comparison data
    # -------------------------------------------------------------------------
    out = RESULTS_DIR / preset
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "solver_settings.yaml").write_text(yaml.safe_dump(solver), encoding="utf-8")
    np.savetxt(out / "comparison.csv", np.column_stack(
        [x, A] + [v for q in ("p", "rho", "M") for v in (numerical[q], reference[q])]),
        delimiter=",", header="x,A,p_fvm,p_exact,rho_fvm,rho_exact,M_fvm,M_exact", comments="")

    # -------------------------------------------------------------------------
    # 7. Plot the steady profiles, save the figure, and display it
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, (q, scale, label) in zip(axes, (
            ("p", p0, "p/p0"), ("rho", exact["rho0"], "rho/rho0"), ("M", 1., "Mach"))):
        ax.plot(xf, exact["shock"][q] / scale, "k-", label="Exact")
        ax.plot(x[1:-1], numerical[q][1:-1] / scale, ".", ms=3, label="FVM")
        ax.set(xlabel="x [m]", ylabel=label)
        ax.legend()
    fig.savefig(out / "comparison.png", dpi=180)

    # Save timing before waiting for the user to close the plot window.
    summary["total_s"] = perf_counter() - started
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {out}")
    if SHOW_FIGURES:
        plt.show(block=True)
    else:
        plt.close(fig)
    if not summary["converged"]:
        raise SystemExit("Residual target not reached; inspect summary.json before using the result.")


if __name__ == "__main__":
    main()
