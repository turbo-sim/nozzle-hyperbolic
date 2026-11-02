"""Steady EulerNozzle baseline: one mesh/method, several back pressures.

Run: poetry run python development/roberto/historical/euler_nozzle/verify_euler_nozzle_steady.py
Add --no-show to save the results without opening a figure window.

The perfect-gas numerical solution is compared with quasi-1D isentropic and
normal-shock references. Reference branches determine the back pressures;
the numerical solver receives only those pressures, not the exact flow,
shock position, or throat Mach number. All cases start from the same rule.
"""

# %% Settings: change these here when deliberately establishing a new baseline.
from pathlib import Path

N_POINTS = 201  # Includes the solver's two boundary/ghost points.
NOZZLE = dict(L=0.2, A_inlet=3.0e-4, A_throat=1.5e-4)
P_STAGNATION = 2.0e5  # Pa
T_STAGNATION = 300.0  # K
# Shock fractions are measured along the divergent section. The last two cases
# take the subsonic area-Mach root on BOTH sides of the throat. M_throat=1 is
# the exactly sonic limiting solution; M_throat<1 is unchoked everywhere.
CASES = (
    dict(label="No shock (supersonic exit)", branch="sup"),
    dict(label="Shock at 35% of divergence", branch="shock", shock_fraction=0.35),
    dict(label="Shock at 70% of divergence", branch="shock", shock_fraction=0.70),
    dict(label="Sonic throat, subsonic exit", branch="subsonic", throat_mach=1.0),
    dict(label="Subsonic throughout", branch="subsonic", throat_mach=0.60),
)
N_EXACT = 1601
SOLVER = dict(
    time_method="fvs_newton", space_method="fvs",
    reconstruction="muscl_van_leer", initialization="simple",
    CFL=10.0, max_iterations=5000, residuals=1.0e-8,
    delta_factor=1.0e-6, newton_iter=1, newton_tol=1.0e-8,
    jacobian="colored", linear_solver="banded", adaptive_dt=True,
    flux_local_jac=True, cell_local_jac=True, fast_increments=True,
    ser_beta=1.0, ser_cfl_max=1.0e6, dt_growth_factor=1.5, eps_filter=0.0,
)
MAX_NORMALIZED_L1 = 0.03  # Baseline check, not a mesh-convergence claim.
MAX_MASS_IMBALANCE = 1.0e-4
OUTPUT_DIR = Path(__file__).resolve().parent / "results_euler_nozzle_steady"

# %% Imports and precision.
import argparse
import json
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import (
    subsonic_reference,
)
from nozzle_hyperbolic.models.euler_nozzle.analytical.general_eos import (
    nozzle_exact_real_gas,
)
from nozzle_hyperbolic.models.euler_nozzle.fvm import (
    fvm_euler_1d,
    make_jax_stadtke_step,
)




def solution_face_mass_flux(primitive, parameters):
    """Audit with the factory's actual flux, reconstruction, and boundaries.

    Rebuilding only the diagnostic callable avoids silently evaluating a
    corrected flux or ordinary Van Leer on a legacy/smoothed solver result.
    No extra timestep is advanced and no numerical formula is duplicated.
    """
    geometry = parameters["geometry"]
    area = np.asarray(parameters["A_nodes"])
    face_area = 0.5 * (area[:-1] + area[1:])
    gradient = np.asarray(geometry.dAdx).copy()
    gradient[1:-1] = np.diff(face_area) / parameters["dx"]
    fluid = parameters["fluid"]
    reservoir = fluid.get_state(jxp.PT_INPUTS, *parameters["pT_stagnation"])
    options = dict(
        recon=parameters["reconstruction"],
        delta_factor=parameters["delta_factor"],
        wall_friction=parameters["wall_friction"],
    )
    for name in ("flux_variant", "smooth_limiter_coefficient", "limiter_scales"):
        if name in parameters:
            options[name] = parameters[name]
    _, _, _, evaluate = make_jax_stadtke_step(
        fluid, area, face_area, gradient, geometry.D_h, geometry.roughness,
        parameters["dx"], parameters["CFL"], parameters["p_out"],
        float(reservoir["h"]), float(reservoir["s"]), **options,
    )
    return np.asarray(evaluate(primitive))


def main(show=True, *, solver_overrides=None, output_dir=None, strict=True):
    """Run the shared cases; optional controls support matched comparisons.

    Defaults retain the standalone verification. Overrides change numerical
    settings only; physical data, initialization, references, and thresholds
    remain shared. strict=False saves and returns failed checks for analysis.
    """
    # The existing solver prints Unicode diagnostics; make Windows redirects safe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = dict(SOLVER)
    if solver_overrides is not None:
        settings.update(solver_overrides)
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    # 1. Build one fluid and one numerical mesh, reused at every back pressure.
    started = perf_counter()
    fluid = jxp.FluidPerfectGas(name="Air")
    x = np.linspace(0.0, NOZZLE["L"], N_POINTS)
    dx = float(x[1] - x[0])
    geometry = get_symmetric_nozzle(x, **NOZZLE)
    area = np.asarray(geometry.A)
    face_area = 0.5 * (area[:-1] + area[1:])
    x_exact = np.linspace(x[0], x[-1], N_EXACT)
    area_exact = np.asarray(get_symmetric_nozzle(x_exact, **NOZZLE).A)
    x_throat = 0.5 * NOZZLE["L"]
    colors = plt.colormaps["plasma"](np.linspace(0.2, 0.8, len(CASES)))
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    summaries, pressure_handles = [], []

    # 2. Each line is a different back pressure; numerical settings stay fixed.
    for index, (case, color) in enumerate(zip(CASES, colors)):
        label = case["label"]
        fraction = case.get("shock_fraction")
        shock_x = None if fraction is None else (
            x_throat + fraction * (x[-1] - x_throat)
        )
        reference_args = dict(
            fluid=fluid, p0=P_STAGNATION, T0=T_STAGNATION,
            x_shock=shock_x, A_throat=NOZZLE["A_throat"],
        )
        # This general-EOS reference reduces to the classical area-Mach and
        # normal-shock relations when supplied with FluidPerfectGas.
        if case["branch"] == "subsonic":
            curve = subsonic_reference(
                area_exact, fluid, case["throat_mach"],
                throat_area=NOZZLE["A_throat"], p0=P_STAGNATION, T0=T_STAGNATION,
            )
            reference = subsonic_reference(
                area, fluid, case["throat_mach"],
                throat_area=NOZZLE["A_throat"], p0=P_STAGNATION, T0=T_STAGNATION,
            )
            rho0 = P_STAGNATION / (float(fluid.constants.R) * T_STAGNATION)
        else:
            exact = nozzle_exact_real_gas(x_exact, area_exact, **reference_args)
            exact_nodes = nozzle_exact_real_gas(x, area, **reference_args)
            curve, reference = exact[case["branch"]], exact_nodes[case["branch"]]
            rho0 = exact["rho0"]
        p_back = float(curve["p"][-1])
        print(f"\n=== {label}: p_back/p0 = {p_back / P_STAGNATION:.6f} ===", flush=True)

        # 3. Call the current EulerNozzle driver. The legacy perfect_gas flag
        # selects an implementation path: False uses general-EOS FVS with the
        # actual perfect-gas fluid object constructed above.
        parameters = dict(
            **settings, fluid=fluid, fluid_name="Air", perfect_gas=False,
            geometry=geometry, A_nodes=area, dx=dx, N_eq=3,
            pT_stagnation=np.array([P_STAGNATION, T_STAGNATION]),
            p_out=p_back, nozzle_type="nozzle", wall_friction=False,
            Q_o=lambda xx: np.zeros((3, len(xx))),
        )
        # Q_o only fills the driver's initial output slot; the FVS path builds
        # its own simple initial flow. No exact field seeds the numerical solve.
        # This is steady pseudo-time marching, so the end time is unlimited.
        fvm_euler_1d(x, np.array([0.0, np.inf]), parameters)
        primitive = jnp.asarray(parameters["U_puh_final"])
        state = fluid.get_state(jxp.HmassP_INPUTS, primitive[:, 2], primitive[:, 0])
        numerical = dict(
            p=np.asarray(primitive[:, 0]), rho=np.asarray(state["rho"]),
            M=np.asarray(primitive[:, 1] / state["a"]),
        )
        if not all(np.isfinite(values).all() for values in numerical.values()):
            raise RuntimeError(f"Non-finite solution for {label}")

        # 4. Quantify errors on physical points. Report the full-domain L1;
        # additionally report smooth-region errors without hiding shock error.
        physical = np.zeros(N_POINTS, dtype=bool)
        physical[1:-1] = True
        smooth = physical.copy()
        if shock_x is not None:
            smooth &= np.abs(x - shock_x) > 3.0 * dx
        scales = dict(p=P_STAGNATION, rho=rho0, M=1.0)
        errors = {
            region: {key: float(np.mean(np.abs(numerical[key][mask] - reference[key][mask])) / scales[key])
                     for key in scales}
            for region, mask in (("all_physical_points", physical), ("away_from_shock", smooth))
        }
        # Match the selected reconstruction and FVS flux, including fallbacks.
        mass_flux = solution_face_mass_flux(primitive, parameters)
        mass_imbalance = float(abs(mass_flux[-1] - mass_flux[0]) / abs(mass_flux[0]))
        stats = parameters["run_stats"].copy()
        stats["pseudo_time_s"] = stats.pop("t_phys")
        converged = bool(stats["ssr"] < settings["residuals"])
        passed = bool(converged and mass_imbalance < MAX_MASS_IMBALANCE
                      and max(errors["all_physical_points"].values()) < MAX_NORMALIZED_L1)
        summary = dict(
            case=label, back_pressure_Pa=p_back, back_pressure_ratio=p_back / P_STAGNATION,
            shock_exact_m=shock_x, **stats, converged=converged,
            normalized_L1=errors, relative_boundary_mass_imbalance=mass_imbalance,
            throat_M_exact=float(reference["M"][np.argmin(area)]),
            throat_M_numerical=float(numerical["M"][np.argmin(area)]),
            maximum_M_numerical=float(np.max(numerical["M"][physical])),
            passed=passed,
        )
        if shock_x is not None:
            # Strongest downstream pressure rise estimates the captured shock.
            faces = np.flatnonzero((x[:-1] > x_throat) & (x[1:] < x[-1]))
            k = faces[np.argmax(np.diff(numerical["p"])[faces])]
            summary["shock_position_error_cells"] = float((0.5 * (x[k] + x[k + 1]) - shock_x) / dx)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

        # 5. Save numerical/reference values and draw matching colors. Solid
        # lines are exact; hollow circles are the numerical solution.
        columns = [x, area] + [v for key in ("p", "rho", "M")
                               for v in (numerical[key], reference[key])]
        np.savetxt(destination / f"case_{index + 1}.csv", np.column_stack(columns),
                   delimiter=",", header="x_m,area_m2,p_num_Pa,p_exact_Pa,rho_num,rho_exact,M_num,M_exact",
                   comments="")
        for ax, key, ylabel in zip(axes, scales, (r"$p/p_0$", r"$\rho/\rho_0$", "Mach number")):
            ax.plot(x_exact, curve[key] / scales[key], color=color, lw=1.5)
            ax.plot(x[physical], numerical[key][physical] / scales[key], "o", color=color,
                    ms=3.2, mfc="none", markevery=max(1, (N_POINTS - 2) // 45))
            ax.set(xlabel="x [m]", ylabel=ylabel)
            ax.grid(alpha=0.2)
        pressure_handles.append(Line2D([], [], color=color,
                                       label=rf"$p_b/p_0={p_back / P_STAGNATION:.3f}$"))

    axes[0].legend(handles=pressure_handles, title="Back pressure", fontsize=9)
    axes[1].legend(handles=[Line2D([], [], color="0.25", label="Exact"),
                           Line2D([], [], color="0.25", marker="o", mfc="none",
                                  linestyle="none", label="FVM")], fontsize=9)
    fig.suptitle(f"EulerNozzle steady: {settings.get('flux_variant', 'installed')} FVS / "
                 f"{settings['reconstruction']} / backward Euler, N={N_POINTS}")
    fig.savefig(destination / "comparison.png", dpi=180)
    fig.savefig(destination / "comparison.pdf")
    report = dict(
        fluid="jaxprop.FluidPerfectGas(Air)", gamma=float(fluid.constants.gamma),
        numerical_points=N_POINTS, geometry=NOZZLE, solver=settings,
        thresholds=dict(normalized_L1=MAX_NORMALIZED_L1, mass_imbalance=MAX_MASS_IMBALANCE),
        cases=summaries, elapsed_s=perf_counter() - started,
    )
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved steady comparison to {destination}", flush=True)
    if show:
        plt.show()
    else:
        plt.close(fig)
    if strict and not all(case["passed"] for case in summaries):
        raise SystemExit("Steady baseline failed; see summary.json for convergence and errors.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening a window.")
    main(show=not parser.parse_args().no_show)
