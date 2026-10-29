"""Independent analytic/constant/MUSCL verification of selected annular cases.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.verify_annular
Numerical starts default to uniform physical cold guesses. The grid-sequence
driver can supply explicit transferred states; analytical profiles never seed a solve.
"""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .conservation_history import ConservationHistoryRecorder
from .case_config import Case, build_case

from nozzle_hyperbolic.models.euler_annular import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.state import primitive_to_conservative
from nozzle_hyperbolic.solvers.convergence_output import strict_json


STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[4]
RECONSTRUCTIONS = ("constant", "muscl_van_leer")
LABELS = ("Analytical reference", "FVM (constant reconstruction)",
          "FVM (MUSCL reconstruction)")
DEFAULT_MESHES = (50, 100, 200, 400)
DEFAULT_CASES = ("radial_outflow", "radial_inflow", "sonic_exit", "swirl_diffusion",
                 "conical_subsonic", "curved_subsonic", "supersonic_expansion",
                 "shock35", "shock70")
CASES = DEFAULT_CASES+("radial_outflow_m005", "radial_outflow_m020", "radial_outflow_swirl")


def make_case(name, fluid):
    """Construct the named verification case from the shared YAML definitions."""
    return build_case(name, fluid)


def reference_fields(case, mesh, W):
    """Equivalent exact cell states, with independent perfect-gas diagnostics."""
    reference = case.reference
    temperature = (W[:, 3]-reference.h_shift)/reference.cp
    density = W[:, 0]/(reference.R*temperature)
    sound = np.sqrt(reference.gamma*reference.R*temperature)
    v_theta = W[:, 2]/mesh.r_center
    return dict(p=W[:, 0], v_m=W[:, 1], v_theta=v_theta, rho=density,
                T=temperature, M_m=W[:, 1]/sound,
                M_theta=v_theta/sound, alpha_deg=np.rad2deg(np.arctan2(v_theta, W[:, 1])),
                M=np.hypot(W[:, 1], v_theta)/sound,
                h0=W[:, 3]+0.5*(W[:, 1]**2+v_theta**2), ell=W[:, 2])


def fingerprints():
    relevant = [
        *list((REPOSITORY/"src/nozzle_hyperbolic/models/euler_annular").rglob("*.py")),
        REPOSITORY/"src/nozzle_hyperbolic/geometry/annular.py",
        REPOSITORY/"src/nozzle_hyperbolic/numerics/characteristic.py",
        REPOSITORY/"src/nozzle_hyperbolic/numerics/jacobians.py",
        REPOSITORY/"src/nozzle_hyperbolic/numerics/linear.py",
        REPOSITORY/"src/nozzle_hyperbolic/numerics/reconstruction.py",
        REPOSITORY/"src/nozzle_hyperbolic/solvers/finite_volume.py",
        REPOSITORY/"src/nozzle_hyperbolic/solvers/residuals.py",
        REPOSITORY/"src/nozzle_hyperbolic/solvers/monitoring.py",
        REPOSITORY/"src/nozzle_hyperbolic/solvers/time_marching/implicit.py",
        REPOSITORY/"src/nozzle_hyperbolic/thermodynamics/adapters.py",
    ]
    return {str(path.relative_to(REPOSITORY)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(relevant)}


def write_json(path, value):
    if isinstance(value, dict) and (value.get("settings", {}).get("controller") == "equation_monitored"):
        path.write_text(strict_json(value), encoding="utf-8")
        return
    def convert(item):
        if isinstance(item, (np.ndarray, jax.Array)):
            return np.asarray(item).tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=True)+"\n", encoding="utf-8")


def verify(case, fluid, cells, output, max_steps, tolerance, cfl, *, mesh=None,
           delta_factor=0.1, flux_method="characteristic", controller="native",
           initial_states=None, initialization_metadata=None):
    supplied_mesh = mesh is not None
    mesh = case.geometry.mesh(cells) if mesh is None else mesh
    if mesh.n_cells != cells:
        raise ValueError("The supplied mesh cell count differs from the requested cells")
    projected = case.reference.cell_average(mesh, shock_m=case.shock_m, return_conserved=True)
    ref_fields = reference_fields(case, mesh, projected["W"])
    inlet_exact = case.reference.profile(0.0)
    outlet_exact = case.reference.profile(case.geometry.length, shock_m=case.shock_m)
    pressure_back = float(outlet_exact["p"])
    pressure0 = float(inlet_exact["p0"])
    pressure_in = float(inlet_exact["p"])
    ref_fields["Cp"] = (ref_fields["p"]-pressure_in)/(pressure0-pressure_in)
    inlet_data = (dict(state=np.asarray(inlet_exact["W"])) if case.inlet_regime == "supersonic"
                  else dict(p0=pressure0, T0=float(case.reference.T0), alpha_in=case.alpha))
    config = case.config or {}
    if controller == "equation_monitored" and not config:
        raise ValueError("Equation-monitored verification requires resolved YAML case inputs")
    if config.get("boundaries", {}).get("back_pressure", "reference") != "reference":
        pressure_back = float(config["boundaries"]["back_pressure"])
    result = dict(case=case.name, cells=cells, m=mesh.m_centers, m_faces=mesh.m_faces,
                  volumes=mesh.volumes, r=mesh.r_center, backpressure=pressure_back,
                  inlet_regime=case.inlet_regime, outlet_regime=case.outlet_regime,
                  alpha_in=case.alpha, shock_m=case.shock_m, reference=ref_fields,
                  fluid="jaxprop.FluidPerfectGas(Air)",
                  reference_invariants={key: float(getattr(case.reference, key)) for key in
                                        ("gamma", "R", "h_shift", "entropy_shift",
                                         "mdot", "h0", "K", "ell")},
                  reservoir=dict(p0=pressure0, T0=float(case.reference.T0)),
                  reference_W=projected["W"], reference_q=projected["q"], methods={},
                  versions={name: metadata.version(name) for name in
                            ("jax", "jaxprop", "nurbspy", "numpy", "scipy")},
                  source_hashes=fingerprints(),
                  case_config=config, case_config_provenance=case.config_provenance,
                  settings=dict(tolerance=tolerance, cfl=cfl, max_steps=max_steps,
                                mesh_definition="provided faces" if supplied_mesh else "uniform arclength",
                                delta_factor=delta_factor, flux_method=flux_method,
                                boundary_formulation="coupled",
                                controller="accepted Newton work; failed timestep halving",
                                conservation_history="signed integrated balance / fixed global flux scales",
                                initial_state="uniform physical cold guess"))
    if initial_states is not None:
        if set(initial_states) != set(RECONSTRUCTIONS):
            raise ValueError("Transferred initial states must include both reconstructions")
        result["settings"]["initial_state"] = "bounded linear primitive transfer from previous converged mesh"
    result["initialization"] = initialization_metadata or dict(strategy="uniform cold physical guess")
    if controller == "cfl_recommended":
        from .recommended_controller import SETTINGS, driver_hashes, solve_recommended
        result["settings"].update(SETTINGS)
        result["settings"]["cfl"] = cfl
        result["settings"]["controller"] = "cfl_recommended"
        result["controller_provenance"] = dict(
            recommendation="REPORT.html#notes-controller", driver_sha256=driver_hashes())
    elif controller == "equation_monitored":
        result["settings"].update(controller=controller, residual_policy=config["residuals"],
                                  monitors=config["monitors"], **{
                                      key: value for key, value in config["run"].items()
                                      if key not in ("cfl", "max_steps", "print_every")})
    for reconstruction in RECONSTRUCTIONS:
        print(f"Starting {case.name} N={cells} {reconstruction}; controller={controller}", flush=True)
        numerical = dict(config.get("numerical", {})) if controller == "equation_monitored" else {}
        numerical.update(reconstruction=reconstruction, delta_factor=delta_factor, flux_method=flux_method)
        if controller == "cfl_recommended":
            numerical["newton_iterations"] = 8
        solver = make_annular_solver(
            fluid, mesh, inlet_data=inlet_data, outlet_pressure=pressure_back,
            inlet_regime=case.inlet_regime, outlet_regime=case.outlet_regime,
            tau_w=config.get("walls", {}).get("shear_stress", 0.0),
            q_w=config.get("walls", {}).get("heat_flux", 0.0), **numerical)
        initial = (solver.initial_state() if initial_states is None
                   else jnp.asarray(initial_states[reconstruction], dtype=jnp.float64))
        if initial.shape != solver.shape or not bool(solver.is_admissible(initial)):
            raise ValueError(f"Invalid initial state for {case.name} N={cells} {reconstruction}")
        def progress(row):
            if row.get("accepted", False) and row["step"] % 500 == 0:
                print(f"{case.name} N={cells} {reconstruction}: step={row['step']} "
                      f"spatial={row['spatial_rms']:.3g} cfl={row['cfl']:.3g}", flush=True)
        controller_audit = None
        if controller == "equation_monitored":
            from nozzle_hyperbolic.solvers.monitoring import MonitorSpec
            from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
            monitors = [MonitorSpec(**item) for item in config["monitors"]]
            options = {key: value for key, value in config["run"].items()
                       if key not in ("print_every", "max_steps", "cfl")}
            solution = solver.solve_steady_monitored(
                initial, residual_policy=ResidualPolicy(**config["residuals"]), monitors=monitors,
                pressure_total=pressure0, max_steps=max_steps, cfl=cfl, callback=progress, **options)
            controller_audit = dict(
                accepted=solution.monitoring["accepted_updates"],
                attempts=solution.monitoring["attempts"],
                rejected=solution.monitoring["rejected_attempts"],
                final_cfl=solution.monitoring["final_cfl"],
                pseudo_time=solution.monitoring["pseudo_time"])
        elif controller == "cfl_recommended":
            solution, controller_audit = solve_recommended(
                solver, initial, tolerance=tolerance, max_steps=max_steps, cfl=cfl)
        else:
            with ConservationHistoryRecorder(solver, progress) as monitor:
                solution = solver.solve_steady(initial, tolerance=tolerance, max_steps=max_steps,
                                               cfl=cfl, callback=monitor)
        fields = {key: np.asarray(value) for key, value in solver.fields(jnp.asarray(solution.state)).items()}
        fields["alpha_deg"] = np.rad2deg(np.arctan2(fields["v_theta"], fields["v_m"]))
        numerical_inlet_pressure = solution.state[0, 0]
        fields["Cp"] = (fields["p"]-numerical_inlet_pressure)/(pressure0-numerical_inlet_pressure)
        conservative = np.asarray(primitive_to_conservative(
            solution.state[1:-1], mesh.r_center, fluid))
        scales = np.asarray([solver.reference_density,
                             solver.reference_density*solver.reference_sound_speed,
                             solver.reference_density*mesh.r_face[0]*solver.reference_sound_speed,
                             solver.reference_density*solver.reference_sound_speed**2])
        error = dict(
            Cp_max=float(np.max(np.abs(fields["Cp"]-ref_fields["Cp"]))),
            Cp_mean_absolute=float(np.sum(mesh.widths*np.abs(fields["Cp"]-ref_fields["Cp"]))/np.sum(mesh.widths)),
            conservative_L1=(np.sum(mesh.volumes[:, None]*np.abs(conservative-projected["q"]), axis=0)
                              /(np.sum(mesh.volumes)*scales)).tolist(),
            p_L1=float(np.sum(mesh.volumes*np.abs(fields["p"]-ref_fields["p"]))
                       /np.sum(mesh.volumes)/pressure0))
        face_flux = np.asarray(solver.face_flux(jnp.asarray(solution.state)))
        integrated_source = np.asarray(solver.source(jnp.asarray(solution.state)))
        balance = face_flux[-1]-face_flux[0]-np.sum(integrated_source, axis=0)
        balance_scales = float(mesh.A_face[0])*solver.reference_sound_speed*scales
        angular_scale = (abs(case.reference.mdot*case.reference.ell)
                         if case.reference.ell != 0 else abs(case.reference.mdot*solver.scales["ell"]))
        energy_scale = max(abs(case.reference.mdot*case.reference.h0),
                           abs(case.reference.mdot*solver.reference_sound_speed**2))
        error["mass_flow_relative"] = float(abs(np.mean(face_flux[:, 0])-case.reference.mdot)
                                            /case.reference.mdot)
        error["face_mass_spread_relative"] = float(np.ptp(face_flux[:, 0])/abs(case.reference.mdot))
        error["angular_flux_spread_scaled"] = float(np.ptp(face_flux[:, 2])/angular_scale)
        error["energy_flux_spread_scaled"] = float(np.ptp(face_flux[:, 3])/energy_scale)
        error["integrated_balance"] = balance.tolist()
        error["integrated_balance_scales"] = balance_scales.tolist()
        error["integrated_balance_scaled"] = (balance/balance_scales).tolist()
        error["angular_flux_normalization"] = angular_scale
        error["energy_flux_normalization"] = energy_scale
        if case.shock_m is not None:
            location_index = int(np.argmin(np.diff(fields["v_m"])/np.diff(mesh.m_centers)))
            shock_estimate = float(np.mean(mesh.m_centers[location_index:location_index+2]))
            error["shock_m"] = shock_estimate
            error["shock_error_cells"] = abs(shock_estimate-case.shock_m)/float(np.max(mesh.widths))
            analytic_pass = error["shock_error_cells"] <= 2
        else:
            analytic_pass = error["Cp_mean_absolute"] <= 0.01 and max(error["conservative_L1"]) < 1e-3
        result["methods"][reconstruction] = dict(
            options=asdict(solver.options), controller_audit=controller_audit,
            success=solution.success, message=solution.message, metrics=solution.final_metrics,
            analytical_target_met=analytic_pass, errors=error, state=solution.state,
            initial_state=np.asarray(initial), fields=fields, face_flux=face_flux,
            history=solution.history, monitoring=solution.monitoring,
            integrated_source=integrated_source,
            boundary_constraints=np.asarray(solver.boundary_residual(jnp.asarray(solution.state))),
            physical_spatial_residual=np.asarray(solver.spatial_residual(jnp.asarray(solution.state))))
        print(f"{case.name} N={cells} {reconstruction}: root={solution.success} "
              f"updates={sum(row['accepted'] and row.get('phase') != 'initial' for row in solution.history)} "
              f"Cp_mean_error={error['Cp_mean_absolute']:.4g} "
              f"spatial={solution.final_metrics['spatial_rms']:.3g}", flush=True)
        del solver
    write_json(output/f"{case.name}_{cells}.json", result)
    plot_result(result, output)
    return result


def plot_result(result, output):
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, 3))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    fields = (("p", "Static pressure [Pa]"), ("rho", "Density [kg/m3]"),
              ("alpha_deg", "Flow angle [deg]"), ("M_m", "Meridional Mach"),
              ("M_theta", "Tangential Mach"), ("M", "Mach magnitude"))
    x = np.asarray(result["m"])
    for ax, (key, ylabel) in zip(axes.ravel(), fields):
        ax.plot(x, result["reference"][key], color=colors[0], label=LABELS[0], lw=1.8)
        for index, (method, style) in enumerate(zip(RECONSTRUCTIONS, ("--", "-.")), 1):
            run = result["methods"][method]
            label = LABELS[index]+(" (not converged)" if not run["success"] else "")
            values = run["fields"].get(key)
            if values is None:
                f = run["fields"]
                values = (np.rad2deg(np.arctan2(f["v_theta"], f["v_m"])) if key=="alpha_deg"
                          else np.asarray(f["v_theta"])/np.asarray(f["a"]))
            ax.plot(x, values, style, color=colors[index], label=label)
        ax.set(xlabel="Meridional arclength [m]", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"{result['case']}, {result['cells']} physical cells; conservative reference projection")
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{result['case']}_{result['cells']}_fields.{suffix}", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for index, method in enumerate(RECONSTRUCTIONS, 1):
        records = [row for row in result["methods"][method]["history"] if row["accepted"]]
        if result["methods"][method].get("monitoring"):
            for k, name in enumerate(("mass", "meridional", "angular", "energy")):
                values = [row["equation_relative"][k] if row["relative_active"][k] else np.nan
                          for row in records]
                ax.semilogy([row["step"] for row in records], values,
                            ("-", "--", "-.", ":")[k], color=colors[index],
                            label=f"{'constant' if index == 1 else 'MUSCL'}: {name}")
        else:
            ax.semilogy([row["step"] for row in records],
                        [max(row["spatial_rms"], row["boundary_max"], row["conservation_max"])
                         for row in records], color=colors[index], label=LABELS[index])
    monitored = result["settings"]["controller"] == "equation_monitored"
    target = result["settings"]["residual_policy"]["relative_tolerance"] if monitored else result["settings"]["tolerance"]
    ax.axhline(target, ls=":", color="0.5", label="Residual target")
    ax.set(xlabel="Accepted implicit updates (initial state at zero)", ylabel=(
        "Equation RMS / initial equation RMS" if monitored else "Max(spatial RMS, BC max, balance defect)"))
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"{result['case']}_{result['cells']}_convergence.{suffix}", dpi=160)
    plt.close(fig)


def report(results, output):
    from .accuracy import assessed_errors, accuracy_target_met
    lines = ["# Annular Euler verification results", "",
             "The general-EOS FVM uses a JAXProp perfect-gas backend; flux and entropy-width settings are saved per case.",
             "Both reconstructions start from uniform cold fields. BCs are simultaneous algebraic blocks.",
             "Cp uses each solution's inlet static pressure and the prescribed reservoir pressure.",
             "Root convergence and analytical accuracy are reported separately; failures are retained.", "",
             "Cp accuracy uses the meridional-length mean absolute difference over all physical cells, with a limit of 0.01 Cp units.",
             "| Case | Cells | Reconstruction | Root | Spatial RMS | BC max | Cp mean absolute error | Analytic target |",
             "|---|---:|---|---|---:|---:|---:|---|"]
    for result in results:
        for method, run in result["methods"].items():
            errors = assessed_errors(result, run)
            lines.append(f"| {result['case']} | {result['cells']} | {method} | "
                         f"{run['success']} | {run['metrics']['spatial_rms']:.3e} | "
                         f"{run['metrics']['boundary_max']:.3e} | {errors['Cp_mean_absolute']:.3e} | "
                         f"{accuracy_target_met(result, run)} |")
    lines.extend(["", "## Field comparisons", ""])
    for result in results:
        name = f"{result['case']}_{result['cells']}"
        lines.extend([f"### {result['case']}, {result['cells']} cells", "",
                      f"![Analytical, constant and MUSCL fields]({name}_fields.png)", "",
                      f"[Vector figure]({name}_fields.pdf). [Numerical convergence]({name}_convergence.png).", ""])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    columns = ("case", "cells", "reconstruction", "root_converged", "analytical_target_met",
               "spatial_rms", "boundary_max", "conservation_max", "Cp_max", "Cp_mean_absolute",
               "conservative_L1_mass", "conservative_L1_momentum", "conservative_L1_angular",
               "conservative_L1_energy", "mass_flow_relative", "face_mass_spread_relative",
               "angular_flux_spread_scaled", "energy_flux_spread_scaled", "shock_error_cells")
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for method, run in result["methods"].items():
                errors = assessed_errors(result, run)
                row = dict(case=result["case"], cells=result["cells"], reconstruction=method,
                           root_converged=run["success"],
                           analytical_target_met=accuracy_target_met(result, run))
                row.update({key: run["metrics"][key] for key in
                            ("spatial_rms", "boundary_max", "conservation_max")})
                row.update({key: errors.get(key, "") for key in
                            ("Cp_max", "Cp_mean_absolute", "mass_flow_relative", "face_mass_spread_relative",
                             "angular_flux_spread_scaled", "energy_flux_spread_scaled",
                             "shock_error_cells")})
                row.update({f"conservative_L1_{name}": value for name, value in
                            zip(("mass", "momentum", "angular", "energy"), errors["conservative_L1"])})
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--cells", nargs="+", type=int, default=list(DEFAULT_MESHES))
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--controller", choices=("native", "cfl_recommended"), default="native")
    parser.add_argument("--delta-factor", type=float, default=0.1,
                        help="Positive characteristic entropy-width factor (same for both reconstructions)")
    parser.add_argument("--flux-method", choices=("characteristic", "rusanov"),
                        default="characteristic")
    parser.add_argument("--output", type=Path, default=STUDY/"outputs"/"verification")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed saved case/grid data; never use them as a numerical initial guess")
    parser.add_argument("--root-only", action="store_true",
                        help="Exploratory convergence audit; do not require finest-grid analytical targets")
    args = parser.parse_args()
    if args.cfl is None:
        args.cfl = 0.2 if args.controller == "cfl_recommended" else 20.0
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        results = []
        for path in sorted(args.output.glob("*.json")):
            if path.name.endswith(".audit.json"):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "case" in value and "methods" in value:
                results.append(value)
        for result in results:
            plot_result(result, args.output)
    else:
        fluid = jxp.FluidPerfectGas(name="Air")
        results = []
        for name in args.cases:
            case = make_case(name, fluid)
            for cells in args.cells:
                saved = args.output/f"{name}_{cells}.json"
                if args.resume and saved.exists():
                    data = json.loads(saved.read_text(encoding="utf-8"))
                    settings = data["settings"]
                    expected = dict(tolerance=args.tolerance, cfl=args.cfl, max_steps=args.max_steps,
                                    delta_factor=args.delta_factor)
                    if args.controller == "cfl_recommended":
                        from .recommended_controller import SETTINGS
                        expected.update(SETTINGS)
                        expected["cfl"] = args.cfl
                    if data.get("source_hashes") != fingerprints():
                        raise ValueError(f"Saved numerical source differs for {saved}; use a new output folder")
                    if (settings.get("controller") == "cfl_recommended") != (args.controller == "cfl_recommended"):
                        raise ValueError(f"Saved controller differs for {saved}")
                    if settings.get("flux_method", "characteristic") != args.flux_method:
                        raise ValueError(f"Saved flux method differs for {saved}; choose a different output folder")
                    if any(settings.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Saved settings differ for {saved}; choose a different output folder")
                    if data["case"] != name or data["cells"] != cells:
                        raise ValueError(f"Saved case identity differs for {saved}")
                    results.append(data)
                    print(f"Reusing completed saved data: {name} N={cells}; no numerical warm start", flush=True)
                    report(results, args.output)
                    continue
                results.append(verify(case, fluid, cells, args.output, args.max_steps,
                                      args.tolerance, args.cfl, delta_factor=args.delta_factor,
                                      flux_method=args.flux_method, controller=args.controller))
                report(results, args.output)
                jax.clear_caches()
    report(results, args.output)
    roots_ok = all(run["success"] for result in results for run in result["methods"].values())
    finest = {}
    for result in results:
        if result["case"] not in finest or result["cells"] > finest[result["case"]]["cells"]:
            finest[result["case"]] = result
    accuracy_ok = all(run["analytical_target_met"]
                      for result in finest.values() for run in result["methods"].values())
    write_json(args.output/"acceptance.json", dict(
        root_only=args.root_only, all_roots=roots_ok, finest_grid_accuracy=accuracy_ok,
        success=roots_ok and (args.root_only or accuracy_ok)))
    return 0 if roots_ok and (args.root_only or accuracy_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
