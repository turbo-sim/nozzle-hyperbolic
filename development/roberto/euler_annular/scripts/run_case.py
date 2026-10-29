"""Run one YAML-defined annular case with live equation and quantity monitors.

From the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.run_case \
      --config development/roberto/euler_annular/cases/verification.yaml \
      --case radial_outflow --cells 100 --reconstruction constant
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import textwrap

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.euler_annular import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.monitoring import EQUATIONS
from nozzle_hyperbolic.solvers.convergence_output import ConvergenceTable, strict_json
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy

from .case_config import (DEFAULT_CONFIG, STUDY, available_cases, build_case,
                          load_case_config, validate_config)


def _message(text):
    print(textwrap.fill(text, width=100))


def source_fingerprints():
    """Capture the full package and runner sources before and after a solve."""
    repository = STUDY.parents[2]
    sources = [*(repository / "src" / "nozzle_hyperbolic").rglob("*.py"),
               Path(__file__), Path(__file__).with_name("case_config.py")]
    return {str(path.relative_to(repository)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)}


def build_solver(case, *, cells=None, reconstruction=None):
    """Resolve physical data once, keeping analytical fields out of the seed."""
    config = deepcopy(case.config)
    if cells is not None:
        config["mesh"]["cells"] = cells
    if reconstruction is not None:
        config["numerical"]["reconstruction"] = reconstruction
    validate_config(config)
    mesh = case.geometry.mesh(config["mesh"]["cells"])
    inlet = config["inlet"]
    inlet_data = (dict(state=np.asarray(case.reference.profile(0.0)["W"]))
                  if case.inlet_regime == "supersonic" else
                  dict(p0=inlet["p0"], T0=inlet["T0"], alpha_in=case.alpha))
    pressure = config["boundaries"]["back_pressure"]
    if pressure == "reference":
        pressure = float(case.reference.profile(case.geometry.length, shock_m=case.shock_m)["p"])
    solver = make_annular_solver(
        case.fluid,
        mesh, inlet_data=inlet_data, outlet_pressure=pressure,
        inlet_regime=case.inlet_regime, outlet_regime=case.outlet_regime,
        tau_w=config["walls"]["shear_stress"], q_w=config["walls"]["heat_flux"],
        **config["numerical"])
    specs = []
    for definition in config["monitors"]:
        definition = deepcopy(definition)
        if definition["quantity"] == "pressure_coefficient":
            definition.setdefault("parameters", {}).setdefault("pressure_total", inlet["p0"])
        specs.append(MonitorSpec(**definition))
    return solver, config, ResidualPolicy(**config["residuals"]), tuple(specs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="YAML input (repository-relative or absolute; default shared verification file)")
    parser.add_argument("--case", help="Named case override; defaults to YAML default_case")
    parser.add_argument("--cells", type=int, help="Override the physical cell count")
    parser.add_argument("--reconstruction", choices=("constant", "linear", "muscl_van_leer"),
                        help="linear is an alias for muscl_van_leer")
    parser.add_argument("--max-steps", type=int, help="Override the accepted-update limit")
    parser.add_argument("--print-every", type=int, help="Print every N accepted updates; retries always print")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs" / "single_case",
                        help="Output directory or explicit .json filename")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved inputs without solving")
    args = parser.parse_args(argv)
    config_path = args.config
    if not config_path.is_file() and (STUDY / config_path).is_file():
        config_path = STUDY / config_path
    if args.list_cases:
        print("\n".join(available_cases(config_path)))
        return 0
    resolved = load_case_config(config_path, args.case)
    config = resolved["config"]
    if args.cells is not None:
        config["mesh"]["cells"] = args.cells
    if args.reconstruction is not None:
        config["numerical"]["reconstruction"] = "muscl_van_leer" if args.reconstruction == "linear" else args.reconstruction
    for argument, key in ((args.max_steps, "max_steps"), (args.print_every, "print_every")):
        if argument is not None:
            config["run"][key] = argument
    validate_config(config)
    if args.dry_run:
        print(strict_json(resolved), end="")
        return 0
    fluid = jxp.FluidPerfectGas(name=config["fluid"]["name"])
    case = build_case(resolved["name"], fluid, config_path=config_path)
    if case.config_provenance != resolved["provenance"]:
        raise RuntimeError("YAML inputs changed while constructing the case")
    case.config = config
    case.fluid = fluid
    source_before = source_fingerprints()
    solver, config, policy, monitors = build_solver(case)
    run_settings = dict(config["run"])
    every = run_settings.pop("print_every", 1)
    table = ConvergenceTable(EQUATIONS, every=every)
    _message(f"Case: {case.name} | {solver.mesh.n_cells} cells | {solver.options.reconstruction}")
    _message(f"Stopping: every active equation <= {policy.relative_tolerance:g} of its initial norm, "
          "with configured absolute/BC/balance safeguards.")
    _message("Integral monitors are saved; configured stability checks remain part of convergence.")
    solution = solver.solve_steady_monitored(
        residual_policy=policy, monitors=monitors, pressure_total=config["inlet"]["p0"],
        callback=table, **run_settings)
    if solution.history and table.last_step != solution.history[-1]["step"]:
        table(solution.history[-1], force=True)
    source_after = source_fingerprints()
    yaml_after = {name: sha256(Path(name).read_bytes()).hexdigest()
                  for name in resolved["provenance"]["source_sha256"]}
    source_unchanged = source_before == source_after
    yaml_unchanged = yaml_after == resolved["provenance"]["source_sha256"]
    integrity_passed = source_unchanged and yaml_unchanged
    success = solution.success and integrity_passed
    message = (solution.message if integrity_passed else
               "Numerical result retained but source or YAML inputs changed during the solve")
    artifact = dict(schema_version=1, case=case.name, cells=solver.mesh.n_cells,
        reconstruction=solver.options.reconstruction, resolved_config=config,
        config_provenance=resolved["provenance"], source_hashes=source_before,
        source_hashes_after=source_after, source_unchanged=source_unchanged,
        yaml_sha256_after=yaml_after, yaml_unchanged=yaml_unchanged,
        initial_state=np.asarray(solver.initial_state()), state=solution.state,
        m=solver.mesh.m_centers, m_faces=solver.mesh.m_faces,
        volumes=solver.mesh.volumes, fields=solver.fields(jnp.asarray(solution.state)),
        success=success, message=message, final_metrics=solution.final_metrics,
        monitoring=solution.monitoring, history=solution.history,
        numerical_options=asdict(solver.options),
        initialization="uniform physical cold guess; reference only selects physical boundary data")
    path = args.output
    if path.suffix.lower() != ".json":
        path = path / f"{case.name}_{solver.mesh.n_cells}_{solver.options.reconstruction}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json(artifact), encoding="utf-8")
    _message(f"Status: {message}")
    _message(f"Saved: {path.resolve()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
