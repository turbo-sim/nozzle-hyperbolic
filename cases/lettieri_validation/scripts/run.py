"""Run equilibrium HEM cases through the shared Euler model and FVM driver.

Examples (from repository root):
  poetry run python -B -m cases.lettieri_validation.scripts.run --prepare-only
  poetry run python -B -m cases.lettieri_validation.scripts.run --case L1 --cells 24 --max-steps 400
  poetry run python -B -m cases.lettieri_validation.scripts.run --case all --cells 100
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from scipy.interpolate import CubicSpline

from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.solvers.fvm import ResidualPolicy, Scheme, SteadySettings, solve_single_grid
from nozzle_hyperbolic.solvers.convergence_output import strict_json
from nozzle_hyperbolic.solvers.residuals import DEFINITION
from .data import ROOT, definitions, experiment, settings, workbook_rows


class BoundedBicubic:
    """Unchanged JAXProp interpolants on an explicitly audited usable rectangle.

    Andrea's broad table includes unsupported low-temperature/low-pressure
    nodes. The study exposes only a rectangle whose complete interpolation
    stencils are populated. No property is clipped, filled, or reconstructed.
    NozzleModel reads these bounds in its normal admissibility checks.
    """

    def __init__(self, source, domain):
        self.source = source
        for name, value in domain.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def get_state(self, input_pair, x, y):
        return self.source.get_state(input_pair, x, y)


def audit_property_domain(fluid, domain):
    """Verify every corner/coefficient used by the fixed admissible rectangle."""
    h = np.asarray(fluid.h_vals)
    p = np.exp(np.asarray(fluid.logP_vals))
    i0, i1 = np.searchsorted(h, [domain["h_min"], domain["h_max"]]) + [-1, 1]
    j0, j1 = np.searchsorted(p, [domain["p_min"], domain["p_max"]]) + [-1, 1]
    if i0 < 0 or j0 < 0 or i1 >= len(h) or j1 >= len(p):
        raise ValueError("Study property domain lacks complete in-table stencils")
    for name in ("density", "temperature", "entropy", "speed_of_sound", "gruneisen", "quality_mass"):
        values = np.asarray(fluid.table[name]["value"])[i0:i1+1, j0:j1+1]
        coefficients = np.asarray(fluid.table[name]["coeffs"])[i0:i1, j0:j1]
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"Nonfinite interpolation stencil for {name}")
        if name in ("density", "temperature", "speed_of_sound") and np.any(values <= 0):
            raise ValueError(f"Unpopulated/invalid interpolation node for {name}")
    return dict(domain=domain, complete_stencil_node_count=int((i1-i0+1)*(j1-j0+1)),
                interpolation_corner_bounds=dict(h_min=float(h[i0]), h_max=float(h[i1]), p_min=float(p[j0]), p_max=float(p[j1])))


def make_mesh(cells, stretch=1.0, width=None):
    """Measured contour with exact spline-integrated volumes on either grid.

    All 33 coordinate pairs come from the upstream workbook. Its one inlet
    half-height typo is corrected explicitly from publication Table 4.
    """
    cfg = settings()["geometry"]
    raw = workbook_rows(ROOT / "data" / cfg["workbook"])[1:]
    x = np.asarray([r["A"] for r in raw])
    y = np.asarray([r["B"] for r in raw])
    y[0] = cfg["inlet_half_height_m"]
    spline = CubicSpline(x, y, extrapolate=False)
    primitive = spline.antiderivative()
    if cells < 2 or not np.isfinite(stretch) or stretch <= 0:
        raise ValueError("At least two cells and positive finite mesh stretch required")
    fraction = np.linspace(0, 1, cells + 1)**stretch
    faces = x[0] + fraction * (x[-1] - x[0])
    centers = 0.5 * (faces[:-1] + faces[1:])
    depth = cfg["width_m"] if width is None else float(width)
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("Positive finite planar depth required")
    area_faces = 2 * depth * spline(faces)
    mesh = Mesh(faces=faces, centers=centers,
                volumes=2 * depth * np.diff(primitive(faces)),
                area_faces=area_faces, area_centers=2 * depth * spline(centers),
                darea_centers=2 * depth * spline(centers, 1),
                area_derivative_integrals=np.diff(area_faces), geometry=spline)
    return mesh, depth


def prepare_fluid(output):
    """Keep selected JAXProp equilibrium bicubic properties and derivatives."""
    cfg = settings()["properties"]
    table_dir = output / "fluid_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table = table_dir / (cfg["table_name"] + ".pkl")
    cached = table.exists()
    log = io.StringIO()
    started = time.perf_counter()
    with redirect_stdout(log), redirect_stderr(log):
        fluid = jxp.FluidBicubic(**cfg, table_dir=str(table_dir))
    elapsed = time.perf_counter() - started
    log_path = output / "properties.log"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(log.getvalue())
    saved_log = log_path.read_text(encoding="utf-8")
    counts = re.findall(r"Successfully evaluated (\d+)/(\d+) points", saved_log)
    generation_times = re.findall(r"Total table generation time: ([\d.]+) s", saved_log)
    domain_audit = audit_property_domain(fluid, settings()["property_domain"])
    metadata = dict(configuration=cfg, preparation_s=elapsed, loaded_existing_table=cached,
                    table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                    table_file=str(table.relative_to(ROOT)), domain_audit=domain_audit,
                    broad_table_generation_counts=counts[-1] if counts else None,
                    initial_generation_s_from_log=float(generation_times[0]) if generation_times else None)
    previous = output / "properties.json"
    preparations = json.loads(previous.read_text(encoding="utf-8")).get("preparations", []) if previous.exists() else []
    preparations.append(dict(metadata))
    previous.write_text(strict_json(metadata | {"preparations": preparations}), encoding="utf-8")
    print(f"Properties prepared in {elapsed:.2f} s; SHA256 {metadata['table_sha256']}", flush=True)
    return BoundedBicubic(fluid, settings()["property_domain"]), metadata


def provenance():
    files = [ROOT / "settings.json", ROOT / "data/publication.json", *sorted((ROOT / "data").glob("*.xlsx"))]
    production = ROOT.parents[1] / "src/nozzle_hyperbolic"
    files += sorted(production.rglob("*.py"))
    return dict(python=platform.python_version(), jax=jax.__version__, jaxprop=getattr(jxp, "__version__", "unknown"),
                backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                files={str(p.relative_to(ROOT.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def run_case(case_id, cells, stretch, max_steps, fluid, property_metadata, output, restart=None, source=None):
    cfg = settings()
    case = definitions()["cases"][case_id]
    tag = f"{case_id}_N{cells}_stretch{stretch:g}"
    folder = output / tag
    attempt = 1
    while folder.exists():
        attempt += 1
        folder = output / f"{tag}_attempt{attempt:02}"
    folder.mkdir(parents=True)
    metadata = dict(case_id=case_id, definition=case, properties=property_metadata, source=source or provenance(), cells=cells,
                    stretch=stretch, residual_definition=DEFINITION, requested_at="2026-11-02",
                    property_domain_audited=True,
                    initialization="model cold Mach ramp" if restart is None else f"explicit restart {restart}")
    start = time.perf_counter()
    try:
        mesh, width = make_mesh(cells, stretch)
        initial = None
        if restart:
            saved = np.load(restart)
            initial = np.column_stack([np.interp(mesh.points, saved["points_m"], saved["state"][:, i]) for i in range(3)])
        model = NozzleModel(mesh, fluid, p0=case["p0_bar"] * 1e5, T0=case["T0_K"],
                            back_pressure=0.25 * case["p0_bar"] * 1e5,
                            outlet_regime="supersonic", initial_state=initial,
                            wall_closure=None, wall_perimeter=2 * (width + 2 * mesh.geometry(mesh.centers)))
        scheme = Scheme(**cfg["scheme"])
        steady = SteadySettings(residual=ResidualPolicy(**cfg["residual"]), **(cfg["steady"] | {"max_steps": max_steps}))
        metadata.update(scheme=asdict(scheme), steady=asdict(steady), variable_scales=model.variable_scales,
                        conservative_scales=model.conservative_scales, global_flux_scales=model.global_flux_scales,
                        reference_time=model.reference_time, width_m=width)
        last_report = [time.perf_counter()]
        def progress(row):
            now = time.perf_counter()
            if now - last_report[0] > 30:
                print(f"{tag}: step={row.get('step')} accepted={row.get('accepted')} equations={row.get('equation_norm')}", flush=True)
                last_report[0] = now
        solution = solve_single_grid(model, scheme, steady, callback=progress)
        fields = {k: np.asarray(v) for k, v in model.fields(solution.state).items()}
        thermo = model.thermodynamic_state(jnp.asarray(solution.state[1:-1]))
        quality = np.asarray(thermo["Q"])
        # Q=-1 identifies a single-phase state in this backend. No value is
        # changed in the governing EOS or in the numerical solution.
        liquid = np.where((quality >= 0) & (quality <= 1), 1 - quality, np.nan)
        xp, pp, position_origin = experiment(case_id)
        predicted = np.interp(xp, mesh.points, solution.state[:, 0])
        errors = (predicted - pp) / 1e5
        mdot = fields["mdot"]
        metadata.update(success=solution.success, reason=solution.reason, elapsed_s=solution.elapsed_s,
                        total_case_s=time.perf_counter()-start, metrics=solution.metrics, history=solution.history,
                        comparison=dict(position_origin=position_origin,
                                        pressure_rmse_bar=float(np.sqrt(np.mean(errors**2))),
                                        pressure_mae_bar=float(np.mean(np.abs(errors))),
                                        pressure_max_error_bar=float(np.max(np.abs(errors))),
                                        meaningful_as_converged_validation=bool(solution.success)),
                        mass_flow=dict(full_channel_kg_s=float(np.mean(mdot)),
                                       half_channel_per_depth_kg_s_m=float(np.mean(mdot)/(2*width)),
                                       cell_range_kg_s=[float(np.min(mdot)), float(np.max(mdot))]))
        np.savez(folder / "solution.npz", state=solution.state, points_m=mesh.points, x_m=mesh.centers,
                 faces_m=mesh.faces, volumes_m3=mesh.volumes, area_m2=mesh.area_centers,
                 liquid_mass_fraction=liquid, quality=quality, **fields)
        print(f"{tag}: success={solution.success}, reason={solution.reason}, pressure RMSE={metadata['comparison']['pressure_rmse_bar']:.6g} bar", flush=True)
    except Exception as exc:
        metadata.update(success=False, reason="exception", error=str(exc), traceback=traceback.format_exc(), total_case_s=time.perf_counter()-start)
        print(f"{tag}: {type(exc).__name__}: {exc}", flush=True)
    (folder / "result.json").write_text(strict_json(metadata), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *definitions()["cases"]], default="L1")
    parser.add_argument("--cells", type=int, default=24)
    parser.add_argument("--stretch", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    source = provenance()
    fluid, prop = prepare_fluid(output)
    if args.prepare_only:
        return
    for case in definitions()["cases"] if args.case == "all" else [args.case]:
        run_case(case, args.cells, args.stretch, args.max_steps, fluid, prop, output, args.restart, source)


if __name__ == "__main__":
    main()
