"""Run independent critical space marching and grid-sequenced production FVM.

Original FVM states and the earlier comparison remain read only. Every new
attempt is retained. Both wall variants use the requested scheme on all grids.
Run from the repository root with the existing .venv Python, for example::

    python -B -m cases.lettieri_validation.scripts.comparison --case L1 --solver space
    python -B -m cases.lettieri_validation.scripts.comparison --solver all
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
import traceback
from uuid import uuid4

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzlex.functions.nozzle_model_critical import CriticalSettings, solve_choked_nozzle
from nozzlex.functions.nozzle_model_solver import NozzleParams
from nozzle_hyperbolic.models.closures.wall import HaalandWall
from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.solvers.convergence_output import strict_json
from nozzle_hyperbolic.solvers.fvm import (Operator, ResidualPolicy, Scheme,
                                         SteadySettings, solve_single_grid)
from .comparison_data import (OUT, NozzleGeometry, baseline_unchanged,
                              cached_fluid)
from .data import ROOT, definitions, settings
from .run import make_mesh, provenance

FIELDS = ("p", "u", "rho", "T", "h", "h0", "M", "quality", "mdot")


def configuration():
    return json.loads((ROOT / "comparison_settings.json").read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(strict_json(value), encoding="utf-8")
    temporary.replace(path)


def source_record():
    import nozzlex
    repo = Path(nozzlex.__file__).resolve().parents[1]
    files = [repo / "nozzlex/functions" / name for name in
             ("nozzle_model_core.py", "nozzle_walls.py", "nozzle_model_critical.py")]
    modified = subprocess.check_output(["git", "status", "--porcelain", "--", *map(str, files)],
                                       cwd=repo, text=True).strip()
    if modified:
        raise RuntimeError("Commit the NozzleX numerical changes before this study: " + modified)
    return dict(hyperbolic=provenance(), nozzlex=dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        files={str(p.relative_to(repo)): sha256(p.read_bytes()).hexdigest() for p in files}))


def new_attempt(tag):
    folder = OUT / tag
    index = 1
    while folder.exists():
        index += 1
        folder = OUT / f"{tag}_attempt{index:02}"
    folder.mkdir(parents=True)
    return folder


def space_case(case_id, physics, fluid, geometry, source, prop, *, refine=False):
    cfg = configuration()
    case = definitions()["cases"][case_id]
    law = cfg["physics"][physics]
    p0 = case["p0_bar"] * 1e5
    reservoir = fluid.get_state(jxp.PT_INPUTS, p0, case["T0_K"])
    params = NozzleParams(fluid=fluid, geometry=geometry, p0_in=p0,
        h0_in=reservoir["h"], d0_in=reservoir["rho"], length=geometry.length,
        L_convergent=geometry.throat_x, roughness=law["roughness_m"],
        wall_friction=law["wall_friction"], heat_transfer=law["heat_transfer"])
    options = dict(cfg["critical"])
    if refine:
        options.update(restart_fraction=options["restart_fraction"] / 2,
                       rtol=options["rtol"] / 4, atol=options["atol"] / 4,
                       critical_tolerance=options["critical_tolerance"] / 4)
    meshes = {n: make_mesh(n)[0] for n in cfg["grids"]}
    points = np.unique(np.r_[np.linspace(0, geometry.length, cfg["reference_profile_points"]),
                            *[m.centers - geometry.origin for m in meshes.values()]])
    folder = new_attempt(f"{physics}_{case_id}_space" + ("_refined" if refine else ""))
    started = time.perf_counter()
    record = dict(case_id=case_id, physics=physics, source=source, properties=prop,
                  definition=case, settings=options, refinement=refine,
                  boundary_conditions="reservoir p0/h0; critical regularity selects mass flow; supersonic exit",
                  source_file=str(folder.relative_to(ROOT)))
    try:
        result = solve_choked_nozzle(params, CriticalSettings(**options), x_eval=points)
        record.update(success=result.success, reason=result.reason, elapsed_s=result.elapsed_s,
                      history=result.history, critical=result.critical, integration=result.integration,
                      x_m=result.x + geometry.origin, fields={k: result.fields[k] for k in FIELDS if k in result.fields})
        if result.success:
            c = record["critical"]
            st = np.asarray(c["state"])
            rho = float(fluid.get_state(jxp.HmassP_INPUTS, st[2], st[0])["rho"])
            c.update(x_sonic_m=float(c["x"] + geometry.origin),
                     mass_flow_kg_s=float(rho * st[1] * geometry(c["x"])[0]))
            valid = np.asarray(result.integration["valid_mask"])
            fields = result.fields
            mdot = np.asarray(fields["mdot"])[valid]
            h0 = np.asarray(fields["h0"])[valid]
            record["audits"] = dict(valid_points=int(valid.sum()),
                sonic_restart_gap_points=int((~valid).sum()),
                mass_flow_relative_range=float(np.ptp(mdot) / np.mean(mdot)),
                energy_relative_range=float(np.ptp(h0) / abs(float(reservoir["h"]))),
                inlet_mach=float(fields["M"][0]), outlet_mach=float(fields["M"][-1]),
                critical_equations_max=float(np.max(np.abs(c["residual"]))),
                sonic_regularity=float(c["regularity"]))
            np.savez(folder / "solution.npz", x_m=record["x_m"], state=result.state,
                     valid_mask=valid, **record["fields"])
    except Exception as exc:
        record.update(success=False, reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    record["total_case_s"] = time.perf_counter() - started
    write(folder / "result.json", record)
    print(f"{physics}/{case_id} space: success={record['success']}; {record['reason']}", flush=True)
    return record


def verify_fvm(model, scheme, solution):
    op = Operator(model, scheme)
    state, mesh = jnp.asarray(solution.state), model.mesh
    raw = np.asarray(op.spatial_residual(state))
    faces = np.asarray(op.face_flux(state))
    loads = np.asarray(model.integrated_source(state))
    dimensional = raw * model.reference_time / (mesh.volumes[:, None] * model.conservative_scales)
    l2 = np.sqrt(np.sum(mesh.volumes[:, None] * dimensional**2, axis=0) / np.sum(mesh.volumes))
    boundary = float(np.max(abs(np.asarray(op.boundary_residual(state, 0.0)))))
    budget = faces[-1] - faces[0] - loads.sum(axis=0)
    defect = abs(budget / model.global_flux_scales)
    identity = abs((raw.sum(axis=0) - budget) / model.global_flux_scales)
    tol = settings()["residual"]
    branch = bool(model.solution_audit(state))
    admissible = bool(op.is_admissible(state))
    verified = bool(solution.success and np.all(l2 <= tol["absolute_tolerance"]) and
        boundary <= tol["boundary_tolerance"] and np.all(defect <= tol["global_tolerance"]) and
        np.all(identity < 1e-12) and admissible and branch)
    return dict(verified=verified, equation_l2=l2, boundary_max=boundary,
        global_defect=defect, signed_boundary_source_budget=budget,
        telescoping_identity_defect=identity, admissible=admissible, physical_branch_pass=branch,
        mass_flow=dict(numerical_inlet_kg_s=float(faces[0, 0]),
            numerical_outlet_kg_s=float(faces[-1, 0]),
            numerical_range_kg_s=[float(faces[:, 0].min()), float(faces[:, 0].max())],
            note="Numerical face flux is the conservative mass flow; cell rho*u*A can vary on a finite grid"))


def fvm_case(case_id, physics, cells, fluid, source, prop, restart=None):
    cfg, base = configuration(), settings()
    law = cfg["physics"][physics]
    case = definitions()["cases"][case_id]
    folder = new_attempt(f"{cfg['campaign_id']}_{physics}_{case_id}_N{cells}")
    record = dict(id=folder.name, case_id=case_id, physics=physics, cells=cells,
        campaign_id=cfg["campaign_id"], grid_sequencing=cfg["grid_sequencing"],
        source_epoch=source, properties=prop, definition=case, source_file=str(folder.relative_to(ROOT)),
        initialization="model cold Mach ramp" if restart is None else f"same-physics grid prolongation: {restart}")
    started = time.perf_counter()
    try:
        mesh, depth = make_mesh(cells)
        perimeter = 2 * (depth + 2 * mesh.geometry(mesh.centers))
        diameter = 4 * mesh.area_centers / perimeter
        initial = None
        if restart is not None:
            saved = np.load(restart)
            initial = np.column_stack([np.interp(mesh.points, saved["points_m"], saved["state"][:, i]) for i in range(3)])
        wall = None if not law["wall_friction"] else HaalandWall(diameter, roughness=law["roughness_m"], heat_flux=0)
        model = NozzleModel(mesh, fluid, p0=case["p0_bar"] * 1e5, T0=case["T0_K"],
            back_pressure=.25 * case["p0_bar"] * 1e5, outlet_regime="supersonic",
            initial_state=initial, wall_closure=wall,
            wall_perimeter=perimeter)
        scheme = Scheme(**(base["scheme"] | cfg["scheme_overrides"]))
        steady = SteadySettings(residual=ResidualPolicy(**base["residual"]),
                               **(base["steady"] | {"max_steps": cfg["fvm_max_steps"]}))
        record.update(scheme=asdict(scheme), steady=asdict(steady), variable_scales=model.variable_scales,
                      conservative_scales=model.conservative_scales, global_flux_scales=model.global_flux_scales,
                      reference_time=model.reference_time)
        last = [time.perf_counter()]
        def progress(row):
            if time.perf_counter() - last[0] > 30:
                print(f"{folder.name}: step={row.get('step')}; residual={row.get('equation_norm')}", flush=True)
                last[0] = time.perf_counter()
        result = solve_single_grid(model, scheme, steady, callback=progress)
        fields = {k: np.asarray(v) for k, v in model.fields(result.state).items()}
        thermo = model.thermodynamic_state(jnp.asarray(result.state[1:-1]))
        fields["quality"] = np.asarray(thermo["Q"])
        verification = verify_fvm(model, scheme, result)
        record.update(success=result.success, reason=result.reason, elapsed_s=result.elapsed_s,
            metrics=result.metrics, history=result.history, saved_verification=verification,
            mass_flow=verification["mass_flow"], x_m=mesh.centers,
            fields={k: fields[k] for k in FIELDS})
        np.savez(folder / "solution.npz", state=result.state, points_m=mesh.points, x_m=mesh.centers,
            faces_m=mesh.faces, volumes_m3=mesh.volumes, area_m2=mesh.area_centers, **record["fields"])
        write(folder / "verification.json", verification)
    except Exception as exc:
        record.update(success=False, reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    record["total_case_s"] = time.perf_counter() - started
    write(folder / "result.json", record)
    print(f"{physics}/{case_id}/N{cells}: success={record['success']}; {record['reason']}", flush=True)
    return record


def match_reference(row, space):
    if not space.get("success") or not row.get("fields"):
        return
    positions = np.asarray(space["x_m"])
    targets = np.asarray(row["x_m"])
    upper = np.clip(np.searchsorted(positions, targets), 0, len(positions) - 1)
    lower = np.maximum(upper - 1, 0)
    indices = np.where(abs(positions[lower] - targets) < abs(positions[upper] - targets), lower, upper)
    if np.max(abs(positions[indices] - targets)) > 5e-17:
        raise RuntimeError("Reference was not sampled exactly at every FVM center")
    ref = {k: np.asarray(v, dtype=float)[indices] for k, v in space["fields"].items()}
    row["reference_fields"] = ref
    row["errors"] = {}
    for key in ("p", "u", "rho", "T", "M"):
        relative = abs((np.asarray(row["fields"][key]) - ref[key]) / ref[key]) * 100
        row["errors"][key] = dict(relative_percent_rms=float(np.sqrt(np.nanmean(relative**2))),
                                  relative_percent_max=float(np.nanmax(relative)))


def assemble(source, prop, geometry):
    cfg = configuration()
    campaign = dict(settings=cfg, provenance=source, properties=prop, geometry=geometry.audit(),
                    original_evidence=baseline_unchanged(), variants={})
    archive = OUT / "campaign_constant_48_96.json"
    previous_campaign = None
    if archive.is_file():
        previous_campaign = json.loads(archive.read_text(encoding="utf-8"))
        campaign["previous_campaign"] = dict(file=str(archive.relative_to(ROOT)),
            sha256=sha256(archive.read_bytes()).hexdigest(),
            description="Preserved HLL constant-reconstruction comparison on 48/96 cells")
    sensitivity = OUT / "reference_sensitivity.json"
    if sensitivity.is_file():
        campaign["reference_sensitivity"] = json.loads(sensitivity.read_text(encoding="utf-8"))
    for physics in cfg["physics"]:
        cases = {}
        for case_id in cfg["cases"]:
            references = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob(f"{physics}_{case_id}_space*/result.json"))
                          if "_refined" not in p.parent.name]
            refinements = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob(f"{physics}_{case_id}_space_refined*/result.json"))]
            space = references[-1] if references else dict(success=False, reason="Not run", fields={}, x_m=[])
            if space.get("x_m"):
                # Distinct inlet-relative floats can collapse after adding the
                # experimental origin. Keep raw attempts intact; expose one
                # observation per identical physical station in the overlay.
                space = dict(space)
                coordinates = np.asarray(space["x_m"])
                unique, indices = np.unique(coordinates, return_index=True)
                if len(unique) != len(coordinates):
                    space["x_m"] = unique.tolist()
                    space["fields"] = {k: np.asarray(v)[indices].tolist() for k, v in space["fields"].items()}
                    space["profile_station_normalization"] = dict(
                        duplicate_physical_stations=len(coordinates)-len(unique),
                        note="Identical physical station floats shown once; all original evaluations retained in space_attempts")
            fvm = []
            for n in cfg["grids"]:
                paths = sorted(OUT.glob(f"{cfg['campaign_id']}_{physics}_{case_id}_N{n}*/result.json"))
                for path in paths:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row.get("cells") != n or row.get("campaign_id") != cfg["campaign_id"]:
                        continue
                    match_reference(row, space)
                    fvm.append(row)
            cases[case_id] = dict(space=space, fvm=fvm, space_attempts=references,
                                  reference_refinements=refinements)
            if previous_campaign is not None:
                old_rows=previous_campaign["variants"][physics]["cases"][case_id]["fvm"]
                previous=next((r for r in old_rows if r["cells"]==96),None)
                if previous is not None:
                    cases[case_id]["previous_constant_comparison"] = dict(cells=96,
                        pressure_rms_percent=previous["errors"]["p"]["relative_percent_rms"],
                        id=previous["id"], archive=campaign["previous_campaign"]["file"])
        campaign["variants"][physics] = dict(label="Inviscid" if physics == "inviscid" else "Haaland friction", cases=cases)
    write(OUT / "campaign.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *definitions()["cases"]], default="all")
    parser.add_argument("--solver", choices=["all", "space", "fvm", "assemble"], default="all")
    parser.add_argument("--physics", choices=["all", "inviscid", "haaland"], default="all")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--cells", type=int, nargs="+", help="Override the configured grid sequence")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_unchanged()
    source = source_record()
    fluid, prop = cached_fluid()
    geometry = NozzleGeometry()
    geometry.audit()
    for case_id in configuration()["cases"] if args.case == "all" else [args.case]:
        if args.solver in ("all", "space"):
            for physics in configuration()["physics"] if args.physics == "all" else [args.physics]:
                space_case(case_id, physics, fluid, geometry, source, prop, refine=args.refine)
        if args.solver in ("all", "fvm"):
            for physics in configuration()["physics"] if args.physics == "all" else [args.physics]:
                restart = None
                for cells in sorted(args.cells or configuration()["grids"]):
                    row = fvm_case(case_id, physics, cells, fluid, source, prop, restart)
                    if not row["success"] or not row.get("saved_verification", {}).get("verified"):
                        print(f"Stopping {case_id}/{physics} grid sequence after failed N{cells} stage", flush=True)
                        break
                    restart = ROOT / row["source_file"] / "solution.npz"
        assemble(source, prop, geometry)
    baseline_unchanged()


if __name__ == "__main__":
    main()
