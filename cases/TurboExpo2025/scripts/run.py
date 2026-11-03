"""Run the paper's spatial methods with the current implicit HEM solver.

Run from the repository root: poetry run python -B -m cases.TurboExpo2025.scripts.run
Completed attempts are immutable; --resume skips already recorded settings.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import zipfile

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import (
    Operator,
    ResidualPolicy,
    Scheme,
    SteadySettings,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.convergence_output import strict_json
from nozzle_hyperbolic.solvers.residuals import DEFINITION
from .physics import build_model, prepare_fluid, resolve_case

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "outputs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json(data), encoding="utf-8")


def fingerprint():
    paths = [ROOT / "settings.json", ROOT / "case_definitions.json"]
    paths += [
        ROOT / "scripts/run.py",
        ROOT / "scripts/physics.py",
        ROOT / "scripts/analytical.py",
    ]
    paths += sorted(
        p for p in (ROOT / "data").iterdir() if p.suffix in (".json", ".xlsx")
    )
    paths += sorted((REPO / "src/nozzle_hyperbolic").rglob("*.py"))
    hashes = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    versions = {}
    for name in ("jax", "jaxlib", "jaxprop", "numpy", "scipy", "equinox"):
        versions[name] = importlib.metadata.version(name)
    return dict(
        identity=identity,
        files=hashes,
        packages=versions,
        python=platform.python_version(),
        executable=(
            platform.python_executable()
            if hasattr(platform, "python_executable")
            else __import__("sys").executable
        ),
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        git_branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ),
    )


def snapshot_source(source):
    """Preserve the actual dirty implementation rather than only its Git HEAD."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"source_{source['identity']}.zip"
    if target.exists():
        return
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source["files"]:
            archive.write(REPO / name, name)
        for name in ("pyproject.toml", "poetry.lock", "AGENTS.md"):
            archive.write(REPO / name, name)
        archive.writestr("source.json", strict_json(source))


def verify(model, scheme, solution, policy, case=None):
    """Independently evaluate the original steady rows and flux budgets."""
    op = solution.operator if solution.operator is not None else Operator(model, scheme)
    state, mesh = jnp.asarray(solution.state), model.mesh
    raw = np.asarray(op.spatial_residual(state))
    faces = np.asarray(op.face_flux(state))
    sources = np.asarray(model.integrated_source(state))
    z = raw * model.reference_time / (mesh.volumes[:, None] * model.conservative_scales)
    rms = np.sqrt(np.sum(mesh.volumes[:, None] * z * z, axis=0) / np.sum(mesh.volumes))
    bc = float(np.max(np.abs(np.asarray(op.boundary_residual(state, 0.0)))))
    budget = faces[-1] - faces[0] - sources.sum(axis=0)
    global_defect = np.abs(budget / model.global_flux_scales)
    identity = np.abs((raw.sum(axis=0) - budget) / model.global_flux_scales)
    admissible = bool(op.is_admissible(state))
    branch = bool(model.solution_audit(state))
    cell_mach = np.asarray(model.fields(state)["M"])
    intended_regime = True
    intended_description = "configured inlet/outlet characteristic regime"
    if case and case["id"] == "G_subsonic":
        intended_regime = bool(np.all(np.isfinite(cell_mach)) and np.max(cell_mach) < 1)
        intended_description = "all-subsonic interior"
    elif case and (
        case["id"].startswith("G_shock")
        or case["geometry"] in ("petruccelli", "berana")
    ):
        intended_regime = bool(
            np.all(np.isfinite(cell_mach)) and np.max(cell_mach) > 1.002
        )
        intended_description = (
            "supersonic interior followed by configured subsonic outlet"
        )
    absolute_pass = bool(
        np.all(rms <= policy.absolute_tolerance)
        and bc <= policy.boundary_tolerance
        and np.all(global_defect <= policy.global_tolerance)
    )
    endpoints = state[jnp.array([0, -1])]
    mach = np.asarray(endpoints[:, 1] / model.thermodynamic_state(endpoints)["a"])
    return dict(
        verified=bool(
            solution.success
            and absolute_pass
            and admissible
            and branch
            and intended_regime
            and np.all(identity < 1e-12)
        ),
        absolute_targets_pass=absolute_pass,
        solver_success=bool(solution.success),
        equation_l2=rms,
        equation_linf=np.max(np.abs(z), axis=0),
        boundary_max=bc,
        global_defect=global_defect,
        signed_boundary_source_budget=budget,
        telescoping_identity_defect=identity,
        admissible=admissible,
        physical_branch_pass=branch and intended_regime,
        boundary_characteristic_regime_pass=branch,
        intended_regime_pass=intended_regime,
        intended_regime=intended_description,
        interior_mach_range=[float(np.min(cell_mach)), float(np.max(cell_mach))],
        endpoint_mach=mach,
        mass_flow=dict(
            numerical_inlet_kg_s=float(faces[0, 0]),
            numerical_outlet_kg_s=float(faces[-1, 0]),
            numerical_range_kg_s=[float(faces[:, 0].min()), float(faces[:, 0].max())],
            cell_mean_kg_s=float(np.mean(model.fields(state)["mdot"])),
        ),
        residual_definition=DEFINITION,
    )


def compare(case, fields, verified, model=None):
    path = ROOT / "data/references.json"
    series = read(path).get(case["id"], []) if path.exists() else []
    if (
        str(case["fluid"]).lower() in ("air", "perfect_gas", "ideal_gas", "ideal air")
        and model is not None
    ):
        from .analytical import profile

        series = [*series, profile(case, model.mesh.geometry, fields["x_m"])]
    errors = []
    for ref in series:
        if ref.get("kind") == "coordinate_diagnostic":
            continue
        if "x_m" not in ref or "p_Pa" not in ref:
            continue
        x, target = np.asarray(ref["x_m"]), np.asarray(ref["p_Pa"])
        valid = (
            np.isfinite(x)
            & np.isfinite(target)
            & (x >= fields["x_m"][0])
            & (x <= fields["x_m"][-1])
        )
        difference = np.interp(x[valid], fields["x_m"], fields["p"]) - target[valid]
        errors.append(
            dict(
                label=ref["label"],
                kind=ref.get("kind"),
                points=int(valid.sum()),
                pressure_rmse_bar=(
                    float(np.sqrt(np.mean(difference**2)) / 1e5)
                    if difference.size
                    else None
                ),
                pressure_max_error_bar=(
                    float(np.max(abs(difference)) / 1e5) if difference.size else None
                ),
                usable_as_converged_comparison=bool(verified),
            )
        )
    return dict(
        series=series, errors=errors, historical_numeric_profile_available=False
    )


def new_folder(case_id, flux, cells):
    base = OUT / case_id / flux / f"N{cells}"
    path, attempt = base, 1
    while path.exists():
        attempt += 1
        path = base.with_name(f"N{cells}_attempt{attempt:02}")
    path.mkdir(parents=True)
    return path


def run_stage(case, flux, cells, fluid, properties, source, config, restart=None):
    folder = new_folder(case["id"], flux, cells)
    scheme = Scheme(**(config["scheme"] | {"flux": flux}))
    policy = ResidualPolicy(**config["residual"])
    steady = SteadySettings(residual=policy, **config["steady"])
    row = dict(
        case_id=case["id"],
        case=case,
        flux=flux,
        cells=cells,
        requested_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        properties=properties,
        scheme=asdict(scheme),
        steady=asdict(steady),
        initialization=(
            (
                "current model cold start"
                + (
                    f"; supersonic pressure seed {case['initial_back_pressure_Pa']:g} Pa (no imposed outlet pressure)"
                    if case.get("initial_back_pressure_Pa") is not None
                    else ""
                )
            )
            if restart is None
            else f"successful same-case same-flux mesh prolongation from {restart.relative_to(ROOT)}"
        ),
        restart_file=None if restart is None else str(restart.relative_to(ROOT)),
        output=str(folder.relative_to(ROOT)),
        success=False,
        elapsed_s=None,
    )
    write(folder / "attempt.json", row)
    started = time.perf_counter()
    try:
        model = build_model(case, cells, fluid)
        if restart is not None:
            old = np.load(restart)
            initial = np.column_stack(
                [
                    np.interp(model.mesh.points, old["points_m"], old["state"][:, k])
                    for k in range(3)
                ]
            )
        else:
            initial = None
        row.update(
            variable_scales=model.variable_scales,
            conservative_scales=model.conservative_scales,
            global_flux_scales=model.global_flux_scales,
            reference_time=model.reference_time,
            reservoir=dict(
                p0_Pa=model.p0,
                T0_K=model.T0,
                h0_J_kg=model.h0,
                s0_J_kgK=model.s0,
                rho0_kg_m3=model.rho0,
                a0_m_s=model.a0,
            ),
        )
        last = [time.perf_counter()]

        def progress(event):
            if time.perf_counter() - last[0] > 25:
                print(
                    f"{case['id']}/{flux}/N{cells}: step={event.get('step')}, phase={event.get('phase')}, RMS={event.get('equation_norm')}",
                    flush=True,
                )
                last[0] = time.perf_counter()

        solution = solve_single_grid(
            model, scheme, steady, initial_state=initial, callback=progress
        )
        row.update(
            success=bool(solution.success),
            reason=solution.reason,
            metrics=solution.metrics,
            history=solution.history,
            elapsed_s=solution.elapsed_s,
        )
        # Save the actual failure state too, without changing quality sentinels.
        mesh = model.mesh
        fields = {k: np.asarray(v) for k, v in model.fields(solution.state).items()}
        thermo = model.thermodynamic_state(jnp.asarray(solution.state[1:-1]))
        quality = np.asarray(thermo["Q"])
        for key in ("s", "a", "Z", "G", "mu"):
            try:
                fields[key] = np.asarray(thermo[key])
            except KeyError:
                pass
        fields.update(
            x_m=mesh.centers,
            quality=quality,
            liquid_mass_fraction=np.where(
                (quality >= 0) & (quality <= 1), 1 - quality, np.nan
            ),
        )
        np.savez(
            folder / "solution.npz",
            state=solution.state,
            points_m=mesh.points,
            faces_m=mesh.faces,
            volumes_m3=mesh.volumes,
            area_m2=mesh.area_centers,
            **fields,
        )
        row["verification"] = verify(model, scheme, solution, policy, case)
        row["comparison"] = compare(
            case, fields, row["verification"]["verified"], model
        )
        write(folder / "verification.json", row["verification"])
        row["accepted_steps"] = sum(
            h.get("phase") == "accepted" for h in solution.history
        )
        row["rejected_attempts"] = sum(
            h.get("phase") == "rejected" for h in solution.history
        )
    except Exception as exc:
        row.update(
            solver_success=row.get("success", False),
            success=False,
            reason="exception",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    row["total_stage_s"] = time.perf_counter() - started
    write(folder / "result.json", row)
    print(
        f"{case['id']}/{flux}/N{cells}: success={row['success']}, verified={row.get('verification',{}).get('verified',False)}, reason={row.get('error',row.get('reason'))}",
        flush=True,
    )
    return row, folder


def matching_saved(case_id, flux, cells, source, config):
    for path in sorted(
        (OUT / case_id / flux).glob(f"N{cells}*/result.json"), reverse=True
    ):
        row = read(path)
        if row.get("cells") != cells:
            continue
        if (
            row.get("source", {}).get("identity") == source["identity"]
            and row.get("scheme")
            == asdict(Scheme(**(config["scheme"] | {"flux": flux})))
            and row.get("steady")
            == asdict(
                SteadySettings(
                    residual=ResidualPolicy(**config["residual"]), **config["steady"]
                )
            )
        ):
            return row, path.parent
    return None


def main():
    definitions = read(ROOT / "case_definitions.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="all", help="case ID or comma-separated IDs; default all"
    )
    parser.add_argument("--family", help="restrict to one physical family")
    parser.add_argument("--flux", choices=("rusanov", "hll", "roe"))
    parser.add_argument(
        "--cells", type=int, help="override publication cell count (pilot only)"
    )
    parser.add_argument(
        "--grids",
        help="comma-separated mesh sequence; requested paper mesh is appended",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = read(ROOT / "settings.json")
    if args.max_steps is not None:
        config["steady"]["max_steps"] = args.max_steps
    selected = definitions["cases"]
    if args.case != "all":
        ids = args.case.split(",")
        selected = [c for c in selected if c["id"] in ids]
        missing = set(ids) - {c["id"] for c in selected}
        if missing:
            parser.error(f"Unknown case IDs: {sorted(missing)}")
    if args.family:
        selected = [c for c in selected if c["family"] == args.family]
    if not selected:
        parser.error("No cases selected")
    source = fingerprint()
    snapshot_source(source)
    write(OUT / "source.json", source)
    fluids = {}
    for original_case in selected:
        case = resolve_case(original_case)
        key = (case["fluid"], case["family"])
        try:
            if key not in fluids:
                fluids[key] = prepare_fluid(case, OUT / "properties")
            fluid, prop = fluids[key]
        except Exception as exc:
            for flux in ([args.flux] if args.flux else case["fluxes"]):
                cells = args.cells or case["cells"]
                folder = new_folder(case["id"], flux, cells)
                write(
                    folder / "result.json",
                    dict(
                        case_id=case["id"],
                        case=case,
                        flux=flux,
                        cells=cells,
                        source=source,
                        success=False,
                        reason="property_preparation_failed",
                        error=str(exc),
                        traceback=traceback.format_exc(),
                        elapsed_s=None,
                        output=str(folder.relative_to(ROOT)),
                    ),
                )
            print(f"{case['id']}: property preparation failed: {exc}", flush=True)
            continue
        if args.prepare_only:
            continue
        for flux in ([args.flux] if args.flux else case["fluxes"]):
            if flux not in case["fluxes"]:
                continue
            final = args.cells or case["cells"]
            grids = (
                sorted(
                    set(
                        [int(n) for n in args.grids.split(",") if int(n) < final]
                        + [final]
                    )
                )
                if args.grids
                else [final]
            )
            restart = None
            for cells in grids:
                saved = (
                    matching_saved(case["id"], flux, cells, source, config)
                    if args.resume
                    else None
                )
                if saved:
                    row, folder = saved
                    print(
                        f"Retained {folder.relative_to(ROOT)}: {row['reason']}",
                        flush=True,
                    )
                else:
                    row, folder = run_stage(
                        case, flux, cells, fluid, prop, source, config, restart
                    )
                restart = (
                    folder / "solution.npz"
                    if row.get("verification", {}).get("verified")
                    else None
                )
                # A failed predecessor leaves the next requested mesh cold;
                # every listed Appendix mesh still receives an actual attempt.
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    main()
