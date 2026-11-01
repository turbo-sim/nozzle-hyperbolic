"""One isolated target solve, or one shared numerical adapted preparation."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from dataclasses import asdict, fields as dataclass_fields
from time import perf_counter
import traceback
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from .common import (OUTPUT, identity, save, load, profile, require_source,
                     configure_compilation_cache, run_id, seed_id)
_compilation_cache = configure_compilation_cache()

from nozzle_hyperbolic.solvers.fvm import (Scheme, SteadySettings, ResidualPolicy, Operator,
                                           Solution, solve_grid_sequence, drivers)
from .problem import target_context, adapted_context, BoundContext
from .audit import evaluate_solution
from .execution import budgeted_solve


def normalized_history(history):
    rows = []
    for value in history:
        row = dict(value)
        if "equation_norm" in row:
            row["residual"] = float(np.max(row["equation_norm"]))
        if "boundary_max" in row:
            row["boundary_residual"] = row["boundary_max"]
        rows.append(row)
    return rows


def record_solution(context, solution, initial, policy, **metadata):
    """Native status and fresh original-equation verification are independent."""
    try:
        assessment = evaluate_solution(context, solution, policy, initial_state=initial)
    except Exception:
        assessment = dict(fields={}, reference_fields={}, errors={}, target_correct=False,
                          audit=dict(passed=False, checks={"evaluation_completed": False},
                                     failed_checks=["evaluation_completed"], error=traceback.format_exc()))
    history = normalized_history(solution.history)
    work = dict(accepted_steps=sum(row.get("phase") == "accepted" for row in history),
                rejected_steps=sum(row.get("phase") == "rejected" for row in history),
                newton_corrections=sum(row.get("inner_iterations", 0) for row in history),
                rejected_newton_trials=sum(row.get("rejected_newton_trials", 0) for row in history))
    model = solution.operator.model
    outlet_options = {field.name: getattr(model.outlet, field.name)
                      for field in dataclass_fields(model.outlet) if field.name != "fluid"}
    outlet_options["wave_backend"] = ("closed-form calorically perfect gas" if model.outlet_formulation == "finite_wave"
                                       else "linearized acoustic compatibility")
    return dict(success=bool(solution.success), native_success=bool(solution.success),
                verified=bool(assessment["audit"]["passed"]), reason=solution.reason,
                state=solution.state, points=np.asarray(model.mesh.points), history=history,
                metrics=solution.metrics, work=work, elapsed_s=solution.elapsed_s,
                initial_state=initial, init_hash=identity(initial),
                scheme=asdict(solution.operator.scheme), outlet_options=outlet_options,
                **metadata, **assessment)


def common_metadata(args, provenance, settings):
    return dict(source_sha256=provenance["source_sha256"], settings_sha256=provenance["settings_sha256"],
                client_sha256=provenance["client_sha256"], execution_controls=settings["execution"],
                flux=args.flux, reconstruction=args.reconstruction)


def prepare_seed(args, provenance, settings, scheme, policy, steady):
    key = seed_id(args.flux, args.reconstruction)
    path = OUTPUT / "seeds" / (key + ".json")
    if path.exists():
        print(f"SAVED SEED {key} verified={load(path)['verified']}", flush=True)
        return
    context = adapted_context()
    records, previous = [], None

    def on_stage(model, solution, stages):
        nonlocal previous
        require_source(provenance)
        stage = stages[-1]
        warm = "prolongation" in stage["initialization"] and not stage["attempts"]
        if warm and previous is not None:
            initial = np.column_stack([np.interp(model.mesh.points, previous["points"], np.asarray(previous["state"])[:, j])
                                       for j in range(model.nvar)])
        else:
            initial = np.asarray(model.initial_state())
        record = record_solution(context, solution, initial, policy,
                                 id=f"{key}__N{model.mesh.n_cells}", case_id="nozzle_adapted",
                                 outlet="acoustic", start="numerical_adapted_preparation", strategy="implicit_euler",
                                 cells=model.mesh.n_cells, attempts=stage["attempts"],
                                 initialization=dict(method=stage["initialization"], analytical_initial_field=False,
                                                     source_cells=previous["cells"] if warm and previous else None),
                                 **common_metadata(args, provenance, settings))
        records.append(record)
        save(OUTPUT / "seed_stages" / key / f"N{model.mesh.n_cells}.json", record)
        previous = record if solution.success else None
        print(f"SEED STAGE {key} N={model.mesh.n_cells} native={record['success']} verified={record['verified']} reason={record['reason']}", flush=True)

    with patch.object(drivers, "solve_single_grid", budgeted_solve):
        solve_grid_sequence(context.build_model, cells=settings["adapted_seed_grids"],
                            scheme=scheme, settings=steady, on_stage=on_stage)
    result = dict(records[-1])
    result.update(id=key, stages=[row["id"] for row in records],
                  preparation_elapsed_s=sum(row["elapsed_s"]+sum(attempt.get("elapsed_s", 0) for attempt in row["attempts"])
                                            for row in records))
    result["seed_sha256"] = identity(dict(state=result["state"], points=result["points"], scheme=result["scheme"],
                                           source_sha256=result["source_sha256"], settings_sha256=result["settings_sha256"]))
    save(path, result)


def main():
    settings = profile()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=settings["cases"])
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--outlet", choices=settings["outlets"], default="acoustic")
    parser.add_argument("--flux", choices=settings["fluxes"], required=True)
    parser.add_argument("--reconstruction", choices=settings["reconstructions"], required=True)
    parser.add_argument("--start", choices=settings["starts"], default="stagnant")
    parser.add_argument("--strategy", choices=settings["strategies"], default="implicit_euler")
    parser.add_argument("--first-step-fraction", type=float, choices=(0.1, 0.25, 1.0),
                        help="Explicit supplemental direct-Newton variant; baseline is 0.25")
    args = parser.parse_args()
    if bool(args.case) == bool(args.seed):
        parser.error("Select one target --case or --seed preparation")
    if args.first_step_fraction is not None and (args.seed or args.strategy != "direct_newton"):
        parser.error("First-step variants apply only to direct-Newton targets")
    provenance = load(OUTPUT / "provenance.json")
    require_source(provenance)
    scheme = Scheme(flux=args.flux, reconstruction=args.reconstruction, **settings["scheme"])
    policy = ResidualPolicy(**settings["residual"])
    steady = SteadySettings(residual=policy, **settings["steady"])
    if args.seed:
        prepare_seed(args, provenance, settings, scheme, policy, steady)
        return
    key = run_id(args.case, args.outlet, args.flux, args.reconstruction, args.start, args.strategy)
    variant = ("fraction_"+str(args.first_step_fraction).replace(".", "p")
               if args.first_step_fraction is not None and args.first_step_fraction != settings["direct"]["first_step_fraction"] else None)
    if variant:
        key += "__"+variant
    path = (OUTPUT / "variants" / variant / "runs" if variant else OUTPUT / "runs") / (key + ".json.gz")
    if path.exists():
        print(f"SAVED {key} verified={load(path)['verified']}", flush=True)
        return
    lock_path = OUTPUT / "locks" / (key + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    context = target_context(args.case)
    model = context.build_model(settings["cells"], args.outlet)
    seed = None
    if args.start == "adapted":
        seed_path = OUTPUT / "seeds" / (seed_id(args.flux, args.reconstruction)+".json")
        if seed_path.exists():
            seed = load(seed_path)
        if seed is None or not seed["verified"]:
            save(path, dict(id=key, case_id=args.case, outlet=args.outlet, start=args.start, strategy=args.strategy,
                            cells=settings["cells"], success=False, native_success=False, verified=False,
                            reason="adapted_seed_not_verified", state=None, points=None, fields={}, reference_fields={},
                            errors={}, history=[], metrics={}, work={}, elapsed_s=0.0, init_hash=None, initial_state=None,
                            initialization=dict(method="blocked: numerical adapted seed unavailable or failed"),
                            target_correct=False, audit=dict(passed=False, checks={"verified_seed":False}, failed_checks=["verified_seed"]),
                            **common_metadata(args, provenance, settings)))
            return
    initial = context.initial_field(model, args.start, seed)
    model._initial_state = initial.copy()
    initialization = dict(method="uniform reservoir p,h; velocity zero" if args.start == "stagnant"
                           else "exact 200-cell numerical adapted field",
                           prescribed_start=args.start, analytical_initial_field=False,
                           target_grid_sequence=False, target_implicit_prefix=False,
                           seed_id=seed["id"] if seed else None, seed_sha256=seed["seed_sha256"] if seed else None)
    metadata = dict(id=key, case_id=args.case, outlet=args.outlet, start=args.start, strategy=args.strategy,
                    cells=settings["cells"], initialization=initialization, variant=variant,
                    **common_metadata(args, provenance, settings))
    direct = None
    if args.strategy == "implicit_euler":
        solution = budgeted_solve(model, scheme, steady, initial_state=initial.copy())
    else:
        from .direct import solve_direct
        operator = Operator(model, scheme)
        bound = BoundContext(context, model, operator, metadata, settings)
        started = perf_counter()
        try:
            direct_options = dict(settings["direct"])
            if args.first_step_fraction is not None:
                direct_options["first_step_fraction"] = args.first_step_fraction
            direct = solve_direct(bound, initial.copy(), policy, direct_options,
                                  wall_seconds=settings["execution"]["wall_seconds_per_attempt"])
        except Exception:
            direct = dict(full_state=initial.copy(), native_success=False, native_status="exception",
                          native_message="adapter exception", exception=traceback.format_exc(), history=[], trials=[],
                          counters={}, options=settings["direct"], elapsed_seconds=perf_counter()-started)
        solution = Solution(np.asarray(direct["full_state"]), direct.get("success", False),
                            direct.get("reason", direct.get("native_message", str(direct.get("native_status")))),
                            direct.get("history", []), direct.get("metrics", {}),
                            elapsed_s=direct.get("elapsed_seconds", perf_counter()-started)+direct.get("compile_seconds", 0.0), operator=operator)
    require_source(provenance)
    result = record_solution(context, solution, initial, policy, **metadata)
    if direct is not None:
        result["native_success"] = bool(direct["native_success"])
        promoted = ("native_status", "native_message", "trials", "counters", "options", "backend", "exception", "common_success")
        result["direct"] = {name: value for name, value in direct.items()
                            if name not in ("full_state", "state", "history", *promoted)}
        for name in promoted:
            if name in direct:
                result[name] = direct[name]
        result["work"].update(direct.get("counters", {}))
        result["compile_s"] = direct.get("compile_seconds", 0.0)
        result["solve_s"] = direct.get("elapsed_seconds")
    result["timing_scope"] = "compile and solve, including monitoring and rejected work; excludes independent final audit and output"
    save(path, result)
    print(f"RUN {key} native={result['success']} verified={result['verified']} target={result['target_correct']} reason={result['reason']} elapsed={result['elapsed_s']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
