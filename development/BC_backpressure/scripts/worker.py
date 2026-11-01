"""Isolated numerical adapted preparation or one target grid sequence."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from dataclasses import asdict, fields as dataclass_fields
from time import perf_counter
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)

from .common import (OUTPUT, chain_id, seed_id, identity, save, load, profile,
                     require_source, configure_compilation_cache)

# Only compiled executables are shared across independent workers. Numerical
# states, controller histories and residual normalizations remain independent.
_compilation_cache = configure_compilation_cache()

import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, ResidualPolicy, solve_grid_sequence, drivers
from .problem import target_context, adapted_context, family_of
from .audit import evaluate_solution
from .execution import budgeted_solve


def normalized_history(history):
    result = []
    for original in history:
        row = dict(original)
        if "equation_norm" in row:
            row["residual"] = float(np.max(row["equation_norm"]))
            row["boundary_residual"] = row["boundary_max"]
        result.append(row)
    return result


def record_solution(context, solution, policy, **metadata):
    try:
        assessment = evaluate_solution(context, solution, policy)
    except Exception:
        assessment = dict(fields={}, reference_fields={}, errors={}, audit=dict(
            passed=False, checks={"evaluation_completed": False}, failed_checks=["evaluation_completed"],
            error=traceback.format_exc()))
    history = normalized_history(solution.history)
    work = dict(accepted_steps=sum(row.get("phase") == "accepted" for row in history),
                rejected_steps=sum(row.get("phase") == "rejected" for row in history),
                newton_corrections=sum(row.get("inner_iterations", 0) for row in history),
                rejected_newton_trials=sum(row.get("rejected_newton_trials", 0) for row in history))
    model = solution.operator.model
    outlet_options = {field.name: getattr(model.outlet, field.name)
                      for field in dataclass_fields(model.outlet) if field.name != "fluid"}
    outlet_options["wave_backend"] = ("closed-form calorically perfect gas" if isinstance(model.fluid, jxp.FluidPerfectGas)
                                      else "numerical general-EOS wave curves") if model.outlet_formulation == "finite_wave" else "linearized acoustic compatibility"
    return dict(success=bool(solution.success), verified=bool(solution.success and assessment["audit"]["passed"]),
                reason=solution.reason, state=solution.state, points=np.asarray(model.mesh.points),
                history=history, metrics=solution.metrics, work=work,
                elapsed_s=solution.elapsed_s, scheme=asdict(solution.operator.scheme),
                table_sha256=context.table_sha256, outlet_options=outlet_options, **metadata, **assessment)


def main():
    settings = profile()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=settings["cases"])
    parser.add_argument("--seed-family")
    parser.add_argument("--outlet", choices=settings["outlets"], default="acoustic")
    parser.add_argument("--flux", choices=settings["fluxes"], required=True)
    parser.add_argument("--reconstruction", choices=settings["reconstructions"], required=True)
    parser.add_argument("--start", choices=settings["starts"], default="stagnant")
    parser.add_argument("--grids", nargs="+", type=int)
    args = parser.parse_args()
    if bool(args.case) == bool(args.seed_family):
        parser.error("Select exactly one target --case or --seed-family")
    provenance = load(OUTPUT / "provenance.json")
    require_source(provenance)
    scheme = Scheme(flux=args.flux, reconstruction=args.reconstruction, **settings["scheme"])
    policy = ResidualPolicy(**settings["residual"])
    steady = SteadySettings(residual=policy, **settings["steady"])
    common = dict(source_sha256=provenance["source_sha256"], settings_sha256=provenance["settings_sha256"],
                  execution_controls=settings["execution"], flux=args.flux, reconstruction=args.reconstruction)

    if args.seed_family:
        key = seed_id(args.seed_family, args.flux, args.reconstruction)
        path = OUTPUT / "seeds" / (key+".json")
        if path.exists():
            old = load(path)
            if any(old[k] != common[k] for k in ("source_sha256", "settings_sha256")):
                raise RuntimeError("Seed execution identity changed")
            print(f"SAVED SEED {key} verified={old['verified']}", flush=True)
            return
        context = adapted_context(args.seed_family)
        model = context.model(settings["adapted_seed_cells"])
        initial = np.asarray(model.initial_state())
        solution = budgeted_solve(model, scheme, steady)
        require_source(provenance)
        result = record_solution(context, solution, policy, id=key, family=args.seed_family,
                                 cells=model.mesh.n_cells, outlet="acoustic", start="numerical_adapted_preparation",
                                 initial_state=initial, initialization=dict(method="canonical physical guess; no analytical field"),
                                 attempts=[], **common)
        result["seed_sha256"] = identity(dict(state=result["state"], points=result["points"], scheme=result["scheme"],
                                               source_sha256=result["source_sha256"], settings_sha256=result["settings_sha256"]))
        save(path, result)
        print(f"SEED {key} verified={result['verified']} reason={result['reason']} elapsed={solution.elapsed_s:.3f}s", flush=True)
        return

    context = target_context(args.case)
    grids = tuple(args.grids or settings["grids"])
    if any(n not in settings["grids"] for n in grids):
        raise ValueError("Requested grids must belong to the declared hierarchy")
    key = chain_id(args.case, args.outlet, args.flux, args.reconstruction, args.start)
    folder = OUTPUT / "runs" / key
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("a+b")
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
    seed = None
    if args.start == "adapted":
        path = OUTPUT / "seeds" / (seed_id(family_of(args.case), args.flux, args.reconstruction)+".json")
        if path.exists():
            seed = load(path)
        if seed is None or not seed["verified"]:
            for count in grids:
                save(folder / f"N{count}.json", dict(
                    id=f"{key}__N{count}", case_id=args.case, outlet=args.outlet, start=args.start,
                    cells=count, success=False, verified=False, reason="adapted_seed_not_verified",
                    table_sha256=context.table_sha256,
                    state=None, points=None, fields={}, reference_fields={}, errors={}, history=[], metrics={},
                    attempts=[], work={}, elapsed_s=0.0,
                    initialization=dict(method="blocked: missing or failed numerical adapted preparation",
                                        seed_id=None if seed is None else seed["id"]),
                    audit=dict(passed=False, checks={"verified_numerical_seed":False},
                               failed_checks=["verified_numerical_seed"]), **common))
            print(f"BLOCKED {key}: numerical adapted seed not verified", flush=True)
            return
        if any(seed[k] != common[k] for k in ("source_sha256", "settings_sha256")):
            raise RuntimeError("Adapted seed has a different execution identity")

    models = {}
    saved = {}
    for count in grids:
        path = folder / f"N{count}.json"
        if path.exists():
            result = load(path)
            if any(result[k] != common[k] for k in ("source_sha256", "settings_sha256")):
                raise RuntimeError("Saved stage has a different execution identity")
            saved[count] = result
    if len(saved) == len(grids):
        print(f"COMPLETE {key}", flush=True)
        return

    def model_factory(count):
        model = context.model(count, args.outlet, start=args.start, seed=seed)
        models[count] = model
        return model

    previous = None
    previous_row = None
    records = []
    started = perf_counter()

    def on_stage(model, solution, stages):
        nonlocal previous_row
        require_source(provenance)
        count = model.mesh.n_cells
        stage = stages[-1]
        method = stage["initialization"]
        warm = "prolongation" in method and not stage["attempts"]
        if warm and previous_row is not None:
            old_points, old_state = np.asarray(previous_row["points"]), np.asarray(previous_row["state"])
            initial = np.column_stack([np.interp(model.mesh.points, old_points, old_state[:, j]) for j in range(model.nvar)])
        else:
            initial = np.asarray(model.initial_state())
        initialization = dict(method=method, prescribed_start=args.start,
                              source_cells=previous_row["cells"] if warm and previous_row else None,
                              source_run_id=previous_row["id"] if warm and previous_row else None,
                              initial_state_sha256=identity(initial),
                              seed_id=seed["id"] if seed is not None else None,
                              seed_sha256=seed["seed_sha256"] if seed is not None else None,
                              actual_cold_method="uniform reservoir p,h; all velocities zero" if args.start == "stagnant"
                              else "interpolated verified numerical adapted solution")
        result = record_solution(context, solution, policy, id=f"{key}__N{count}", case_id=args.case,
                                 outlet=args.outlet, start=args.start, cells=count,
                                 initial_state=initial, initialization=initialization,
                                 attempts=stage["attempts"], **common)
        result["final_attempt_elapsed_s"] = result["elapsed_s"]
        result["elapsed_s"] += sum(float(attempt.get("elapsed_s", 0)) for attempt in stage["attempts"])
        save(folder / f"N{count}.json", result)
        records.append(result)
        previous_row = result if solution.success else None
        print(f"STAGE {key} N={count} success={solution.success} verified={result['verified']} reason={solution.reason} accepted={result['work']['accepted_steps']} elapsed={solution.elapsed_s:.3f}s", flush=True)

    try:
        with patch.object(drivers, "solve_single_grid", budgeted_solve):
            index = 0
            while index < len(grids):
                count = grids[index]
                if count in saved:
                    row = saved[count]
                    records.append(row)
                    previous_row = row if row["success"] else None
                    previous = None
                    if row["success"]:
                        model = model_factory(count)
                        previous = SimpleNamespace(success=True, state=np.asarray(row["state"]),
                                                   operator=SimpleNamespace(mesh=model.mesh))
                    index += 1
                    continue
                end = index
                while end < len(grids) and grids[end] not in saved:
                    end += 1
                previous, _ = solve_grid_sequence(model_factory, cells=grids[index:end],
                                                  scheme=scheme, settings=steady, on_stage=on_stage,
                                                  initial_solution=previous)
                index = end
        save(folder / "chain.json", dict(chain=key, completed=True, elapsed_s=perf_counter()-started,
                                          records=[row["id"] for row in records]))
    except Exception:
        save(folder / "chain.json", dict(chain=key, completed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
