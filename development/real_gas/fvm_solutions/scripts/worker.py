"""One physical-case/flux/reconstruction chain in an isolated Python process."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, ResidualPolicy, solve_grid_sequence
from .common import STUDY, OUTPUT, clean, identity, save, source_manifest
from .problem import load_context
from .audit import evaluate_solution
from .execution import budgeted_solve, CONTROLS
from .profiles import register, compatible
from nozzle_hyperbolic.solvers.fvm import drivers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluid-case", required=True)
    parser.add_argument("--regime", choices=("adapted", "shock"), required=True)
    parser.add_argument("--flux", choices=("rusanov", "hll", "roe"), required=True)
    parser.add_argument("--reconstruction", choices=("constant", "muscl_van_leer"), required=True)
    parser.add_argument("--grids", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = json.loads((STUDY / "settings.json").read_text())
    grids = tuple(args.grids or profile["grids"])
    case_id = args.fluid_case + "_" + args.regime
    chain = "__".join((case_id, args.flux, args.reconstruction))
    folder = OUTPUT / "runs" / chain
    folder.mkdir(parents=True, exist_ok=True)
    # An advisory per-chain lock permits independent campaign slices without
    # duplicate solves or overwriting another active process's stage evidence.
    lock = (folder / "worker.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"BUSY {chain}: another process owns this chain", flush=True)
        return
    start = perf_counter()
    source_before = source_manifest()
    provenance_file = OUTPUT / "provenance.json"
    if provenance_file.exists():
        campaign_source = json.loads(provenance_file.read_text())["source_before"]
        if source_before != campaign_source:
            raise RuntimeError("Source changed since campaign start; refusing a mixed-source campaign")
    context = load_context(args.fluid_case, args.regime)
    scheme = Scheme(flux=args.flux, reconstruction=args.reconstruction, **profile["scheme"])
    policy = ResidualPolicy(**profile["residual"])
    settings = SteadySettings(residual=policy, **profile["steady"])
    model_factory = lambda cells: context.model(cells, cold_start=profile.get("cold_start"))
    profile_sha = identity(profile)
    register(profile)
    initial_solution, previous_cells = None, None
    records = []
    remaining = list(grids)
    if args.resume or any(folder.glob("N*.json")):
        for cells in grids:
            path = folder / f"N{cells}.json"
            if not path.exists():
                break
            result = json.loads(path.read_text())
            if not compatible(result.get("profile_sha256"), profile_sha, args.fluid_case, args.regime) or result.get("source_sha256") != identity(source_before):
                raise RuntimeError("Refusing resume with changed settings or source")
            records.append(result)
            remaining.remove(cells)
            previous_cells = cells
            if result["success"]:
                model = model_factory(cells)
                initial_solution = SimpleNamespace(
                    success=True, state=np.asarray(result["state"]),
                    operator=SimpleNamespace(mesh=model.mesh),
                )
            else:
                initial_solution = None

    def on_stage(model, solution, stages):
        nonlocal previous_cells
        cells = model.mesh.n_cells
        if source_before != source_manifest():
            raise RuntimeError("Source changed during grid sequence; stop and keep the completed-stage evidence")
        stage = stages[-1]
        print(f"AUDIT {chain} N={cells} native={solution.success} reason={solution.reason}", flush=True)
        try:
            evaluation = evaluate_solution(context, solution, policy)
        except Exception:
            evaluation = dict(fields={}, errors={}, audit=dict(passed=False, error=traceback.format_exc()))
        history = []
        for row in solution.history:
            row = dict(row)
            if "equation_norm" in row:
                row["residual"] = float(np.max(row["equation_norm"]))
                row["boundary_residual"] = row["boundary_max"]
            history.append(row)
        work = dict(
            accepted_steps=sum(row.get("phase") == "accepted" for row in history),
            rejected_steps=sum(row.get("phase") == "rejected" for row in history),
            newton_corrections=sum(row.get("inner_iterations", 0) for row in history),
            rejected_newton_trials=sum(row.get("rejected_newton_trials", 0) for row in history),
        )
        retained_attempt_time = sum(float(attempt.get("elapsed_s", 0)) for attempt in stage["attempts"])
        result = dict(
            id=f"{chain}__N{cells}", case_id=case_id, flux=args.flux,
            reconstruction=args.reconstruction, cells=cells,
            success=solution.success, verified=bool(solution.success and evaluation["audit"].get("passed", False)),
            reason=solution.reason, elapsed_s=solution.elapsed_s + retained_attempt_time,
            final_attempt_elapsed_s=solution.elapsed_s,
            retained_attempt_elapsed_s=retained_attempt_time,
            chain_elapsed_s=perf_counter()-start,
            initialization=dict(method=stage["initialization"], source_cells=previous_cells
                                if "prolongation" in stage["initialization"] else None),
            attempts=stage["attempts"], state=solution.state, history=history,
            metrics=solution.metrics, work=work, scheme=asdict(scheme),
            profile_sha256=profile_sha, source_sha256=identity(source_before),
            roe_metadata=getattr(model, "roe_metadata", None) if args.flux == "roe" else None,
            execution_controls=CONTROLS,
            table_sha256=context.eos.table_sha256,
            **evaluation,
        )
        save(folder / f"N{cells}.json", result)
        records.append(clean(result))
        previous_cells = cells if solution.success else None
        print(f"SAVED {chain} N={cells} verified={result['verified']} accepted={work['accepted_steps']} rejected={work['rejected_steps']} elapsed={solution.elapsed_s:.3f}s", flush=True)

    try:
        if remaining:
            with patch.object(drivers, "solve_single_grid", budgeted_solve):
                solve_grid_sequence(model_factory, cells=remaining, scheme=scheme, settings=settings,
                                    on_stage=on_stage, initial_solution=initial_solution)
        completed_cells = [r["cells"] for r in records]
        save(folder / "chain.json", dict(chain=chain,
                                        completed=set(profile["grids"]).issubset(completed_cells),
                                        requested_grids=grids, completed_grids=completed_cells,
                                        elapsed_s=perf_counter()-start,
                                        src_unchanged=source_before == source_manifest(),
                                        records=[r["id"] for r in records]))
    except Exception:
        save(folder / "chain.json", dict(chain=chain, completed=False, error=traceback.format_exc(),
                                        elapsed_s=perf_counter()-start,
                                        src_unchanged=source_before == source_manifest()))
        raise


if __name__ == "__main__":
    main()
