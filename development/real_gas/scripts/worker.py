"""One physical-case/flux/reconstruction chain in an isolated Python process."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from dataclasses import asdict
import json
from time import perf_counter
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, ResidualPolicy, solve_grid_sequence
from .common import (STUDY, OUTPUT, clean, identity, save, source_manifest,
                     source_for_flux, require_current_source)
from .problem import load_context
from .audit import evaluate_solution
from .execution import budgeted_solve, controls_for
from .configuration import case_settings_id, scheme_options, scheme_matches
from nozzle_hyperbolic.solvers.fvm import drivers


def main():
    profile = json.loads((STUDY / "settings.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluid-case", required=True)
    parser.add_argument("--regime", choices=("adapted", "shock"), required=True)
    parser.add_argument("--flux", choices=profile['fluxes'], required=True)
    parser.add_argument("--reconstruction", choices=("constant", "muscl_van_leer"), required=True)
    parser.add_argument("--grids", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--retry-budget-stops', action='store_true',
                        help='Recompute wall-budget stops, retaining their full attempts')
    args = parser.parse_args()
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
    provenance_file = OUTPUT / "provenance.json"
    if not provenance_file.exists():
        raise RuntimeError("Use scripts.run to initialize the campaign provenance")
    metadata = json.loads(provenance_file.read_text())
    source_id = identity(source_for_flux(metadata, args.flux))
    case_settings_sha = case_settings_id(profile, args.fluid_case, args.regime)
    initial_solution, previous_cells = None, None
    records = []
    saved = {}
    if args.resume or any(folder.glob("N*.json")):
        for cells in grids:
            path = folder / f"N{cells}.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if (result.get("case_settings_sha256") != case_settings_sha
                    or result.get("source_sha256") != source_id
                    or not scheme_matches(profile, args.flux, result['scheme'])):
                raise RuntimeError("Refusing resume with changed settings or source")
            saved[cells] = result
    remaining = [cells for cells in grids if cells not in saved or (
        args.retry_budget_stops and saved[cells]['reason'] == 'study_wall_time_budget')]
    if not remaining:
        print(f"COMPLETE {chain}: all requested stages are saved", flush=True)
        return
    require_current_source(metadata)
    require_current_source(metadata, args.flux)
    source_files = source_manifest()
    context = load_context(args.fluid_case, args.regime)
    scheme = Scheme(reconstruction=args.reconstruction, **scheme_options(profile, args.flux))
    policy = ResidualPolicy(**profile["residual"])
    settings = SteadySettings(residual=policy, **profile["steady"])
    model_factory = lambda cells: context.model(cells, cold_start=profile.get("cold_start"))

    def on_stage(model, solution, stages):
        nonlocal previous_cells
        cells = model.mesh.n_cells
        if source_files != source_manifest():
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
        attempts = [dict(attempt, execution_controls=controls_for(scheme))
                    for attempt in stage['attempts']]
        if cells in saved:
            prior = saved[cells]
            prior_attempts = [dict(attempt, execution_controls=attempt.get(
                'execution_controls', prior.get('execution_controls')))
                for attempt in prior.get('attempts', [])]
            attempts = [*prior_attempts, dict(
                initialization=prior['initialization']['method'],
                success=prior['success'], reason=prior['reason'],
                state=prior['state'], metrics=prior['metrics'], history=prior['history'],
                elapsed_s=prior.get('final_attempt_elapsed_s', prior['elapsed_s']),
                execution_controls=prior.get('execution_controls'),
                source_sha256=prior['source_sha256'], scheme=prior['scheme'],
                audit=prior['audit'],
            ), *attempts]
        retained_attempt_time = sum(float(attempt.get("elapsed_s", 0)) for attempt in attempts)
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
            attempts=attempts, state=solution.state, history=history,
            metrics=solution.metrics, work=work, scheme=asdict(scheme),
            case_settings_sha256=case_settings_sha, source_sha256=source_id,
            roe_metadata=getattr(model, "roe_metadata", None) if args.flux == "roe" else None,
            execution_controls=controls_for(scheme),
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
                index = 0
                while index < len(grids):
                    cells = grids[index]
                    if cells not in remaining:
                        result = saved[cells]
                        records.append(result)
                        previous_cells = cells if result['success'] else None
                        if result['success']:
                            model = model_factory(cells)
                            initial_solution = SimpleNamespace(
                                success=True, state=np.asarray(result['state']),
                                operator=SimpleNamespace(mesh=model.mesh))
                        else:
                            initial_solution = None
                        index += 1
                        continue
                    end = index
                    while end < len(grids) and grids[end] in remaining:
                        end += 1
                    initial_solution, _ = solve_grid_sequence(
                        model_factory, cells=grids[index:end], scheme=scheme, settings=settings,
                        on_stage=on_stage, initial_solution=initial_solution)
                    index = end
        completed_cells = [r["cells"] for r in records]
        save(folder / "chain.json", dict(chain=chain,
                                        completed=set(profile["grids"]).issubset(completed_cells),
                                        requested_grids=grids, completed_grids=completed_cells,
                                        records=[r["id"] for r in records]))
    except Exception:
        save(folder / "chain.json", dict(chain=chain, completed=False, error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
