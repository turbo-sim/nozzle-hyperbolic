"""Study-only execution budgets, preserving the last accepted production state.

The production equations, time integrator and convergence policy are unchanged.
A declared lack-of-progress budget prevents spending thousands of iterations
on a state whose CFL has collapsed. Budget stops are failures, never successes.
"""
from time import perf_counter
from unittest.mock import patch
import json

import numpy as np
from nozzle_hyperbolic.solvers.fvm import drivers, Solution
from .common import STUDY


CONTROLS = dict(wall_seconds=180.0, plateau_window=64,
                plateau_relative_improvement=1e-7, plateau_cfl_max=1e-6)


def controls_for(scheme):
    """Execution budgets are separate from numerical convergence tolerances."""
    options = json.loads((STUDY / 'settings.json').read_text())
    limits = options.get('execution_limits', {})
    controls = dict(CONTROLS)
    controls['wall_seconds'] = limits.get(
        'osher_wall_seconds' if scheme.flux == 'osher' else 'default_wall_seconds',
        CONTROLS['wall_seconds'])
    return controls


class StudyBudgetStop(RuntimeError):
    pass


def budgeted_solve(model, scheme, settings, **kwargs):
    """Call the actual driver; observe its accepted state without replacing math."""
    production_solve = _PRODUCTION_SOLVE
    production_operator = drivers.Operator
    controls = controls_for(scheme)
    observer = kwargs.pop("callback", None)
    cache = {}
    history, accepted = [], []
    started = perf_counter()

    def bind(*args, **options):
        operator = production_operator(*args, **options)
        compiled_step = operator.step
        cache["operator"] = operator
        def tracked_step(state, *step_args):
            cache["state"] = state
            candidate, info = compiled_step(state, *step_args)
            cache["candidate"] = candidate
            return candidate, info
        operator.step = tracked_step
        return operator

    def observe(row):
        history.append(row)
        if observer is not None:
            observer(row)
        if row.get("phase") == "accepted":
            cache["state"] = cache["candidate"]
            accepted.append(row)
        if row.get("phase") == "initial":
            cache["initial_norm"] = np.asarray(row["equation_norm"])
        converged = all(row.get(key, False) for key in
                        ("finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed"))
        if converged:
            return
        if perf_counter() - started >= controls["wall_seconds"]:
            raise StudyBudgetStop("study_wall_time_budget")
        window = controls["plateau_window"]
        if len(accepted) >= window and row.get("phase") == "accepted":
            recent = accepted[-window:]
            residual = np.array([np.max(r["equation_norm"]) for r in recent])
            cfl = np.array([r["cfl"] for r in recent])
            scale = max(float(residual[0]), np.finfo(float).tiny)
            improvement = (float(residual[0]) - float(np.min(residual))) / scale
            if np.all(cfl <= controls["plateau_cfl_max"]) and improvement < controls["plateau_relative_improvement"]:
                raise StudyBudgetStop("study_collapsed_cfl_plateau")

    try:
        with patch.object(drivers, "Operator", bind):
            return production_solve(model, scheme, settings, callback=observe, **kwargs)
    except StudyBudgetStop as stop:
        operator = cache["operator"]
        state = cache.get("state")
        if state is None:
            state = kwargs.get("initial_state")
            if state is None:
                state = model.initial_state()
        sample = drivers.sampler(operator, settings.residual)
        metrics = sample(state, cache.get("initial_norm"))
        metrics["study_execution_stop"] = str(stop)
        return Solution(np.asarray(state), False, str(stop), history, metrics,
                        elapsed_s=perf_counter()-started, operator=operator)


_PRODUCTION_SOLVE = drivers.solve_single_grid
