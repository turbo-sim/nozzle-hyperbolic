"""Declared execution stops observe production solves without altering equations."""
from time import perf_counter
from unittest.mock import patch

import traceback
import numpy as np
from nozzle_hyperbolic.solvers.fvm import drivers, Solution
from .common import profile

_PRODUCTION_SOLVE = drivers.solve_single_grid


class BudgetStop(RuntimeError):
    pass


def budgeted_solve(model, scheme, settings, **kwargs):
    controls = profile()["execution"]
    started = perf_counter()
    production_operator = drivers.Operator
    observer = kwargs.pop("callback", None)
    history, accepted, cache = [], [], {}

    def bind(*args, **options):
        operator = production_operator(*args, **options)
        compiled = operator.step
        cache["operator"] = operator

        def step(state, *arguments):
            cache["state"] = state
            candidate, info = compiled(state, *arguments)
            cache["candidate"] = candidate
            return candidate, info

        operator.step = step
        return operator

    def observe(row):
        history.append(row)
        if observer is not None:
            observer(row)
        if row.get("phase") == "initial":
            cache["initial_norm"] = np.asarray(row["equation_norm"])
        if row.get("phase") == "accepted":
            cache["state"] = cache["candidate"]
            accepted.append(row)
        if all(row.get(key, False) for key in ("finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed")):
            return
        if perf_counter()-started >= controls["wall_seconds_per_attempt"]:
            raise BudgetStop("study_wall_time_budget")
        window = controls["plateau_window"]
        if row.get("phase") == "accepted" and len(accepted) >= window:
            recent = accepted[-window:]
            residual = np.asarray([np.max(value["equation_norm"]) for value in recent])
            cfl = np.asarray([value["cfl"] for value in recent])
            improvement = (residual[0]-residual.min())/max(float(residual[0]), np.finfo(float).tiny)
            if np.all(cfl <= controls["plateau_cfl_max"]) and improvement < controls["plateau_relative_improvement"]:
                raise BudgetStop("study_collapsed_cfl_plateau")

    try:
        with patch.object(drivers, "Operator", bind):
            return _PRODUCTION_SOLVE(model, scheme, settings, callback=observe, **kwargs)
    except Exception as stop:
        operator = cache.get("operator")
        if operator is None:
            raise
        state = cache.get("state", kwargs.get("initial_state"))
        if state is None:
            state = model.initial_state()
        metrics = drivers.sampler(operator, settings.residual)(state, cache.get("initial_norm"))
        metrics["study_execution_stop"] = str(stop)
        if not isinstance(stop, BudgetStop):
            metrics["exception"] = traceback.format_exc()
        return Solution(np.asarray(state), False, str(stop) if isinstance(stop, BudgetStop) else "exception", history, metrics,
                        elapsed_s=perf_counter()-started, operator=operator)
