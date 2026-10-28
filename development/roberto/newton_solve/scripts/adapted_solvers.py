"""Controlled PySolverView methods for the adapted-field initialization study.

Every model evaluation uses the existing steady problem. This module supplies
only solver configuration, accepted-point monitoring and detached results.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np
import pysolver_view as psv

from development.roberto.newton_solve.scripts.direct_solvers import _history_record


def solve_custom(
    problem,
    initial,
    method,
    *,
    max_iterations=300,
    native_tolerance=1e-10,
    residual_rms_tol=1e-8,
    first_step_fraction=0.25,
    initial_radius=0.25,
    step_growth_factor=1.5,
    max_wall_s=90,
    observation_sink=None,
):
    """Use custom Newton/HYBR/LM or constrained SLSQP, never vanilla SciPy.

    Newton/SLSQP use a direction-fraction schedule. HYBR/LM instead use an
    absolute trust radius in the same dimensionless coordinates; these are
    distinct controls, not numerically equivalent initial step lengths.
    """
    point = np.array(initial, dtype=float, copy=True)
    history = []

    def publish(entry, residual):
        _history_record(history, entry, residual, perf_counter() - started)
        record = history[-1]
        record.update(
            x=np.array(entry["x"], copy=True),
            accepted=True,
            observation_index=len(history) - 1,
            cached_residual_rms=record["residual_rms"],
        )
        if observation_sink is not None:
            observation_sink(deepcopy(record))
        if len(history) == 1 or len(history) % 25 == 0:
            print(
                f'    {method}: accepted {len(history)-1}, RMS={record["residual_rms"]:.3e}',
                flush=True,
            )
        return False if perf_counter() - started >= max_wall_s else None

    if method == "slsqp":
        lower = np.tile([1e-8, -np.inf, 1e-8], point.size // 3)
        optimization = psv.as_optimization_problem(
            problem, n_variables=point.size, bounds=(lower, np.full(point.size, np.inf))
        )

        def observe_optimization(entry):
            # Accepted custom SLSQP entries expose the original equality values.
            return publish(entry, np.asarray(entry["fitness"])[1:])

        solver = psv.OptimizationSolver(
            optimization,
            library="pysolver",
            method="slsqp",
            tolerance=native_tolerance,
            max_iterations=max_iterations,
            problem_scale=None,
            check_kkt=False,
            extra_options=dict(
                first_step_fraction=first_step_fraction,
                max_step_fraction=1.0,
                step_growth_factor=step_growth_factor,
                feasibility_tol=residual_rms_tol,
            ),
            update_on="iteration",
            backend_callback=observe_optimization,
            print_convergence=False,
            plot_convergence=False,
        )
    else:
        options = dict(history_level="full")
        if method == "newton":
            options.update(
                first_step_fraction=first_step_fraction,
                step_growth_factor=step_growth_factor,
                backtracking=dict(step_limits=None),
            )
        elif method in ("hybr", "lm"):
            options.update(
                initial_radius=initial_radius,
                max_radius=100.0,
                growth_factor=step_growth_factor,
                max_nfev=5000,
            )
        else:
            raise ValueError("Expected newton, hybr, lm or slsqp")

        def observe_root(_point, _count):
            entry = solver.iteration_history[-1]
            return publish(entry, entry["fun"])

        solver = psv.NonlinearSystemSolver(
            problem,
            library="pysolver",
            method=method,
            tolerance=residual_rms_tol * np.sqrt(point.size),
            max_iterations=max_iterations,
            options=options,
            update_on="iteration",
            callback_func=observe_root,
            print_convergence=False,
            plot_convergence=False,
        )

    problem.reset_counters()
    started = perf_counter()
    error = None
    returned = None
    try:
        returned = np.array(solver.solve(point.copy()), dtype=float, copy=True)
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception))
    elapsed = perf_counter() - started
    backend = solver.backend_result
    if (
        returned is None
        and backend is not None
        and getattr(backend, "x", None) is not None
    ):
        returned = np.array(backend.x, dtype=float, copy=True)
    last = None if not history else history[-1]["x"]
    candidate = (
        returned if returned is not None else last if last is not None else point
    )
    source = (
        "returned"
        if returned is not None
        else "last_observation" if last is not None else "initial"
    )
    native = solver.result
    return dict(
        method=method,
        x=returned,
        authoritative_returned_point=returned is not None,
        audited_point=candidate,
        diagnostic_point_source=source,
        last_observed_candidate=last,
        history=history,
        wall_s=elapsed,
        exception=error,
        native_success=bool(solver.success) if native is not None else False,
        native_status=None if backend is None else int(backend.status),
        native_message=str(error["message"]) if error else str(solver.message),
        iterations=None if backend is None else int(backend.nit),
        nfev=None if native is None else int(native.nfev),
        njev=None if native is None else int(native.njev),
        model_work=problem.model_work(),
        first_observed_seed_max_difference=(
            None if not history else float(np.max(np.abs(history[0]["x"] - point)))
        ),
        backend=dict(
            library=solver.library,
            method=solver.method,
            effective_options=(
                {} if backend is None else deepcopy(backend.effective_options)
            ),
            history_note="Initial and accepted custom-driver points.",
            initial_step_controls_available=True,
        ),
    )
