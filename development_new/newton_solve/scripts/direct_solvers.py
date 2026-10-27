"""Unshifted steady Newton through the custom PySolverView backend.

The problem supplies a dimensionless residual and its exact Jacobian through
``residual(x)`` and ``gradient(x)``. The adapter records the initial state and
accepted steps, using residual values already computed by the backend. Model
domain failures should raise ``pysolver_view.DomainError`` so the backends can
backtrack; unrelated exceptions propagate to the caller.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
import pysolver_view as psv


def _initial_point(x0):
    """Copy a finite, nonempty vector of dimensionless solver coordinates."""
    point = np.array(x0, dtype=float, copy=True)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("x0 must be a finite, nonempty one-dimensional vector.")
    return point


def _history_record(history, entry, residual, elapsed_s):
    """Append an accepted record without evaluating the physical problem.

    ``residual`` contains dimensionless equation values already available in
    the backend callback. ``accepted_index`` counts stored states from zero;
    ``native_iteration`` retains the backend's own iteration label.
    """
    residual = np.asarray(residual, dtype=float)
    residual_l2 = float(np.linalg.norm(residual))
    record = {
        "accepted_index": len(history),
        "native_iteration": int(entry["iteration"]),
        "elapsed_s": float(elapsed_s),
        "residual_rms": residual_l2 / np.sqrt(residual.size),
        "residual_l2": residual_l2,
        "nfev": int(entry["nfev"]),
        "njev": int(entry["njev"]),
    }
    for name in (
        "step_fraction",
        "backtracks",
        "domain_rejections",
        "radius",
        "step_norm",
        "scaled_step_norm",
    ):
        if name in entry:
            record[name] = entry[name]
    history.append(record)


def _solution_result(solver, history, wall_s, history_started_at_s):
    """Detach the final point and summarize backend status and counted work.

    ``native_success`` retains the frontend's backend-based success decision.
    The caller must also check the final physical steady residual against its
    common acceptance criterion before comparing methods or reporting speedups.
    ``history_started_at_s`` is the process-local ``perf_counter`` origin of
    accepted-history timestamps, used to align stages within an outer timer.
    """
    result = solver.result
    backend_result = solver.backend_result
    backend = {
        "library": solver.library,
        "method": solver.method,
        "pysolver_view_version": psv.__version__,
        "status": int(backend_result.status),
        "effective_options": deepcopy(backend_result.effective_options),
    }
    for name in (
        "solver_success",
        "feasible",
        "ntrials",
        "nrejected",
        "ninvalid",
        "nbacktracks",
        "domain_rejections",
        "derivative_rejections",
        "derivative_nfev",
        "failed_njev",
        "linear_algebra",
        "nupdates",
        "radius",
        "max_trial_step_scaled",
        "max_radius_used",
        "max_radius_ratio",
        "kkt_enabled",
        "kkt_evaluations",
        "kkt_time",
    ):
        if name in backend_result:
            backend[name] = deepcopy(backend_result[name])
    return {
        "x": np.array(solver.x_final, dtype=float, copy=True),
        "history": history,
        "wall_s": float(wall_s),
        "history_started_at_s": float(history_started_at_s),
        "native_success": bool(solver.success),
        "message": str(solver.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "iterations": int(result.nit),
        "backend": backend,
    }


def solve_newton(
    problem,
    x0,
    residual_rms_tol=1e-8,
    max_iterations=150,
    first_step_fraction=0.25,
    step_growth_factor=1.5,
):
    """Solve a square steady system with fraction-controlled Newton steps.

    Parameters
    ----------
    problem : object
        Provides ``residual(x)`` of shape ``(n,)`` and ``gradient(x)`` of shape
        ``(n, n)`` in dimensionless variables and equation units.
    x0 : array_like, shape (n,)
        Shared dimensionless initial state used by the comparison methods.
    residual_rms_tol : float, optional
        RMS residual target. PySolverView's Newton target is a two-norm, so
        this is converted to ``residual_rms_tol * sqrt(n)``.
    max_iterations : int, optional
        Maximum backend trial attempts, including rejected backtracking trials.
    first_step_fraction : float, optional
        Initial fraction of the Newton direction, in the interval ``(0, 1]``.
    step_growth_factor : float, optional
        Growth of the direction-fraction cap after each accepted move, up to 1.

    Returns
    -------
    dict
        Final coordinates, accepted residual history, solve wall time in
        seconds, backend success/status, and evaluation/iteration counts.

    Notes
    -----
    Armijo backtracking remains enabled. Its separate absolute displacement
    envelope is disabled, leaving the requested fraction schedule to control
    initial step size. The solve timer includes callbacks and backend interface
    work, and excludes solver construction. No callback evaluates the model.
    """
    point = _initial_point(x0)
    history = []

    def record_accepted(_point, _callback_count):
        """Timestamp a Newton record already stored by the public frontend."""
        entry = solver.iteration_history[-1]
        _history_record(history, entry, entry["fun"], perf_counter() - started)

    solver = psv.NonlinearSystemSolver(
        problem,
        library="pysolver",
        method="newton_raphson",
        tolerance=residual_rms_tol * np.sqrt(point.size),
        max_iterations=max_iterations,
        options={
            "first_step_fraction": first_step_fraction,
            "step_growth_factor": step_growth_factor,
            "history_level": "full",
            "backtracking": {"step_limits": None},
        },
        update_on="iteration",
        callback_func=record_accepted,
        print_convergence=False,
        plot_convergence=False,
    )
    started = perf_counter()
    solver.solve(point)
    wall_s = perf_counter() - started
    return _solution_result(solver, history, wall_s, started)
