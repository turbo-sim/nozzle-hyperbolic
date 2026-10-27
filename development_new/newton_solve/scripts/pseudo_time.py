"""Measured implicit pseudo-time reference using the production step kernel."""

from time import perf_counter

import jax.numpy as jnp
import numpy as np


def solve_implicit(
    problem, residual_rms_tol=1.0e-8, max_iterations=5000, initial_state=None
):
    """Record accepted pseudo-time updates against the common steady norm.

    Uses the baseline CFL, SER growth, and adaptive increment controls. Each
    outer update contains the production implicit Newton solve. Compilation and
    its discarded warm-up step are handled by problem.warm_up() beforehand.
    Rejected timestep trials are counted, never recorded as accepted iterates.
    The returned ``history_started_at_s`` is the process-local ``perf_counter``
    origin used for history timestamps, allowing alignment with an outer timer.
    ``initial_state`` optionally supplies a full primitive startup field;
    omitting it retains the existing simple initialization.
    """
    state = jnp.asarray(
        problem.initial_state if initial_state is None else initial_state
    )
    if (
        state.shape != problem.initial_state.shape
        or not np.isfinite(np.asarray(state)).all()
    ):
        raise ValueError("initial_state must be a finite full primitive field")
    s = problem.settings
    history = []
    total_trials = rejected_trials = 0
    first_ssr = None
    dt_previous = float(problem.compute_dt(state)[0])
    started = perf_counter()

    def record(index):
        """Evaluate the shared stopping residual once per accepted state."""
        r = problem.residual(problem.pack(state))
        rms = float(np.linalg.norm(r) / np.sqrt(r.size))
        history.append(
            dict(
                accepted_index=index,
                native_iteration=index,
                elapsed_s=perf_counter() - started,
                residual_rms=rms,
                residual_l2=float(np.linalg.norm(r)),
                implicit_step_trials=total_trials,
                rejected_trials=rejected_trials,
            )
        )
        return rms

    rms = record(0)
    previous_ssr = np.inf
    message = "Maximum pseudo-time iterations reached"
    for index in range(1, max_iterations + 1):
        if rms <= residual_rms_tol:
            message = "Common steady residual target reached"
            break
        dt_cap, sound = problem.compute_dt(state)
        dt_cap = float(dt_cap)
        if first_ssr is not None and np.isfinite(previous_ssr):
            multiplier = np.clip(
                (first_ssr / max(previous_ssr, 1.0e-300)) ** s["ser_beta"],
                1.0,
                s["ser_cfl_max"] / s["CFL"],
            )
            dt_cap *= float(multiplier)
        dt = min(dt_previous * 1.2, dt_cap)
        old = np.asarray(state)
        accepted = False
        for _ in range(9):
            candidate, ssr = problem.step(state, jnp.float64(dt))
            total_trials += 1
            new = np.asarray(candidate)
            increments = (
                np.max(np.abs(new[:, 0] - old[:, 0]) / (old[:, 0] + 1.0e-10)),
                np.max(np.abs(new[:, 2] - old[:, 2]) / (np.abs(old[:, 2]) + 1.0e-10)),
                np.max(
                    np.abs(new[:, 1] - old[:, 1])
                    / (np.abs(old[:, 1]) + np.asarray(sound) + 1.0e-10)
                ),
            )
            if (
                np.isfinite(new).all()
                and np.all(new[1:-1] > 0.0)
                and all(v <= limit for v, limit in zip(increments, (0.2, 0.1, 0.2)))
            ):
                accepted = True
                break
            rejected_trials += 1
            dt *= 0.5
        if not accepted:
            message = "Adaptive implicit update failed its increment/domain checks"
            break
        state = candidate
        dt_previous = dt
        previous_ssr = float(ssr)
        if first_ssr is None:
            first_ssr = previous_ssr
        rms = record(index)
    if rms <= residual_rms_tol:
        message = "Common steady residual target reached"
    wall_s = perf_counter() - started
    return dict(
        x=problem.pack(state),
        history=history,
        wall_s=wall_s,
        history_started_at_s=started,
        native_success=bool(rms <= residual_rms_tol),
        message=message,
        iterations=len(history) - 1,
        nfev=None,
        njev=None,
        backend=dict(
            method="backward Euler / Newton / banded",
            implicit_step_trials=total_trials,
            rejected_trials=rejected_trials,
            note="Inner residual/Jacobian evaluation counts are unavailable; monitor calls are separate.",
        ),
    )
