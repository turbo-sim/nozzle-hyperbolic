"""Study-local fixed-CFL-growth controller using unchanged implicit kernels.

Adapted from newton_solve/scripts/pseudo_time.py; unlike its SER policy, CFL
grows after each accepted update and shrinks after each rejected trial.
"""

from time import perf_counter

import jax.numpy as jnp
import numpy as np


def solve_implicit(
    problem,
    residual_rms_tol=1.0e-8,
    max_iterations=5000,
    initial_state=None,
    *,
    dt_growth_factor=None,
    max_halvings=12,
    record_controls=False,
    state_is_admissible=None,
):
    """Apply the annular investigation's fixed-growth CFL recommendation.

    Retain source full-step Newton, adding independent acceptance checks on
    its relative correction and on the scaled backward-Euler residual.
    All rejected work and inner correction counts are retained.
    """
    if dt_growth_factor is None:
        dt_growth_factor = problem.settings.get("dt_growth_factor", 1.2)
    if not np.isfinite(dt_growth_factor) or dt_growth_factor < 1:
        raise ValueError("dt_growth_factor must be finite and at least one")
    if (
        isinstance(max_halvings, bool)
        or int(max_halvings) != max_halvings
        or max_halvings < 0
    ):
        raise ValueError("max_halvings must be a nonnegative integer")
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
    trial_history = []
    accepted_control = dict(dt_s=None, effective_cfl=None, existing_ssr=None)
    total_trials = rejected_trials = 0
    current_cfl = float(s["CFL"])
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
        if record_controls:
            history[-1].update(accepted_control)
        return rms

    rms = record(0)
    message = "Maximum pseudo-time iterations reached"
    for index in range(1, max_iterations + 1):
        if rms <= residual_rms_tol:
            message = "Common steady residual target reached"
            break
        dt_cap, sound = problem.compute_dt(state)
        dt_cap = float(dt_cap)
        base_dt = dt_cap
        dt = base_dt * current_cfl / s["CFL"]
        old = np.asarray(state)
        accepted = False
        for halving in range(int(max_halvings) + 1):
            candidate, ssr, inner_info, implicit_rms = problem.checked_step(
                state, jnp.float64(dt)
            )
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
            valid = (
                bool(inner_info.converged)
                and float(implicit_rms) <= 1e-10
                and np.isfinite(new).all()
                and (
                    np.all(new[1:-1] > 0.0)
                    if state_is_admissible is None
                    else bool(state_is_admissible(new))
                )
                and all(v <= limit for v, limit in zip(increments, (0.2, 0.1, 0.2)))
            )
            if record_controls:
                trial_history.append(
                    dict(
                        trial_index=total_trials,
                        anchor_index=index - 1,
                        dt_s=dt,
                        effective_cfl=dt * s["CFL"] / base_dt,
                        halving=halving,
                        accepted=bool(valid),
                        existing_ssr=float(ssr),
                        inner_iterations=int(inner_info.iterations),
                        inner_relative_update=float(inner_info.relative_update),
                        implicit_residual_rms=float(implicit_rms),
                        max_dp_relative=float(increments[0]),
                        max_dh_relative=float(increments[1]),
                        max_du_scaled=float(increments[2]),
                    )
                )
            if valid:
                accepted = True
                break
            rejected_trials += 1
            dt *= 0.5
        if not accepted:
            message = "Adaptive implicit update failed its Newton/residual/increment/domain checks"
            break
        state = candidate
        current_cfl = min(dt * s["CFL"] / base_dt * dt_growth_factor, s["ser_cfl_max"])
        previous_ssr = float(ssr)
        accepted_control = dict(
            dt_s=dt, effective_cfl=dt * s["CFL"] / base_dt, existing_ssr=previous_ssr
        )
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
            newton_corrections=sum(t["inner_iterations"] for t in trial_history),
            note="Includes rejected Newton work; source full-step Newton without inner line search.",
        ),
        **(dict(trial_history=trial_history) if record_controls else {}),
    )
