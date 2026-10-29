"""Shared Newton corrections for implicit square balance systems.

Model callbacks own physical residuals, scales and admissibility. Algebraic
boundary equations are ordinary rows here; no boundary solve or state clipping
is performed by this layer.
"""

import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp


class NewtonInfo(NamedTuple):
    """Traceable inner-solve status; a small update alone is not a root audit.

    status: 0=selected convergence criteria met, 1=iteration cap/no criterion,
    2=invalid initial state, 3=trial acceptance exhausted, 4=nonfinite result.
    residual_norm is NaN for the historical unchecked full-step specialization.
    """

    iterations: jax.Array
    relative_update: jax.Array
    residual_norm: jax.Array
    converged: jax.Array
    admissible: jax.Array
    rejected_trials: jax.Array
    status: jax.Array


def make_newton_solver(
    residual, direction, scale, *, max_iterations, tolerance=0.0,
    admissible=None, backtracking=False, max_backtracks=12, armijo=1e-4,
    residual_tolerance=0.0,
):
    """Return solve(initial_flat, *residual_args) -> (solution, NewtonInfo).

    ``direction(point,*args)`` solves J*correction=-residual. ``scale(point)``
    supplies positive variable scales; residual rows must already be scaled.
    With backtracking, accept an admissible finite trial only if its actual
    residual RMS meets Armijo decrease or residual_tolerance. Exhausted trials
    retain the preceding admissible iterate and report failure for the outer
    timestep controller. No clipping, boundary projection or relaxed tolerance
    is applied. An admissibility callback must itself evaluate safely.

    The specialization backtracking=False/admissible=None/residual_tolerance=0
    retains the existing Euler full-step Newton arithmetic and iteration rules.
    It deliberately does not introduce extra residual evaluations or a new
    convergence criterion; its caller must independently audit the final root.
    """
    max_iterations = operator.index(max_iterations)
    max_backtracks = operator.index(max_backtracks)
    if max_iterations < 0 or max_backtracks < 0:
        raise ValueError("Newton iteration/trial limits must be nonnegative")
    if not math.isfinite(tolerance) or not math.isfinite(residual_tolerance):
        raise ValueError("Newton tolerances must be finite")
    if residual_tolerance < 0 or not 0 < armijo < 1:
        raise ValueError("residual_tolerance must be nonnegative and 0 < armijo < 1")

    legacy_full_steps = not backtracking and admissible is None and residual_tolerance == 0

    def solve(initial_flat, *args):
        if legacy_full_steps:
            if tolerance > 0:
                def condition(carry):
                    return (carry[0] < max_iterations) & (carry[2] > tolerance)

                def body(carry):
                    correction = direction(carry[1], *args)
                    return (
                        carry[0] + 1, carry[1] + correction,
                        jnp.max(jnp.abs(correction) / (scale(carry[1]) + 1e-300)),
                    )

                iterations, point, relative = jax.lax.while_loop(
                    condition, body, (0, initial_flat, jnp.asarray(jnp.inf, initial_flat.dtype))
                )
            else:
                point = jax.lax.fori_loop(
                    0, max_iterations,
                    lambda _, current: current + direction(current, *args),
                    initial_flat,
                )
                iterations = jnp.asarray(max_iterations)
                relative = jnp.asarray(jnp.nan, initial_flat.dtype)
            finite = jnp.all(jnp.isfinite(point))
            converged = finite & (tolerance > 0) & (relative <= tolerance)
            status = jnp.where(~finite, 4, jnp.where(converged, 0, 1))
            return point, NewtonInfo(
                iterations, relative, jnp.asarray(jnp.nan, initial_flat.dtype),
                converged, finite, jnp.asarray(0), status,
            )

        def valid(point):
            finite = jnp.all(jnp.isfinite(point))
            if admissible is None:
                return finite
            return jax.lax.cond(finite, lambda: jnp.all(admissible(point)), lambda: jnp.asarray(False))

        def norm(point):
            value = residual(point, *args)
            return jnp.sqrt(jnp.mean(value**2))

        initial_valid = valid(initial_flat)
        initial_norm = jax.lax.cond(
            initial_valid, lambda: norm(initial_flat),
            lambda: jnp.asarray(jnp.inf, initial_flat.dtype),
        )
        initial_valid = initial_valid & jnp.isfinite(initial_norm)

        def criteria(relative, current_norm):
            update_ok = (tolerance <= 0) | (relative <= tolerance)
            residual_ok = (residual_tolerance <= 0) | (current_norm <= residual_tolerance)
            configured = (tolerance > 0) | (residual_tolerance > 0)
            return configured & update_ok & residual_ok

        initial_relative = jnp.where(
            (residual_tolerance > 0) & (initial_norm <= residual_tolerance), 0.0, jnp.inf
        ).astype(initial_flat.dtype)

        def condition(carry):
            iteration, _, relative, current_norm, state_valid, failed, _ = carry
            return ((iteration < max_iterations) & state_valid & ~failed
                    & ~criteria(relative, current_norm))

        def body(carry):
            iteration, point, _, current_norm, _, _, rejections = carry
            correction = direction(point, *args)
            ntrials = max_backtracks + 1 if backtracking else 1

            def trial_condition(trial):
                return (trial[0] < ntrials) & ~trial[2]

            def trial_body(trial):
                count, alpha, _, accepted_point, accepted_norm, accepted_alpha, rejected = trial
                candidate = point + alpha * correction
                candidate_valid = valid(candidate)
                candidate_norm = jax.lax.cond(
                    candidate_valid, lambda: norm(candidate),
                    lambda: jnp.asarray(jnp.inf, point.dtype),
                )
                accepted = candidate_valid & jnp.isfinite(candidate_norm)
                if backtracking:
                    decreased = candidate_norm <= (1.0 - armijo * alpha) * current_norm
                    at_root = ((residual_tolerance > 0)
                               & (candidate_norm <= residual_tolerance))
                    accepted = accepted & (decreased | at_root)
                return (
                    count + 1, alpha * 0.5, accepted,
                    jnp.where(accepted, candidate, accepted_point),
                    jnp.where(accepted, candidate_norm, accepted_norm),
                    jnp.where(accepted, alpha, accepted_alpha),
                    rejected + (~accepted).astype(jnp.int32),
                )

            trial = jax.lax.while_loop(
                trial_condition, trial_body,
                (0, jnp.asarray(1.0, point.dtype), jnp.asarray(False), point,
                 current_norm, jnp.asarray(0.0, point.dtype), jnp.asarray(0, jnp.int32)),
            )
            _, _, accepted, candidate, candidate_norm, alpha, rejected = trial
            relative = jnp.where(
                accepted,
                jnp.max(jnp.abs(alpha * correction) / (scale(point) + 1e-300)),
                jnp.inf,
            )
            return (iteration + 1, candidate, relative, candidate_norm,
                    jnp.asarray(True), ~accepted, rejections + rejected)

        result = jax.lax.while_loop(
            condition, body,
            (0, initial_flat, initial_relative, initial_norm, initial_valid,
             jnp.asarray(False), jnp.asarray(0, jnp.int32)),
        )
        iterations, point, relative, final_norm, state_valid, failed, rejections = result
        converged = state_valid & ~failed & criteria(relative, final_norm)
        status = jnp.where(~initial_valid, 2,
                           jnp.where(failed, 3, jnp.where(converged, 0, 1)))
        return point, NewtonInfo(
            iterations, relative, final_norm, converged, state_valid, rejections, status
        )

    return solve
