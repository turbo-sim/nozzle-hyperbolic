"""Conservative forward Euler and SSPRK3 with algebraic stage constraints."""

import jax
import jax.numpy as jnp

from .implicit import NewtonInfo


def make_explicit_step(
    storage,
    spatial,
    recover,
    complete_boundaries,
    balance,
    admissible,
    *,
    method,
    evaluate_stage=None
):
    """Build a step using only model-independent numerical callbacks.

    recover(contents, physical_guess) and complete_boundaries(full,time) return
    a state and NewtonInfo. balance(full) returns the signed boundary/source
    rate used for an independent global budget. Stage combinations apply to
    integrated conserved contents, never to primitive iteration variables.
    Optional evaluate_stage(full) returns both spatial residual and balance in
    one evaluation so expensive model properties are reused by the assembler.
    """
    if method not in ("euler-explicit", "runge-kutta-3"):
        raise ValueError("Unsupported explicit integrator")

    def step(old, dt, time=0.0):
        def normalize(info):
            return NewtonInfo(
                jnp.asarray(info.iterations, jnp.int32),
                jnp.asarray(info.relative_update, old.dtype),
                jnp.asarray(info.residual_norm, old.dtype),
                jnp.asarray(info.converged, bool),
                jnp.asarray(info.admissible, bool),
                jnp.asarray(info.rejected_trials, jnp.int32),
                jnp.asarray(info.status, jnp.int32),
            )

        # time denotes the end of the requested step, as for backward Euler.
        start = time - dt
        initial, initial_info = complete_boundaries(old, start)
        initial_info = normalize(initial_info)
        initial_ok = (
            initial_info.converged & initial_info.admissible & admissible(initial)
        )
        zero = jnp.zeros((old.shape[1],), old.dtype)

        def advance(target, seed, stage_time):
            physical, recovery = recover(target, seed[1:-1])
            recovery = normalize(recovery)
            provisional = seed.at[1:-1].set(physical)

            def boundary(_):
                full, info = complete_boundaries(provisional, stage_time)
                info = normalize(info)
                okay = info.converged & info.admissible & admissible(full)
                return full, info._replace(converged=okay, admissible=admissible(full))

            skipped = recovery._replace(
                iterations=jnp.int32(0),
                relative_update=jnp.asarray(0.0, old.dtype),
                residual_norm=jnp.asarray(0.0, old.dtype),
                converged=jnp.asarray(False),
                rejected_trials=jnp.int32(0),
            )
            full, bc = jax.lax.cond(
                recovery.converged & recovery.admissible,
                boundary,
                lambda _: (seed, skipped),
                operand=None,
            )
            info = NewtonInfo(
                recovery.iterations + bc.iterations,
                jnp.maximum(recovery.relative_update, bc.relative_update),
                jnp.maximum(recovery.residual_norm, bc.residual_norm),
                recovery.converged & bc.converged,
                recovery.admissible & bc.admissible,
                recovery.rejected_trials + bc.rejected_trials,
                jnp.where(~recovery.converged, recovery.status, bc.status),
            )
            return full, info

        def rates(full):
            return (
                (spatial(full), balance(full))
                if evaluate_stage is None
                else evaluate_stage(full)
            )

        def body(_):
            q0 = storage(initial[1:-1])
            residual0, rate0 = rates(initial)
            q1 = q0 - dt * residual0
            first, info1 = advance(q1, initial, start + dt)
            if method == "euler-explicit":
                return first, info1, dt * rate0

            def remaining(_):
                residual1, rate1 = rates(first)
                q2 = 0.75 * q0 + 0.25 * (storage(first[1:-1]) - dt * residual1)
                second, info2 = advance(q2, first, start + 0.5 * dt)

                def last(_):
                    residual2, rate2 = rates(second)
                    q3 = q0 / 3.0 + (2.0 / 3.0) * (
                        storage(second[1:-1]) - dt * residual2
                    )
                    final, info3 = advance(q3, second, start + dt)
                    info = info3._replace(
                        iterations=info1.iterations
                        + info2.iterations
                        + info3.iterations,
                        rejected_trials=info1.rejected_trials
                        + info2.rejected_trials
                        + info3.rejected_trials,
                        residual_norm=jnp.maximum(
                            info1.residual_norm,
                            jnp.maximum(info2.residual_norm, info3.residual_norm),
                        ),
                    )
                    return (
                        final,
                        info,
                        dt * (rate0 / 6.0 + rate1 / 6.0 + (2.0 / 3.0) * rate2),
                    )

                return jax.lax.cond(
                    info2.converged, last, lambda _: (second, info2, zero), operand=None
                )

            return jax.lax.cond(
                info1.converged, remaining, lambda _: (first, info1, zero), operand=None
            )

        new, info, increment = jax.lax.cond(
            initial_ok,
            body,
            lambda _: (old, initial_info._replace(converged=jnp.asarray(False)), zero),
            operand=None,
        )
        info = info._replace(
            iterations=info.iterations + initial_info.iterations,
            rejected_trials=info.rejected_trials + initial_info.rejected_trials,
        )
        return new, info, increment

    return jax.jit(step)
