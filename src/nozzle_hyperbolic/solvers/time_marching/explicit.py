"""Conservative Euler, SSPRK3 and classical RK4 with algebraic stage constraints."""

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
    evaluate_stage=None,
    recover_initial=False,
):
    """Build a step using only model-independent numerical callbacks.

    recover(contents, physical_guess) and complete_boundaries(full,time) return
    a state and NewtonInfo. balance(full) returns the signed boundary/source
    rate used for an independent global budget. Stage combinations apply to
    integrated conserved contents, never to primitive iteration variables.
    Optional evaluate_stage(full) returns both spatial residual and balance in
    one evaluation so expensive model properties are reused by the assembler.
    Constrained models set recover_initial=True to recover the initial cell
    algebraic state from unchanged contents before any source or flux is read.
    """
    if method not in ("euler-explicit", "runge-kutta-3", "runge-kutta-4"):
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
        # The transported count need not equal the iteration-state count.
        zero = jnp.zeros((storage(old[1:-1]).shape[-1],), old.dtype)
        if recover_initial:
            cells, recovery = recover(storage(old[1:-1]), old[1:-1])
            recovery = normalize(recovery)
            provisional = old.at[1:-1].set(cells)

            def initial_boundary(_):
                full, info = complete_boundaries(provisional, start)
                return full, normalize(info)

            skipped = recovery._replace(
                iterations=jnp.int32(0), relative_update=jnp.asarray(0., old.dtype),
                residual_norm=jnp.asarray(0., old.dtype), converged=jnp.asarray(False),
                rejected_trials=jnp.int32(0),
            )
            initial, bc = jax.lax.cond(
                recovery.converged & recovery.admissible,
                initial_boundary, lambda _: (old, skipped), operand=None,
            )
            initial_info = NewtonInfo(
                recovery.iterations + bc.iterations,
                jnp.maximum(recovery.relative_update, bc.relative_update),
                jnp.maximum(recovery.residual_norm, bc.residual_norm),
                recovery.converged & bc.converged,
                recovery.admissible & bc.admissible,
                recovery.rejected_trials + bc.rejected_trials,
                jnp.where(~recovery.converged, recovery.status, bc.status),
            )
        else:
            initial, initial_info = complete_boundaries(old, start)
            initial_info = normalize(initial_info)
        initial_ok = (
            initial_info.converged & initial_info.admissible & admissible(initial)
        )

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
            if method == "runge-kutta-4":
                # Classical RK4 on contents. Every rate sees recovered cell
                # constraints and completed traces at its own stage time.
                first, info1 = advance(q0 - 0.5 * dt * residual0, initial, start + 0.5 * dt)

                def combine(previous, current):
                    return current._replace(
                        iterations=previous.iterations + current.iterations,
                        rejected_trials=previous.rejected_trials + current.rejected_trials,
                        residual_norm=jnp.maximum(previous.residual_norm, current.residual_norm),
                        relative_update=jnp.maximum(previous.relative_update, current.relative_update),
                        converged=previous.converged & current.converged,
                        admissible=previous.admissible & current.admissible,
                    )

                def stage_two(_):
                    residual1, rate1 = rates(first)
                    second, info2 = advance(q0 - 0.5 * dt * residual1, first, start + 0.5 * dt)
                    info2 = combine(info1, info2)

                    def stage_three(_):
                        residual2, rate2 = rates(second)
                        third, info3 = advance(q0 - dt * residual2, second, start + dt)
                        info3 = combine(info2, info3)

                        def stage_four(_):
                            residual3, rate3 = rates(third)
                            target = q0 - dt * (residual0 + 2 * residual1 + 2 * residual2 + residual3) / 6.0
                            final, info4 = advance(target, third, start + dt)
                            info4 = combine(info3, info4)
                            increment = dt * (rate0 + 2 * rate1 + 2 * rate2 + rate3) / 6.0
                            return final, info4, jnp.where(info4.converged, increment, zero)

                        return jax.lax.cond(
                            info3.converged, stage_four,
                            lambda _: (third, info3, zero), operand=None,
                        )

                    return jax.lax.cond(
                        info2.converged, stage_three,
                        lambda _: (second, info2, zero), operand=None,
                    )

                return jax.lax.cond(
                    info1.converged, stage_two,
                    lambda _: (first, info1, zero), operand=None,
                )

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
            # The failed-initial branch already returns initial_info; do not
            # count its work twice.
            iterations=info.iterations + jnp.where(initial_ok, initial_info.iterations, 0),
            rejected_trials=info.rejected_trials + jnp.where(initial_ok, initial_info.rejected_trials, 0),
        )
        return new, info, increment

    return jax.jit(step)
