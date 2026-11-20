"""Model-independent nonlinear completion of zero-storage endpoint traces."""

import jax
import jax.numpy as jnp

from .time_marching.implicit import make_newton_solver


def make_boundary_solver(
    boundary_residual,
    shape,
    variable_scales,
    *,
    admissible=None,
    max_iterations=20,
    update_tolerance=1e-10,
    residual_tolerance=1e-12,
    max_backtracks=20
):
    """Return ``solve(full_state, time, *args) -> (completed_state, NewtonInfo)``.

    Physical cells remain fixed. The same boundary_residual callback can be
    appended to a global implicit system. Explicit RK stages call this solve
    separately for each stage; endpoint traces are never time-integrated.
    No physics, EOS, boundary regime or variable sign is assumed here.
    Optional trailing arguments, such as model run-time parameters, are passed
    unchanged to ``boundary_residual(full, time, *args)``.
    """
    npoints, nvar = shape
    if npoints < 4 or nvar < 1:
        raise ValueError("Expected at least two cells and two boundary traces")
    scale = jnp.broadcast_to(jnp.asarray(variable_scales), (2, nvar)).ravel()

    def insert(flat, full):
        trace = flat.reshape(2, nvar)
        return full.at[0].set(trace[0]).at[-1].set(trace[1])

    def residual(flat, full, time, *args):
        return boundary_residual(insert(flat, full), time, *args).ravel()

    jacobian = jax.jacfwd(residual, argnums=0)

    def direction(flat, full, time, *args):
        return jnp.linalg.solve(
            jacobian(flat, full, time, *args), -residual(flat, full, time, *args)
        )

    def valid(flat, full):
        return (
            jnp.all(jnp.isfinite(flat))
            if admissible is None
            else admissible(insert(flat, full))
        )

    # The common Newton callback receives only the candidate. Bind the fixed
    # physical field lexically inside solve; tracing retains a pure JAX graph.
    def solve(full, time=0.0, *args):
        newton = make_newton_solver(
            residual,
            direction,
            lambda _: scale,
            max_iterations=max_iterations,
            tolerance=update_tolerance,
            residual_tolerance=residual_tolerance,
            admissible=lambda flat: valid(flat, full),
            backtracking=True,
            max_backtracks=max_backtracks,
        )
        result, info = newton(
            jnp.stack((full[0], full[-1])).ravel(), full, time, *args
        )
        return insert(result, full), info

    return jax.jit(solve)


def make_boundary_eliminator(boundary_residual, complete_boundaries):
    """Return differentiable nonlinear elimination of zero-storage traces.

    ``eliminate(physical, seed_full, time, *args)`` returns a full state whose
    physical cells equal ``physical``. Endpoint roots are selected by the
    supplied seed and solved with ``complete_boundaries``. A failed completion
    returns NaNs, allowing the enclosing safeguarded Newton solve to reject the
    trial. Optional trailing arguments are model run-time parameters.

    The custom JVP differentiates the converged boundary equations:
    ``dtrace = -B_trace^-1 * (B_cells*dphysical + B_time*dtime + B_args*dargs)``.
    It does not differentiate the Newton iteration history. On a fixed smooth
    root branch, the result is independent of the seed. This requires a locally
    nonsingular boundary Jacobian with respect to the two endpoint traces.
    """

    def insert(physical, seed):
        return seed.at[1:-1].set(physical)

    @jax.custom_jvp
    def eliminate(physical, seed_full, time, *args):
        full, info = complete_boundaries(insert(physical, seed_full), time, *args)
        okay = info.converged & info.admissible & jnp.all(jnp.isfinite(full))
        return jnp.where(okay, full, jnp.full_like(full, jnp.nan))

    @eliminate.defjvp
    def implicit_jvp(primals, tangents):
        physical, seed_full, time, *args = primals
        dphysical, _, dtime, *dargs = tangents
        full = eliminate(physical, seed_full, time, *args)
        nvar = full.shape[-1]

        def linearized(_):
            trace = jnp.stack((full[0], full[-1])).ravel()

            def trace_rows(flat):
                endpoints = flat.reshape(2, nvar)
                candidate = full.at[0].set(endpoints[0]).at[-1].set(endpoints[1])
                return boundary_residual(candidate, time, *args).ravel()

            boundary_matrix = jax.jacfwd(trace_rows)(trace)
            _, forcing = jax.jvp(
                lambda cells, instant, *values: boundary_residual(
                    insert(cells, full), instant, *values
                ).ravel(),
                (physical, time, *args),
                (dphysical, dtime, *dargs),
            )
            dtrace = jnp.linalg.solve(boundary_matrix, -forcing).reshape(2, nvar)
            return jnp.concatenate((dtrace[:1], dphysical, dtrace[1:]), axis=0)

        tangent = jax.lax.cond(
            jnp.all(jnp.isfinite(full)),
            linearized,
            lambda _: jnp.full_like(full, jnp.nan),
            operand=None,
        )
        return full, tangent

    return eliminate
