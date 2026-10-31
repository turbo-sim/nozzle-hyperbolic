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
    """Return solve(full_state, time) -> (completed_state, NewtonInfo).

    Physical cells remain fixed. The same boundary_residual callback can be
    appended to a global implicit system. Explicit RK stages call this solve
    separately for each stage; endpoint traces are never time-integrated.
    No physics, EOS, boundary regime or variable sign is assumed here.
    """
    npoints, nvar = shape
    if npoints < 4 or nvar < 1:
        raise ValueError("Expected at least two cells and two boundary traces")
    scale = jnp.broadcast_to(jnp.asarray(variable_scales), (2, nvar)).ravel()

    def insert(flat, full):
        trace = flat.reshape(2, nvar)
        return full.at[0].set(trace[0]).at[-1].set(trace[1])

    def residual(flat, full, time):
        return boundary_residual(insert(flat, full), time).ravel()

    jacobian = jax.jacfwd(residual, argnums=0)

    def direction(flat, full, time):
        return jnp.linalg.solve(jacobian(flat, full, time), -residual(flat, full, time))

    def valid(flat, full):
        return (
            jnp.all(jnp.isfinite(flat))
            if admissible is None
            else admissible(insert(flat, full))
        )

    # The common Newton callback receives only the candidate. Bind the fixed
    # physical field lexically inside solve; tracing retains a pure JAX graph.
    def solve(full, time=0.0):
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
        result, info = newton(jnp.stack((full[0], full[-1])).ravel(), full, time)
        return insert(result, full), info

    return jax.jit(solve)
