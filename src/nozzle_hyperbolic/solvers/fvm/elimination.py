"""Reduced backward-Euler solves with fully converged nonlinear boundaries.

This module owns numerical elimination only. The same model boundary callback
defines both the eliminated traces and the full coupled residual audit.
"""

import jax
import jax.numpy as jnp

from ..boundaries import make_boundary_eliminator
from ..time_marching.implicit import NewtonInfo, make_newton_solver
from ...numerics.jacobians import make_block_operators


def make_eliminated_implicit_step(
    *,
    boundary_residual,
    complete_boundaries,
    physical_residual,
    coupled_residual,
    admissible,
    shape,
    variable_scales,
    radius,
    scheme,
):
    """Build an N*V Newton system and its full-state backward-Euler step.

    Reduced callbacks take ``(cells_flat, old_full_flat, dt, time)``. Every
    residual/trial solves the nonlinear boundary subproblem at the current
    cells. Its JVP uses the implicit-function derivative of that root. Local
    source/BC stencil declarations bound the reduced band; models with global
    boundary coupling already require the dense backend at operator binding.

    NewtonInfo.iterations and rejected_trials count the outer reduced Newton
    work, excluding nested boundary solves. Its final residual_norm audits the
    original full backward-Euler and boundary rows; success requires both the
    reduced solve and final boundary completion to converge.
    """
    npoints, nvar = shape
    ncells = npoints - 2
    eliminate = make_boundary_eliminator(boundary_residual, complete_boundaries)
    scale = jnp.broadcast_to(jnp.asarray(variable_scales), (ncells, nvar)).ravel()

    def full_state(flat, old_flat, time):
        return eliminate(flat.reshape(ncells, nvar), old_flat.reshape(shape), time)

    def residual(flat, old_flat, dt, time):
        full = full_state(flat, old_flat, time)
        return jax.lax.cond(
            jnp.all(jnp.isfinite(full)),
            lambda: physical_residual(full, old_flat.reshape(shape), dt).ravel(),
            lambda: jnp.full_like(flat, jnp.nan),
        )

    blocks = make_block_operators(
        residual,
        ncells,
        nvar,
        min(radius, ncells - 1),
        jacobian_mode=scheme.jacobian_mode,
        linear_solver=scheme.linear_solver,
    )

    def step(old, dt, time=0.0):
        old_flat = old.ravel()
        solve = make_newton_solver(
            residual,
            blocks["direction"],
            lambda _: scale,
            max_iterations=scheme.newton_iterations,
            tolerance=scheme.newton_update_tolerance,
            residual_tolerance=scheme.newton_residual_tolerance,
            admissible=lambda flat: admissible(
                full_state(flat, old_flat, time).ravel()
            ),
            backtracking=True,
            max_backtracks=scheme.max_backtracks,
        )
        physical, outer = solve(old[1:-1].ravel(), old_flat, dt, time)
        full, boundary = complete_boundaries(
            old.at[1:-1].set(physical.reshape(ncells, nvar)), time
        )
        valid = admissible(full.ravel())
        norm = jax.lax.cond(
            valid,
            lambda: jnp.sqrt(
                jnp.mean(coupled_residual(full.ravel(), old_flat, dt, time) ** 2)
            ),
            lambda: jnp.asarray(jnp.inf, old.dtype),
        )
        root_ok = jnp.isfinite(norm)
        if scheme.newton_residual_tolerance > 0:
            root_ok = root_ok & (norm <= scheme.newton_residual_tolerance)
        converged = (
            outer.converged & boundary.converged & boundary.admissible & valid & root_ok
        )
        status = jnp.where(
            ~outer.converged,
            outer.status,
            jnp.where(
                ~boundary.converged,
                boundary.status,
                jnp.where(
                    ~valid,
                    2,
                    jnp.where(~jnp.isfinite(norm), 4, jnp.where(converged, 0, 1)),
                ),
            ),
        )
        return full, NewtonInfo(
            outer.iterations,
            outer.relative_update,
            norm,
            converged,
            valid,
            outer.rejected_trials,
            status,
        )

    return dict(step=step, residual=residual, **blocks)
