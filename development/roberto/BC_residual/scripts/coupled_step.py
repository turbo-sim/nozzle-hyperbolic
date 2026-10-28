"""Source-style JIT Newton/colored/banded timestep with simultaneous BC rows."""

import jax
import jax.numpy as jnp
import jaxprop as jxp

from development.roberto.BC_residual.scripts.boundary_system import (
    boundary_constraints,
)
from development.roberto.BC_residual.scripts.source_adapter import (
    independent_ghost_operators,
    banded_newton_for_full_state,
    value,
)


class CoupledStep:
    """Use the source Newton machinery with six additional algebraic unknowns.

    Full-state cell ordering preserves the local width-two Jacobian stencil.
    Interior equations retain the exact source V*dQ + dt*F units. BC rows
    are scaled to primitive-variable units; they have no pseudo-time storage.
    """

    def __init__(self, base):
        self.base = base
        self.operators = independent_ghost_operators(base)
        self.constraints = boundary_constraints(base, self.operators)
        count = len(base.x)
        primitive_scale = jnp.asarray([base.p0, base.a0, base.h0])
        raw = self.operators["transient"]

        def residual(new_flat, old_flat, dt):
            full = new_flat.reshape(count, 3)
            old = old_flat.reshape(count, 3)
            # The source residual borrows its ghost slots from its old-state
            # argument before applying BCs. For independent ghosts supply the
            # current ghost unknowns there, preserving old INTERIOR storage.
            mixed = jnp.concatenate([full[:1], old[1:-1], full[-1:]])
            interior = raw(full[1:-1].ravel(), mixed.ravel(), dt).reshape(count - 2, 3)
            bc = self.constraints(full) * primitive_scale
            return jnp.concatenate([bc[:1], interior, bc[1:]]).ravel()

        self.residual = jax.jit(residual)
        direction, blocks, solve = banded_newton_for_full_state(base, residual)
        self.blocks = jax.jit(blocks)
        self.linear_solve = jax.jit(solve)
        self.dense_jacobian = jax.jit(jax.jacfwd(residual, argnums=0))
        scale = value(base.step, "_newton_scale")
        maximum = value(base.step, "newton_iter")
        tolerance = value(base.step, "newton_tol")

        @jax.jit
        def metrics(full):
            physical = self.operators["spatial"](full).reshape(count - 2, 3)
            reference = jnp.asarray(base.residual_scale).reshape(count - 2, 3)
            spatial_rms = jnp.sqrt(jnp.mean((physical / reference) ** 2))
            boundary_rms = jnp.sqrt(jnp.mean(self.constraints(full) ** 2))
            props = base.fluid.get_state(jxp.HmassP_INPUTS, full[:, 2], full[:, 0])
            t_char = base.dx / (jnp.max(jnp.abs(full[:, 1]) + props["a"]) + 1e-300)
            conservative = base.operators["conservative_state"](full)
            volume = base.operators["cell_volumes"][1:-1, None]
            ssr = jnp.sqrt(
                jnp.mean(
                    (
                        t_char
                        * physical
                        / (volume * jnp.abs(conservative[1:-1]) + 1e-300)
                    )
                    ** 2
                )
            )
            return spatial_rms, boundary_rms, ssr

        self.metrics = metrics

        @jax.jit
        def step(old, dt):
            old_flat = old.ravel()
            if tolerance > 0:

                def condition(carry):
                    return (carry[0] < maximum) & (carry[2] > tolerance)

                def body(carry):
                    correction = direction(carry[1], old_flat, dt)
                    relative = jnp.max(jnp.abs(correction) / (scale(carry[1]) + 1e-300))
                    return carry[0] + 1, carry[1] + correction, relative

                iterations, result, relative = jax.lax.while_loop(
                    condition, body, (0, old_flat, jnp.float64(jnp.inf))
                )
            else:
                result = jax.lax.fori_loop(
                    0,
                    maximum,
                    lambda _, current: current + direction(current, old_flat, dt),
                    old_flat,
                )
                iterations, relative = maximum, jnp.float64(jnp.nan)
            full = result.reshape(count, 3)
            # Match the source step return contract. Extra audits are outside timing.
            return full, metrics(full)[2]

        self.step = step
