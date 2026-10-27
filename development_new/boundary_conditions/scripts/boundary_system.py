"""Current source elimination and experimental simultaneous boundary equations."""

import jax
import jax.numpy as jnp
import jaxprop as jxp


def boundary_constraints(base, operators):
    """Dimensionless inlet/outlet algebraic equations; no nested inlet solve."""
    scale = jnp.asarray([base.p0, base.a0, base.h0])

    @jax.jit
    def evaluate(full):
        inlet = base.fluid.get_state(jxp.HmassP_INPUTS, full[:1, 2], full[:1, 0])
        first = jnp.array(
            [
                (inlet["s"][0] - base.s0) / (base.h0 / base.T0),
                (full[0, 1] - (2 * full[1, 1] - full[2, 1])) / base.a0,
                (full[0, 2] + 0.5 * full[0, 1] ** 2 - base.h0) / base.h0,
            ]
        )
        return jnp.stack([first, (full[-1] - operators["outlet"](full)[-1]) / scale])

    return evaluate
