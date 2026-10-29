"""Wall-law placeholders for annular flow, with prescribed zero-default data.

The state is W=(p,v_m,ell,h), r is the local radius, and fluid is the supplied
EOS object. Keeping those arguments explicit allows later state-dependent
constitutive laws without changing the source assembly. No empirical friction
or heat-transfer correlation is assumed here. The factory validates prescribed
profiles outside JAX tracing; these small kernels support JIT and AD.
"""

import jax.numpy as jnp


def shear_stress_wall(W, r, fluid, *, prescribed=0.0):
    """Return nonnegative prescribed shear magnitude in Pa; default is zero.

    A scalar or cell-shaped profile is broadcast to W's leading dimensions.
    The direction opposing the local velocity is applied by the source kernel.
    State, radius and EOS arguments are available for future wall-law closures.
    """
    W = jnp.asarray(W)
    return jnp.broadcast_to(jnp.asarray(prescribed, dtype=W.dtype), W.shape[:-1])


def heat_flux_wall(W, r, fluid, *, prescribed=0.0):
    """Return prescribed heat flux in W/m^2, positive into the fluid; default zero.

    A scalar or cell-shaped profile is broadcast to W's leading dimensions.
    This placeholder does not infer a wall temperature or transfer coefficient.
    """
    W = jnp.asarray(W)
    return jnp.broadcast_to(jnp.asarray(prescribed, dtype=W.dtype), W.shape[:-1])
