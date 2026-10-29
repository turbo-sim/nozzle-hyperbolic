"""Explicit, general-EOS initial guesses; independent boundary unknowns remain free."""

import jax.numpy as jnp


def uniform_initial_state(r, p, h, v_m, v_theta=0.0, alpha_in=None):
    """Build (p,v_m,ell,h) guesses at supplied radii, including boundary traces.

    Pressure and enthalpy are physical setup data, never ideal-gas estimates.
    If an inlet angle is supplied, use that angle in the initial velocity field.
    This prepares a guess; it does not project a trial state onto its BC rows.
    """
    r = jnp.asarray(r)
    p, h, v_m = (jnp.broadcast_to(jnp.asarray(value), r.shape) for value in (p, h, v_m))
    v_theta = jnp.broadcast_to(jnp.asarray(v_theta), r.shape)
    if alpha_in is not None:
        v_theta = v_m*jnp.tan(alpha_in)
    return jnp.stack((p, v_m, r*v_theta, h), axis=-1)
