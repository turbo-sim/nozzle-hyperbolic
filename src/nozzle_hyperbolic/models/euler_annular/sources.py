"""Integrated inviscid annular geometry sources; no empirical wall closures."""

import jax.numpy as jnp

from .state import state_property, thermodynamic_state


def integrated_source(W, r, r_prime, A_left, A_right, cell_volume, fluid):
    """Balance face pressure areas exactly at rest and midpoint swirl forcing.

    Angular momentum and total energy have no geometric source. The remaining
    centrifugal term uses the same lumped cell state/radius as storage.
    """
    W, r = jnp.asarray(W), jnp.asarray(r)
    rho = state_property(thermodynamic_state(W, fluid), "rho")
    momentum = (W[..., 0]*(jnp.asarray(A_right)-jnp.asarray(A_left))
                +jnp.asarray(cell_volume)*rho*(W[..., 2]/r)**2*jnp.asarray(r_prime)/r)
    zero = jnp.zeros_like(momentum)
    return jnp.stack((zero, momentum, zero, zero), axis=-1)
