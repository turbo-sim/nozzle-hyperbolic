"""Annular Euler physical fluxes, independent of the thermodynamic backend."""

import jax.numpy as jnp

from .state import state_property, thermodynamic_state


def transverse_flux(W, r, fluid, rho=None):
    """Unweighted normal flux of (rho, rho*v_m, rho*v_theta, rho*E)."""
    W, r = jnp.asarray(W), jnp.asarray(r)
    if rho is None:
        rho = state_property(thermodynamic_state(W, fluid), "rho")
    p, v_m, ell, h = (W[..., k] for k in range(4))
    rho = jnp.broadcast_to(jnp.asarray(rho), p.shape)
    v_theta = ell/r
    h0 = h+0.5*(v_m**2+v_theta**2)
    return jnp.stack((rho*v_m, rho*v_m**2+p, rho*v_m*v_theta,
                      rho*v_m*h0), axis=-1)


def physical_flux(W, r, fluid):
    """Unweighted angular-momentum flux (mass, momentum, torque, energy)."""
    flux = transverse_flux(W, r, fluid)
    return flux.at[..., 2].set(flux[..., 2]*jnp.asarray(r))


def characteristic_speeds(W, sound_speed):
    """Acoustic, entropy and transverse characteristic speeds, in wave order."""
    v_m = jnp.asarray(W)[..., 1]
    return jnp.stack((v_m+sound_speed, v_m-sound_speed, v_m, v_m), axis=-1)
