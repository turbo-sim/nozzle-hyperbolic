"""Four-variable general-EOS characteristic flux on one shared face geometry."""

import jax.numpy as jnp

from ...numerics.characteristic import stadtke_flux_with_entropy_viscosity
from .equations import characteristic_speeds, transverse_flux
from .state import state_property, thermodynamic_state, transverse_conservative


def conservative_projectors(W, r, rho, a, G):
    """Conservative transverse-Euler projectors from C P C^-1.

    The primitive eigenbasis is (p, v_m, v_theta, h). Transforming with the
    storage Jacobian, rather than the flux Jacobian, stays regular at rest and
    sonic states. The two convective projectors represent entropy and swirl.
    """
    W, r = jnp.asarray(W), jnp.asarray(r)
    v_m, v_theta, h = W[..., 1], W[..., 2]/r, W[..., 3]
    rho, a, G = (jnp.broadcast_to(jnp.asarray(value), v_m.shape)
                 for value in (rho, a, G))
    zero, one = jnp.zeros_like(v_m), jnp.ones_like(v_m)
    rho_p, rho_h = (1+G)/a**2, -rho*G/a**2
    h0 = h+0.5*(v_m**2+v_theta**2)
    C = jnp.stack((
        jnp.stack((rho_p, zero, zero, rho_h), axis=-1),
        jnp.stack((v_m*rho_p, rho, zero, v_m*rho_h), axis=-1),
        jnp.stack((v_theta*rho_p, zero, rho, v_theta*rho_h), axis=-1),
        jnp.stack((h0*rho_p-one, rho*v_m, rho*v_theta, h0*rho_h+rho), axis=-1),
    ), axis=-2)
    chi = 0.5*(v_m**2+v_theta**2)-h
    C_inverse = jnp.stack((
        jnp.stack((a**2+G*chi, -G*v_m, -G*v_theta, G), axis=-1),
        jnp.stack((-v_m/rho, one/rho, zero, zero), axis=-1),
        jnp.stack((-v_theta/rho, zero, one/rho, zero), axis=-1),
        jnp.stack(((a**2+(one+G)*chi)/rho, -(one+G)*v_m/rho,
                   -(one+G)*v_theta/rho, (one+G)/rho), axis=-1),
    ), axis=-2)
    right = jnp.stack((
        jnp.stack((rho*a, one, zero, a), axis=-1),
        jnp.stack((-rho*a, one, zero, -a), axis=-1),
        jnp.stack((zero, zero, zero, one), axis=-1),
        jnp.stack((zero, zero, one, zero), axis=-1),
    ), axis=-2)
    left = jnp.stack((
        jnp.stack((0.5/(rho*a), 0.5*one, zero, zero), axis=-1),
        jnp.stack((-0.5/(rho*a), 0.5*one, zero, zero), axis=-1),
        jnp.stack((-one/rho, zero, zero, one), axis=-1),
        jnp.stack((zero, zero, one, zero), axis=-1),
    ), axis=-2)
    conservative_right = jnp.einsum("...ij,...kj->...ki", C, right)
    conservative_left = jnp.einsum("...ki,...ij->...kj", left, C_inverse)
    return conservative_right[..., :, None]*conservative_left[..., None, :]


def numerical_flux(W_L, W_R, r_face, A_face, fluid, delta_factor=0.1,
                   method="stadtke"):
    """Integrated angular flux, with the same radius and area on both sides.

    ``method`` is a factory-static choice. Stadtke reuses the original shared
    regularized contraction; Rusanov is a diagnostic. This retained low-level
    baseline does not implement the shared Scheme's custom-correction flags.
    A strictly positive entropy width is required for characteristic fluxes.
    """
    if method not in ("stadtke", "rusanov"):
        raise ValueError(f"Unknown annular flux method: {method}")
    if delta_factor <= 0:
        raise ValueError("delta_factor must be strictly positive")
    W_L, W_R = jnp.asarray(W_L), jnp.asarray(W_R)
    average = 0.5*(W_L+W_R)
    state_L, state_R = thermodynamic_state(W_L, fluid), thermodynamic_state(W_R, fluid)
    a_L, a_R = state_property(state_L, "a"), state_property(state_R, "a")
    rho_L, rho_R = state_property(state_L, "rho"), state_property(state_R, "rho")
    flux_L = transverse_flux(W_L, r_face, fluid, rho=rho_L)
    flux_R = transverse_flux(W_R, r_face, fluid, rho=rho_R)
    q_L = transverse_conservative(W_L, r_face, fluid, rho=rho_L)
    q_R = transverse_conservative(W_R, r_face, fluid, rho=rho_R)
    if method == "rusanov":
        speed = jnp.maximum(jnp.abs(W_L[..., 1])+a_L, jnp.abs(W_R[..., 1])+a_R)
        flux = 0.5*(flux_L+flux_R-speed[..., None]*(q_R-q_L))
    else:
        state_average = thermodynamic_state(average, fluid)
        rho, a, G = (state_property(state_average, name) for name in ("rho", "a", "G"))
        projectors = conservative_projectors(average, r_face, rho, a, G)
        eigenvalues = characteristic_speeds(average, a)
        spread = characteristic_speeds(W_R, a_R)-characteristic_speeds(W_L, a_L)
        delta0 = delta_factor*a[..., None]
        positive_spread = 0.5*(spread+jnp.sqrt(spread**2+delta0**2))
        width = delta0+0.5*positive_spread
        flux = stadtke_flux_with_entropy_viscosity(
            flux_L, flux_R, q_R-q_L, projectors, eigenvalues, width)
    flux = flux.at[..., 2].set(flux[..., 2]*jnp.asarray(r_face))
    return flux*jnp.asarray(A_face)[..., None]
