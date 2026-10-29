"""Integrated geometry, wall-torque and heating sources for annular Euler."""

import jax.numpy as jnp

from .state import state_property, thermodynamic_state
from .closures import heat_flux_wall, shear_stress_wall


def integrated_source(W, r, r_prime, A_left, A_right, cell_volume, fluid,
                      *, height=1.0, tau_w=0.0, q_w=0.0):
    """Balance pressure areas and add cell-state swirl, wall and heat forcing.

    Geometry produces no angular-momentum or total-energy source. The angular
    wall source is torque (radius times tangential drag), not tangential force.
    Stationary-wall friction converts kinetic into internal energy without a
    total-energy sink; only heat flux enters the total-energy source. The normal
    height is irrelevant when both prescribed inputs are zero. At zero speed
    shear direction is defined as zero using finite, differentiable arithmetic.
    """
    W, r = jnp.asarray(W), jnp.asarray(r)
    rho = state_property(thermodynamic_state(W, fluid), "rho")
    momentum = (W[..., 0]*(jnp.asarray(A_right)-jnp.asarray(A_left))
                +jnp.asarray(cell_volume)*rho*(W[..., 2]/r)**2*jnp.asarray(r_prime)/r)
    v_m, v_theta = W[..., 1], W[..., 2]/r
    speed_squared = v_m**2+v_theta**2
    safe_speed = jnp.sqrt(jnp.where(speed_squared > 0, speed_squared, 1.0))
    wall_measure = 2*jnp.asarray(cell_volume)/jnp.asarray(height)
    shear = wall_measure*shear_stress_wall(W, r, fluid, prescribed=tau_w)
    momentum = momentum-shear*v_m/safe_speed
    torque = -r*shear*v_theta/safe_speed
    heating = wall_measure*heat_flux_wall(W, r, fluid, prescribed=q_w)
    return jnp.stack((jnp.zeros_like(momentum), momentum, torque, heating), axis=-1)
