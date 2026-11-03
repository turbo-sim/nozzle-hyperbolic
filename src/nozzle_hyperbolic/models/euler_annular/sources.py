"""Integrated geometry, wall-torque and heating sources for annular Euler."""

import jax.numpy as jnp
from .closures.wall import heat_flux_wall, shear_stress_wall
from .state import state_property, thermodynamic_state


def integrated_source(
    W,
    r,
    r_prime,
    area_derivative_integral,
    cell_volume,
    fluid,
    *,
    height=1.0,
    tau_w=0.0,
    q_w=0.0,
    wall_closure=None
):
    """Add geometric pressure/swirl loading, wall torque and signed heat input.

    ``area_derivative_integral`` comes from the smooth fixed physical geometry.
    No derivative is inferred from neighboring cell samples. Shear removes
    momentum and supplies a radius-weighted angular torque. A stationary
    adiabatic wall has no total-energy sink; heat alone changes total energy.
    A directional closure supplies ``directional_stress(W,r,fluid)`` returning
    signed ``(tau_m,tau_theta,q)`` in Pa, Pa and W/m^2; positive stresses are
    subtracted from the respective momentum balances. Otherwise the optional
    closure evaluates physical ``(W,r,fluid)->(tau,q)`` and can
    expose a smooth ``drag_coefficient`` to resolve the laminar zero-speed
    limit. Prescribed stress defines zero directional drag exactly at rest.
    """
    W, r = jnp.asarray(W), jnp.asarray(r)
    rho = state_property(thermodynamic_state(W, fluid), "rho")
    volume = jnp.asarray(cell_volume)
    vm, vtheta = W[..., 1], W[..., 2] / r
    momentum = (
        W[..., 0] * jnp.asarray(area_derivative_integral)
        + volume * rho * vtheta**2 * jnp.asarray(r_prime) / r
    )
    speed_squared = vm**2 + vtheta**2
    safe_speed = jnp.sqrt(jnp.where(speed_squared > 0.0, speed_squared, 1.0))
    wall_measure = 2 * volume / jnp.asarray(height)
    directional = getattr(wall_closure, "directional_stress", None)
    if callable(directional):
        tau_m, tau_theta, heat = directional(W, r, fluid)
    else:
        if wall_closure is None:
            shear = shear_stress_wall(W, r, fluid, prescribed=tau_w)
            heat = heat_flux_wall(W, r, fluid, prescribed=q_w)
        else:
            shear, heat = wall_closure(W, r, fluid)
        coefficient = getattr(wall_closure, "drag_coefficient", None)
        drag = coefficient(W, r, fluid) if callable(coefficient) else shear / safe_speed
        tau_m, tau_theta = drag * vm, drag * vtheta
    momentum = momentum - wall_measure * tau_m
    torque = -r * wall_measure * tau_theta
    heating = wall_measure * heat
    return jnp.stack((jnp.zeros_like(momentum), momentum, torque, heating), axis=-1)
