"""Dimension-independent conservative numerical-flux algebra."""

import jax.numpy as jnp


def rusanov(left_flux, right_flux, jump, left_bounds, right_bounds):
    """Local Lax--Friedrichs flux from model-provided signal bounds."""
    lo_l, hi_l = left_bounds
    lo_r, hi_r = right_bounds
    speed = jnp.maximum(
        jnp.maximum(jnp.abs(lo_l), jnp.abs(hi_l)),
        jnp.maximum(jnp.abs(lo_r), jnp.abs(hi_r)),
    )
    return 0.5 * (left_flux + right_flux - speed[..., None] * jump)


def hll(left_flux, right_flux, jump, left_bounds, right_bounds):
    """HLL's two-wave conservative flux, including a safe zero-speed limit."""
    low = jnp.minimum(left_bounds[0], right_bounds[0])
    high = jnp.maximum(left_bounds[1], right_bounds[1])
    denominator = jnp.where(high > low, high - low, 1.0)
    middle = (
        high[..., None] * left_flux
        - low[..., None] * right_flux
        + (low * high)[..., None] * jump
    ) / denominator[..., None]
    return jnp.where(
        (low >= 0)[..., None],
        left_flux,
        jnp.where((high <= 0)[..., None], right_flux, middle),
    )
