"""Exact scalar reference profiles, independent of the finite-volume machinery.

These JAX-compatible functions may also specify initial and inflow data. They
return point values, not conservative cell averages. A finite-domain comparison
must supply compatible inflow data or stop before a Riemann wave reaches an end.
"""

import jax.numpy as jnp


def translated_trigonometric(coordinate, time=0.0, *, speed=1.0, length=1.0):
    """Translate the package's smooth advection fixture without distortion.

    For ``q_t + speed*q_x = 0``, characteristics give ``q(x,t)=q_0(x-speed*t)``.
    Here ``q_0(x)=sin(2*pi*x/length)+cos(4*pi*x/length)/4``; this particular
    initial profile is a verification choice made in this repository, not a
    literature formula. ``length`` must be positive and all inputs finite.

    The characteristic solution is described by Ketcheson, LeVeque and del Razo,
    *Riemann Problems and Jupyter Solutions*, Advection chapter:
    https://www.clawpack.org/riemann_book/html/Advection.html

    A spatially repeating analytic profile does not impose periodic numerical
    boundary conditions: the production model prescribes its incoming trace.
    """
    phase = 2.0 * jnp.pi * (jnp.asarray(coordinate) - speed * time) / length
    return jnp.sin(phase) + 0.25 * jnp.cos(2.0 * phase)


def burgers_riemann(coordinate, time=0.0, *, left=1.0, right=-0.5, location=0.5):
    """Sample the entropy solution of ``q_t + (q**2/2)_x = 0``.

    The initial discontinuity is at ``location``. When ``left > right``, the
    Rankine--Hugoniot speed is ``(left+right)/2``. Otherwise the centered
    rarefaction is ``clip((x-location)/t, left, right)`` for positive time.
    At time zero and exactly on a discontinuity, the right state is returned.
    Inputs must be finite and ``time >= 0``. Safe inactive arithmetic permits
    JIT compilation and automatic differentiation away from wave fronts.

    These standard relations, including the entropy condition, are given in
    Ketcheson, LeVeque and del Razo, *Riemann Problems and Jupyter Solutions*,
    Burgers chapter: https://www.clawpack.org/riemann_book/html/Burgers.html
    The sampling convention at a discontinuity is a repository choice.
    """
    x = jnp.asarray(coordinate) - location
    shock = jnp.where(x < 0.5 * (left + right) * time, left, right)
    safe_time = jnp.where(jnp.asarray(time) > 0, time, 1.0)
    fan = jnp.clip(x / safe_time, left, right)
    initial = jnp.where(x < 0, left, right)
    rarefaction = jnp.where(time > 0, fan, initial)
    return jnp.where(jnp.asarray(left) > right, shock, rarefaction)
