"""Dimension-independent characteristic flux regularization.

The caller owns the physical fluxes, conservative variables, projectors and
wave speeds. These kernels contain no thermodynamic or geometry assumptions.
"""

import jax.numpy as jnp


def harten_absolute(eigenvalues, width):
    """Return a positive C1 regularized absolute value for positive widths.

    The quadratic inner branch matches abs(lambda) and its first derivative
    at +/-width. Its second derivative jumps there: this is C1, not C2.
    Every traced branch is finite, including lambda=0, when width>0.
    """
    absolute = jnp.abs(eigenvalues)
    inner = (eigenvalues**2 + width**2) / (2.0 * width)
    return jnp.where(absolute < width, inner, absolute)


def characteristic_flux_with_entropy_viscosity(
    flux_left, flux_right, conservative_jump, projectors, eigenvalues, width
):
    """Combine characteristic FVS with a conservative sonic viscosity term.

    phi is the regularized absolute eigenvalue and s=lambda/phi. The first
    characteristic term is a smooth sum-to-one weighting of physical fluxes.
    The additional coefficient phi-s*lambda is nonnegative. For a frozen
    linear system, the complete viscosity is exactly phi rather than the
    smaller lambda**2/phi supplied by smoothing the flux weights alone.

    Projectors have shape (..., waves, balances, conserved variables). The
    number of waves and variables is supplied by the physical model.
    """
    absolute = harten_absolute(eigenvalues, width)
    sign = eigenvalues / absolute
    extra_viscosity = absolute - sign * eigenvalues
    projected_flux_jump = jnp.einsum(
        "...kij,...j->...ki", projectors, flux_right - flux_left
    )
    projected_state_jump = jnp.einsum(
        "...kij,...j->...ki", projectors, conservative_jump
    )
    dissipation = jnp.sum(
        sign[..., None] * projected_flux_jump
        + extra_viscosity[..., None] * projected_state_jump,
        axis=-2,
    )
    return 0.5 * (flux_left + flux_right) - 0.5 * dissipation
