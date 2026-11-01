"""Conservative Roe flux contraction from model-owned characteristic data.

No averaging rule, EOS, equation identity or wave ordering is assumed here.
Models supply compatible conservative spectral projectors and entropy widths.
"""

import jax.numpy as jnp


def roe_absolute(eigenvalues, width):
    """Harten absolute value, with an exact safe zero-width ``abs`` branch.

    Widths must be nonnegative. A zero width leaves that field unregularized;
    it must not evaluate a zero denominator in an inactive JAX/AD branch.
    The unregularized absolute value retains its mathematical kink at zero.
    """
    absolute = jnp.abs(eigenvalues)
    positive = width > 0
    denominator = jnp.where(positive, width, jnp.ones_like(width))
    inner = (eigenvalues**2 + width**2) / (2.0 * denominator)
    return jnp.where(positive & (absolute < width), inner, absolute)


def roe(flux_left, flux_right, conservative_jump, projectors, eigenvalues, width):
    """Return the Roe flux in the model's unweighted conservative face frame.

    Projectors have shape (..., waves, variables, variables); the wave count
    and variable count are independent. Repeated speeds may use a grouped
    higher-rank projector. Callers handle validity and apply geometry once.
    """
    jumps = jnp.einsum("...kij,...j->...ki", projectors, conservative_jump)
    dissipation = jnp.sum(roe_absolute(eigenvalues, width)[..., None] * jumps, axis=-2)
    return 0.5 * (flux_left + flux_right - dissipation)
