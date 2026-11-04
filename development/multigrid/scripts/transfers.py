"""Generic conservative, limited-linear initialization on nested 1D grids.

These transfers use transported densities and geometry only. Storage recovery,
physical admissibility, trace completion and damping decisions belong to the
caller. There are no Euler or EOS-specific identities here.
"""

import jax
import jax.numpy as jnp


def _minmod(left, right):
    return jnp.where(
        jnp.sign(left) == jnp.sign(right),
        jnp.sign(left) * jnp.minimum(jnp.abs(left), jnp.abs(right)),
        jnp.zeros_like(left),
    )


@jax.jit
def prolong_initial_density(
    coarse_density, coarse_centers, fine_centers, fine_volumes, theta=1.0
):
    """Give two children limited linear density with an exact parent mean.

    Parameters
    ----------
    coarse_density : array, shape (N, T)
        Parent transported densities, with arbitrary transported-field count T.
    coarse_centers : array, shape (N,)
        Strictly increasing physical parent centers, with N at least two.
    fine_centers, fine_volumes : arrays, shape (2*N,)
        Ordered child centers and positive physical volumes. Each consecutive
        pair belongs to one parent; its volumes must sum to the parent volume.
    theta : scalar
        Explicit slope multiplier. Zero is piecewise-constant prolongation.

    Notes
    -----
    Interior slopes are componentwise minmod of the two physical secants;
    endpoints use their one-sided secant. The child offsets are recentered by
    their volume-weighted mean. Thus the two integrated child contents sum to
    parent volume times parent density even with unequal child volumes and
    parent centers that are not physical-volume centroids. An arbitrary theta
    preserves this identity. Linear profiles are reproduced relative to each
    parent's volume mean; pointwise exactness also requires centroid-consistent
    parent centers. A local density limiter alone does not establish EOS or
    algebraic admissibility, which the caller must check through recovery.
    """
    density = jnp.asarray(coarse_density)
    xc = jnp.asarray(coarse_centers)
    xf = jnp.asarray(fine_centers)
    volumes = jnp.asarray(fine_volumes)
    if density.ndim != 2 or density.shape[0] < 2:
        raise ValueError("Use at least two parent cells and a 2D density array")
    nc, nt = density.shape
    if xc.shape != (nc,) or xf.shape != (2 * nc,) or volumes.shape != (2 * nc,):
        raise ValueError("Coordinates and volumes must match a 2:1 hierarchy")
    if jnp.ndim(theta) != 0:
        raise ValueError("The slope damping factor must be scalar")
    dtype = jnp.result_type(density, xc, xf, volumes, 1.0)
    density, xc, xf, volumes = (
        jnp.asarray(value, dtype=dtype) for value in (density, xc, xf, volumes)
    )
    secant = jnp.diff(density, axis=0) / jnp.diff(xc)[:, None]
    slopes = jnp.concatenate(
        (secant[:1], _minmod(secant[:-1], secant[1:]), secant[-1:]), axis=0
    )
    weights = volumes.reshape(nc, 2)
    offsets = xf.reshape(nc, 2) - xc[:, None]
    weighted_mean = jnp.sum(weights * offsets, axis=1) / jnp.sum(weights, axis=1)
    centered = offsets - weighted_mean[:, None]
    children = (
        density[:, None, :]
        + jnp.asarray(theta, dtype=dtype) * centered[:, :, None] * slopes[:, None, :]
    )
    return children.reshape(2 * nc, nt)
