"""Coordinate-aware generic MUSCL reconstruction and retained NumPy helpers."""

import numpy as np
import jax.numpy as jnp


def reconstruct_faces(
    full,
    points,
    faces,
    method="muscl_van_leer",
    *,
    variable_scales=None,
    smooth_coefficient=1.0
):
    """Coordinate-based MUSCL on physical cells and independent endpoint traces.

    full/points have N+2 rows, faces has N+1 positions. End traces lie exactly at
    endpoints and have zero slope. Component reconstruction is model agnostic.
    Optional C2 limiter smoothing uses dimensionless gradients with a fixed
    reference length; its width vanishes quadratically with mean mesh spacing.
    """
    if method == "constant":
        return full[:-1], full[1:]
    gradients = (full[1:] - full[:-1]) / (points[1:] - points[:-1])[:, None]
    if method == "muscl_van_leer":
        interior = _van_leer_jax_safe(gradients[:-1], gradients[1:])
    elif method in _JAX_LIMITERS:
        interior = _JAX_LIMITERS[method](gradients[:-1], gradients[1:])
    elif method == "muscl_van_leer_smooth":
        length = faces[-1] - faces[0]
        gradient_scale = jnp.asarray(variable_scales) / length
        epsilon = smooth_coefficient / (faces.shape[0] - 1) ** 2
        interior = gradient_scale * _van_leer_jax_regularized(
            gradients[:-1] / gradient_scale, gradients[1:] / gradient_scale, epsilon
        )
    else:
        raise ValueError("Unsupported reconstruction")
    slopes = jnp.zeros_like(full).at[1:-1].set(interior)
    return (
        full[:-1] + slopes[:-1] * (faces - points[:-1])[:, None],
        full[1:] + slopes[1:] * (faces - points[1:])[:, None],
    )


# ---------------------------------------------------------------------------
# MUSCL reconstruction  (2nd-order spatial accuracy)
# ---------------------------------------------------------------------------


def _minmod(a, b):
    return np.where(
        np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0
    )


def _van_leer(a, b):
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / (a + b + 1e-300), 0.0)


def _van_leer_jax_safe(a, b):
    """Return the original JAX Van Leer slope with a safe inactive branch.

    The active formula is unchanged. For opposite differences, including
    ``a == -b``, replace the denominator before dividing so reverse AD never
    encounters an overflowing inactive quotient. This does not smooth the
    mathematical limiter at its axes or origin.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    active = a * b > 0.0
    denominator = jnp.where(active, a + b + 1e-300, jnp.ones_like(a + b))
    return jnp.where(active, 2.0 * a * b / denominator, 0.0)


def _van_leer_jax_regularized(a, b, epsilon):
    """Return a C2 regularization of Van Leer for dimensionless differences.

    ``epsilon`` must be finite and strictly positive. Normalize physical
    component differences with fixed reference scales before calling, then
    multiply the returned slope by the same scales. A mesh-dependent choice
    such as ``epsilon = coefficient * (dx / length)**2`` makes the smoothing
    vanish with refinement without introducing state-dependent nonlocality.

    With ``t = max(a*b, 0)`` and ``s = a+b``, this computes

        2*t**3*s / ((t**2 + epsilon**4)*(s**2 + epsilon**2)).

    For same-sign differences it is Van Leer multiplied by two factors in
    [0, 1]; opposite-sign and zero differences give exactly zero. Thus it
    retains componentwise slope/reconstruction bounds and preserves constants.
    Those properties alone do not establish TVD or update positivity for a
    coupled Euler solver. Linear slopes are damped near the regularization
    scale and are recovered as epsilon tends to zero, rather than preserved
    exactly at finite epsilon.

    The positive-part cubic makes the extension across each axis C2. Factored
    evaluation avoids division by a small ``a+b`` and any exponential branch.
    No custom derivative is needed for JAX forward or reverse AD.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    epsilon = jnp.asarray(epsilon)
    product = jnp.maximum(a * b, 0.0)
    total = a + b
    product_squared = product * product
    epsilon_squared = epsilon * epsilon
    product_weight = product_squared / (
        product_squared + epsilon_squared * epsilon_squared
    )
    return (2.0 * product * total / (total * total + epsilon_squared)) * product_weight


def _same_sign(a, b):
    """Strict same-sign test without a potentially overflowing product."""
    return ((a > 0) & (b > 0)) | ((a < 0) & (b < 0))


def minmod(a, b):
    """JAX minmod slope; zero at extrema, homogeneous and odd in the inputs."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    return jnp.where(_same_sign(a, b), jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def superbee(a, b):
    """JAX Superbee, selecting the larger magnitude for either slope sign.

    A signed maximum of the two minmod candidates is wrong for negative
    gradients. This fixes Andrea's active JAX implementation.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    aa, bb = jnp.abs(a), jnp.abs(b)
    magnitude = jnp.maximum(jnp.minimum(2 * aa, bb), jnp.minimum(aa, 2 * bb))
    return jnp.where(_same_sign(a, b), jnp.sign(a) * magnitude, 0.0)


def van_albada(a, b):
    """Sign-preserving Van Albada slope, with safe inactive JAX arithmetic.

    For like signs, sigma=ab(a+b)/(a*a+b*b); otherwise sigma=0. This explicit
    extrema gate differs from the retained ungated NumPy formula. Normalizing
    before squaring avoids underflow/overflow and needs no dimensional epsilon.
    The origin remains mathematically nonsmooth; AD selects a finite branch.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    active = _same_sign(a, b)
    scale = jnp.maximum(jnp.abs(a), jnp.abs(b))
    safe_scale = jnp.where(active, scale, 1.0)
    an = jnp.where(active, a, 0.0) / safe_scale
    bn = jnp.where(active, b, 0.0) / safe_scale
    denominator = jnp.where(active, an * an + bn * bn, 1.0)
    numerator = jnp.where(active, an, 0.0) * jnp.where(active, bn, 0.0) * (an + bn)
    return safe_scale * numerator / denominator


_JAX_LIMITERS = {
    "muscl_minmod": minmod,
    "muscl_superbee": superbee,
    "muscl_van_albada": van_albada,
}


def _superbee(a, b):
    s1 = np.where(
        np.sign(a) == np.sign(b),
        np.sign(a) * np.minimum(2.0 * np.abs(a), np.abs(b)),
        0.0,
    )
    s2 = np.where(
        np.sign(a) == np.sign(b),
        np.sign(a) * np.minimum(np.abs(a), 2.0 * np.abs(b)),
        0.0,
    )
    return np.where(np.abs(s1) > np.abs(s2), s1, s2)


def _van_albada(a, b):
    """Van Albada limiter — smooth and differentiable everywhere (no hard sign test).
    Combines minmod (low compression) and superbee (high compression) properties:
    reduces to minmod for |a|<<|b| and to superbee for |a|≈|b|.
    σ = (a²b + ab²) / (a² + b² + ε)
    JAX autodiff works correctly through this limiter, making it preferable
    for Newton solvers where the Jacobian must be accurate near discontinuities.
    This matches the limiter described by Städtke (2006) §8.4, Eq. 8.37."""
    return (a**2 * b + a * b**2) / (a**2 + b**2 + 1e-300)


def _mc(a, b):
    """Monotonized Central (MC) limiter — between minmod and superbee.
    σ = minmod(2a, 2b, (a+b)/2)
    More compressive than minmod, less than superbee; TVD for all smooth flows."""
    same_sign = np.sign(a) == np.sign(b)
    mag = np.sign(a) * np.minimum(
        np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)), 0.5 * np.abs(a + b)
    )
    return np.where(same_sign, mag, 0.0)


_LIMITERS = {
    "muscl_minmod": _minmod,
    "muscl_van_leer": _van_leer,
    "muscl_superbee": _superbee,
    "muscl_van_albada": _van_albada,
    "muscl_mc": _mc,
}


def muscl_van_leer_ph(U):
    """Van Leer MUSCL reconstruction in primitive (p, u, h) space.

    U   : (N, 3) JAX array  — columns are [p, u, h]
    Returns (U_L, U_R) each (N-1, 3) — left/right states at each face.
    Boundary cells (index 0 and N-1) are assigned zero slope, so the
    face adjacent to a ghost cell uses a 1st-order (cell-centred) value.
    """
    dU = U[1:] - U[:-1]  # (N-1, 3)
    vl = _van_leer_jax_safe
    sigma = jnp.zeros_like(U).at[1:-1].set(vl(dU[:-1], dU[1:]))  # (N, 3)
    U_L = U[:-1] + 0.5 * sigma[:-1]
    U_R = U[1:] - 0.5 * sigma[1:]
    return U_L, U_R


def _muscl_reconstruct(Q, method):
    """
    MUSCL piecewise-linear reconstruction in conservative variables.

    Returns Q_L, Q_R at each of the Nj-1 interfaces (shape (3, Nj-1)).
    Ghost cells (i=0 and i=Nj-1) use zero slope so the BC values are
    passed through unchanged.
    """
    limiter = _LIMITERS[method]

    dQ = Q[:, 1:] - Q[:, :-1]  # (3, Nj-1): dQ[:, i] = Q_{i+1} - Q_i

    # Slopes for interior cells 1 … Nj-2; zero at ghost cells
    sigma = np.zeros_like(Q)  # (3, Nj)
    sigma[:, 1:-1] = limiter(dQ[:, :-1], dQ[:, 1:])  # (3, Nj-2)

    Q_L = Q[:, :-1] + 0.5 * sigma[:, :-1]  # left  state at each interface
    Q_R = Q[:, 1:] - 0.5 * sigma[:, 1:]  # right state at each interface
    return Q_L, Q_R
