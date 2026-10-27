"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# MUSCL reconstruction  (2nd-order spatial accuracy)
# ---------------------------------------------------------------------------

def _minmod(a, b):
    return np.where(np.sign(a) == np.sign(b),
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _van_leer(a, b):
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / (a + b + 1e-300), 0.0)


def _superbee(a, b):
    s1 = np.where(np.sign(a) == np.sign(b),
                  np.sign(a) * np.minimum(2.0 * np.abs(a), np.abs(b)), 0.0)
    s2 = np.where(np.sign(a) == np.sign(b),
                  np.sign(a) * np.minimum(np.abs(a), 2.0 * np.abs(b)), 0.0)
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
        np.minimum(2.0 * np.abs(a), 2.0 * np.abs(b)),
        0.5 * np.abs(a + b))
    return np.where(same_sign, mag, 0.0)


_LIMITERS = {
    'muscl_minmod':      _minmod,
    'muscl_van_leer':    _van_leer,
    'muscl_superbee':    _superbee,
    'muscl_van_albada':  _van_albada,
    'muscl_mc':          _mc,
}


def muscl_van_leer_ph(U):
    """Van Leer MUSCL reconstruction in primitive (p, u, h) space.

    U   : (N, 3) JAX array  — columns are [p, u, h]
    Returns (U_L, U_R) each (N-1, 3) — left/right states at each face.
    Boundary cells (index 0 and N-1) are assigned zero slope, so the
    face adjacent to a ghost cell uses a 1st-order (cell-centred) value.
    """
    dU    = U[1:] - U[:-1]                                              # (N-1, 3)
    vl    = lambda a, b: jnp.where(a * b > 0, 2.0 * a * b / (a + b + 1e-300), 0.0)
    sigma = jnp.zeros_like(U).at[1:-1].set(vl(dU[:-1], dU[1:]))        # (N, 3)
    U_L   = U[:-1] + 0.5 * sigma[:-1]
    U_R   = U[1:]  - 0.5 * sigma[1:]
    return U_L, U_R


def _muscl_reconstruct(Q, method):
    """
    MUSCL piecewise-linear reconstruction in conservative variables.

    Returns Q_L, Q_R at each of the Nj-1 interfaces (shape (3, Nj-1)).
    Ghost cells (i=0 and i=Nj-1) use zero slope so the BC values are
    passed through unchanged.
    """
    limiter = _LIMITERS[method]

    dQ = Q[:, 1:] - Q[:, :-1]          # (3, Nj-1): dQ[:, i] = Q_{i+1} - Q_i

    # Slopes for interior cells 1 … Nj-2; zero at ghost cells
    sigma = np.zeros_like(Q)            # (3, Nj)
    sigma[:, 1:-1] = limiter(dQ[:, :-1], dQ[:, 1:])   # (3, Nj-2)

    Q_L = Q[:, :-1] + 0.5 * sigma[:, :-1]   # left  state at each interface
    Q_R = Q[:, 1:]  - 0.5 * sigma[:, 1:]    # right state at each interface
    return Q_L, Q_R
