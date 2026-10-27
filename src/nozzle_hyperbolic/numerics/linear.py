"""Unchanged definitions relocated from the TwoFluid solver modules."""
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Equilibrated linear solves
# ---------------------------------------------------------------------------
# The two-fluid matrices mix units: mass rows ~O(10), momentum ~O(1e6), energy
# ~O(1e8) → kappa(rho_mat) ~ 1e13 from SCALING alone (after one max-abs row+col
# equilibration it drops to ~1e2; measured at the throat checkerboard cell).
# A raw solve at kappa~1e13 injects ~kappa*eps ≈ 1e-3 RELATIVE roundoff noise
# into the wave decomposition every step — a stochastic forcing that lands
# hardest on the small-magnitude VOID row (the observed throat checkerboard,
# and the SSR plateau at ~1e-3 = the noise floor).  Equilibration is an exact
# identity in real arithmetic — it changes nothing but the roundoff.

def _equil_solve(A, B):
    """solve(A, B) with max-abs row+col equilibration (B may be (6,) or (6,m))."""
    dr = 1.0 / jnp.maximum(jnp.max(jnp.abs(A), axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / jnp.maximum(jnp.max(jnp.abs(As), axis=0), 1e-300)
    As = As * dc[None, :]
    Bs = B * (dr[:, None] if B.ndim == 2 else dr)
    Y  = jnp.linalg.solve(As, Bs)
    return Y * (dc[:, None] if B.ndim == 2 else dc)


def _equil_lstsq(A, B, rcond=1e-14):
    """lstsq(A, B) with the same equilibration (rcond acts on the SCALED matrix,
    so it no longer risks truncating genuine directions hidden by bad units)."""
    dr = 1.0 / jnp.maximum(jnp.max(jnp.abs(A), axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / jnp.maximum(jnp.max(jnp.abs(As), axis=0), 1e-300)
    As = As * dc[None, :]
    Bs = B * (dr[:, None] if B.ndim == 2 else dr)
    Y  = jnp.linalg.lstsq(As, Bs, rcond=rcond)[0]
    return Y * (dc[:, None] if B.ndim == 2 else dc)
