"""Unchanged definitions relocated from the TwoFluid solver modules."""
import operator

import jax
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


def solve_block_banded(blocks, rhs, radius=None):
    """Solve a square block-banded system without constructing a dense matrix.

    ``blocks[i, d]`` is the derivative of row block i with respect to block
    i+d-radius, shape ``(nblocks, 2*radius+1, nvar, nvar)``. ``rhs`` is flat.
    Missing off-domain blocks must be zero. Grouping radius adjacent blocks
    produces a block-tridiagonal system; a block Thomas elimination follows.
    Padding adds identity rows for an incomplete final group.

    Pivoting is local to a superblock, not across superblocks. A nonsingular
    full matrix can still defeat this factorization; callers must check the
    correction and linear defect. This is the existing Euler algorithm with
    its variable count and bandwidth parameterized.
    """
    nblocks, ndiag, nvar, ncol = blocks.shape
    radius = (ndiag - 1) // 2 if radius is None else operator.index(radius)
    if radius < 0 or ndiag != 2 * radius + 1 or nvar != ncol:
        raise ValueError("Expected square blocks with 2*radius+1 diagonals")
    if nblocks < 1 or nvar < 1 or rhs.shape != (nblocks * nvar,):
        raise ValueError("Block count, variable count and flat rhs must agree")
    cells_per_group = max(radius, 1)
    ngroups = -(-nblocks // cells_per_group)
    padded_count = ngroups * cells_per_group
    group_size = nvar * cells_per_group
    padded = jnp.pad(blocks, ((0, padded_count - nblocks), (0, 0), (0, 0), (0, 0)))
    if padded_count > nblocks:
        padded = padded.at[nblocks:, radius].set(jnp.eye(nvar, dtype=padded.dtype))
    diagonal = jnp.zeros((ngroups, group_size, group_size), padded.dtype)
    upper = jnp.zeros_like(diagonal)
    lower = jnp.zeros_like(diagonal)
    for a in range(cells_per_group):
        for d in range(ndiag):
            off = a + d - radius
            block = padded[a::cells_per_group, d]
            rows = slice(nvar * a, nvar * a + nvar)
            if 0 <= off < cells_per_group:
                diagonal = diagonal.at[:, rows, nvar*off:nvar*off+nvar].set(block)
            elif off >= cells_per_group:
                o = off - cells_per_group
                upper = upper.at[:, rows, nvar*o:nvar*o+nvar].set(block)
            else:
                o = off + cells_per_group
                lower = lower.at[:, rows, nvar*o:nvar*o+nvar].set(block)
    b = jnp.pad(rhs, (0, nvar * (padded_count - nblocks))).reshape(ngroups, group_size)

    def forward(carry, inputs):
        previous_upper, previous_rhs = carry
        diagonal_i, upper_i, lower_i, rhs_i = inputs
        reduced_diagonal = diagonal_i - lower_i @ previous_upper
        reduced_rhs = rhs_i - lower_i @ previous_rhs
        solved = jnp.linalg.solve(
            reduced_diagonal, jnp.concatenate([upper_i, reduced_rhs[:, None]], axis=1)
        )
        result = (solved[:, :-1], solved[:, -1])
        return result, result

    zero = (jnp.zeros((group_size, group_size), padded.dtype),
            jnp.zeros((group_size,), padded.dtype))
    _, (upper_factors, rhs_factors) = jax.lax.scan(
        forward, zero, (diagonal, upper, lower, b)
    )

    def backward(next_x, inputs):
        upper_i, rhs_i = inputs
        x_i = rhs_i - upper_i @ next_x
        return x_i, x_i

    _, solution = jax.lax.scan(
        backward, jnp.zeros((group_size,), padded.dtype),
        (upper_factors, rhs_factors), reverse=True,
    )
    return solution.reshape(-1)[:nvar * nblocks]
