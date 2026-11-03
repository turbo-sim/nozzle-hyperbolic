"""Block solves and colored Jacobians agree with independent dense algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.jacobians import make_block_operators


@pytest.mark.parametrize(
    "nblocks,nvar,radius", [(2, 3, 1), (3, 4, 2), (6, 4, 2), (5, 3, 0)]
)
def test_colored_blocks_and_banded_direction_match_independent_matrix(
    nblocks, nvar, radius
):
    """Exercise odd/padded blocks, four-variable rows and a block diagonal limit."""
    count = nblocks * nvar
    random = np.random.default_rng(31 + count)
    matrix = random.normal(scale=0.02, size=(count, count))
    block_ids = np.arange(count) // nvar
    matrix[np.abs(block_ids[:, None] - block_ids[None, :]) > radius] = 0
    matrix += 2.0 * np.eye(count)
    storage = np.ones(count)
    storage[:nvar] = storage[-nvar:] = 0.0
    point = jnp.linspace(0.7, 1.3, count)
    old = point * 0.9
    dt = 0.13

    def residual(new, previous, timestep):
        return jnp.asarray(storage) * (new - previous) + timestep * (
            jnp.asarray(matrix) @ new + 0.03 * new**2 - 0.6
        )

    operators = make_block_operators(
        residual, nblocks, nvar, radius, jacobian_mode="colored", linear_solver="banded"
    )
    expected = np.diag(storage + dt * 0.06 * np.asarray(point)) + dt * matrix
    actual = np.asarray(jax.jit(operators["colored_jacobian"])(point, old, dt))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        jax.jit(operators["dense_jacobian"])(point, old, dt),
        expected,
        rtol=2e-14,
        atol=2e-14,
    )
    correction = np.asarray(jax.jit(operators["direction"])(point, old, dt))
    defect = expected @ correction + np.asarray(residual(point, old, dt))
    assert np.linalg.norm(defect, ord=np.inf) < 1e-12


def test_declared_band_guard_detects_a_wider_boundary_dependency():
    """The sparsity guard must expose a radius-two BC in a radius-one layout."""

    def residual(point):
        rows = point.reshape(6, 4)
        return (2 * rows).at[0].add(0.3 * rows[2]).ravel()

    operators = make_block_operators(residual, 6, 4, 1)
    assert float(jax.jit(operators["jacobian_difference"])(jnp.ones(24))) > 0.1
