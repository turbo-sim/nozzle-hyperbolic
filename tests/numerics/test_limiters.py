"""Limiter signs, reconstruction geometry and finite inactive derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.reconstruction import (
    minmod,
    superbee,
    van_albada,
    reconstruct_faces,
)


@pytest.mark.parametrize("limiter", [minmod, superbee, van_albada])
def test_limiter_symmetry_extrema_and_finite_ad(limiter):
    a = jnp.array([0.0, 1.0, 2.0, -1.0, -2.0, 1.0, 0.0])
    b = jnp.array([0.0, 2.0, 1.0, -2.0, -1.0, -1.0, 3.0])
    actual = jax.jit(limiter)(a, b)
    np.testing.assert_allclose(actual, -limiter(-a, -b), atol=0)
    np.testing.assert_allclose(actual, limiter(b, a), atol=0)
    np.testing.assert_array_equal(np.asarray(actual)[[0, 5, 6]], 0)
    for derivative in (jax.jacfwd, jax.jacrev):
        assert np.all(
            np.isfinite(derivative(lambda z: limiter(z[0], z[1]))(jnp.stack((a, b))))
        )
    np.testing.assert_allclose(limiter(jnp.ones(3), jnp.ones(3)), 1)


def test_compression_and_negative_superbee_regression():
    a, b = jnp.array([-1.0, 1.0]), jnp.array([-3.0, 3.0])
    np.testing.assert_allclose(minmod(a, b), [-1, 1])
    np.testing.assert_allclose(van_albada(a, b), [-1.2, 1.2])
    np.testing.assert_allclose(superbee(a, b), [-2, 2])
    np.testing.assert_array_equal(
        van_albada(jnp.array([0.0, 1e300]), jnp.array([0.0, -1e300])), 0
    )


@pytest.mark.parametrize(
    "method", ["muscl_minmod", "muscl_superbee", "muscl_van_albada"]
)
def test_coordinate_reconstruction_preserves_linear_profiles_on_nonuniform_mesh(method):
    faces = jnp.array([0.0, 0.1, 0.35, 0.7, 1.0])
    points = jnp.concatenate((faces[:1], 0.5 * (faces[:-1] + faces[1:]), faces[-1:]))
    full = jnp.stack((2 + 3 * points, 7 - 2 * points), axis=1)
    left, right = reconstruct_faces(full, points, faces, method)
    expected = jnp.stack((2 + 3 * faces, 7 - 2 * faces), axis=1)
    np.testing.assert_allclose(left, expected, atol=2e-15)
    np.testing.assert_allclose(right, expected, atol=2e-15)
