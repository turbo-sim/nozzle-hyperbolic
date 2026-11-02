"""Local derivative reuse preserves values, AD and changing linearization points."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.local_derivatives import local_jvp
from nozzle_hyperbolic.solvers.fvm import Scheme

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("width", [1, 2, 4, 7])
def test_local_face_derivatives_with_nonuniform_opaque_context(width):
    left = jnp.linspace(0.2, 1.3, 5 * width).reshape(5, width)
    right = 0.7 * left + 0.1
    weights = jnp.linspace(0.8, 1.2, 5)[:, None]

    def physical_map(a, b):
        return weights * (jnp.sin(a) + b**2 + 0.1 * a * b)

    reused = local_jvp(physical_map)
    np.testing.assert_array_equal(reused(left, right), physical_map(left, right))
    for transform in (jax.jacfwd, jax.jacrev):
        actual = transform(reused, argnums=(0, 1))(left, right)
        expected = transform(physical_map, argnums=(0, 1))(left, right)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=3e-14, atol=3e-14)
    # A later Newton point must use new derivatives rather than stale blocks.
    for factor in (1.0, 1.4):
        point = (factor * left, right - 0.03 * factor)
        tangent = (jnp.cos(left), jnp.sin(right))
        actual = jax.jit(lambda a, b: jax.jvp(reused, (a, b), tangent))(*point)
        expected = jax.jvp(physical_map, point, tangent)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=3e-14, atol=3e-14)


def test_local_storage_and_arguments_with_different_widths():
    state = jnp.linspace(0.3, 1.1, 15).reshape(5, 3)
    context = jnp.linspace(0.4, 0.8, 10).reshape(5, 2)

    def storage(a):
        return a + 0.03 * a**3

    def two_widths(a, b):
        return jnp.stack((a[:, 0] * b[:, 1], a[:, 2] + b[:, 0] ** 2), axis=1)

    np.testing.assert_allclose(
        jax.jacfwd(local_jvp(storage))(state), jax.jacfwd(storage)(state),
        rtol=3e-14, atol=3e-14,
    )
    for actual, expected in zip(
        jax.jacfwd(local_jvp(two_widths), argnums=(0, 1))(state, context),
        jax.jacfwd(two_widths, argnums=(0, 1))(state, context),
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


def test_derivative_reuse_is_not_a_public_scheme_input():
    with pytest.raises(TypeError, match="local_derivative_reuse"):
        Scheme(local_derivative_reuse=False)
