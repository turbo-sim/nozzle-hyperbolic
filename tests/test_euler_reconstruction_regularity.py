"""Local limiter bounds and JAX regularity, without solving a nozzle PDE."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.numerics.reconstruction import (
    _van_leer_jax_regularized,
    _van_leer_jax_safe,
)


def test_safe_limiter_preserves_original_active_formula():
    """Inactive-branch protection must not change same-sign reconstruction."""
    a = jnp.array([0.25, 1.0, 4.0, -0.25, -1.0, -4.0])
    b = jnp.array([2.0, 3.0, 0.5, -2.0, -3.0, -0.5])
    original = jax.jit(lambda x, y: 2.0 * x * y / (x + y + 1e-300))
    actual = jax.jit(_van_leer_jax_safe)(a, b)
    np.testing.assert_array_equal(actual, original(a, b))


def test_safe_limiter_has_finite_forward_and_reverse_ad_on_inactive_branches():
    """Opposite slopes and exact cancellations must not poison reverse AD."""
    points = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1e4, -1e4],
        ]
    )
    slope = lambda point: _van_leer_jax_safe(point[0], point[1])
    value = jax.jit(jax.vmap(slope))(points)
    forward = jax.jit(jax.vmap(jax.jacfwd(slope)))(points)
    reverse = jax.jit(jax.vmap(jax.jacrev(slope)))(points)
    np.testing.assert_array_equal(value, np.zeros(len(points)))
    assert np.isfinite(forward).all()
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, reverse, atol=0.0, rtol=0.0)


def test_regularized_limiter_is_c2_at_axes_origin_and_opposite_slopes():
    """Both AD modes and second derivatives agree with the flat extensions."""
    points = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    slope = lambda point: _van_leer_jax_regularized(point[0], point[1], 0.2)
    value = jax.jit(jax.vmap(slope))(points)
    forward = jax.jit(jax.vmap(jax.jacfwd(slope)))(points)
    reverse = jax.jit(jax.vmap(jax.jacrev(slope)))(points)
    hessian = jax.jit(jax.vmap(jax.hessian(slope)))(points)
    for actual in (value, forward, reverse, hessian):
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_regularized_second_derivatives_approach_axis_and_origin_limits():
    """Check continuity near branch boundaries, not just their AD values."""
    slope = lambda point: _van_leer_jax_regularized(point[0], point[1], 0.2)
    evaluate = jax.jit(jax.vmap(jax.hessian(slope)))
    axis = np.asarray(evaluate(jnp.array([[1e-4, 1.0], [1e-5, 1.0], [1e-6, 1.0]])))
    origin = np.asarray(evaluate(jnp.array([[1e-2, 1e-2], [1e-3, 1e-3], [1e-4, 1e-4]])))
    axis_norms = np.linalg.norm(axis.reshape(3, -1), axis=1)
    origin_norms = np.linalg.norm(origin.reshape(3, -1), axis=1)
    assert np.all(axis_norms[1:] < 0.2 * axis_norms[:-1])
    assert np.all(origin_norms[1:] < 0.01 * origin_norms[:-1])
    assert axis_norms[-1] < 0.01


def test_regularized_slope_bounds_constants_and_symmetry():
    """Smoothing attenuates Van Leer rather than enlarging limited slopes."""
    values = jnp.array([-1.0, -0.1, -1e-3, 0.0, 1e-3, 0.1, 1.0])
    a, b = jnp.meshgrid(values, values, indexing="ij")
    evaluate = jax.jit(_van_leer_jax_regularized)
    actual = np.asarray(evaluate(a, b, 0.05))
    original = np.asarray(jax.jit(_van_leer_jax_safe)(a, b))
    active = np.asarray(a * b > 0.0)
    np.testing.assert_array_equal(actual[~active], np.zeros(np.count_nonzero(~active)))
    assert np.all(np.sign(actual[active]) == np.sign(np.asarray(a)[active]))
    assert np.all(np.abs(actual) <= np.abs(original) + 1e-15)
    assert np.all(np.abs(actual) <= 2.0 * np.minimum(np.abs(a), np.abs(b)) + 1e-15)
    np.testing.assert_allclose(actual, evaluate(b, a, 0.05), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual, -evaluate(-a, -b, 0.05), rtol=0.0, atol=0.0)


def test_regularized_primitive_interfaces_remain_between_neighbor_values():
    """Fixed-scale component bounds preserve positive interface primitives."""
    states = jnp.array(
        [
            [2e5, 100.0, 3e5],
            [1.7e5, 150.0, 2.9e5],
            [1e5, 300.0, 2.55e5],
            [1.5e5, 200.0, 2.8e5],
            [1.6e5, 150.0, 2.9e5],
            [1.8e5, 120.0, 3e5],
        ]
    )
    scales = jnp.array([2e5, 340.0, 3e5])

    @jax.jit
    def reconstruct(physical):
        differences = (physical[1:] - physical[:-1]) / scales
        slopes = (
            jnp.zeros_like(physical)
            .at[1:-1]
            .set(
                scales
                * _van_leer_jax_regularized(differences[:-1], differences[1:], 1e-3)
            )
        )
        return physical[:-1] + 0.5 * slopes[:-1], physical[1:] - 0.5 * slopes[1:]

    left, right = reconstruct(states)
    lower = np.minimum(states[:-1], states[1:])
    upper = np.maximum(states[:-1], states[1:])
    for interface in (left, right):
        assert np.all(interface >= lower)
        assert np.all(interface <= upper)
        assert np.all(interface > 0.0)
    constant = jnp.broadcast_to(states[0], states.shape)
    for interface in reconstruct(constant):
        np.testing.assert_array_equal(interface, constant[:-1])


def test_regularized_linear_slopes_recover_van_leer_as_epsilon_vanishes():
    """Finite smoothing damps slopes; the zero-smoothing limit is consistent."""
    epsilon = jnp.array([1e-2, 1e-3, 1e-4])
    for difference in (0.2, -0.2):
        actual = np.asarray(
            jax.jit(_van_leer_jax_regularized)(difference, difference, epsilon)
        )
        relative_error = np.abs(actual / difference - 1.0)
        assert np.all(relative_error[1:] < relative_error[:-1])
        assert relative_error[-1] < 1e-6
    a = jnp.array([0.5, 1.0, -0.5, -1.0])
    b = jnp.array([1.0, 0.5, -1.0, -0.5])
    np.testing.assert_allclose(
        jax.jit(_van_leer_jax_regularized)(a, b, 1e-5),
        jax.jit(_van_leer_jax_safe)(a, b),
        rtol=1e-9,
        atol=0.0,
    )
