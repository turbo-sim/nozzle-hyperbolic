"""Independent characteristic and jump checks for scalar reference profiles."""

import jax
import numpy as np

from nozzle_hyperbolic.models.scalar.analytical import (
    burgers_riemann,
    translated_trigonometric,
)


def test_advection_translates_in_either_direction_and_is_jittable():
    x = np.array([0.1, 0.3, 0.7])
    for speed in (-2.0, 0.0, 1.5):
        shifted = x - speed * 0.2
        phase = 2 * np.pi * shifted / 1.7
        expected = np.sin(phase) + 0.25 * np.cos(2 * phase)
        actual = jax.jit(translated_trigonometric)(x, 0.2, speed=speed, length=1.7)
        np.testing.assert_allclose(actual, expected, atol=5e-7)


def test_burgers_shock_satisfies_jump_speed_and_right_trace_convention():
    # f(2)-f(-1) = s*(2-(-1)) gives s=0.5.
    front = 0.5 + 0.5 * 0.4
    result = burgers_riemann(np.array([front - 0.01, front + 0.01]), 0.4,
                             left=2.0, right=-1.0)
    np.testing.assert_array_equal(result, [2.0, -1.0])
    np.testing.assert_array_equal(burgers_riemann(np.array([0.4, 0.5]), 0.0),
                                  [1.0, -0.5])


def test_burgers_rarefaction_initial_state_and_jitted_fan():
    x = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    np.testing.assert_array_equal(burgers_riemann(x, 0.0, left=-1.0, right=1.0),
                                  [-1.0, -1.0, 1.0, 1.0, 1.0])
    sampled = jax.jit(burgers_riemann)(x, 0.2, left=-1.0, right=1.0)
    np.testing.assert_allclose(sampled, [-1.0, -0.5, 0.0, 0.5, 1.0], atol=2e-7)
