"""Newton corrections respect residual tolerances, convergence and singular systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver


def test_newton_backtracking_rejects_invalid_trials_and_solves_original_equation():
    def residual(point):
        return jnp.log(point)

    def direction(point):
        return -point * jnp.log(point)

    solve = jax.jit(
        make_newton_solver(
            residual,
            direction,
            lambda point: jnp.ones_like(point),
            max_iterations=20,
            tolerance=1e-11,
            residual_tolerance=1e-11,
            admissible=lambda point: jnp.all(point > 0),
            backtracking=True,
        )
    )
    result, info = solve(jnp.array([10.0]))
    np.testing.assert_allclose(result, [1.0], rtol=1e-11, atol=1e-11)
    assert bool(info.converged) and int(info.status) == 0
    assert int(info.rejected_trials) >= 2
    assert float(jnp.max(jnp.abs(residual(result)))) < 1e-11


def test_failed_newton_trials_retain_previous_iterate_and_report_failure():
    solve = jax.jit(
        make_newton_solver(
            lambda point: point - 1.0,
            lambda point: jnp.ones_like(point),
            lambda point: jnp.ones_like(point),
            max_iterations=4,
            backtracking=True,
            max_backtracks=3,
            residual_tolerance=1e-10,
        )
    )
    initial = jnp.array([2.0])
    result, info = solve(initial)
    np.testing.assert_array_equal(result, initial)
    assert not bool(info.converged)
    assert int(info.status) == 3 and int(info.rejected_trials) == 4


@pytest.mark.parametrize("tolerance", [0.0, 1e-10])
def test_unchecked_newton_preserves_historical_full_step_iteration_rule(tolerance):
    def direction(point):
        return -(point**2 - 2.0) / (2.0 * point)

    initial = jnp.array([1.0, 2.0, 3.0])
    expected = np.asarray(initial).copy()
    relative = np.inf
    iterations = 0
    while iterations < 8 and (tolerance <= 0 or relative > tolerance):
        correction = -(expected**2 - 2.0) / (2.0 * expected)
        relative = np.max(np.abs(correction) / (np.abs(expected) + 1e-300))
        expected += correction
        iterations += 1
    solve = jax.jit(
        make_newton_solver(
            lambda point: point**2 - 2.0,
            direction,
            jnp.abs,
            max_iterations=8,
            tolerance=tolerance,
        )
    )
    actual, info = solve(initial)
    # NumPy's unfused host arithmetic can round the final sqrt(2) iterate
    # one ULP differently from XLA. The iteration rule/count remains exact.
    np.testing.assert_array_max_ulp(np.asarray(actual), expected, maxulp=1)
    assert int(info.iterations) == iterations
