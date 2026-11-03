"""Termination requires original equations and retained failure diagnostics."""

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import ResidualPolicy, SteadySettings
from nozzle_hyperbolic.solvers.residuals import assess_residuals


@pytest.mark.parametrize(
    "criterion,expected",
    [
        ("relative", [True, False]),
        ("absolute", [False, True]),
        ("either", [True, True]),
        ("both", [False, False]),
    ],
)
def test_relative_absolute_stopping_modes_have_distinct_meanings(criterion, expected):
    policy = ResidualPolicy(criterion=criterion)
    # First equation meets only relative; second equation meets only absolute.
    result = assess_residuals(
        jnp.asarray([1e-5, 1e-10]), jnp.asarray([100.0, 1e-5]), policy
    )
    np.testing.assert_array_equal(result["equation_passed"], expected)
    assert policy.relative_tolerance == 1e-6
    assert policy.absolute_tolerance == 1e-9
    assert SteadySettings().residual == ResidualPolicy()


@pytest.mark.parametrize("criterion", ["relative", "absolute", "either", "both"])
def test_initially_zero_equations_use_an_explicit_absolute_fallback(criterion):
    policy = ResidualPolicy(criterion=criterion)
    current = jnp.asarray([0.0, 5e-10, 2e-9, 5e-10])
    initial = jnp.asarray([0.0, 0.0, 0.0, policy.initial_floor / 2])
    result = assess_residuals(current, initial, policy)
    np.testing.assert_array_equal(result["relative_active"], [False] * 4)
    np.testing.assert_array_equal(result["equation_passed"], [True, True, False, True])
    assert not bool(result["all_equations_passed"])
