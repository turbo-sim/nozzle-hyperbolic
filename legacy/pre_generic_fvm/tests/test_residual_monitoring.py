"""Independent mathematical checks for model-independent convergence tools."""

import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.convergence_output import ConvergenceTable
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec, MonitorTracker
from nozzle_hyperbolic.solvers.residuals import (
    ResidualPolicy, assess_residuals, reduce_residuals, validate_residual_setup)


def test_weighted_norms_and_cancellation_eager_jit():
    # Dimensionless residual density = [[2,-3],[-2,1]], weights=[1/4,3/4].
    volumes = jnp.array([1., 3.])
    scales = jnp.array([2., 4.])
    integrated = jnp.array([[2., -6.], [-6., 6.]])
    for reducer in (reduce_residuals, jax.jit(reduce_residuals)):
        result = reducer(integrated, volumes, scales, 2.)
        np.testing.assert_allclose(result["l1"], [2., 1.5])
        np.testing.assert_allclose(result["l2"], [2., np.sqrt(3.)])
        np.testing.assert_allclose(result["linf"], [2., 3.])
        np.testing.assert_array_equal(result["worst_cell"], [0, 0])
    opposite = reduce_residuals(jnp.array([[1.], [-1.]]), jnp.ones(2), jnp.ones(1), 1.)
    assert float(opposite["l2"][0]) == 1.  # A zero global sum cannot hide local defects.


def test_volume_subdivision_and_consistent_units_preserve_norms():
    original = reduce_residuals(jnp.array([[2.], [9.]]), jnp.array([1., 3.]), jnp.array([2.]), 1.)
    refined = reduce_residuals(jnp.array([[2.], [3.], [6.]]), jnp.array([1., 1., 2.]), jnp.array([2.]), 1.)
    units = reduce_residuals(jnp.array([[20.], [90.]]), jnp.array([1., 3.]), jnp.array([20.]), 1.)
    for norm in ("l1", "l2", "linf"):
        np.testing.assert_allclose(original[norm], refined[norm])
        np.testing.assert_allclose(original[norm], units[norm])


def test_cell_weighting_accepts_integer_inputs_without_truncating_weights():
    result = jax.jit(lambda rows: reduce_residuals(
        rows, jnp.array([1, 1]), jnp.array([1]), 1., weighting="cell"))(
            jnp.array([[2], [4]]))
    np.testing.assert_allclose(result["l1"], [3.])
    np.testing.assert_allclose(result["l2"], [np.sqrt(10.)])
    np.testing.assert_allclose(result["linf"], [4.])


def test_equation_isolation_absolute_safeguard_and_initial_zero():
    policy = ResidualPolicy()
    assess = jax.jit(lambda values, initial: assess_residuals(values, initial, policy))
    initial = jnp.array([1., 1., 0., 100.])
    result = assess(jnp.array([1e-10, 2e-9, 1e-10, 1e-8]), initial)
    assert not bool(result["all_equations_passed"])
    np.testing.assert_array_equal(result["equation_passed"], [True, False, True, True])
    np.testing.assert_array_equal(result["relative_active"], [True, True, False, True])
    assert not bool(assess(jnp.array([0., 0., 2e-8, 0.]), initial)["all_equations_passed"])
    # Large initial residual: nine orders alone do not meet the absolute safeguard.
    assert not bool(assess(jnp.array([1e-7]), jnp.array([1e3]))["all_equations_passed"])
    assert not bool(assess(jnp.array([jnp.nan]), jnp.array([1.]))["all_equations_passed"])


def test_norms_do_not_overflow_when_squaring_large_finite_values():
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    magnitude = 1e150 if jax.config.x64_enabled else 1e30
    result = reduce_residuals(jnp.array([[magnitude], [-magnitude]], dtype=dtype),
                              jnp.ones(2, dtype=dtype), jnp.ones(1, dtype=dtype), 1.)
    np.testing.assert_allclose(result["l2"], [magnitude], rtol=1e-6)


@pytest.mark.parametrize("kwargs", [{"norm": "max"}, {"weighting": "area"},
    {"relative_tolerance": 0}, {"absolute_tolerance": np.nan}, {"initial_floor": -1}])
def test_invalid_policies_rejected(kwargs):
    with pytest.raises(ValueError):
        ResidualPolicy(**kwargs)


def test_setup_validation():
    validate_residual_setup([1., 2.], [1., 3.], 1.)
    with pytest.raises(ValueError):
        validate_residual_setup([1., 0.], [1.], 1.)


def test_monitor_full_window_fixed_scale_and_multiple_checks():
    spec = MonitorSpec("cp", "any_model_quantity", "mean", convergence=True, window=3,
                       absolute_tolerance=.01, relative_tolerance=.01, reference=2.)
    tracker = MonitorTracker([spec, MonitorSpec("mdot", "other", "outlet")])
    assert not tracker.update({"cp": 0., "mdot": 1.})["cp"]["stable"]
    tracker.update({"cp": .02, "mdot": 1.})
    assert not tracker.passed
    third = tracker.update({"cp": -.005, "mdot": 1.})
    assert tracker.passed
    assert third["cp"]["threshold"] == .03
    tracker.update({"cp": .5, "mdot": 1.})
    assert not tracker.passed
    tracker.update({"cp": .5, "mdot": float("nan")})
    assert not tracker.passed


def test_monitor_configuration_and_value_contract():
    with pytest.raises(ValueError):
        MonitorTracker([MonitorSpec("same", "a", "b"), MonitorSpec("same", "a", "b")])
    with pytest.raises(ValueError):
        MonitorSpec("cp", "a", "b", window=1)
    with pytest.raises(ValueError):
        MonitorTracker([MonitorSpec("cp", "a", "b")]).update({})


def test_console_labels_initial_inactive_and_rejected_states():
    stream = io.StringIO()
    table = ConvergenceTable(["mass", "angular"], ["cp"], stream=stream)
    row = dict(step=0, phase="initial", cfl=.2, equation_relative=[1., 0.],
               relative_active=[True, False], monitors={"cp": {"value": .2}})
    table(row)
    table(dict(step=1, accepted=False))
    output = stream.getvalue()
    assert "R1=mass" in output and "R2=angular" in output and "initial" in output
    assert "retry" in output and "--" in output
    assert "cp" not in output
    assert max(map(len, output.splitlines())) <= 100
