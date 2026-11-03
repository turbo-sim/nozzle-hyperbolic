"""Physical source and boundary signal limits constrain explicit timesteps."""

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, solve_transient

from tests._support.generic_models import _UserAdvection, _UserFilteredAdvection


@pytest.mark.parametrize("integrator", ["euler-explicit", "runge-kutta-3"])
@pytest.mark.parametrize("cfl", [0.2, 1.0, 3.0])
def test_explicit_source_timestep_is_an_absolute_cap_after_wave_cfl(integrator, cfl):
    model = _UserAdvection(0.7)
    source_limit = 2e-4
    model.explicit_timestep = lambda full: jnp.asarray(source_limit)
    operator = Operator(model, Scheme(flux="hll", time_integrator=integrator))
    limits = operator.timestep(model.initial_state(), cfl)
    assert bool(limits.valid)
    assert float(limits.source_dt) == source_limit
    expected = 1.0 / (1.0 / (cfl * float(limits.wave_dt)) + 1.0 / source_limit)
    np.testing.assert_allclose(limits.dt, expected, rtol=2e-15)
    assert float(limits.dt) <= source_limit
    np.testing.assert_allclose(
        operator.compute_dt(model.initial_state(), cfl), limits.dt
    )


@pytest.mark.parametrize("integrator", ["euler-explicit", "runge-kutta-3"])
def test_transient_driver_does_not_multiply_absolute_source_cap_by_cfl(integrator):
    model = _UserAdvection(0.7)
    source_limit = 2e-4
    model.explicit_timestep = lambda full: jnp.asarray(source_limit)
    (final,) = solve_transient(
        model, (7e-4,), Scheme(flux="hll", time_integrator=integrator), cfl=3.0
    )
    assert final.success, final.reason
    accepted = [row for row in final.history if row["accepted"]]
    assert len(accepted) == 4
    assert all(row["dt"] <= source_limit for row in accepted)
    assert np.max(np.abs(final.metrics["conservation_budget"])) < 1e-12


@pytest.mark.parametrize("bad_limit", [0.0, -1.0, np.nan])
def test_invalid_explicit_source_bounds_are_rejected_and_implicit_does_not_apply_them(
    bad_limit,
):
    model = _UserAdvection(0.7)
    model.explicit_timestep = lambda full: jnp.asarray(bad_limit)
    explicit = Operator(model, Scheme(flux="hll", time_integrator="euler-explicit"))
    limits = explicit.timestep(model.initial_state(), 0.4)
    assert not bool(limits.valid)
    assert np.isnan(limits.dt)
    implicit = Operator(model, Scheme(flux="hll", time_integrator="euler-implicit"))
    limits = implicit.timestep(model.initial_state(), 3.0)
    assert bool(limits.valid)
    assert np.isinf(limits.source_dt)
    np.testing.assert_allclose(limits.dt, 3.0 * limits.wave_dt)


def test_infinite_source_bound_leaves_wave_cfl_unchanged():
    model = _UserAdvection(0.7)
    model.explicit_timestep = lambda full: jnp.asarray(jnp.inf)
    operator = Operator(model, Scheme(flux="hll", time_integrator="euler-explicit"))
    limits = operator.timestep(model.initial_state(), 0.4)
    assert bool(limits.valid)
    np.testing.assert_allclose(limits.dt, 0.4 * limits.wave_dt)


def test_zero_signal_speed_and_infinite_source_bound_are_safe():
    model = _UserFilteredAdvection(0.0)
    model.boundary_residual = lambda full, time: jnp.stack(
        (full[0] - full[1], full[-1] - full[-2])
    )
    model.explicit_timestep = lambda full: jnp.asarray(jnp.inf)
    operator = Operator(
        model, Scheme(flux="hll", time_integrator="euler-explicit", filter_strength=1.0)
    )
    limits = operator.timestep(model.initial_state(), 0.4)
    assert bool(limits.valid) and np.isfinite(limits.dt) and limits.dt > 0
    (final,) = solve_transient(model, (0.03,), operator.scheme)
    assert final.success
    np.testing.assert_array_equal(final.state[1:-1], model.initial_state()[1:-1])
