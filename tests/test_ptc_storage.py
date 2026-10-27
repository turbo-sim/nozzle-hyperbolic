"""Physical mass scaling, locality, and characteristic times for steady PTC."""

from copy import copy
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from development_new.newton_solve.scripts.ptc_storage import PhysicalStorage
from nozzle_hyperbolic.models.euler_nozzle.fvm import make_jax_stadtke_step
from tests.test_euler_steady_operator import _factory_inputs


@pytest.fixture(scope="module")
def problem():
    """Use the actual 13-node operator without an analytic reference solve."""
    arguments, options, varying, _ = _factory_inputs("muscl_van_leer")
    fluid, area, _, _, _, _, dx, cfl, _, h0, _ = arguments
    step, compute_dt, _, _, operators = make_jax_stadtke_step(
        *arguments, **options, return_operators=True
    )
    stagnation = fluid.get_state(jxp.PT_INPUTS, 2e5, 300.0)
    a0, rho0 = float(stagnation["a"]), float(stagnation["rho"])
    variable_scale = np.tile([2e5, a0, h0], len(area) - 2)
    residual_scale = (
        (area[1:-1] * dx)[:, None] * (rho0 * np.array([1.0, a0, h0]))[None, :] * a0 / dx
    ).ravel()
    x0 = np.asarray(varying)[1:-1].ravel() / variable_scale
    result = SimpleNamespace(
        fluid=fluid,
        operators=operators,
        variable_scale=variable_scale,
        residual_scale=residual_scale,
        settings=dict(CFL=cfl),
        dx=dx,
        x0=x0,
        compute_dt=compute_dt,
        step=step,
    )
    result._validate = lambda point: np.asarray(point, dtype=float)
    result.full_state = lambda point: operators["state_from_interior"](
        jnp.asarray(point * variable_scale)
    )
    return result


@pytest.fixture(scope="module")
def storage(problem):
    return PhysicalStorage(problem)


def _finite_difference_matrix(adapter, point):
    columns = []
    for index in range(point.size):
        delta = np.zeros(point.size)
        delta[index] = 1e-6 * max(1.0, abs(point[index]))
        columns.append(
            (
                adapter.conservative_storage(point + delta)
                - adapter.conservative_storage(point - delta)
            )
            / (2.0 * delta[index])
        )
    return np.column_stack(columns)


def test_storage_matrix_matches_independent_finite_differences(problem, storage):
    expected = _finite_difference_matrix(storage, problem.x0)
    actual = storage.matrix(problem.x0)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-13)


def test_storage_matrix_matches_production_transient_storage_ad(problem, storage):
    """Compare against the actual conservative path used at implicit dt=0."""
    volumes = jnp.asarray(problem.operators["cell_volumes"])[1:-1, None]
    residual_scale = jnp.asarray(problem.residual_scale.reshape(-1, 3))

    @jax.jit
    def production_storage(point):
        full = problem.full_state(point)
        conservative = problem.operators["conservative_state"](full)
        return (volumes * conservative[1:-1] / residual_scale).ravel()

    expected = np.asarray(
        jax.jit(jax.jacfwd(production_storage))(jnp.asarray(problem.x0))
    )
    actual = storage.matrix(problem.x0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        storage.conservative_storage(problem.x0),
        production_storage(problem.x0),
        rtol=1e-14,
    )


def test_storage_is_local_and_not_an_identity_shift(problem, storage):
    actual = storage.matrix(problem.x0)
    indices = np.arange(problem.x0.size)
    off_block = indices[:, None] // 3 != indices[None, :] // 3
    np.testing.assert_array_equal(
        actual[off_block], np.zeros(np.count_nonzero(off_block))
    )
    assert np.isfinite(actual).all()
    assert not np.allclose(actual, np.diag(np.diag(actual)), rtol=0.0, atol=1e-15)


def test_storage_uses_eos_ad_instead_of_perfect_gas_specific_derivatives(problem):
    """A pressure-offset EOS changes d(rho)/dw and must change the matrix."""

    class ShiftedGas:
        def get_state(self, _pair, enthalpy, pressure):
            return {"rho": 1.4 * (pressure + 2e5) / (0.4 * enthalpy)}

    other = copy(problem)
    other.fluid = ShiftedGas()
    adapter = PhysicalStorage(other)
    actual = adapter.matrix(other.x0)
    expected = _finite_difference_matrix(adapter, other.x0)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-13)
    assert not np.allclose(
        actual, PhysicalStorage(problem).matrix(problem.x0), rtol=1e-5, atol=1e-15
    )


def test_characteristic_time_uses_requested_cfl_and_existing_kernel(problem, storage):
    state = problem.full_state(problem.x0)
    baseline_dt = float(problem.compute_dt(state)[0])
    for cfl in (0.5, 1.0, 7.0):
        values = storage.wave_timescales(problem.x0, cfl)
        expected = baseline_dt * cfl / problem.settings["CFL"]
        np.testing.assert_allclose(values["global_dt"], expected, rtol=1e-14)
        np.testing.assert_allclose(np.min(values["local_dt"]), expected, rtol=1e-14)
        assert values["local_dt"].shape == (len(state) - 2,)
    for cfl in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="CFL"):
            storage.characteristic_dt(problem.x0, cfl)


def test_warmup_and_counters_are_independent_of_spatial_calls(problem, storage):
    costs = storage.warm_up()
    assert all(np.isfinite(value) and value >= 0.0 for value in costs.values())
    assert all(value == 0 for value in storage.model_work().values())
    storage.matrix(problem.x0)
    storage.conservative_storage(problem.x0)
    storage.characteristic_dt(problem.x0)
    work = storage.model_work()
    assert work["matrix_calls"] == work["storage_calls"] == work["timescale_calls"] == 1
