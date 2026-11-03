"""Osher path quadrature, sonic roots and binding contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar.advection import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme

from tests._support.osher_models import CONFIGURATIONS, sonic_burgers, LinearBalance


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_operator_configuration_jit_outer_ad_and_conservation(
    quadrature, integration, reconstruction
):
    model = LinearBalance()
    operator = Operator(
        model,
        Scheme(
            flux="osher",
            quadrature_points=quadrature,
            integration=integration,
            reconstruction=reconstruction,
        ),
    )
    state = jnp.asarray([[0.3], [0.45], [0.72], [1.0], [1.4], [1.7]])
    left, _ = operator.reconstruct(state)
    np.testing.assert_allclose(operator.face_flux(state), left, rtol=3e-14, atol=3e-14)
    derivative = jax.jit(jax.jacfwd(operator.face_flux))(state)
    assert np.isfinite(np.asarray(derivative)).all()
    direction = jnp.linspace(-0.1, 0.2, state.size).reshape(state.shape)
    step = 1e-6
    fd = (
        operator.face_flux(state + step * direction)
        - operator.face_flux(state - step * direction)
    ) / (2 * step)
    ad = jnp.einsum("aibj,bj->ai", derivative, direction)
    np.testing.assert_allclose(ad, fd, rtol=2e-8, atol=2e-10)
    flux = np.asarray(operator.face_flux(state))
    raw = np.asarray(operator.spatial_residual(state))
    np.testing.assert_allclose(raw.sum(axis=0), flux[-1] - flux[0], atol=3e-14)


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_operator_requires_characteristic_model_capability(quadrature, integration):
    with pytest.raises(ValueError, match="characteristic"):
        Operator(
            AdvectionModel(4),
            Scheme(flux="osher", quadrature_points=quadrature, integration=integration),
        )


def test_operator_path_is_shape_checked_at_binding():
    model = LinearBalance()
    model.face_state_path = lambda left, right, parameter: (left[:, 0], right[:, 0])
    with pytest.raises(ValueError, match="face_state_path"):
        Operator(model, Scheme(flux="osher"))


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_operator_rejects_model_invalid_sampled_path(quadrature, integration):
    class PositivePath(LinearBalance):
        def face_state_path(self, left, right, parameter):
            return (
                left + parameter * (right - left) - 8 * parameter * (1 - parameter),
                right - left - 8 * (1 - 2 * parameter),
            )

        def admissible(self, full, left, right):
            return (
                super().admissible(full, left, right)
                & jnp.all(left > 0)
                & jnp.all(right > 0)
            )

    model = PositivePath()
    operator = Operator(
        model,
        Scheme(flux="osher", quadrature_points=quadrature, integration=integration),
    )
    assert not bool(operator.is_admissible(jnp.ones((6, 1))))


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_operator_cfl_bounds_regularized_viscosity(quadrature, integration):
    model = LinearBalance(width=4.0)
    operator = Operator(
        model,
        Scheme(
            flux="osher",
            quadrature_points=quadrature,
            integration=integration,
            time_integrator="euler-explicit",
        ),
    )
    info = operator.timestep(model.initial_state(), 0.4)
    expected = np.min(model.mesh.widths) / (17.0 / 8.0)
    assert bool(info.valid)
    np.testing.assert_allclose(info.wave_dt, expected, rtol=2e-15)
    np.testing.assert_allclose(info.dt, 0.4 * expected, rtol=2e-15)


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_operator_cfl_keeps_endpoint_physical_bounds(quadrature, integration):
    model = LinearBalance(width=4.0, physical_face_bound=3.0)
    operator = Operator(
        model,
        Scheme(
            flux="osher",
            quadrature_points=quadrature,
            integration=integration,
            time_integrator="euler-explicit",
        ),
    )
    info = operator.timestep(model.initial_state(), 0.4)
    expected = np.min(model.mesh.widths) / 3.0
    assert bool(info.valid)
    np.testing.assert_allclose(info.wave_dt, expected, rtol=2e-15)
    np.testing.assert_allclose(info.dt, 0.4 * expected, rtol=2e-15)


@pytest.mark.parametrize("flux", ["hll", "rusanov"])
def test_other_fluxes_ignore_osher_options(flux):
    model = AdvectionModel(4)
    default = Operator(model, Scheme(flux=flux))
    custom = Operator(
        model, Scheme(flux=flux, quadrature_points=9, integration="sonic_partitioned")
    )
    np.testing.assert_array_equal(
        default.face_flux(model.initial_state()),
        custom.face_flux(model.initial_state()),
    )
