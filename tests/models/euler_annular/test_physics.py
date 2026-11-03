"""Independent mathematical contracts for general-EOS annular Euler kernels."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_annular.boundaries import make_boundary_residual
from nozzle_hyperbolic.models.euler_annular.equations import (
    physical_flux,
    transverse_flux,
)
from nozzle_hyperbolic.models.euler_annular.flux_jax import (
    conservative_projectors,
    numerical_flux,
)
from nozzle_hyperbolic.models.euler_annular.initialization import uniform_initial_state
from nozzle_hyperbolic.models.euler_annular.sources import integrated_source
from nozzle_hyperbolic.models.euler_annular.state import (
    admissible,
    conservative_to_primitive,
    primitive_fields,
    primitive_to_conservative,
    thermodynamic_state,
    transverse_conservative,
)
from nozzle_hyperbolic.models.euler_nozzle.flux_jax import _jax_F_fvs_stadtke
from nozzle_hyperbolic.thermodynamics.adapters import HostFluidJAXAdapter


class RestrictedHPFluid:
    """Independent stiffened-gas fixture exposing only the declared EOS fields.

    Pressure and energy reference shifts expose numerical ideal-gas shortcuts.
    No public heat-capacity ratio or gas-constant property is available.
    """

    def __init__(self, pressure_offset=0.0, enthalpy_shift=0.0):
        self.pressure_offset = pressure_offset
        self.enthalpy_shift = enthalpy_shift

    def get_state(self, pair, first, second):
        if pair == jxp.HmassP_INPUTS:
            h, p = first, second
        elif pair == jxp.DmassUmass_INPUTS:
            rho, e = first, second
            p = 0.4 * rho * (e - self.enthalpy_shift) - 1.4 * self.pressure_offset
            h = e + p / rho
        else:
            raise AssertionError("The numerical boundary may only evaluate the HP pair")
        effective_h = h - self.enthalpy_shift
        rho = 3.5 * (p + self.pressure_offset) / effective_h
        temperature = effective_h / 1004.5
        return dict(
            rho=rho,
            a=jnp.sqrt(0.4 * effective_h),
            G=jnp.full_like(p, 0.4),
            T=temperature,
            s=1004.5 * jnp.log(temperature / 300)
            - 287.0 * jnp.log((p + self.pressure_offset) / 1e5),
            p=p,
            h=h,
        )


@pytest.mark.parametrize("offset,shift", [(0.0, 0.0), (1e5, -5e5)])
def test_general_eos_storage_round_trip_and_reference_shift_admissibility(
    offset, shift
):
    fluid = RestrictedHPFluid(offset, shift)
    radius = jnp.asarray([0.10, 0.17, 0.23])
    W = jnp.asarray(
        [
            [1e5, 20.0, 3.0, 3e5 + shift],
            [1.2e5, 300.0, -4.0, 3.2e5 + shift],
            [0.8e5, 600.0, 7.0, 3.4e5 + shift],
        ]
    )
    q = primitive_to_conservative(W, radius, fluid)
    np.testing.assert_allclose(
        conservative_to_primitive(q, radius, fluid), W, rtol=2e-14, atol=2e-10
    )
    assert np.all(admissible(W, radius, fluid))
    fields = primitive_fields(W, radius, fluid)
    np.testing.assert_allclose(q[:, 2], fields["rho"] * W[:, 2], rtol=1e-14)
    np.testing.assert_allclose(q[:, 3], fields["rho"] * fields["E"], rtol=1e-14)


def test_setup_conservative_recovery_supports_an_hp_only_eos_and_reports_bad_brackets():
    class HPOnlyFluid(RestrictedHPFluid):
        def get_state(self, pair, first, second):
            if pair != jxp.HmassP_INPUTS:
                raise NotImplementedError("Only HP states are available")
            return super().get_state(pair, first, second)

    fluid = HPOnlyFluid(1e5, -5e5)
    W = jnp.asarray([1.3e5, 120.0, 7.0, -1.8e5])
    q = primitive_to_conservative(W, 0.2, fluid)
    actual = conservative_to_primitive(q, 0.2, fluid, pressure_bounds=(1e4, 1e6))
    np.testing.assert_allclose(actual, W, rtol=2e-12, atol=2e-7)
    with pytest.raises(ValueError):
        conservative_to_primitive(q, 0.2, fluid, pressure_bounds=(1e4, 2e4))


def test_setup_conservative_recovery_falls_back_after_wrapped_host_du_returns_nan():
    class HostHPOnlyPerfectGas:
        name = "HP-only perfect-gas recovery fixture"

        def __init__(self):
            self.native = jxp.FluidPerfectGas(name="Air")

        def get_state(self, pair, first, second):
            if pair != jxp.HmassP_INPUTS:
                raise ValueError("This host backend supports only HP states")
            return self.native.get_state(pair, float(first), float(second))

    fluid = HostFluidJAXAdapter(HostHPOnlyPerfectGas())
    radii = jnp.asarray([0.13, 0.2])
    W = jnp.asarray([[1.3e5, 120.0, 7.0, 3.2e5], [0.8e5, -30.0, -2.0, 2.8e5]])
    q = primitive_to_conservative(W, radii, fluid)
    unsupported = fluid.get_state(
        jxp.DmassUmass_INPUTS,
        q[:, 0],
        q[:, 3] / q[:, 0] - 0.5 * (W[:, 1] ** 2 + (W[:, 2] / radii) ** 2),
    )
    assert np.all(np.isnan(unsupported["p"]))
    actual = conservative_to_primitive(q, radii, fluid, pressure_bounds=(1e4, 1e6))
    np.testing.assert_allclose(actual, W, rtol=2e-12, atol=2e-7)
    np.testing.assert_allclose(
        primitive_to_conservative(actual, radii, fluid), q, rtol=2e-12, atol=2e-7
    )
    with pytest.raises(ValueError, match="explicit pressure_bounds or pressure_guess"):
        conservative_to_primitive(q, radii, fluid)


def test_native_du_recovery_preserves_jit_and_inverse_forward_reverse_ad():
    fluid = RestrictedHPFluid(pressure_offset=1e5, enthalpy_shift=-5e5)
    radius = 0.17
    W = jnp.asarray([1.2e5, 120.0, -4.0, -1.8e5])
    q = primitive_to_conservative(W, radius, fluid)
    recover = lambda conservative: conservative_to_primitive(
        conservative, radius, fluid
    )
    np.testing.assert_allclose(jax.jit(recover)(q), W, rtol=2e-14, atol=2e-10)
    forward = jax.jit(jax.jacfwd(recover))(q)
    reverse = jax.jit(jax.jacrev(recover))(q)
    np.testing.assert_allclose(forward, reverse, rtol=2e-13, atol=2e-10)
    storage_jacobian = jax.jacfwd(
        lambda primitive: primitive_to_conservative(primitive, radius, fluid)
    )(W)
    np.testing.assert_allclose(forward @ storage_jacobian, np.eye(4), atol=2e-10)


@pytest.mark.parametrize("v_m", [0.0, np.sqrt(1.2e5), -np.sqrt(1.2e5), 700.0])
def test_four_projectors_resolve_the_independent_flux_jacobian(v_m):
    fluid = RestrictedHPFluid(1e5, -5e5)
    radius = 0.17
    W = jnp.asarray([1e5, v_m, 23.0, -2e5])
    state = thermodynamic_state(W, fluid)
    projectors = np.asarray(
        conservative_projectors(W, radius, state["rho"], state["a"], state["G"])
    )
    storage_jacobian = np.asarray(
        jax.jacfwd(lambda x: transverse_conservative(x, radius, fluid))(W)
    )
    flux_jacobian = np.asarray(
        jax.jacfwd(lambda x: transverse_flux(x, radius, fluid))(W)
    )
    operator = flux_jacobian @ np.linalg.inv(storage_jacobian)
    eigenvalues = [v_m + float(state["a"]), v_m - float(state["a"]), v_m, v_m]
    scale = np.asarray([1.0, 400.0, 400.0, 400.0**2])
    normalize = lambda matrix: matrix * scale[None, :] / scale[:, None]
    np.testing.assert_allclose(normalize(projectors.sum(axis=0)), np.eye(4), atol=1e-12)
    for i in range(4):
        np.testing.assert_allclose(
            normalize(operator @ projectors[i] - eigenvalues[i] * projectors[i]),
            0,
            atol=2e-9,
        )
        for j in range(4):
            expected = projectors[i] if i == j else 0.0
            np.testing.assert_allclose(
                normalize(projectors[i] @ projectors[j] - expected), 0, atol=2e-12
            )


@pytest.mark.parametrize("method", ["stadtke", "rusanov"])
def test_uniform_flux_consistency_uses_one_face_radius_and_area(method):
    fluid = RestrictedHPFluid(1e5)
    W = jnp.asarray(
        [[1e5, 0.0, 8.0, 3e5], [1.1e5, 340.0, -4.0, 3.2e5], [0.9e5, -500.0, 6.0, 3.4e5]]
    )
    radius, area = jnp.asarray([0.1, 0.2, 0.3]), jnp.asarray([0.003, 0.004, 0.005])
    actual = numerical_flux(W, W, radius, area, fluid, method=method)
    np.testing.assert_allclose(
        actual, physical_flux(W, radius, fluid) * area[:, None], rtol=3e-13, atol=3e-10
    )
    np.testing.assert_allclose(actual[:, 2], actual[:, 0] * W[:, 2], rtol=3e-13)


def test_zero_swirl_reduces_to_the_existing_three_balance_flux():
    fluid = RestrictedHPFluid(1e5)
    left = jnp.asarray([[1e5, 0.0, 0.0, 3e5], [1.1e5, 380.0, 0.0, 3.2e5]])
    right = jnp.asarray([[0.9e5, 20.0, 0.0, 2.9e5], [1e5, 410.0, 0.0, 3.1e5]])
    annular = numerical_flux(left, right, 0.2, 1.0, fluid)
    nozzle = _jax_F_fvs_stadtke(left[:, [0, 1, 3]], right[:, [0, 1, 3]], fluid)
    np.testing.assert_allclose(annular[:, [0, 1, 3]], nozzle, rtol=2e-12, atol=2e-8)
    np.testing.assert_array_equal(annular[:, 2], 0.0)


def test_batched_characteristics_accept_scalar_eos_constants():
    class ScalarGFluid(RestrictedHPFluid):
        def get_state(self, pair, first, second):
            state = super().get_state(pair, first, second)
            state["G"] = jnp.asarray(0.4)
            return state

    fluid = ScalarGFluid()
    W = jnp.asarray([[1e5, 50.0, 8.0, 3e5], [1.1e5, 340.0, -4.0, 3.2e5]])
    np.testing.assert_allclose(
        numerical_flux(W, W, 0.2, 1.0, fluid),
        physical_flux(W, 0.2, fluid),
        rtol=3e-13,
        atol=3e-10,
    )


def test_characteristic_flux_has_finite_consistent_forward_and_reverse_ad():
    fluid = RestrictedHPFluid(1e5)
    sound, radius = np.sqrt(1.2e5), 0.2
    scales = jnp.asarray([1e5, sound, radius * sound, 3e5])
    center = jnp.asarray([1e5, sound, 16.0, 3e5])

    def scaled_flux(x):
        left, right = center + x[:4] * scales, center + x[4:] * scales
        return numerical_flux(left, right, radius, 1.0, fluid) / jnp.asarray(
            [1e3, 1e5, 1e4, 1e8]
        )

    x = jnp.zeros(8)
    forward = jax.jit(jax.jacfwd(scaled_flux))(x)
    reverse = jax.jit(jax.jacrev(scaled_flux))(x)
    assert np.all(np.isfinite(forward))
    np.testing.assert_allclose(forward, reverse, rtol=2e-11, atol=2e-11)


def test_pressure_area_source_preserves_rest_and_angular_energy_have_no_source():
    fluid = RestrictedHPFluid()
    area = jnp.asarray([0.003, 0.004, 0.0035, 0.006])
    W = jnp.tile(jnp.asarray([1e5, 0.0, 0.0, 3e5]), (3, 1))
    radius = jnp.asarray([0.1, 0.13, 0.17])
    source = integrated_source(
        W, radius, jnp.ones(3), jnp.diff(area), jnp.ones(3) * 1e-4, fluid
    )
    traces = jnp.tile(W[0], (4, 1))
    flux = numerical_flux(traces, traces, jnp.linspace(0.08, 0.2, 4), area, fluid)
    np.testing.assert_allclose(flux[1:] - flux[:-1] - source, 0, atol=2e-13)
    moving = W.at[:, 1].set(50.0).at[:, 2].set(6.0)
    outward = integrated_source(
        moving, radius, jnp.ones(3), jnp.diff(area), jnp.ones(3) * 1e-4, fluid
    )
    inward = integrated_source(
        moving, radius, -jnp.ones(3), jnp.diff(area), jnp.ones(3) * 1e-4, fluid
    )
    np.testing.assert_array_equal(outward[:, [0, 2, 3]], 0.0)
    np.testing.assert_allclose(
        outward[:, 1] + inward[:, 1], 2 * source[:, 1], atol=2e-12
    )


def boundary_fixture(outlet_regime="subsonic", outlet_pressure=1e5):
    fluid = RestrictedHPFluid(1e5)
    radius = 0.2
    W = jnp.asarray([1e5, 40.0, 4.0, 3e5])
    state = thermodynamic_state(W, fluid)
    data = dict(
        h0=float(W[3] + 0.5 * (W[1] ** 2 + (W[2] / radius) ** 2)),
        s0=float(state["s"]),
        alpha_in=float(np.arctan2(W[2] / radius, W[1])),
    )
    scales = dict(p=1e5, a=400.0, ell=80.0, h=1.6e5, s=500.0)
    residual = make_boundary_residual(
        fluid,
        [0.02, 0.08, 0.17],
        [0, 0.04, 0.12, 0.22],
        [radius] * 4,
        data,
        outlet_pressure,
        scales,
        outlet_regime=outlet_regime,
    )
    return residual, jnp.tile(W, (5, 1))


def test_coupled_boundary_constraints_are_zero_on_uniform_current_state():
    residual, z = boundary_fixture()
    np.testing.assert_allclose(jax.jit(residual)(z), 0, atol=3e-15)
    derivative = jax.jacfwd(residual)(z)
    assert np.all(np.isfinite(derivative))
    assert np.linalg.norm(np.asarray(derivative[0, :, 0, :])) > 0
    assert np.linalg.norm(np.asarray(derivative[0, :, 2, :])) > 0
    assert np.linalg.norm(np.asarray(derivative[1, :, -1, :])) > 0
    assert np.linalg.norm(np.asarray(derivative[1, :, -3, :])) > 0
    changed = z.at[0, 0].multiply(1.01)
    assert np.linalg.norm(np.asarray(residual(changed)[0])) > 1e-4


def test_fixed_angle_allows_changing_inlet_angular_momentum():
    residual, z = boundary_fixture()
    changed = z.at[0, 1].multiply(1.2).at[0, 2].multiply(1.2)
    assert float(changed[0, 2]) != float(z[0, 2])
    np.testing.assert_allclose(residual(changed)[0, 2], 0, atol=1e-15)


def test_supersonic_outlet_rows_do_not_depend_on_backpressure():
    first, z = boundary_fixture("supersonic", 1e5)
    second, _ = boundary_fixture("supersonic", 8e5)
    np.testing.assert_array_equal(first(z), second(z))


def test_uniform_initial_guess_uses_physical_enthalpy_and_named_velocity_components():
    radii = jnp.asarray([0.10, 0.15, 0.20])
    state = uniform_initial_state(radii, 1e5, -2e5, 20.0, alpha_in=0.3)
    np.testing.assert_array_equal(state[:, 3], -2e5)
    np.testing.assert_allclose(state[:, 2] / radii, 20 * np.tan(0.3), rtol=1e-14)


def test_jaxprop_perfect_gas_uses_the_same_batched_general_eos_kernels():
    fluid = jxp.FluidPerfectGas(name="Air")
    radii = jnp.asarray([0.10, 0.20])
    W = jnp.asarray([[1e5, 40.0, 6.0, 3e5], [1.1e5, 100.0, 4.0, 3.2e5]])
    actual = jax.jit(lambda states: numerical_flux(states, states, radii, 1.0, fluid))(
        W
    )
    np.testing.assert_allclose(
        actual, physical_flux(W, radii, fluid), rtol=3e-13, atol=3e-9
    )
    q = primitive_to_conservative(W, radii, fluid)
    recovered = conservative_to_primitive(q, radii, fluid, pressure_guess=W[:, 0])
    np.testing.assert_allclose(recovered, W, rtol=2e-12, atol=2e-7)


def test_admissibility_rejects_a_silently_floored_backend_without_enthalpy_sign_assumptions():
    fluid = jxp.FluidPerfectGas(name="Air")
    state = jnp.asarray([1e5, 40.0, 4.0, -1e5])
    raw = thermodynamic_state(state, fluid)
    assert float(raw["T"]) > 0 and float(raw["rho"]) > 0 and float(raw["a"]) > 0
    assert not bool(jax.jit(lambda x: admissible(x, 0.2, fluid))(state))
    shifted = RestrictedHPFluid(enthalpy_shift=-5e5)
    assert bool(admissible(state, 0.2, shifted))


def test_admissibility_rejects_returned_coordinates_that_do_not_satisfy_the_trial_hp_pair():
    class InconsistentCoordinates(RestrictedHPFluid):
        def get_state(self, pair, first, second):
            state = super().get_state(pair, first, second)
            state["h"] = state["h"] + 1e3
            return state

    assert not bool(
        admissible(jnp.asarray([1e5, 40.0, 4.0, 3e5]), 0.2, InconsistentCoordinates())
    )


def test_declared_thermodynamic_bounds_are_checked_before_the_eos_call():
    def unexpected_evaluation(_):
        raise AssertionError("An out-of-domain state must not reach the backend")

    class BoundedFluid(RestrictedHPFluid):
        h_min = 1e5
        h_max = 4e5

        def get_state(self, pair, first, second):
            jax.debug.callback(unexpected_evaluation, first)
            return super().get_state(pair, first, second)

    fluid = BoundedFluid()
    state = jnp.asarray([1e5, 40.0, 4.0, 5e4])
    assert not bool(jax.jit(lambda x: admissible(x, 0.2, fluid))(state))
