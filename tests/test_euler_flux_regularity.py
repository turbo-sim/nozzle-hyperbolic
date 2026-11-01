"""Consistency and differentiability contracts for the regularized Euler flux."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_nozzle.flux_jax import (
    _jax_F_fvs_stadtke,
    _jax_F_fvs_stadtke_legacy,
    _jax_stadtke_flux_with_entropy_viscosity,
    _jax_harten_absolute,
)


class ThermodynamicGas:
    """Ideal/stiffened gas with the general-EOS density derivative identities."""

    gamma = 1.4
    cp = 1004.5

    def __init__(self, pressure_offset=0.0):
        self.pressure_offset = pressure_offset

    def get_state(self, _inputs, h, p):
        """Return differentiable rho, a, and Gruneisen for (h,p)."""
        gruneisen = jnp.full_like(p, self.gamma - 1)
        rho = self.gamma * (p + self.pressure_offset) / (gruneisen * h)
        return {"rho": rho, "a": jnp.sqrt(gruneisen * h), "G": gruneisen}


FLUID = ThermodynamicGas()
PRESSURE = 1.0e5
ENTHALPY = FLUID.cp * 300.0
SOUND = np.sqrt((FLUID.gamma - 1) * ENTHALPY)
DENSITY = FLUID.gamma * PRESSURE / ((FLUID.gamma - 1) * ENTHALPY)
FLUX_SCALE = DENSITY * SOUND * np.array([1.0, SOUND, ENTHALPY])


def physical_flux(states, fluid=FLUID):
    """Compute the reference physical flux independently of its splitting."""
    p, u, h = states[:, 0], states[:, 1], states[:, 2]
    density = fluid.get_state(None, h, p)["rho"]
    return jnp.stack(
        [density * u, density * u**2 + p, density * u * (h + 0.5 * u**2)], axis=-1
    )


def state(mach, pressure=PRESSURE, enthalpy=ENTHALPY):
    """Return one primitive face state with the specified local Mach number."""
    sound = np.sqrt((FLUID.gamma - 1) * enthalpy)
    return np.array([pressure, mach * sound, enthalpy])


@pytest.mark.parametrize("pressure_offset", [0.0, 1.0e5])
def test_uniform_flux_is_consistent_at_rest_sonic_and_reverse_flow(pressure_offset):
    """A sonic floor must preserve physical flux for ideal and general EOS."""
    fluid = ThermodynamicGas(pressure_offset)
    mach = np.array(
        [-2, -1.000001, -1, -0.999999, -0.1, 0, 1e-8, 0.1, 0.999999, 1, 1.000001, 2]
    )
    states = jnp.asarray(np.array([state(value) for value in mach]))
    evaluate = jax.jit(lambda left, right: _jax_F_fvs_stadtke(left, right, fluid, 1e-6))
    actual = np.asarray(evaluate(states, states))
    expected = np.asarray(physical_flux(states, fluid))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual / FLUX_SCALE, expected / FLUX_SCALE, atol=1e-13, rtol=1e-13
    )


def test_entropy_activation_threshold_has_no_finite_flux_jump():
    """The legacy 20% projector jump must vanish under shrinking perturbations."""
    right = jnp.asarray(state(1.1)[None, :])
    evaluate = jax.jit(lambda left: _jax_F_fvs_stadtke(left, right, FLUID, 1e-6))
    differences = []
    for epsilon in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        minus = jnp.asarray(state(1 - epsilon)[None, :])
        plus = jnp.asarray(state(1 + epsilon)[None, :])
        difference = np.linalg.norm(
            (np.asarray(evaluate(plus)) - np.asarray(evaluate(minus))) / FLUX_SCALE
        )
        differences.append(difference)
    assert np.all(np.array(differences[1:]) < 0.2 * np.array(differences[:-1]))
    assert differences[-1] < 1e-7
    minus = jnp.asarray(state(1 - 1e-9)[None, :])
    plus = jnp.asarray(state(1 + 1e-9)[None, :])
    legacy_jump = np.linalg.norm(
        (
            np.asarray(_jax_F_fvs_stadtke_legacy(plus, right, FLUID, 1e-6))
            - np.asarray(_jax_F_fvs_stadtke_legacy(minus, right, FLUID, 1e-6))
        )
        / FLUX_SCALE
    )
    assert legacy_jump > 1e-3


@pytest.mark.parametrize("mach", [-1.0, 0.0, 1.0])
def test_jit_forward_and_reverse_ad_are_finite_and_agree_at_singular_legacy_states(
    mach,
):
    """Both AD modes must differentiate the same finite sonic/rest flux."""
    variable_scale = jnp.asarray([PRESSURE, SOUND, ENTHALPY] * 2)

    def scaled_flux(z):
        primitive = (z * variable_scale).reshape(2, 3)
        flux = _jax_F_fvs_stadtke(primitive[:1], primitive[1:], FLUID, 1e-6)
        return flux[0] / jnp.asarray(FLUX_SCALE)

    z = jnp.asarray(np.tile(state(mach), 2)) / variable_scale
    forward = np.asarray(jax.jit(jax.jacfwd(scaled_flux))(z))
    reverse = np.asarray(jax.jit(jax.jacrev(scaled_flux))(z))
    assert np.isfinite(forward).all()
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, reverse, atol=1e-11, rtol=1e-11)
    direction = np.array([0.1, -0.2, 0.15, -0.15, 0.12, -0.1])
    epsilon = 1e-8
    fd = np.asarray(
        (scaled_flux(z + epsilon * direction) - scaled_flux(z - epsilon * direction))
        / (2 * epsilon)
    )
    error = np.linalg.norm(forward @ direction - fd) / np.linalg.norm(
        forward @ direction
    )
    assert error < 2e-5


def test_velocity_reflection_and_left_right_reversal_symmetry():
    """Reflecting an interface reverses mass/energy but preserves momentum flux."""
    left = np.array([state(0.95), state(0), state(1.8)])
    right = np.array(
        [
            state(1.1, pressure=1.1e5, enthalpy=0.95 * ENTHALPY),
            state(0.2),
            state(2.0, pressure=0.9e5),
        ]
    )
    actual = np.asarray(
        _jax_F_fvs_stadtke(jnp.asarray(left), jnp.asarray(right), FLUID, 1e-6)
    )
    reflected_left, reflected_right = right.copy(), left.copy()
    reflected_left[:, 1] *= -1
    reflected_right[:, 1] *= -1
    reflected = np.asarray(
        _jax_F_fvs_stadtke(
            jnp.asarray(reflected_left), jnp.asarray(reflected_right), FLUID, 1e-6
        )
    )
    np.testing.assert_allclose(
        reflected / FLUX_SCALE,
        actual * [-1, 1, -1] / FLUX_SCALE,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("direction", [-1, 1])
def test_supersonic_flow_uses_the_upwind_physical_flux(direction):
    """Interfaces whose characteristic speeds lie beyond the bands are upwind."""
    left = jnp.asarray(state(direction * 2.0)[None, :])
    right = jnp.asarray(
        state(direction * 2.2, pressure=0.9e5, enthalpy=0.95 * ENTHALPY)[None, :]
    )
    actual = np.asarray(_jax_F_fvs_stadtke(left, right, FLUID, 1e-6))
    expected = np.asarray(physical_flux(left if direction > 0 else right))
    np.testing.assert_allclose(
        actual / FLUX_SCALE, expected / FLUX_SCALE, rtol=1e-12, atol=1e-12
    )


def test_regularized_flux_matches_legacy_outside_all_sonic_bands():
    """The repair retains the intended characteristic FVS away from sonic flow."""
    left = jnp.asarray([state(0.2), state(2), state(-2)])
    right = jnp.asarray([state(0.25, pressure=0.95e5), state(2.1), state(-2.1)])
    actual = np.asarray(_jax_F_fvs_stadtke(left, right, FLUID, 1e-6))
    legacy = np.asarray(_jax_F_fvs_stadtke_legacy(left, right, FLUID, 1e-6))
    np.testing.assert_allclose(
        actual / FLUX_SCALE, legacy / FLUX_SCALE, atol=1e-12, rtol=1e-12
    )


def test_frozen_linear_characteristic_viscosity_is_harten_absolute():
    """The added conservative jump restores phi, including the zero-speed mode."""
    eigenvalues = jnp.asarray([[-1.0, 0.0, 0.5]])
    width = jnp.asarray([[0.2, 0.4, 1.0]])
    q_left = jnp.asarray([[0.2, -0.3, 0.4]])
    q_right = jnp.asarray([[-0.1, 0.5, 0.7]])
    flux_left, flux_right = eigenvalues * q_left, eigenvalues * q_right
    projectors = jnp.asarray(
        np.array([np.outer(row, row) for row in np.eye(3)])[None, :]
    )
    actual = _jax_stadtke_flux_with_entropy_viscosity(
        flux_left, flux_right, q_right - q_left, projectors, eigenvalues, width
    )
    # These scalar absolute eigenvalues follow independently from the
    # outer |lambda| and inner (lambda^2+delta^2)/(2 delta) definitions.
    expected_absolute = jnp.asarray([[1.0, 0.2, 0.625]])
    expected = 0.5 * (flux_left + flux_right) - 0.5 * expected_absolute * (
        q_right - q_left
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=1e-14, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(_jax_harten_absolute(eigenvalues, width)),
        np.asarray(expected_absolute),
    )
