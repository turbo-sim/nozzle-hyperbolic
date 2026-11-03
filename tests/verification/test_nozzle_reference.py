"""Continuous-reference invariants and published perfect-gas limiting values."""

from types import SimpleNamespace

import jaxprop as jxp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle
from nozzle_hyperbolic.models.euler_nozzle.analytical import (
    PerfectGasReference,
    area_mach,
    mach_from_area_ratio,
    normal_shock,
    nozzle_exact_real_gas,
    sod_exact,
    sod_exact_with_sound_speeds,
    subsonic_reference,
)


def test_area_mach_and_normal_shock_classical_values():
    # gamma=1.4, M1=2: NACA 1135 normal-shock relations and area ratio.
    assert_allclose(area_mach(2.0, 1.4), 1.6875, rtol=1e-14)
    assert_allclose(mach_from_area_ratio(1.6875, 1.4, "sup"), 2.0, atol=1e-13)
    assert mach_from_area_ratio(1.0, 1.4) == 1.0
    p_ratio, m2, total_ratio = normal_shock(2.0, 1.4)
    assert_allclose(p_ratio, 4.5, rtol=1e-14)
    assert_allclose(m2, np.sqrt(1 / 3), rtol=1e-14)
    assert_allclose(total_ratio, 0.7208738614847455, rtol=1e-14)
    with pytest.raises(ValueError, match="branch"):
        mach_from_area_ratio(2.0, 1.4, "typo")


@pytest.mark.parametrize(
    "regime",
    [
        "subsonic",
        "critical_sonic",
        "internal_shock",
        "shock_at_exit",
        "adapted",
        "underexpanded_external",
    ],
)
def test_operating_map_mass_energy_and_regimes(regime):
    reference = PerfectGasReference(
        SimpleNamespace(constants=SimpleNamespace(gamma=1.4, R=287.0))
    )
    pressure_ratios = dict(
        subsonic=0.99,
        critical_sonic=reference.critical,
        internal_shock=reference.shock_data(0.16)[0],
        shock_at_exit=reference.shock_exit,
        adapted=reference.adapted,
        underexpanded_external=reference.adapted / 2,
    )
    fields = reference.profile(np.linspace(0, 0.2, 101), pressure_ratios[regime])
    assert fields["regime"] == regime
    assert_allclose(fields["mdot"], fields["mdot"][0], rtol=2e-10)
    assert_allclose(
        reference.cp * fields["T"] + 0.5 * fields["u"] ** 2,
        reference.cp * reference.T0,
        rtol=1e-14,
    )
    if regime == "internal_shock":
        assert_allclose(fields["shock_x"], 0.16, atol=2e-13)
        assert fields["shock_ds"] > 0
    if regime == "shock_at_exit":
        # Domain endpoint belongs to the pre-shock limit by definition.
        assert_allclose(fields["p"][-1] / reference.p0, reference.adapted)
        assert fields["downstream_exit_pressure"] > fields["p"][-1]


def test_subsonic_virtual_critical_area_and_stagnant_integer_coordinates():
    fluid = jxp.FluidPerfectGas(name="Air")
    reference = PerfectGasReference(fluid)
    area = np.array([3e-4, 1.5e-4, 3e-4])
    fields = subsonic_reference(area, fluid, 0.6, throat_area=1.5e-4, p0=2e5, T0=300.0)
    assert_allclose(fields["M"][1], 0.6, atol=1e-13)
    assert_allclose(fields["p"][[0, 2]], fields["p"][0])
    stagnant = reference.profile(np.array([0]), 1.0)
    assert_allclose(stagnant["p"], 2e5)
    assert_allclose(stagnant["u"], 0.0)


def test_perfect_gas_reference_rejects_incompatible_eos_and_accepts_enthalpy_shift():
    native = jxp.FluidPerfectGas(name="Air")

    class DifferentEOS:
        constants = native.constants

        def get_state(self, pair, pressure, temperature):
            state = native.get_state(pair, pressure, temperature)
            return dict(rho=1.1 * state["rho"], h=state["h"], a=state["a"])

    class ShiftedEnthalpy:
        constants = native.constants

        def get_state(self, pair, pressure, temperature):
            state = native.get_state(pair, pressure, temperature)
            return dict(rho=state["rho"], h=state["h"] - 1e6, a=state["a"])

    with pytest.raises(ValueError, match="inconsistent.*perfect-gas"):
        PerfectGasReference(DifferentEOS())
    with pytest.raises(ValueError, match="inconsistent.*perfect-gas"):
        subsonic_reference(
            [3e-4, 1.5e-4], DifferentEOS(), 0.6, throat_area=1.5e-4, p0=2e5, T0=300.0
        )
    shifted = PerfectGasReference(ShiftedEnthalpy())
    assert_allclose(shifted.mdot_star, PerfectGasReference(native).mdot_star)


@pytest.mark.parametrize(
    "changes",
    [dict(p0=np.nan), dict(T0=np.inf), dict(length=np.nan), dict(inlet_area=np.inf)],
)
def test_perfect_gas_reference_requires_finite_inputs(changes):
    constants = SimpleNamespace(constants=SimpleNamespace(gamma=1.4, R=287.0))
    with pytest.raises(ValueError, match="finite|positive"):
        PerfectGasReference(constants, **changes)


@pytest.mark.parametrize(
    "mach,gamma", [(np.nan, 1.4), (np.inf, 1.4), (2.0, np.nan), (2.0, 1.0)]
)
def test_normal_shock_requires_finite_physical_inputs(mach, gamma):
    with pytest.raises(ValueError, match="finite"):
        normal_shock(mach, gamma)


def test_sod_star_states_and_both_public_apis():
    # Standard nondimensional Sod data; Toro chapter 4 tabulates this test.
    x = np.array([-0.5, -0.01, 0.1, 0.3, 0.5])
    args = (x, 0.2, 0.0, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    rho, velocity, pressure = sod_exact(*args)
    assert_allclose(pressure[1:4], 0.3031301780506468, rtol=2e-10)
    assert_allclose(velocity[1:4], 0.9274526200489499, rtol=2e-10)
    assert_allclose(rho[[1, 2]], 0.4263194281784952, rtol=2e-10)
    assert_allclose(rho[3], 0.2655737117053071, rtol=2e-10)
    supplied = sod_exact_with_sound_speeds(
        *args, np.sqrt(1.4), np.sqrt(1.4 * 0.1 / 0.125)
    )
    for actual, expected in zip(supplied, (rho, velocity, pressure)):
        assert_allclose(actual, expected, rtol=2e-10)


def test_sod_zero_time_integer_input_and_unsupported_pattern():
    args = (np.array([-1, 0, 1]), 0.0, 0.0, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    for result in (
        sod_exact(*args),
        sod_exact_with_sound_speeds(*args, np.sqrt(1.4), np.sqrt(1.12)),
    ):
        assert_allclose(result[0], [1.0, 0.125, 0.125])
        assert_allclose(result[2], [1.0, 0.1, 0.1])
    with pytest.raises(ValueError, match="Sod pattern"):
        sod_exact(np.array([0.0]), 0.1, 0.0, 0.125, 0.0, 0.1, 1.0, 0.0, 1.0, 1.4)


def test_general_eos_reference_perfect_gas_limit_and_shock_conservation():
    fluid = jxp.FluidPerfectGas(name="Air")
    x = np.linspace(0.0, 0.2, 41)
    area = get_symmetric_nozzle(x, L=0.2, A_inlet=3e-4, A_throat=1.5e-4).A
    fields = nozzle_exact_real_gas(x, area, fluid, 2e5, 300.0, x_shock=0.16)
    reference = PerfectGasReference(fluid)
    perfect = reference.profile(x, reference.shock_data(0.16)[0])
    assert_allclose(fields["shock"]["p"], perfect["p"], rtol=4e-8)
    assert_allclose(fields["mdot"], reference.mdot_star, rtol=2e-12)
    assert_allclose(
        fields["shock"]["rho"] * fields["shock"]["u"] * area, fields["mdot"], rtol=3e-10
    )
    assert_allclose(
        fields["shock"]["h"] + 0.5 * fields["shock"]["u"] ** 2, fields["h0"], rtol=1e-14
    )
    pre, post = fields["pre"], fields["post"]
    assert_allclose(pre["rho"] * pre["u"], post["rho"] * post["u"], rtol=1e-14)
    assert_allclose(
        pre["p"] + pre["rho"] * pre["u"] ** 2,
        post["p"] + post["rho"] * post["u"] ** 2,
        rtol=1e-14,
    )
    assert pre["M"] > 1 > post["M"]
    assert post["s"] > pre["s"]


def test_general_eos_reference_shifted_enthalpy_datum_with_explicit_bounds():
    native = jxp.FluidPerfectGas(name="Air")
    shift = -1e6

    class ShiftedEOS:
        def get_state(self, pair, first, second):
            if pair == jxp.HmassP_INPUTS:
                first = np.asarray(first) - shift
            state = native.get_state(pair, first, second)
            return {
                key: np.asarray(state[key]) + (shift if key == "h" else 0.0)
                for key in ("h", "rho", "a", "T", "s")
            }

    x = np.linspace(0.0, 0.2, 9)
    area = get_symmetric_nozzle(x, L=0.2, A_inlet=3e-4, A_throat=1.5e-4).A
    ordinary = nozzle_exact_real_gas(x, area, native, 2e5, 300.0, h_min=4e4)
    shifted = nozzle_exact_real_gas(
        x, area, ShiftedEOS(), 2e5, 300.0, h_min=4e4 + shift
    )
    assert_allclose(shifted["sup"]["p"], ordinary["sup"]["p"], rtol=4e-8)
    assert_allclose(shifted["sup"]["h"] - shift, ordinary["sup"]["h"], rtol=4e-8)


def test_general_eos_bounds_fail_instead_of_returning_clipped_reference():
    fluid = jxp.FluidPerfectGas(name="Air")
    x = np.linspace(0.0, 0.2, 7)
    area = get_symmetric_nozzle(x, L=0.2, A_inlet=3e-4, A_throat=1.5e-4).A
    with pytest.raises(ValueError, match="search range|bracket"):
        nozzle_exact_real_gas(x, area, fluid, 2e5, 300.0, p_bounds=(1.9e5, 3e5))
