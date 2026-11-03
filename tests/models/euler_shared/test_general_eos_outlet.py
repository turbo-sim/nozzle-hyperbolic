"""Independent nonlinear outlet checks using a shifted stiffened-gas EOS."""

import jax
import jax.numpy as jnp
import jaxprop as jxp

from math import erf, exp, log, pi, sqrt

import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_outlet import EulerOutlet


class ShiftedStiffenedGas:
    """HP-only EOS; no public perfect-gas constants or backend inheritance."""

    def __init__(self, offset=2e4, shift=-5e5):
        self.offset, self.shift = offset, shift

    def get_state(self, pair, h, p):
        assert pair == jxp.HmassP_INPUTS
        thermal_h = h - self.shift
        rho = 3.5 * (p + self.offset) / thermal_h
        return dict(
            rho=rho,
            a=jnp.sqrt(0.4 * thermal_h),
            s=1004.5 * jnp.log(thermal_h / 3e5)
            - 287.0 * jnp.log((p + self.offset) / 2e5),
        )


def setup(mach, ratio, *, nvar=3, fluid=None):
    fluid = ShiftedStiffenedGas() if fluid is None else fluid
    W = jnp.array([1e5, mach * np.sqrt(1.2e5), -2e5])
    scales = [1e5, 350.0, 1e5]
    if nvar == 4:
        W = jnp.array([W[0], W[1], -8.0, W[2]])
        scales = [1e5, 350.0, 70.0, 1e5]
    return fluid, W, EulerOutlet(fluid, ratio * 1e5, scales, 400.0, 0.5, "finite_wave")


@pytest.mark.parametrize("nvar", [3, 4])
def test_stiffened_compression_satisfies_independent_conservative_jumps(nvar):
    fluid, W, outlet = setup(1.2, 8.0, nvar=nvar)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    assert int(wave.mode) == 3 and float(wave.head_speed) < 0
    left, right = outlet.thermodynamics(W), outlet.thermodynamics(wave.state)
    wl, wr = W[1] - wave.head_speed, wave.state[1] - wave.head_speed
    np.testing.assert_allclose(left["rho"] * wl, right["rho"] * wr, rtol=3e-13)
    np.testing.assert_allclose(
        W[0] + left["rho"] * wl**2,
        wave.state[0] + right["rho"] * wr**2,
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        W[-1] + wl**2 / 2, wave.state[-1] + wr**2 / 2, rtol=3e-13
    )
    assert float(right["s"]) > float(left["s"])
    if nvar == 4:
        assert wave.state[2] == W[2]


@pytest.mark.parametrize("back_ratio", [0.8, 0.01])
def test_stiffened_expansion_matches_closed_form_including_sonic_sampling(back_ratio):
    fluid, W, outlet = setup(0.3, back_ratio)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    ai = np.sqrt(1.2e5)
    a = ai * ((outlet.pressure + fluid.offset) / (float(W[0]) + fluid.offset)) ** (
        1 / 7
    )
    u = float(W[1]) + 5 * (ai - a)
    expected_p = outlet.pressure
    mode = 4
    if u > a:
        a = (0.4 * float(W[1]) + 2 * ai) / 2.4
        u = a
        expected_p = (float(W[0]) + fluid.offset) * (a / ai) ** 7 - fluid.offset
        mode = 5
    h = float(W[-1]) + (a * a - ai * ai) / 0.4
    np.testing.assert_allclose(wave.state, [expected_p, u, h], rtol=3e-10, atol=3e-6)
    assert int(wave.mode) == mode
    np.testing.assert_allclose(
        outlet.thermodynamics(wave.state)["s"], outlet.thermodynamics(W)["s"], atol=3e-8
    )


@pytest.mark.parametrize("ratio", [8.0, 0.8, 0.01])
def test_wave_forward_reverse_ad_agree_with_independent_finite_difference(ratio):
    _, W, outlet = setup(0.3, ratio)
    cells = jnp.stack((W, W))
    scales = jnp.array([1e5, 350.0, 1e5])

    def evaluate(q):
        return outlet.wave(cells.at[-1].set(W + q * scales)).state / scales

    origin = jnp.zeros(3)
    forward = jax.jit(jax.jacfwd(evaluate))(origin)
    reverse = jax.jit(jax.jacrev(evaluate))(origin)
    epsilon = 2e-5
    finite = np.column_stack(
        [
            (
                np.asarray(evaluate(origin.at[k].set(epsilon)))
                - np.asarray(evaluate(origin.at[k].set(-epsilon)))
            )
            / (2 * epsilon)
            for k in range(3)
        ]
    )
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    np.testing.assert_allclose(forward, reverse, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(forward, finite, rtol=3e-6, atol=3e-7)


def test_weak_shock_has_finite_correct_acoustic_limit_without_reference_zero_assumption():
    _, W, outlet = setup(0.3, 1 + 1e-10)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    state = outlet.thermodynamics(W)
    expected_speed = W[1] - state["a"]
    np.testing.assert_allclose(wave.head_speed, expected_speed, rtol=2e-10)
    assert np.isfinite(
        jax.jit(jax.jacfwd(lambda cells: outlet.wave(cells).state))(jnp.stack((W, W)))
    ).all()


def test_outgoing_expansion_ignores_unreachable_external_pressure():
    _, W, outlet = setup(1.2, 0.01)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    assert int(wave.mode) == 1
    np.testing.assert_array_equal(wave.state, W)


def test_invalid_wave_path_and_hugoniot_are_rejected_without_clipping():
    fluid = ShiftedStiffenedGas()
    fluid.h_min, fluid.h_max = -2.05e5, -1.95e5
    for ratio in (0.5, 8.0):
        _, W, outlet = setup(0.3, ratio, fluid=fluid)
        wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
        assert np.isnan(wave.state).all()


def test_expansion_budget_and_defect_are_explicit_rejections():
    fluid, W, _ = setup(0.3, 0.5)
    for options in (dict(wave_panels=1), dict(wave_tolerance=1e-20)):
        outlet = EulerOutlet(
            fluid, 5e4, [1e5, 350.0, 1e5], 400.0, 0.5, "finite_wave", **options
        )
        assert np.isnan(jax.jit(outlet.wave)(jnp.stack((W, W))).state).all()


def test_expansion_refines_localized_property_curvature_without_relaxing_error_budget():
    class LocalizedEOS:
        """Known PH wave integral with a narrow smooth specific-volume pulse.

        This is an integration fixture, not a thermodynamic consistency claim.
        Constant sound keeps the characteristic fan strictly monotone while
        the pulse forces refinement inside one otherwise smooth log-p panel.
        """

        def get_state(self, pair, h, p):
            assert pair == jxp.HmassP_INPUTS
            pulse = jnp.exp(-(((jnp.log(p / 1e5) + 0.025) / 0.003) ** 2))
            return dict(
                rho=1 / (0.1 * (1 + 4 * pulse)), a=jnp.asarray(300.0, dtype=h.dtype)
            )

    W = jnp.array([1e5, 0.0, 3e5])
    cells = jnp.stack((W, W))
    outlet = EulerOutlet(
        LocalizedEOS(), 5e4, [1e5, 300.0, 9e4], 400.0, 0.5, "finite_wave"
    )
    wave = jax.jit(outlet.wave)(cells)
    center, width = log(1e5) - 0.025, 0.003

    def primitive(pressure):
        return (
            exp(center + width**2 / 4)
            * width
            * sqrt(pi)
            / 2
            * erf((log(pressure) - center) / width - width / 2)
        )

    increment = 0.1 * (5e4 + 4 * (primitive(1e5) - primitive(5e4)))
    expected = np.array([5e4, increment / 300.0, 3e5 - increment])
    np.testing.assert_allclose(
        (wave.state - expected) / np.array([1e5, 300.0, 9e4]), 0.0, atol=2e-7
    )
    assert int(wave.mode) == 4
    # Fourteen panels suffice only for the original maximum log-p spacing;
    # local rejection/subdivision is essential to resolve this pulse.
    limited = EulerOutlet(
        LocalizedEOS(),
        5e4,
        [1e5, 300.0, 9e4],
        400.0,
        0.5,
        "finite_wave",
        wave_panels=14,
    )
    assert np.isnan(jax.jit(limited.wave)(cells).state).all()


def test_expansion_accepts_exact_pressure_domain_anchors_without_clipping():
    fluid = ShiftedStiffenedGas()
    fluid.p_min, fluid.p_max = 5e4, 1e5
    fluid.h_min, fluid.h_max = -2.5e5, -1.5e5
    _, W, outlet = setup(0.3, 0.5, fluid=fluid)
    wave = jax.jit(outlet.wave)(jnp.stack((W, W)))
    assert np.isfinite(wave.state).all()
    assert wave.state[0] == fluid.p_min
    # The same path with a genuinely excluded pressure must still reject.
    fluid.p_min = 50001.0
    excluded = EulerOutlet(fluid, 5e4, [1e5, 350.0, 1e5], 400.0, 0.5, "finite_wave")
    assert np.isnan(jax.jit(excluded.wave)(jnp.stack((W, W))).state).all()


def test_exact_zero_pressure_jump_retains_the_correct_differential_wave_curve():
    base = jxp.FluidPerfectGas(name="Air")

    class HPOnly:
        def get_state(self, pair, h, p):
            assert pair == jxp.HmassP_INPUTS
            return base.get_state(pair, h, p)

    state = base.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    W = jnp.array([1e5, 0.3 * state["a"], state["h"]])
    cells = jnp.stack((W, W))
    scales = jnp.array([1e5, state["a"], state["a"] ** 2])
    numerical = EulerOutlet(HPOnly(), 1e5, scales, 400.0, 0.5, "finite_wave")
    closed = EulerOutlet(base, 1e5, scales, 400.0, 0.5, "finite_wave")

    def evaluate(outlet, q):
        return outlet.wave(cells.at[-1].set(W + scales * q)).state / scales

    origin = jnp.zeros(3)
    expected = jax.jit(jax.jacfwd(lambda q: evaluate(closed, q)))(origin)
    forward = jax.jit(jax.jacfwd(lambda q: evaluate(numerical, q)))(origin)
    reverse = jax.jit(jax.jacrev(lambda q: evaluate(numerical, q)))(origin)
    np.testing.assert_array_equal(jax.jit(numerical.wave)(cells).state, W)
    np.testing.assert_allclose(forward, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(reverse, expected, rtol=3e-13, atol=3e-13)
