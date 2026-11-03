"""Osher path quadrature, sonic roots and binding contracts."""

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.osher import OsherFlux
from nozzle_hyperbolic.solvers.fvm import Scheme

from tests._support.osher_models import (
    INTEGRATIONS,
    CONFIGURATIONS,
    Advection,
    Burgers,
    MixedStorage,
    flux_function,
    WideHartenAdvection,
    sonic_burgers,
    OscillatoryScalar,
    InvalidInteriorPath,
)


@pytest.mark.parametrize("speed", [-2.0, 2.0])
@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_linear_advection_is_upwind_for_both_signs(speed, quadrature, integration):
    model = Advection(speed)
    evaluate = flux_function(model, OsherFlux(quadrature, integration))
    pair = jnp.asarray([[0.3], [1.2]])
    actual = jax.jit(evaluate)(pair)
    expected = speed * pair[0 if speed > 0 else 1, 0]
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    derivative = np.asarray(jax.jit(jax.jacfwd(evaluate))(pair))[:, 0]
    np.testing.assert_allclose(
        derivative, [speed, 0.0] if speed > 0 else [0.0, speed], atol=2e-14
    )


@pytest.mark.parametrize("integration", INTEGRATIONS)
def test_zero_width_stationary_wave_has_finite_flux_and_derivative(integration):
    evaluate = flux_function(Advection(0.0), OsherFlux(5, integration), 0.0)
    pair = jnp.asarray([[0.3], [1.2]])
    np.testing.assert_array_equal(jax.jit(evaluate)(pair), 0.0)
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(evaluate))(pair), 0.0)


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_nonlinear_storage_and_explicit_path_are_covariant_in_mixed_balances(
    quadrature, integration
):
    model, flux = MixedStorage(), OsherFlux(quadrature, integration)
    left = jnp.asarray([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7]])
    right = jnp.asarray([[0.7, 0.2], [0.1, -0.4], [0.3, 0.2]])

    def evaluate(pair):
        l, r = pair
        return flux(
            model,
            l,
            r,
            0.1,
            model.face_data(l)["physical_flux"],
            model.face_data(r)["physical_flux"],
        )

    def expected(pair):
        l, r = (model.base_storage(value) for value in pair)
        return (jnp.where(model.speeds > 0, l, r) * model.speeds) @ model.transform.T

    pair = jnp.stack((left, right))
    actual = jax.jit(evaluate)(pair)
    np.testing.assert_allclose(actual, expected(pair), atol=2e-14, rtol=2e-14)
    direction = (
        jnp.arange(pair.size, dtype=pair.dtype).reshape(pair.shape) * 0.01 + 0.03
    )
    _, derivative = jax.jit(
        lambda point, tangent: jax.jvp(evaluate, (point,), (tangent,))
    )(pair, direction)
    _, reference = jax.jvp(expected, (pair,), (direction,))
    np.testing.assert_allclose(derivative, reference, atol=3e-13, rtol=3e-13)
    diagnostics = flux.diagnostics(model, left, right, 0.1)
    np.testing.assert_allclose(
        diagnostics["finite_jump_quadrature_defect"], 0.0, atol=3e-14
    )
    assert bool(
        jax.jit(lambda l, r: flux.path_admissible(model, pair[0], l, r, 0.1))(
            left, right
        )
    )


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_explicit_guard_includes_viscosity_larger_than_the_physical_speed(
    quadrature, integration
):
    model, flux = WideHartenAdvection(4.0), OsherFlux(quadrature, integration)
    left, right = jnp.asarray([[0.0], [0.2]]), jnp.asarray([[1.0], [0.8]])
    maximum = jax.jit(lambda l, r: flux.max_sampled_speed(model, l, r, 0.1))(
        left, right
    )
    # Harten phi=(lambda^2+width^2)/(2 width)=17/8 exceeds |lambda|=1.
    assert maximum.shape == (2,)
    np.testing.assert_allclose(maximum, 17.0 / 8.0, atol=2e-14)
    assert np.all(np.asarray(maximum) > abs(model.speed))


@pytest.mark.parametrize("quadrature,integration", CONFIGURATIONS)
def test_explicit_guard_samples_interior_regularized_speed_maxima(
    quadrature, integration
):
    class InteriorWidth(Advection):
        def face_characteristics(self, left, right, delta_factor):
            projectors, eigenvalues, _ = super().face_characteristics(
                left, right, delta_factor
            )
            state = 0.5 * (left + right)
            return projectors, eigenvalues, 1.0 + 32.0 * state * (1.0 - state)

    model, flux = InteriorWidth(1.0), OsherFlux(quadrature, integration)
    left, right = jnp.zeros((1, 1)), jnp.ones((1, 1))
    maximum = jax.jit(lambda l, r: flux.max_sampled_speed(model, l, r, 0.1))(
        left, right
    )
    # Width is one at both endpoints and nine at the midpoint, a node of
    # every supported odd Gauss rule. The physical wave speed stays one.
    np.testing.assert_allclose(maximum, (1.0 + 9.0**2) / (2.0 * 9.0), atol=2e-14)


@pytest.mark.parametrize("integration", INTEGRATIONS)
def test_invalid_width_does_not_turn_into_a_finite_explicit_speed_bound(integration):
    model, flux = WideHartenAdvection(-1.0), OsherFlux(5, integration)
    left, right = jnp.zeros((1, 1)), jnp.ones((1, 1))
    maximum = jax.jit(lambda l, r: flux.max_sampled_speed(model, l, r, 0.1))(
        left, right
    )
    assert not np.isfinite(np.asarray(maximum)).any()


@pytest.mark.parametrize("points", [3, 5, 9])
def test_unsplit_burgers_counterexample_is_retained(points):
    """A signed chain-rule check is not an absolute-viscosity error bound."""
    model, flux = Burgers(), OsherFlux(points)
    pair = jnp.asarray([[-1.0], [1.0]])
    value = flux_function(model, flux)(pair)
    assert float(value) > 0  # Fails the scalar E-flux condition, by design.
    audit = flux.diagnostics(model, pair[:1], pair[1:], 0.1)
    np.testing.assert_allclose(audit["finite_jump_quadrature_defect"], 0.0, atol=3e-15)


def test_sonic_partition_integrates_regularized_burgers_rarefaction(sonic_burgers):
    evaluate, derivative = sonic_burgers
    pair = jnp.asarray([[-1.0], [1.0]])
    # Integral of the quadratic Harten band gives -delta^2/6 exactly.
    np.testing.assert_allclose(evaluate(pair), -(0.1**2) / 6, atol=3e-15, rtol=3e-13)
    assert np.isfinite(np.asarray(derivative(pair))).all()


@pytest.mark.parametrize(
    "values", [(-1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.1, 0.1), (-0.1, 0.1)]
)
def test_sonic_endpoint_and_coincident_root_derivatives_match_finite_differences(
    sonic_burgers, values
):
    evaluate, derivative = sonic_burgers
    pair = jnp.asarray(values).reshape(2, 1)
    actual = np.asarray(derivative(pair))[:, 0]
    finite_difference = []
    epsilon = 1e-6
    for variable in range(2):
        direction = jnp.zeros_like(pair).at[variable, 0].set(epsilon)
        finite_difference.append(
            (evaluate(pair + direction) - evaluate(pair - direction)) / (2 * epsilon)
        )
    assert np.isfinite(float(evaluate(pair))) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, finite_difference, atol=2e-8, rtol=2e-8)


def test_compact_capacity_overflow_keeps_all_captured_roots():
    model, flux = OscillatoryScalar(), OsherFlux(5, "sonic_partitioned")
    pair = jnp.asarray([[0.0], [1.0]])
    evaluate = flux_function(model, flux, 0.01)
    audit = flux.diagnostics(model, pair[:1], pair[1:], 0.01)
    assert audit["compact_overflow"] is True
    assert (
        int(audit["raw_active_bracket_count_per_face"][0]) > audit["compact_capacity"]
    )
    # Independent high-order integral, partitioned at analytically known roots
    # of sin(7*pi*u)+0.012 = -delta, 0, +delta.
    cuts = [0.0, 1.0]
    for level in (-0.01, 0.0, 0.01):
        angle = np.arcsin(level - 0.012)
        for period in range(-1, 5):
            for root in (
                angle + 2 * np.pi * period,
                np.pi - angle + 2 * np.pi * period,
            ):
                if 0 < root < 7 * np.pi:
                    cuts.append(root / (7 * np.pi))
    nodes, weights = np.polynomial.legendre.leggauss(32)
    integral = 0.0
    for lower, upper in zip(sorted(cuts)[:-1], sorted(cuts)[1:]):
        coordinate = lower + 0.5 * (nodes + 1) * (upper - lower)
        speed = np.sin(7 * np.pi * coordinate) + 0.012
        absolute = np.where(
            abs(speed) < 0.01, 0.5 * (speed**2 / 0.01 + 0.01), abs(speed)
        )
        integral += 0.5 * (upper - lower) * np.dot(weights, absolute)
    expected = 0.5 * (
        float(model.face_physical_flux(pair[0, 0]))
        + float(model.face_physical_flux(pair[1, 0]))
        - integral
    )
    np.testing.assert_allclose(jax.jit(evaluate)(pair), expected, atol=1e-7, rtol=1e-7)
    derivative = np.asarray(jax.jit(jax.jacfwd(evaluate))(pair))[:, 0]
    epsilon = 1e-6
    fd = [
        (
            evaluate(pair + jnp.eye(2)[:, index : index + 1] * epsilon)
            - evaluate(pair - jnp.eye(2)[:, index : index + 1] * epsilon)
        )
        / (2 * epsilon)
        for index in range(2)
    ]
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(derivative, fd, atol=3e-8, rtol=3e-8)


@pytest.mark.parametrize("integration", INTEGRATIONS)
def test_finite_but_model_invalid_interior_nodes_are_rejected(integration):
    model, flux = InvalidInteriorPath(), OsherFlux(5, integration)
    state = jnp.ones((2, 1))
    assert bool(model.admissible(state, state, state))
    check = jax.jit(lambda l, r: flux.path_admissible(model, state, l, r, 0.1))
    assert not bool(check(state, state))


def test_invalid_node_rejection_precedes_characteristic_data_evaluation():
    class CheckedModel(InvalidInteriorPath):
        def face_data(self, state):
            def positive_only(value):
                assert np.all(
                    np.asarray(value) > 0
                ), "Invalid node reached characteristic data"

            jax.debug.callback(positive_only, state)
            return dict(conserved=state, physical_flux=state, auxiliary=None)

    model, flux = CheckedModel(), OsherFlux(5)
    state = jnp.ones((2, 1))
    assert not bool(
        jax.jit(lambda: flux.path_admissible(model, state, state, state, 0.1))()
    )


@pytest.mark.parametrize(
    "options",
    [
        {"quadrature_points": 4},
        {"integration": "unsupported"},
        {"intervals": 0},
        {"bisections": 0},
        {"fast_iterations": -1},
        {"root_residual_relative_tolerance": 0.0},
    ],
)
def test_invalid_quadrature_and_root_policies_are_rejected(options):
    with pytest.raises(ValueError):
        OsherFlux(**options)


def test_default_and_flat_configuration_round_trip():
    options = dict(flux="osher", quadrature_points=5, integration="gauss")
    scheme = Scheme(**options)
    assert all(asdict(scheme)[key] == value for key, value in options.items())
    assert Scheme(**asdict(scheme)) == scheme
    assert OsherFlux().quadrature_points == 5
    assert OsherFlux().integration == "gauss"
    assert not OsherFlux().sonic_partition


@pytest.mark.parametrize("points", [None, True, False, 0, 1, 4, 17, -3, 3.0])
def test_config_and_kernel_reject_unsupported_quadrature(points):
    with pytest.raises(ValueError, match="quadrature"):
        OsherFlux(quadrature_points=points)
    with pytest.raises(ValueError, match="quadrature_points"):
        Scheme(flux="osher", quadrature_points=points)


@pytest.mark.parametrize("integration", [None, True, False, 0, [], "unsupported"])
def test_config_and_kernel_reject_unsupported_integration(integration):
    with pytest.raises(ValueError, match="integration"):
        OsherFlux(integration=integration)
    with pytest.raises(ValueError, match="integration"):
        Scheme(flux="osher", integration=integration)


def test_invalid_public_flux_selector_is_rejected():
    with pytest.raises(ValueError, match="Unsupported numerical flux"):
        Scheme(flux="unsupported_flux")
