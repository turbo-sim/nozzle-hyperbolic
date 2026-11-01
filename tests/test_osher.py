"""Generic path-flux contracts, sonic quadrature and differentiated roots."""

from dataclasses import asdict

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.osher import OsherFlux
from nozzle_hyperbolic.models.scalar.advection import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


INTEGRATIONS = ("gauss", "sonic_partitioned")
CONFIGURATIONS = [(points, policy) for points in (3, 5, 9) for policy in INTEGRATIONS]


class Advection:
    def __init__(self, speed):
        self.speed = speed

    def face_conserved(self, state):
        return state

    def face_physical_flux(self, state):
        return self.speed * state

    def face_characteristics(self, left, right, delta_factor):
        shape = left.shape[:-1] + (1,)
        return (
            jnp.ones(shape + (1, 1)),
            jnp.full(shape, self.speed),
            jnp.full(shape, delta_factor),
        )

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )


class Burgers(Advection):
    def __init__(self):
        super().__init__(0.0)

    def face_physical_flux(self, state):
        return 0.5 * state**2

    def face_characteristics(self, left, right, delta_factor):
        eigenvalues = 0.5 * (left + right)
        width = delta_factor * jnp.maximum(jnp.maximum(abs(left), abs(right)), 1.0)
        return jnp.ones(left.shape[:-1] + (1, 1, 1)), eigenvalues, width


class MixedStorage:
    """Fused-only model: nonlinear storage, mixed balances and both wave signs."""

    faces = 3
    transform = jnp.asarray([[1.0, 0.4], [-0.3, 1.2]])
    inverse = jnp.linalg.inv(transform)
    speeds = jnp.asarray([-0.7, 2.1])

    def base_storage(self, state):
        return state + jnp.asarray([0.2, 0.1]) * state**3

    def face_data(self, state):
        # A path quadrature axis must never be flattened into model geometry.
        assert state.shape == (self.faces, 2)
        base = self.base_storage(state)
        return dict(
            conserved=base @ self.transform.T,
            physical_flux=(base * self.speeds) @ self.transform.T,
            auxiliary=state,
        )

    def face_characteristics_from_data(
        self, left, right, delta_factor, left_data, right_data
    ):
        projectors = jnp.einsum("ik,kj->kij", self.transform, self.inverse)
        return (
            jnp.broadcast_to(projectors, (self.faces, 2, 2, 2)),
            jnp.broadcast_to(self.speeds, left.shape),
            jnp.full(left.shape, delta_factor),
        )

    def face_state_path(self, left, right, parameter):
        assert parameter.shape == (self.faces, 1)
        curvature = jnp.asarray([0.2, -0.1])
        return (
            left + parameter * (right - left) + parameter * (1 - parameter) * curvature,
            right - left + (1 - 2 * parameter) * curvature,
        )

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )


def flux_function(model, flux, delta_factor=0.1):
    def evaluate(pair):
        left, right = pair[0:1], pair[1:2]
        return flux(
            model,
            left,
            right,
            delta_factor,
            model.face_physical_flux(left),
            model.face_physical_flux(right),
        )[0, 0]

    return evaluate


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


class WideHartenAdvection(Advection):
    def __init__(self, width):
        super().__init__(1.0)
        self.width = width

    def face_characteristics(self, left, right, delta_factor):
        projectors, eigenvalues, _ = super().face_characteristics(
            left, right, delta_factor
        )
        return projectors, eigenvalues, jnp.full_like(eigenvalues, self.width)


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


@pytest.fixture(scope="module", params=(3, 5, 9))
def sonic_burgers(request):
    evaluate = flux_function(Burgers(), OsherFlux(request.param, "sonic_partitioned"))
    return jax.jit(evaluate), jax.jit(jax.jacfwd(evaluate))


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


class OscillatoryScalar(Advection):
    def __init__(self):
        super().__init__(0.0)

    def face_physical_flux(self, state):
        return (1 - jnp.cos(7 * jnp.pi * state)) / (7 * jnp.pi) + 0.012 * state

    def face_characteristics(self, left, right, delta_factor):
        eigenvalues = jnp.sin(7 * jnp.pi * 0.5 * (left + right)) + 0.012
        return (
            jnp.ones(left.shape[:-1] + (1, 1, 1)),
            eigenvalues,
            jnp.full_like(eigenvalues, delta_factor),
        )


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


class InvalidInteriorPath(Advection):
    def __init__(self):
        super().__init__(1.0)

    def face_state_path(self, left, right, parameter):
        return left + parameter * (right - left) - 8 * parameter * (
            1 - parameter
        ), right - left - 8 * (1 - 2 * parameter)

    def admissible(self, full, left, right):
        return (
            super().admissible(full, left, right)
            & jnp.all(left > 0)
            & jnp.all(right > 0)
        )


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


class LinearBalance(AdvectionModel):
    """Complete EOS-free FVM model with a declared characteristic width."""

    def __init__(self, width=0.1, physical_face_bound=1.0):
        super().__init__(4, speed=1.0)
        self.width, self.physical_face_bound = width, physical_face_bound

    def face_characteristics(self, left, right, delta_factor):
        return (
            jnp.ones((left.shape[0], 1, 1, 1)),
            jnp.ones_like(left),
            jnp.full_like(left, self.width),
        )

    def face_speed_bounds(self, state):
        bound = jnp.full(state.shape[0], self.physical_face_bound)
        return -bound, bound


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
