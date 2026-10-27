"""Independent AD and linear-solve contracts for current steady Euler kernels."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.models.euler_nozzle.fvm import make_jax_stadtke_step


def _factory_inputs(reconstruction):
    """Build a small perfect-gas problem without an analytic nozzle solve."""
    count = 13
    length = 0.2
    x = np.linspace(0.0, length, count)
    dx = float(x[1] - x[0])
    coordinate = x / length
    area = 3.0e-4 - 6.0e-4 * coordinate * (1.0 - coordinate)
    face_area = 0.5 * (area[:-1] + area[1:])
    area_derivative = np.gradient(area, dx)
    area_derivative[1:-1] = np.diff(face_area) / dx
    diameter = 2.0 * np.sqrt(area / np.pi)
    fluid = jxp.FluidPerfectGas(name="Air")
    p0, temperature0 = 2e5, 300.0
    stagnation = fluid.get_state(jxp.PT_INPUTS, p0, temperature0)
    h0, s0, a0 = (float(stagnation[key]) for key in ("h", "s", "a"))
    arguments = (
        fluid,
        area,
        face_area,
        area_derivative,
        diameter,
        0.0,
        dx,
        20.0,
        0.9 * p0,
        h0,
        s0,
    )
    options = dict(
        delta_factor=1e-6,
        newton_iter=1,
        wall_friction=False,
        recon=reconstruction,
        jacobian_mode="colored",
        linear_solver="banded",
        flux_local_jac=True,
        cell_local_jac=True,
        flux_variant="regularized",
        limiter_scales=[p0, a0, h0],
        smooth_limiter_coefficient=1.0,
    )
    velocity = a0 * (0.25 + 0.15 * np.sin(np.pi * coordinate))
    varying = np.column_stack(
        [p0 * (0.98 - 0.08 * coordinate), velocity, h0 - 0.5 * velocity**2]
    )
    flat = np.broadcast_to(
        [0.95 * p0, 0.25 * a0, h0 - 0.5 * (0.25 * a0) ** 2], varying.shape
    ).copy()
    return arguments, options, jnp.asarray(varying), jnp.asarray(flat)


@pytest.fixture(
    scope="module",
    params=("constant", "muscl_van_leer", "muscl_van_leer_smooth"),
)
def steady_kernels(request):
    """Share compiled 13-node kernels across independent operator checks."""
    arguments, options, varying, flat = _factory_inputs(request.param)
    extended = make_jax_stadtke_step(*arguments, **options, return_operators=True)
    operators = extended[-1]
    dense = jax.jit(jax.jacfwd(operators["steady_residual"]))
    return operators, (varying[1:-1].ravel(), flat[1:-1].ravel()), dense


def test_colored_steady_jacobian_matches_independent_dense_ad(steady_kernels):
    """Spatial AD must cancel storage and preserve the reconstruction stencil."""
    operators, states, dense = steady_kernels
    for interior in states:
        expected = np.asarray(dense(interior))
        actual = np.asarray(operators["steady_jacobian"](interior))
        assert np.isfinite(expected).all()
        assert np.isfinite(actual).all()
        relative = np.linalg.norm(actual - expected, ord=np.inf) / max(
            np.linalg.norm(expected, ord=np.inf), 1e-300
        )
        assert relative < 1e-11
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_banded_steady_direction_solves_independent_linearization(steady_kernels):
    """A banded direction must solve the full spatial system, not its own copy."""
    operators, states, dense = steady_kernels
    for interior in states:
        residual = np.asarray(operators["steady_residual"](interior))
        direction = np.asarray(operators["steady_newton_step"](interior))
        jacobian = np.asarray(dense(interior))
        assert np.isfinite(direction).all()
        relative = np.linalg.norm(jacobian @ direction + residual) / max(
            np.linalg.norm(residual), 1e-300
        )
        assert relative < 1e-8


def test_steady_reverse_ad_is_finite_and_matches_forward_on_flat_states(steady_kernels):
    """Exact zero interior slopes must not poison an adjoint contraction."""
    operators, states, forward = steady_kernels
    flat = states[1]
    weights = jnp.linspace(0.5, 1.5, flat.size)
    reverse = jax.jit(
        jax.grad(lambda point: jnp.dot(weights, operators["steady_residual"](point)))
    )
    expected = np.asarray(forward(flat)).T @ np.asarray(weights)
    actual = np.asarray(reverse(flat))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_enabling_steady_operators_preserves_current_implicit_step():
    """Check four/five-return parity of the current flux, not legacy physics."""
    arguments, options, state, _ = _factory_inputs("muscl_van_leer")
    original_api = make_jax_stadtke_step(*arguments, **options)
    extended_api = make_jax_stadtke_step(*arguments, **options, return_operators=True)
    assert len(original_api) == 4
    assert len(extended_api) == 5
    # Tiny physical timestep keeps this a single safe kernel check, not a
    # pseudo-time convergence experiment or a performance measurement.
    dt = jnp.float64(1e-8)
    expected_state, expected_ssr = original_api[0](state, dt)
    actual_state, actual_ssr = extended_api[0](state, dt)
    assert np.isfinite(expected_state).all()
    assert np.isfinite(expected_ssr)
    np.testing.assert_array_equal(actual_state, expected_state)
    np.testing.assert_array_equal(actual_ssr, expected_ssr)


@pytest.mark.parametrize("width", [0.0, -0.1, np.nan, np.inf])
def test_regularized_factory_rejects_invalid_entropy_widths(width):
    """A positive finite sonic width is part of the regularity contract."""
    arguments, options, _, _ = _factory_inputs("constant")
    options["delta_factor"] = width
    with pytest.raises(ValueError, match="delta_factor"):
        make_jax_stadtke_step(*arguments, **options)


def test_legacy_factory_allows_zero_width_for_reproducing_archived_runs():
    """Legacy construction remains available without claiming sonic regularity."""
    arguments, options, _, _ = _factory_inputs("constant")
    options.update(flux_variant="legacy", delta_factor=0.0)
    assert len(make_jax_stadtke_step(*arguments, **options)) == 4


@pytest.mark.parametrize("coefficient", [0.0, -0.1, np.nan, np.inf])
def test_smooth_reconstruction_rejects_invalid_width_coefficients(coefficient):
    """The smooth limiter requires a positive finite regularization scale."""
    arguments, options, _, _ = _factory_inputs("muscl_van_leer_smooth")
    options["smooth_limiter_coefficient"] = coefficient
    with pytest.raises(ValueError, match="smooth_limiter_coefficient"):
        make_jax_stadtke_step(*arguments, **options)
