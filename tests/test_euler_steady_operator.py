"""Independent spatial AD and linear-solve contracts for shared Euler kernels."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


def _problem(reconstruction, **options):
    mesh = SymmetricNozzleGeometry().build_mesh(11)
    model = NozzleModel(mesh, jxp.FluidPerfectGas(name="Air"), back_pressure=1.8e5)
    operator = Operator(
        model, Scheme(reconstruction=reconstruction, delta_factor=1e-6, **options)
    )
    coordinate = mesh.points / 0.2
    velocity = model.a0 * (0.25 + 0.15 * np.sin(np.pi * coordinate))
    varying = jnp.asarray(
        np.column_stack(
            [
                model.p0 * (0.98 - 0.08 * coordinate),
                velocity,
                model.h0 - 0.5 * velocity**2,
            ]
        )
    )
    flat = jnp.broadcast_to(
        jnp.asarray(
            [0.95 * model.p0, 0.25 * model.a0, model.h0 - 0.5 * (0.25 * model.a0) ** 2]
        ),
        varying.shape,
    )
    return operator, varying, flat


@pytest.fixture(
    scope="module", params=("constant", "muscl_van_leer", "muscl_van_leer_smooth")
)
def steady_kernels(request):
    operator, varying, flat = _problem(request.param)
    spatial = make_block_operators(
        operator.steady_residual,
        operator.shape[0],
        3,
        operator.radius,
        jacobian_mode="colored",
        linear_solver="banded",
    )
    dense = jax.jit(jax.jacfwd(operator.steady_residual))
    return operator, spatial, (varying.ravel(), flat.ravel()), dense


def test_colored_steady_jacobian_matches_independent_dense_ad(steady_kernels):
    operator, blocks, states, dense = steady_kernels
    scales = np.tile(operator.model.variable_scales, operator.shape[0])
    for point in states:
        expected = np.asarray(dense(point))
        actual = np.asarray(blocks["jacobian"](point))
        assert np.isfinite(expected).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual * scales, expected * scales, rtol=1e-10, atol=1e-10
        )


def test_banded_steady_direction_solves_independent_linearization(steady_kernels):
    operator, blocks, states, dense = steady_kernels
    for point in states:
        residual = np.asarray(operator.steady_residual(point))
        direction = np.asarray(blocks["direction"](point))
        jacobian = np.asarray(dense(point))
        assert np.isfinite(direction).all()
        assert (
            np.linalg.norm(jacobian @ direction + residual) / np.linalg.norm(residual)
            < 1e-8
        )


def test_steady_reverse_ad_is_finite_and_matches_forward_on_flat_states(steady_kernels):
    operator, _, states, forward = steady_kernels
    flat = states[1]
    weights = jnp.linspace(0.5, 1.5, flat.size)
    reverse = jax.jit(
        jax.grad(lambda point: jnp.dot(weights, operator.steady_residual(point)))
    )
    expected = np.asarray(forward(flat)).T @ np.asarray(weights)
    actual = np.asarray(reverse(flat))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_dense_and_colored_implicit_steps_solve_the_same_coupled_equations():
    banded, state, _ = _problem("muscl_van_leer")
    dense = Operator(
        banded.model,
        Scheme(
            reconstruction="muscl_van_leer",
            delta_factor=1e-6,
            jacobian_mode="dense",
            linear_solver="dense",
        ),
    )
    first, info = banded.step(state, jnp.float64(1e-8), 0.0)
    second, other = dense.step(state, jnp.float64(1e-8), 0.0)
    assert bool(info.converged & info.admissible & other.converged & other.admissible)
    np.testing.assert_allclose(first, second, rtol=2e-12, atol=2e-9)


@pytest.mark.parametrize("width", [0.0, -0.1, np.nan, np.inf])
def test_factory_rejects_invalid_entropy_widths(width):
    with pytest.raises(ValueError):
        Scheme(delta_factor=width)


def test_retired_legacy_flux_is_an_explicit_api_error():
    with pytest.raises(ValueError):
        Scheme(flux="legacy")


@pytest.mark.parametrize("coefficient", [0.0, -0.1, np.nan, np.inf])
def test_smooth_reconstruction_rejects_invalid_width_coefficients(coefficient):
    with pytest.raises(ValueError, match="smooth_limiter_coefficient"):
        Scheme(
            reconstruction="muscl_van_leer_smooth",
            smooth_limiter_coefficient=coefficient,
        )
