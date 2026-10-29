"""Independent contracts for shared square FVM/DAE numerical machinery."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.solvers.finite_volume import (
    flux_divergence,
    make_coupled_residual,
)
from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver


@pytest.mark.parametrize("nblocks,nvar,radius", [(2, 3, 1), (3, 4, 2), (6, 4, 2), (5, 3, 0)])
def test_colored_blocks_and_banded_direction_match_independent_matrix(nblocks, nvar, radius):
    """Exercise odd/padded blocks, four-variable rows and a block diagonal limit."""
    count = nblocks * nvar
    random = np.random.default_rng(31 + count)
    matrix = random.normal(scale=0.02, size=(count, count))
    block_ids = np.arange(count) // nvar
    matrix[np.abs(block_ids[:, None] - block_ids[None, :]) > radius] = 0
    matrix += 2.0 * np.eye(count)
    storage = np.ones(count)
    storage[:nvar] = storage[-nvar:] = 0.0
    point = jnp.linspace(0.7, 1.3, count)
    old = point * 0.9
    dt = 0.13

    def residual(new, previous, timestep):
        return jnp.asarray(storage) * (new - previous) + timestep * (
            jnp.asarray(matrix) @ new + 0.03 * new**2 - 0.6
        )

    operators = make_block_operators(
        residual, nblocks, nvar, radius, jacobian_mode="colored", linear_solver="banded"
    )
    expected = np.diag(storage + dt * 0.06 * np.asarray(point)) + dt * matrix
    actual = np.asarray(jax.jit(operators["colored_jacobian"])(point, old, dt))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        jax.jit(operators["dense_jacobian"])(point, old, dt), expected,
        rtol=2e-14, atol=2e-14,
    )
    correction = np.asarray(jax.jit(operators["direction"])(point, old, dt))
    defect = expected @ correction + np.asarray(residual(point, old, dt))
    assert np.linalg.norm(defect, ord=np.inf) < 1e-12


def test_declared_band_guard_detects_a_wider_boundary_dependency():
    """The sparsity guard must expose a radius-two BC in a radius-one layout."""
    def residual(point):
        rows = point.reshape(6, 4)
        return (2 * rows).at[0].add(0.3 * rows[2]).ravel()

    operators = make_block_operators(residual, 6, 4, 1)
    assert float(jax.jit(operators["jacobian_difference"])(jnp.ones(24))) > 0.1


def test_coupled_residual_has_current_boundary_fluxes_and_zero_boundary_storage():
    nphysical, nvar = 3, 4
    physical_source = jnp.arange(nphysical * nvar).reshape(nphysical, nvar) * 0.01

    def storage(physical):
        return physical**2 + 0.3 * physical

    def faces(full):
        return 0.7 * full[:-1] + 0.3 * full[1:]

    def spatial(full):
        return flux_divergence(faces(full)) - physical_source

    def boundaries(full):
        return jnp.stack([full[0] - full[1], full[-1] - 2 * full[-2] + full[-3]])

    scales = jnp.array([1., 2., 3., 4.])
    operators = make_coupled_residual(
        storage, spatial, boundaries, nphysical=nphysical, nvar=nvar,
        content_scale=scales, flux_scale=scales * 2,
    )
    full = jnp.linspace(0.8, 1.8, 20).reshape(5, 4)
    previous = full * 0.97
    residual = jax.jit(operators["residual"])
    actual = residual(full.ravel(), previous.ravel(), 0.4).reshape(5, 4)
    changed_old = previous.at[0].set(30.).at[-1].set(-20.)
    np.testing.assert_array_equal(
        residual(full.ravel(), changed_old.ravel(), 0.4).reshape(5, 4), actual
    )
    zero_dt = residual(full.ravel(), previous.ravel(), 0.).reshape(5, 4)
    boundary_indices = jnp.array([0, -1])
    np.testing.assert_array_equal(zero_dt[boundary_indices], actual[boundary_indices])
    np.testing.assert_allclose(
        zero_dt[1:-1], (storage(full[1:-1]) - storage(previous[1:-1])) / scales
    )
    # Current independent boundary values must enter the physical flux rows.
    changed = full.at[0, 0].add(0.2)
    assert not np.array_equal(
        residual(changed.ravel(), previous.ravel(), 0.4).reshape(5, 4)[1], actual[1]
    )
    expected_budget = faces(full)[-1] - faces(full)[0] - physical_source.sum(axis=0)
    np.testing.assert_allclose(operators["spatial"](full.ravel()).sum(axis=0), expected_budget)
    steady = operators["steady"](full.ravel()).reshape(5, 4)
    np.testing.assert_array_equal(steady[boundary_indices], boundaries(full))
    np.testing.assert_allclose(steady[1:-1], spatial(full) / (2 * scales))


def test_newton_backtracking_rejects_invalid_trials_and_solves_original_equation():
    def residual(point):
        return jnp.log(point)

    def direction(point):
        return -point * jnp.log(point)

    solve = jax.jit(make_newton_solver(
        residual, direction, lambda point: jnp.ones_like(point),
        max_iterations=20, tolerance=1e-11, residual_tolerance=1e-11,
        admissible=lambda point: jnp.all(point > 0), backtracking=True,
    ))
    result, info = solve(jnp.array([10.]))
    np.testing.assert_allclose(result, [1.], rtol=1e-11, atol=1e-11)
    assert bool(info.converged) and int(info.status) == 0
    assert int(info.rejected_trials) >= 2
    assert float(jnp.max(jnp.abs(residual(result)))) < 1e-11


def test_failed_newton_trials_retain_previous_iterate_and_report_failure():
    solve = jax.jit(make_newton_solver(
        lambda point: point - 1., lambda point: jnp.ones_like(point),
        lambda point: jnp.ones_like(point), max_iterations=4,
        backtracking=True, max_backtracks=3, residual_tolerance=1e-10,
    ))
    initial = jnp.array([2.])
    result, info = solve(initial)
    np.testing.assert_array_equal(result, initial)
    assert not bool(info.converged)
    assert int(info.status) == 3 and int(info.rejected_trials) == 4


@pytest.mark.parametrize("tolerance", [0., 1e-10])
def test_unchecked_newton_preserves_historical_full_step_iteration_rule(tolerance):
    def direction(point):
        return -(point**2 - 2.) / (2. * point)

    initial = jnp.array([1., 2., 3.])
    expected = np.asarray(initial).copy()
    relative = np.inf
    iterations = 0
    while iterations < 8 and (tolerance <= 0 or relative > tolerance):
        correction = -(expected**2 - 2.) / (2. * expected)
        relative = np.max(np.abs(correction) / (np.abs(expected) + 1e-300))
        expected += correction
        iterations += 1
    solve = jax.jit(make_newton_solver(
        lambda point: point**2 - 2., direction, jnp.abs,
        max_iterations=8, tolerance=tolerance,
    ))
    actual, info = solve(initial)
    # NumPy's unfused host arithmetic can round the final sqrt(2) iterate
    # one ULP differently from XLA. The iteration rule/count remains exact.
    np.testing.assert_array_max_ulp(np.asarray(actual), expected, maxulp=1)
    assert int(info.iterations) == iterations
