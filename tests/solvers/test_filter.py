"""Conservative filtering respects contents, declared support and stage checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Mesh, Operator, Scheme

from tests._support.generic_models import _UserFilteredAdvection


def test_filter_widens_the_colored_stencil_and_preserves_boundary_fluxes():
    model = _UserFilteredAdvection(0.7)
    unfiltered = Operator(model, Scheme(flux="hll"))
    filtered = Operator(model, Scheme(flux="hll", filter_strength=0.03))
    state = model.initial_state()
    assert unfiltered.radius == 1
    assert filtered.radius == 2
    plain_flux, corrected_flux = np.asarray(unfiltered.face_flux(state)), np.asarray(
        filtered.face_flux(state)
    )
    np.testing.assert_array_equal(plain_flux[[0, -1]], corrected_flux[[0, -1]])
    assert np.max(np.abs(corrected_flux[1:-1] - plain_flux[1:-1])) > 1e-6
    args = (state.ravel(), state.ravel() * 0.97, 0.03, 0.0)
    dense = np.asarray(filtered.dense_jacobian(*args))
    np.testing.assert_allclose(filtered.jacobian(*args), dense, rtol=3e-14, atol=3e-14)
    i, j = np.indices(dense.shape)
    assert np.max(np.abs(dense[np.abs(i - j) == 2])) > 1e-5
    correction = np.asarray(filtered.direction(*args))
    np.testing.assert_allclose(
        dense @ correction, -filtered.residual(*args), rtol=0, atol=3e-13
    )


@pytest.mark.parametrize("integrator", ["euler-explicit", "runge-kutta-3"])
def test_filter_is_included_in_conservative_explicit_stage_budgets(integrator):
    model = _UserFilteredAdvection(0.7)
    filtered = Operator(
        model, Scheme(flux="hll", time_integrator=integrator, filter_strength=0.03)
    )
    unfiltered = Operator(model, Scheme(flux="hll", time_integrator=integrator))
    old = model.initial_state()
    new, info, increment = filtered.step_with_balance(old, 0.001, 0.001)
    plain, plain_info = unfiltered.step(old, 0.001, 0.001)
    assert bool(info.converged) and bool(plain_info.converged)
    assert np.max(np.abs(np.asarray(new[1:-1] - plain[1:-1]))) > 1e-8
    actual_change = np.asarray(
        filtered.storage(new[1:-1]) - filtered.storage(old[1:-1])
    ).sum(axis=0)
    np.testing.assert_allclose(actual_change, increment, rtol=0, atol=3e-13)
    assert np.max(np.abs(model.boundary_residual(new, 0.001))) < 1e-12


def test_combined_wave_filter_timestep_damps_an_interior_checkerboard():
    model = _UserFilteredAdvection(1.0, 64)
    faces = np.linspace(0.0, 1.0, 65)
    model.mesh = Mesh(
        faces, (faces[:-1] + faces[1:]) / 2.0, np.diff(faces), np.ones(65)
    )
    operator = Operator(
        model, Scheme(flux="hll", time_integrator="euler-explicit", filter_strength=1.0)
    )
    state = (
        jnp.ones((66, 1))
        .at[1:-1, 0]
        .add(1e-3 * jnp.where(jnp.arange(64) % 2 == 0, 1.0, -1.0))
    )
    limits = operator.timestep(state, 0.45)
    assert bool(limits.valid)
    expected = 1.0 / (
        1.0 / (0.45 * float(limits.wave_dt)) + 1.0 / float(limits.filter_dt)
    )
    np.testing.assert_allclose(limits.dt, expected, rtol=3e-15)
    new, info = operator.step(state, limits.dt, limits.dt)
    assert bool(info.converged)
    before = np.max(np.abs(np.asarray(state[10:-10]) - 1.0))
    after = np.max(np.abs(np.asarray(new[10:-10]) - 1.0))
    assert after < before
