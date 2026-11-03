"""Binding user-defined laws, validating callbacks and declaring stencil reach."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import Operator, Scheme

from tests._support.generic_models import _UserBalance


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_user_model_radius_three_coloring_direction_and_conservation(reconstruction):
    model = _UserBalance()
    colored = Operator(model, Scheme(flux="hll", reconstruction=reconstruction))
    # Keep direct AD as an independent test reference, not a public setting.
    from nozzle_hyperbolic.solvers.fvm import operator as engine

    with patch.object(engine, "local_jvp", lambda function: function):
        dense = Operator(
            model,
            Scheme(
                flux="hll",
                reconstruction=reconstruction,
                jacobian_mode="dense",
                linear_solver="dense",
            ),
        )
    assert colored.radius == 3
    state = model.initial_state()
    old, dt, time = state * 0.97, 0.017, 0.13
    args = (state.ravel(), old.ravel(), dt, time)
    jacobian = np.asarray(dense.dense_jacobian(*args))
    np.testing.assert_allclose(
        colored.jacobian(*args), jacobian, rtol=3e-14, atol=3e-14
    )
    assert float(colored.jacobian_difference(*args)) < 3e-14
    # Independent central directional derivative checks the differentiated graph.
    direction = np.random.default_rng(13).normal(size=state.size)
    eps = 2e-6
    finite_difference = (
        np.asarray(colored.residual(state.ravel() + eps * direction, *args[1:]))
        - np.asarray(colored.residual(state.ravel() - eps * direction, *args[1:]))
    ) / (2 * eps)
    np.testing.assert_allclose(
        jacobian @ direction, finite_difference, rtol=2e-8, atol=2e-9
    )
    expected_direction = np.linalg.solve(jacobian, -np.asarray(colored.residual(*args)))
    np.testing.assert_allclose(
        colored.direction(*args), expected_direction, rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        dense.direction(*args), expected_direction, rtol=3e-13, atol=3e-13
    )
    flux, source = np.asarray(colored.face_flux(state)), np.asarray(
        colored.source(state)
    )
    np.testing.assert_allclose(
        np.asarray(colored.spatial_residual(state)).sum(axis=0),
        flux[-1] - flux[0] - source.sum(axis=0),
        rtol=3e-14,
        atol=3e-14,
    )


def test_boundary_traces_have_zero_storage_and_nonlinear_completion_freezes_cells():
    model = _UserBalance()
    operator = Operator(model, Scheme(flux="rusanov"))
    full = model.initial_state()
    old = full * 0.9
    changed_old = old.at[0].set(300.0).at[-1].set(-400.0)
    dt, time = 0.04, 0.2
    actual = operator.residual(full.ravel(), old.ravel(), dt, time).reshape(full.shape)
    np.testing.assert_array_equal(
        operator.residual(full.ravel(), changed_old.ravel(), dt, time), actual.ravel()
    )
    zero_dt = operator.residual(full.ravel(), old.ravel(), 0.0, time).reshape(
        full.shape
    )
    np.testing.assert_array_equal(
        zero_dt[jnp.asarray([0, -1])], actual[jnp.asarray([0, -1])]
    )
    np.testing.assert_array_equal(
        actual[jnp.asarray([0, -1])], model.boundary_residual(full, time)
    )
    expected_storage = (
        np.asarray(operator.storage(full[1:-1])) - operator.storage(old[1:-1])
    ) / (model.mesh.volumes[:, None] * model.conservative_scales)
    np.testing.assert_allclose(zero_dt[1:-1], expected_storage, rtol=3e-15, atol=3e-15)
    completed, info = operator.complete_boundaries(full, time)
    assert bool(info.converged) and bool(info.admissible)
    np.testing.assert_array_equal(completed[1:-1], full[1:-1])
    assert np.max(np.abs(model.boundary_residual(completed, time))) < 1e-12
    assert not np.array_equal(np.asarray(completed)[[0, -1]], np.asarray(full)[[0, -1]])


def test_optional_characteristics_and_declared_global_stencil_guards():
    model = _UserBalance()
    Operator(model, Scheme(flux="rusanov"))
    Operator(model, Scheme(flux="hll"))
    with pytest.raises(ValueError, match="characteristic"):
        Operator(
            model,
            Scheme(
                flux="stadtke",
                custom_jump_correction=False,
                custom_endpoint_envelope=False,
            ),
        )
    model.boundary_coupling = "global"
    with pytest.raises(ValueError, match="Global boundary"):
        Operator(model, Scheme(flux="hll"))
    model.boundary_residual = lambda full, time: jnp.stack(
        (full[0] - full[-2], full[-1] - full[1])
    )
    operator = Operator(
        model, Scheme(flux="hll", jacobian_mode="dense", linear_solver="dense")
    )
    state = model.initial_state().ravel()
    jacobian = np.asarray(operator.dense_jacobian(state, state, 0.03, 0.0))
    assert np.any(jacobian[:2, -4:-2] != 0)
    assert np.any(jacobian[-2:, 2:4] != 0)
    np.testing.assert_allclose(
        jacobian @ operator.direction(state, state, 0.03, 0.0),
        -operator.residual(state, state, 0.03, 0.0),
        atol=3e-13,
    )


@pytest.mark.parametrize(
    "integrator", ["euler-implicit", "euler-explicit", "runge-kutta-3"]
)
def test_nonlinear_storage_model_uses_conservative_stages_and_algebraic_traces(
    integrator,
):
    model = _UserBalance()
    operator = Operator(model, Scheme(flux="hll", time_integrator=integrator))
    old = model.initial_state()
    new, info, increment = operator.step_with_balance(old, 0.001, 0.13)
    assert bool(info.converged) and bool(info.admissible)
    assert np.max(np.abs(model.boundary_residual(new, 0.13))) < 1e-12
    actual_change = np.asarray(
        operator.storage(new[1:-1]) - operator.storage(old[1:-1])
    ).sum(axis=0)
    np.testing.assert_allclose(actual_change, increment, rtol=0, atol=3e-12)


def test_binding_rejects_bad_callback_shapes_and_stencil_declarations():
    model = _UserBalance()
    model.source_dependency_radius = None
    with pytest.raises(ValueError, match="source_dependency_radius"):
        Operator(model, Scheme(flux="hll"))
    model.source_dependency_radius = 3
    model.cell_conserved = lambda physical: physical[:, 0]
    with pytest.raises(ValueError, match="cell_conserved.*shape"):
        Operator(model, Scheme(flux="hll"))
