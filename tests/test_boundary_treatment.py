"""Coupled and eliminated boundary solves satisfy the same physical step."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.boundaries import (
    make_boundary_eliminator,
    make_boundary_solver,
)
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme


def _boundary_rows(full, time):
    return jnp.stack(
        (
            full[0] ** 2 - 2.0 - 0.3 * full[1] - 0.2 * full[2] - 0.1 * time,
            full[-1] ** 3 - 1.0 - 0.4 * full[-2] + 0.2 * time,
        )
    )


def _nonlinear_model(cells=11):
    model = AdvectionModel(cells, speed=0.7)
    model.boundary_dependency_radius = 2
    model.boundary_residual = _boundary_rows
    model.initial_state = lambda: jnp.linspace(1.1, 1.3, cells + 2)[:, None]
    return model


@pytest.mark.parametrize(
    "method", ["euler-implicit", "euler-explicit", "runge-kutta-3"]
)
def test_auto_preserves_existing_boundary_treatment_defaults(method):
    scheme = Scheme(time_integrator=method)
    assert scheme.newton_iterations == 10
    assert scheme.newton_backtracking is False
    expected = "coupled" if method == "euler-implicit" else "segregated"
    assert scheme.effective_boundary_treatment == expected
    operator = Operator(AdvectionModel(3), Scheme(flux="hll", time_integrator=method))
    assert operator.effective_boundary_treatment == expected


@pytest.mark.parametrize("method", ["euler-explicit", "runge-kutta-3"])
def test_explicit_coupled_boundary_treatment_is_rejected(method):
    with pytest.raises(ValueError, match="Explicit.*segregated"):
        Scheme(time_integrator=method, boundary_treatment="coupled")
    assert (
        Scheme(
            time_integrator=method, boundary_treatment="segregated"
        ).effective_boundary_treatment
        == "segregated"
    )


def test_unknown_boundary_treatment_is_rejected():
    with pytest.raises(ValueError, match="boundary_treatment"):
        Scheme(boundary_treatment="one-pass")


@pytest.mark.parametrize("mode", ["coupled", "segregated"])
def test_implicit_newton_default_takes_full_valid_correction(mode):
    model = AdvectionModel(3, speed=0.7)
    model.cell_conserved = lambda cells: cells**3
    model.face_physical_flux = lambda states: jnp.zeros_like(states)
    model.integrated_source = lambda full: jnp.asarray(model.mesh.volumes)[:, None]
    model.boundary_residual = lambda full, time: jnp.stack(
        (full[0] - 0.1, full[-1] - 0.1)
    )
    old = jnp.full((5, 1), 0.1)
    operators = [
        Operator(
            model,
            Scheme(
                flux="hll",
                boundary_treatment=mode,
                newton_iterations=1,
                newton_backtracking=flag,
            ),
        )
        for flag in (False, True)
    ]
    full, full_info = operators[0].step(old, 1.0, 0.0)
    damped, damped_info = operators[1].step(old, 1.0, 0.0)
    # The first Newton correction of q^3 - old_q^3 = 1 is 1/(3*0.1^2).
    # Full Newton may increase the residual; Armijo must shorten this update.
    np.testing.assert_allclose(full[1:-1], 0.1 + 1.0 / 0.03, rtol=1e-12)
    assert float(jnp.max(damped[1:-1])) < float(jnp.min(full[1:-1]))
    assert bool(full_info.admissible) and bool(damped_info.admissible)
    assert int(full_info.rejected_trials) == 0
    assert int(damped_info.rejected_trials) > 0
    assert not bool(full_info.converged)


def test_newton_backtracking_requires_a_boolean():
    with pytest.raises(ValueError, match="newton_backtracking"):
        Scheme(newton_backtracking="false")


def test_eliminated_trace_jvp_is_the_implicit_function_derivative_in_cells_and_time():
    physical = jnp.asarray([[1.2], [1.3], [1.4]])
    seed = jnp.ones((5, 1)) * 1.5
    complete = make_boundary_solver(
        _boundary_rows,
        seed.shape,
        [1.0],
        admissible=lambda full: jnp.all(jnp.isfinite(full)) & jnp.all(full > 0),
    )
    eliminate = make_boundary_eliminator(_boundary_rows, complete)
    time = jnp.float64(0.2)
    dphysical, dtime = jnp.asarray([[0.3], [-0.2], [0.1]]), jnp.float64(0.7)
    full, tangent = jax.jit(
        lambda cells, instant: jax.jvp(
            lambda c, t: eliminate(c, seed, t), (cells, instant), (dphysical, dtime)
        )
    )(physical, time)
    left = np.sqrt(2.0 + 0.3 * 1.2 + 0.2 * 1.3 + 0.1 * 0.2)
    right = np.cbrt(1.0 + 0.4 * 1.4 - 0.2 * 0.2)
    expected = np.concatenate(
        (
            [[(0.3 * 0.3 - 0.2 * 0.2 + 0.1 * 0.7) / (2.0 * left)]],
            np.asarray(dphysical),
            [[(0.4 * 0.1 - 0.2 * 0.7) / (3.0 * right**2)]],
        )
    )
    np.testing.assert_allclose(full[[0, -1], 0], [left, right], rtol=2e-14)
    np.testing.assert_allclose(tangent, expected, rtol=3e-13, atol=3e-14)
    # Changing only the Newton seed cannot change the derivative of a fixed root.
    seed_tangent = jax.jvp(
        lambda guess: eliminate(physical, guess, time), (seed,), (jnp.ones_like(seed),)
    )[1]
    np.testing.assert_array_equal(seed_tangent, 0.0)


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_reduced_jacobian_is_the_full_nonlinear_schur_complement(reconstruction):
    model = _nonlinear_model()
    operator = Operator(
        model,
        Scheme(
            flux="hll", reconstruction=reconstruction, boundary_treatment="segregated"
        ),
    )
    old = model.initial_state()
    cells, dt, time = old[1:-1].ravel() * 1.01, 0.017, 0.13
    full, info = operator.complete_boundaries(old.at[1:-1].set(cells[:, None]), time)
    assert bool(info.converged)
    coupled = np.asarray(operator.dense_jacobian(full.ravel(), old.ravel(), dt, time))
    interior = np.arange(1, len(full) - 1)
    traces = np.asarray([0, len(full) - 1])
    expected = coupled[np.ix_(interior, interior)] - coupled[
        np.ix_(interior, traces)
    ] @ np.linalg.solve(
        coupled[np.ix_(traces, traces)], coupled[np.ix_(traces, interior)]
    )
    args = (cells, old.ravel(), dt, time)
    dense = np.asarray(operator.reduced_dense_jacobian(*args))
    np.testing.assert_allclose(dense, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(
        operator.reduced_jacobian(*args), dense, rtol=3e-13, atol=3e-13
    )
    direction = np.asarray(operator.reduced_direction(*args))
    np.testing.assert_allclose(
        dense @ direction, -operator.reduced_residual(*args), atol=3e-13
    )
    perturbation = np.random.default_rng(17).normal(size=cells.shape)
    epsilon = 2e-6
    numerical = (
        operator.reduced_residual(cells + epsilon * perturbation, *args[1:])
        - operator.reduced_residual(cells - epsilon * perturbation, *args[1:])
    ) / (2 * epsilon)
    np.testing.assert_allclose(dense @ perturbation, numerical, rtol=2e-8, atol=2e-9)


def test_coupled_and_segregated_steps_solve_identical_backward_euler_and_budgets():
    model = _nonlinear_model()
    operators = [
        Operator(model, Scheme(flux="hll", boundary_treatment=mode))
        for mode in ("coupled", "segregated")
    ]
    old, dt, time = model.initial_state(), 0.017, 0.13
    results = []
    for operator in operators:
        new, info, increment = operator.step_with_balance(old, dt, time)
        assert bool(info.converged) and bool(info.admissible)
        np.testing.assert_allclose(
            operator.residual(new.ravel(), old.ravel(), dt, time), 0.0, atol=2e-12
        )
        np.testing.assert_allclose(model.boundary_residual(new, time), 0.0, atol=2e-12)
        change = np.asarray(
            operator.storage(new[1:-1]) - operator.storage(old[1:-1])
        ).sum(axis=0)
        np.testing.assert_allclose(change, increment, atol=2e-12)
        results.append(np.asarray(new))
    np.testing.assert_allclose(results[0], results[1], rtol=2e-12, atol=2e-12)


def test_global_boundary_elimination_retains_dense_corner_dependencies():
    model = _nonlinear_model(8)
    model.boundary_coupling = "global"
    model.boundary_residual = lambda full, time: jnp.stack(
        (full[0] ** 2 - 1.0 - 0.2 * full[-2], full[-1] ** 3 - 1.0 - 0.3 * full[1])
    )
    with pytest.raises(ValueError, match="Global boundary"):
        Operator(model, Scheme(flux="hll", boundary_treatment="segregated"))
    operator = Operator(
        model,
        Scheme(
            flux="hll",
            boundary_treatment="segregated",
            jacobian_mode="dense",
            linear_solver="dense",
        ),
    )
    old = model.initial_state()
    args = (old[1:-1].ravel(), old.ravel(), 0.02, 0.0)
    jacobian = np.asarray(operator.reduced_jacobian(*args))
    assert abs(jacobian[0, -1]) > 1e-4
    np.testing.assert_allclose(
        jacobian @ operator.reduced_direction(*args),
        -operator.reduced_residual(*args),
        atol=3e-13,
    )
    new, info = operator.step(old, 0.02, 0.0)
    assert bool(info.converged)
    np.testing.assert_allclose(
        operator.residual(new.ravel(), old.ravel(), 0.02, 0.0), 0.0, atol=2e-12
    )


def test_failed_boundary_elimination_is_not_reported_as_an_implicit_root():
    model = _nonlinear_model(4)
    # A finite boundary state cannot satisfy this algebraic constraint.
    model.boundary_residual = lambda full, time: jnp.ones((2, 1))
    operator = Operator(model, Scheme(flux="hll", boundary_treatment="segregated"))
    old = model.initial_state()
    new, info = operator.step(old, 0.01, 0.0)
    assert not bool(info.converged)
    assert int(info.status) != 0
    assert np.all(np.isfinite(new))


def _euler_model(kind):
    fluid = jxp.FluidPerfectGas("air")
    pressure, temperature, velocity = 1.5e5, 310.0, 20.0
    swirl = 0.0 if kind == "nozzle" else 5.0
    state = fluid.get_state(jxp.PT_INPUTS, pressure, temperature)
    h0 = state["h"] + 0.5 * (velocity**2 + swirl**2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, h0, state["s"])
    if kind == "nozzle":
        mesh = ConstantAreaGeometry(length=0.04, area=3e-4).build_mesh(6)
        full = jnp.broadcast_to(jnp.asarray([pressure, velocity, state["h"]]), (8, 3))
        return NozzleModel(
            mesh,
            fluid,
            p0=float(reservoir["p"]),
            T0=float(reservoir["T"]),
            back_pressure=pressure,
            initial_state=full.at[3, 1].add(2.0),
        )
    mesh, native = build_annular_mesh(conical_channel(0.04, 0.1, 0.005, phi=0.0), 6)
    full = jnp.broadcast_to(
        jnp.asarray([pressure, velocity, 0.1 * swirl, state["h"]]), (8, 4)
    )
    return AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=dict(
            p0=float(reservoir["p"]),
            T0=float(reservoir["T"]),
            h0=float(h0),
            s0=float(state["s"]),
            ell_in=0.1 * swirl,
        ),
        outlet_pressure=pressure,
        initial_state=full.at[3, 1].add(2.0),
    )


@pytest.mark.parametrize("kind", ["nozzle", "annular"])
@pytest.mark.parametrize(
    "method,mode",
    [
        ("euler-implicit", "coupled"),
        ("euler-implicit", "segregated"),
        ("euler-explicit", "segregated"),
        ("runge-kutta-3", "segregated"),
    ],
)
def test_euler_models_accept_every_supported_integrator_boundary_combination(
    kind, method, mode
):
    model = _euler_model(kind)
    operator = Operator(model, Scheme(time_integrator=method, boundary_treatment=mode))
    old = model.initial_state()
    new, info, increment = operator.step_with_balance(old, 1e-6, 1e-6)
    assert operator.effective_boundary_treatment == mode
    assert bool(info.converged) and bool(info.admissible)
    assert bool(operator.is_admissible(new))
    assert np.max(np.abs(operator.boundary_residual(new, 1e-6))) < 1e-10
    change = np.asarray(operator.storage(new[1:-1]) - operator.storage(old[1:-1])).sum(
        axis=0
    )
    scale = np.sum(model.mesh.volumes) * np.asarray(model.conservative_scales)
    assert np.max(np.abs((change - increment) / scale)) < 2e-11


def test_strong_pressure_release_with_automatic_finite_wave_outlet_matches_in_both_modes():
    fluid = jxp.FluidPerfectGas("air")
    static = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    mesh = ConstantAreaGeometry(length=0.2, area=1.0).build_mesh(8)
    initial = jnp.broadcast_to(jnp.asarray([1e5, 0.0, static["h"]]), (10, 3))
    model = NozzleModel(
        mesh,
        fluid,
        p0=1e5,
        T0=300.0,
        back_pressure=1e3,
        outlet_regime="auto",
        outlet_formulation="finite_wave",
        initial_state=initial,
    )
    results = []
    for mode in ("coupled", "segregated"):
        operator = Operator(model, Scheme(boundary_treatment=mode))
        new, info, increment = operator.step_with_balance(initial, 1e-6, 1e-6)
        assert bool(info.converged) and bool(info.admissible)
        assert np.max(np.abs(operator.boundary_residual(new, 1e-6))) < 1e-10
        change = np.asarray(
            operator.storage(new[1:-1]) - operator.storage(initial[1:-1])
        ).sum(axis=0)
        scale = np.sum(mesh.volumes) * np.asarray(model.conservative_scales)
        assert np.max(np.abs((change - increment) / scale)) < 2e-11
        boundary = model.thermodynamic_state(new[-1])
        np.testing.assert_allclose(new[-1, 1] / boundary["a"], 1.0, atol=2e-10)
        results.append(np.asarray(new))
    np.testing.assert_allclose(
        (results[1] - results[0]) / model.variable_scales, 0.0, atol=2e-11
    )
