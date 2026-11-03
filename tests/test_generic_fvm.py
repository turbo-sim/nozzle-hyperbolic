"""The public FVM engine exercised by independent user-programmed models."""

import json
import subprocess
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import (
    Mesh,
    Operator,
    ResidualPolicy,
    Scheme,
    SteadySettings,
    solve_grid_sequence,
    solve_single_grid,
    solve_transient,
)
from nozzle_hyperbolic.solvers.residuals import assess_residuals
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec

jax.config.update("jax_enable_x64", True)


def _mesh(count=8):
    faces = np.linspace(0.0, 1.0, count + 1) ** 1.2
    return Mesh(
        faces, (faces[:-1] + faces[1:]) / 2.0, np.diff(faces), np.ones(count + 1)
    )


class _UserBalance:
    """A non-Euler two-equation law with nonlinear storage and a wide source.

    This class intentionally has no EOS or characteristic methods. Its
    artificial source exercises the declaration of a radius-three stencil.
    """

    nvar = 2
    equation_names = ("first_content", "second_content")
    conservative_scales = np.asarray([1.0, 2.0])
    variable_scales = np.asarray([1.0, 2.0])
    global_flux_scales = np.asarray([1.0, 2.0])
    reference_time = 0.7
    source_dependency_radius = 3
    boundary_dependency_radius = 3
    boundary_coupling = "local"

    def __init__(self):
        self.mesh = _mesh()

    def cell_conserved(self, physical):
        return physical + 0.03 * physical**3

    face_conserved = cell_conserved

    def face_physical_flux(self, physical):
        return self.cell_conserved(physical) * jnp.asarray([0.7, -0.4])

    def face_speed_bounds(self, physical):
        return (jnp.full(physical.shape[:-1], -0.4), jnp.full(physical.shape[:-1], 0.7))

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        indices = jnp.arange(1, self.mesh.n_cells + 1)
        left = full[jnp.maximum(indices - 3, 0)]
        right = full[jnp.minimum(indices + 3, self.mesh.n_cells + 1)]
        return (
            0.04
            * jnp.asarray(self.mesh.volumes)[:, None]
            * (left - 2.0 * full[1:-1] + right)
        )

    def boundary_residual(self, full, time):
        return jnp.stack(
            (full[0] ** 2 - 0.2 * full[3] - 1.0 - time, full[-1] - 0.3 * full[-4] - 0.8)
        )

    def wave_speed(self, physical):
        return jnp.full(physical.shape[:-1], 0.7)

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full)) & jnp.all(full > 0)

    def initial_state(self):
        return jnp.linspace(1.0, 1.5, (self.mesh.n_cells + 2) * self.nvar).reshape(
            -1, 2
        )


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
                flux="hll", reconstruction=reconstruction,
                jacobian_mode="dense", linear_solver="dense",
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
            Scheme(flux="stadtke", custom_jump_correction=False, custom_endpoint_envelope=False),
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


class _UserAdvection:
    nvar = 1
    equation_names = ("advected_content",)
    conservative_scales = variable_scales = global_flux_scales = np.ones(1)
    reference_time = 1.0
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, speed, count=16):
        self.mesh, self.speed = _mesh(count), speed

    def cell_conserved(self, physical):
        return physical

    face_conserved = cell_conserved
    flux_to_conserved_frame = cell_conserved

    def face_physical_flux(self, physical):
        return self.speed * physical

    def face_speed_bounds(self, physical):
        speed = jnp.full(physical.shape[:-1], self.speed)
        return speed, speed

    def wave_speed(self, physical):
        return jnp.full(physical.shape[:-1], abs(self.speed))

    def integrated_source(self, full):
        return jnp.zeros_like(full[1:-1])

    @staticmethod
    def exact(x, time, speed):
        return 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * (x - speed * time))

    def boundary_residual(self, full, time):
        if self.speed > 0:
            return jnp.stack(
                (full[0] - self.exact(0.0, time, self.speed), full[-1] - full[-2])
            )
        return jnp.stack(
            (full[0] - full[1], full[-1] - self.exact(1.0, time, self.speed))
        )

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full))

    def initial_state(self):
        return self.exact(jnp.asarray(self.mesh.points), 0.0, self.speed)[:, None]


@pytest.mark.parametrize("speed", [-0.7, 0.7])
@pytest.mark.parametrize("flux", ["hll", "rusanov"])
@pytest.mark.parametrize(
    "integrator", ["euler-implicit", "euler-explicit", "runge-kutta-3"]
)
def test_signed_user_advection_all_integrators_without_eigenstructure(
    speed, flux, integrator
):
    errors = []
    for count in (16, 32):
        model = _UserAdvection(speed, count)
        (final,) = solve_transient(
            model, (0.03,), Scheme(flux=flux, time_integrator=integrator), cfl=0.4
        )
        assert final.success, final.reason
        error = np.abs(
            final.state[1:-1, 0] - model.exact(model.mesh.centers, 0.03, speed)
        )
        errors.append(np.dot(model.mesh.volumes, error))
        assert np.max(np.abs(final.metrics["conservation_budget"])) < 1e-12
        assert np.max(np.abs(model.boundary_residual(final.state, final.time))) < 1e-12
    # First-order face reconstruction must converge toward the translated wave.
    assert errors[1] < 0.7 * errors[0], errors


def test_public_steady_driver_solves_a_user_programmed_model_and_audits_original_rows():
    model = _UserAdvection(0.7)
    result = solve_single_grid(
        model,
        Scheme(flux="hll"),
        SteadySettings(cfl=0.5, cfl_growth=2.0, max_steps=100),
    )
    assert result.success, result.reason
    np.testing.assert_allclose(result.state, 1.0, rtol=0, atol=2e-9)
    assert (
        np.max(
            np.abs(result.operator.steady_residual(jnp.asarray(result.state).ravel()))
        )
        < 1e-8
    )
    assert result.metrics["boundary_passed"] and result.metrics["global_passed"]


def test_grid_sequence_cold_retry_retains_a_user_quantity_evaluator_factory():
    coarse_model = _UserAdvection(0.7, 8)
    coarse = solve_single_grid(
        coarse_model, Scheme(flux="hll"), initial_state=np.ones((10, 1))
    )
    assert coarse.success

    def factory(count):
        # A stricter model domain makes the prolonged coarse state invalid,
        # while the model's own cold guess is already the new steady root.
        model = _UserAdvection(0.7, count)
        model.initial_state = lambda: jnp.full((count + 2, 1), 1.5)
        model.admissible = lambda full, left, right: jnp.all(
            jnp.isfinite(full)
        ) & jnp.all(full > 1.2)
        model.boundary_residual = lambda full, time: jnp.stack(
            (full[0] - 1.5, full[-1] - full[-2])
        )
        return model

    def quantities(model):
        return lambda state, flux: {"mean": float(jnp.mean(state[1:-1]))}

    result, stages = solve_grid_sequence(
        factory,
        (16,),
        Scheme(flux="hll"),
        initial_solution=coarse,
        monitor_specs=(MonitorSpec("mean", "mean", "custom"),),
        quantity_evaluator_factory=quantities,
    )
    assert result.success, result.reason
    assert stages[0]["attempts"][0]["reason"] == "invalid_initial_state"
    assert stages[0]["initialization"] == "cold restart after retained warm failure"
    np.testing.assert_allclose(result.metrics["monitors"]["mean"]["value"], 1.5)


@pytest.mark.parametrize("integrator", ["euler-explicit", "runge-kutta-3"])
@pytest.mark.parametrize("cfl", [0.2, 1.0, 3.0])
def test_explicit_source_timestep_is_an_absolute_cap_after_wave_cfl(integrator, cfl):
    model = _UserAdvection(0.7)
    source_limit = 2e-4
    model.explicit_timestep = lambda full: jnp.asarray(source_limit)
    operator = Operator(model, Scheme(flux="hll", time_integrator=integrator))
    limits = operator.timestep(model.initial_state(), cfl)
    assert bool(limits.valid)
    assert float(limits.source_dt) == source_limit
    expected = 1.0 / (1.0 / (cfl * float(limits.wave_dt)) + 1.0 / source_limit)
    np.testing.assert_allclose(limits.dt, expected, rtol=2e-15)
    assert float(limits.dt) <= source_limit
    np.testing.assert_allclose(
        operator.compute_dt(model.initial_state(), cfl), limits.dt
    )


@pytest.mark.parametrize("integrator", ["euler-explicit", "runge-kutta-3"])
def test_transient_driver_does_not_multiply_absolute_source_cap_by_cfl(integrator):
    model = _UserAdvection(0.7)
    source_limit = 2e-4
    model.explicit_timestep = lambda full: jnp.asarray(source_limit)
    (final,) = solve_transient(
        model, (7e-4,), Scheme(flux="hll", time_integrator=integrator), cfl=3.0
    )
    assert final.success, final.reason
    accepted = [row for row in final.history if row["accepted"]]
    assert len(accepted) == 4
    assert all(row["dt"] <= source_limit for row in accepted)
    assert np.max(np.abs(final.metrics["conservation_budget"])) < 1e-12


@pytest.mark.parametrize("bad_limit", [0.0, -1.0, np.nan])
def test_invalid_explicit_source_bounds_are_rejected_and_implicit_does_not_apply_them(
    bad_limit,
):
    model = _UserAdvection(0.7)
    model.explicit_timestep = lambda full: jnp.asarray(bad_limit)
    explicit = Operator(model, Scheme(flux="hll", time_integrator="euler-explicit"))
    limits = explicit.timestep(model.initial_state(), 0.4)
    assert not bool(limits.valid)
    assert np.isnan(limits.dt)
    implicit = Operator(model, Scheme(flux="hll", time_integrator="euler-implicit"))
    limits = implicit.timestep(model.initial_state(), 3.0)
    assert bool(limits.valid)
    assert np.isinf(limits.source_dt)
    np.testing.assert_allclose(limits.dt, 3.0 * limits.wave_dt)


def test_infinite_source_bound_leaves_wave_cfl_unchanged():
    model = _UserAdvection(0.7)
    model.explicit_timestep = lambda full: jnp.asarray(jnp.inf)
    operator = Operator(model, Scheme(flux="hll", time_integrator="euler-explicit"))
    limits = operator.timestep(model.initial_state(), 0.4)
    assert bool(limits.valid)
    np.testing.assert_allclose(limits.dt, 0.4 * limits.wave_dt)


class _UserFilteredAdvection(_UserAdvection):
    filter_conserved = _UserAdvection.cell_conserved

    def filter_signal_speed(self, full):
        return jnp.full((self.mesh.n_cells + 1,), abs(self.speed))


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


def test_zero_signal_speed_and_infinite_source_bound_are_safe():
    model = _UserFilteredAdvection(0.0)
    model.boundary_residual = lambda full, time: jnp.stack(
        (full[0] - full[1], full[-1] - full[-2])
    )
    model.explicit_timestep = lambda full: jnp.asarray(jnp.inf)
    operator = Operator(
        model, Scheme(flux="hll", time_integrator="euler-explicit", filter_strength=1.0)
    )
    limits = operator.timestep(model.initial_state(), 0.4)
    assert bool(limits.valid) and np.isfinite(limits.dt) and limits.dt > 0
    (final,) = solve_transient(model, (0.03,), operator.scheme)
    assert final.success
    np.testing.assert_array_equal(final.state[1:-1], model.initial_state()[1:-1])


@pytest.mark.parametrize(
    "criterion,expected",
    [
        ("relative", [True, False]),
        ("absolute", [False, True]),
        ("either", [True, True]),
        ("both", [False, False]),
    ],
)
def test_relative_absolute_stopping_modes_have_distinct_meanings(criterion, expected):
    policy = ResidualPolicy(criterion=criterion)
    # First equation meets only relative; second equation meets only absolute.
    result = assess_residuals(
        jnp.asarray([1e-5, 1e-10]), jnp.asarray([100.0, 1e-5]), policy
    )
    np.testing.assert_array_equal(result["equation_passed"], expected)
    assert policy.relative_tolerance == 1e-6
    assert policy.absolute_tolerance == 1e-9
    assert SteadySettings().residual == ResidualPolicy()


@pytest.mark.parametrize("criterion", ["relative", "absolute", "either", "both"])
def test_initially_zero_equations_use_an_explicit_absolute_fallback(criterion):
    policy = ResidualPolicy(criterion=criterion)
    current = jnp.asarray([0.0, 5e-10, 2e-9, 5e-10])
    initial = jnp.asarray([0.0, 0.0, 0.0, policy.initial_floor / 2])
    result = assess_residuals(current, initial, policy)
    np.testing.assert_array_equal(result["relative_active"], [False] * 4)
    np.testing.assert_array_equal(result["equation_passed"], [True, True, False, True])
    assert not bool(result["all_equations_passed"])


def test_binding_rejects_bad_callback_shapes_and_stencil_declarations():
    model = _UserBalance()
    model.source_dependency_radius = None
    with pytest.raises(ValueError, match="source_dependency_radius"):
        Operator(model, Scheme(flux="hll"))
    model.source_dependency_radius = 3
    model.cell_conserved = lambda physical: physical[:, 0]
    with pytest.raises(ValueError, match="cell_conserved.*shape"):
        Operator(model, Scheme(flux="hll"))


def test_plain_package_and_numerical_engine_import_no_models_eos_or_plotting():
    code = """
import json
import sys
import nozzle_hyperbolic
plain = [name for name in sys.modules if name.startswith(('nozzle_hyperbolic.models', 'jaxprop', 'matplotlib'))]
import nozzle_hyperbolic.solvers.fvm
core = [name for name in sys.modules if name.startswith(('nozzle_hyperbolic.models', 'jaxprop', 'matplotlib'))]
print(json.dumps({'plain': plain, 'core': core}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {"plain": [], "core": []}
