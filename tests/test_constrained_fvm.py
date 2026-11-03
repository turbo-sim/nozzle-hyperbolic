"""Independent constrained-cell laws exercising the generic DAE machinery."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.fvm import (
    Mesh, Operator, Scheme, ResidualPolicy, SteadySettings,
    sampler, solve_single_grid, solve_transient,
)

jax.config.update("jax_enable_x64", True)


class ConstrainedReaction:
    """u_t=-z, z=u**2: exact solution u(t)=u(0)/(1+t*u(0))."""

    nvar, ntransport, nconstraints, nboundary = 2, 1, 1, 2
    nonconservative_mode = "none"
    equation_names = ("content",)
    constraint_names = ("quadratic manifold",)
    variable_scales = np.ones(2)
    conservative_scales = global_flux_scales = np.ones(1)
    constraint_scales = np.array([2.0])
    reference_time = 1.0
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, nonuniform=False, reaction=1.0):
        faces = np.linspace(0, 1, 5) ** (1.6 if nonuniform else 1)
        self.mesh = Mesh(faces, (faces[1:] + faces[:-1]) / 2, np.diff(faces), np.ones(5))
        self.reaction = reaction

    def cell_conserved(self, state):
        return state[:, :1]

    face_conserved = cell_conserved

    def cell_constraints(self, state):
        return state[:, 1:] - state[:, :1] ** 2

    def face_physical_flux(self, state):
        return jnp.zeros_like(state[:, :1])

    def face_speed_bounds(self, state):
        return jnp.zeros(state.shape[0]), jnp.zeros(state.shape[0])

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return -self.reaction * jnp.asarray(self.mesh.volumes)[:, None] * full[1:-1, 1:]

    def boundary_residual(self, full, time=0.0):
        traces = full[jnp.array([0, -1])]
        adjacent = full[jnp.array([1, -2])]
        return jnp.concatenate((traces[:, :1] - adjacent[:, :1],
                                self.cell_constraints(traces) / self.constraint_scales), axis=1)

    def wave_speed(self, cells):
        return jnp.ones(cells.shape[0])

    def cell_admissible(self, cells):
        return jnp.all(jnp.isfinite(cells)) & jnp.all(cells > 0)

    def admissible(self, full, left, right):
        return self.cell_admissible(full)

    def initial_state(self):
        u = 1 + 0.1 * jnp.asarray(self.mesh.points)
        return jnp.stack((u, u**2), axis=1)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_rectangular_storage_recovery_zero_storage_constraints_and_jacobian(nonuniform):
    model = ConstrainedReaction(nonuniform)
    op = Operator(model, Scheme(flux="rusanov"))
    state = model.initial_state()
    target = op.storage(state[1:-1]) * 0.85
    recovered, info = op.recover_storage(target, state[1:-1])
    assert info.converged
    np.testing.assert_allclose(op.storage(recovered), target, atol=2e-14)
    np.testing.assert_allclose(model.cell_constraints(recovered), 0, atol=2e-13)
    trial = state.at[2, 1].add(0.2)
    expected = op.cell_constraints(trial[1:-1])
    for dt in (0.0, 0.1, 1e8):
        packed = op.residual(trial.ravel(), state.ravel(), dt, 0).reshape(state.shape)
        np.testing.assert_array_equal(packed[1:-1, 1:], expected)
    args = (trial.ravel(), state.ravel(), 0.1, 0.0)
    np.testing.assert_allclose(op.jacobian(*args), op.dense_jacobian(*args), atol=2e-14)
    direction = np.random.default_rng(2).normal(size=state.size)
    eps = 1e-6
    finite_difference = (op.residual(args[0] + eps * direction, *args[1:])
                         - op.residual(args[0] - eps * direction, *args[1:])) / (2 * eps)
    np.testing.assert_allclose(op.jacobian(*args) @ direction, finite_difference, atol=2e-9)


def test_coupled_and_eliminated_constrained_cells_solve_same_backward_euler():
    model = ConstrainedReaction(True)
    outputs = []
    for treatment in ("coupled", "segregated"):
        op = Operator(model, Scheme(flux="rusanov", boundary_treatment=treatment))
        state = model.initial_state()
        result, info = op.step(state, 0.1, 0.1)
        assert info.converged
        expected = 2 * state[1:-1, 0] / (1 + jnp.sqrt(1 + 0.4 * state[1:-1, 0]))
        np.testing.assert_allclose(result[1:-1, 0], expected, atol=2e-13)
        np.testing.assert_allclose(op.cell_constraints(result[1:-1]), 0, atol=2e-13)
        outputs.append(result)
    np.testing.assert_allclose(*outputs, atol=2e-13)


@pytest.mark.parametrize("method", ["euler-explicit", "runge-kutta-3", "runge-kutta-4", "euler-implicit"])
def test_transient_constrained_budget_and_stage_recovery(method):
    model = ConstrainedReaction(True)
    solution = solve_transient(model, [0.01], Scheme(flux="rusanov", time_integrator=method))[0]
    assert solution.success, solution.reason
    assert solution.metrics["constraint_max"] < 2e-12
    assert solution.metrics["conservation_budget"].shape == (1,)
    np.testing.assert_allclose(solution.metrics["conservation_budget"], 0, atol=2e-12)
    if method == "runge-kutta-4":
        expected = model.initial_state()[1:-1, 0] / (1 + 0.01 * model.initial_state()[1:-1, 0])
        np.testing.assert_allclose(solution.state[1:-1, 0], expected, atol=1e-11)


def test_constraints_have_independent_fixed_absolute_monitor_and_stopping():
    model = ConstrainedReaction(reaction=0)
    state = jnp.ones((6, 2)).at[2, 1].add(2e-5)
    op = Operator(model, Scheme(flux="rusanov"))
    metrics = sampler(op, ResidualPolicy())(state)
    assert metrics["all_equations_passed"]
    assert metrics["constraint_max"] == pytest.approx(1e-5)
    assert not metrics["constraints_passed"]
    result = solve_single_grid(model, Scheme(flux="rusanov"), SteadySettings(max_steps=0), initial_state=state)
    assert not result.success


def test_explicit_timestep_uses_recovered_algebraic_state():
    class AlgebraicSpeed(ConstrainedReaction):
        def wave_speed(self, cells):
            return cells[:, 1]
    model = AlgebraicSpeed()
    op = Operator(model, Scheme(flux="rusanov", time_integrator="runge-kutta-4"))
    state = jnp.ones((6, 2))
    off_manifold = state.at[:, 1].set(4.)
    np.testing.assert_allclose(op.timestep(off_manifold, 0.2).dt,
                               op.timestep(state, 0.2).dt, atol=1e-14)


class NonconservativeReaction(ConstrainedReaction):
    """Synthetic flux-difference correction with explicit stabilization order."""

    nonconservative_mode = "flux_difference"
    supported_fluxes = ("rusanov",)
    nonconservative_dependency_radius = 1
    stabilization_dependency_radius = 1

    def face_physical_flux(self, state):
        return state[:, :1] ** 2 / 2

    def integrated_nonconservative(self, full, raw):
        return 0.3 * full[1:-1, :1] * (raw[1:] - raw[:-1])

    def stabilization_flux(self, full, raw):
        return -0.07 * (full[1:, :1] - full[:-1, :1])


def test_nonconservative_ordering_budget_and_stencil():
    model = NonconservativeReaction(True)
    op = Operator(model, Scheme(flux="rusanov"))
    state = model.initial_state()
    raw = op.unfiltered_face_flux(state)
    parts = op.residual_contributions(state)
    np.testing.assert_allclose(parts["nonconservative"], model.integrated_nonconservative(state, raw))
    expected = parts["conservative"] + parts["stabilization"] + parts["nonconservative"] - parts["source"]
    np.testing.assert_allclose(op.spatial_residual(state), expected, atol=2e-15)
    np.testing.assert_allclose(op.balance_rate(state), -jnp.sum(expected, axis=0), atol=2e-15)
    assert float(op.jacobian_difference(state.ravel(), state.ravel(), 0.02, 0.0)) < 2e-14
    with pytest.raises(ValueError, match="not supported"):
        Operator(model, Scheme(flux="hll"))


def test_equation_policy_retains_physical_defects_and_invariant_gate():
    class SelectedEquations(ConstrainedReaction):
        has_equation_pins = True
        steady_only = True
        equation_dependency_radius = 0
        invariant_projections = np.ones((1, 1))
        invariant_names = ("content",)
        invariant_flux_scales = np.ones(1)

        def transform_cell_equations(self, full, rows, dt=None):
            return jnp.asarray(self.mesh.volumes)[:, None] * (full[1:-1, :1] - 1)

    model = SelectedEquations()
    state = jnp.ones((6, 2))
    op = Operator(model, Scheme(flux="rusanov"))
    metrics = sampler(op, ResidualPolicy())(state)
    assert metrics["all_equations_passed"]
    np.testing.assert_allclose(metrics["unmodified_equation_norm"], [1.])
    np.testing.assert_allclose(metrics["invariant_defect"], [1.])
    assert not metrics["invariants_passed"]
    result = solve_single_grid(model, Scheme(flux="rusanov"), SteadySettings(max_steps=0), initial_state=state)
    assert not result.success
    with pytest.raises(ValueError, match="equation-changing"):
        Operator(model, Scheme(flux="rusanov", time_integrator="runge-kutta-4"))
    with pytest.raises(ValueError, match="steady-only"):
        solve_transient(model, [0.1], Scheme(flux="rusanov"))
