"""Physical continuation retains attempts and recovers rectangular storage."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, Solution, SteadySettings, solve_grid_sequence
from nozzle_hyperbolic.solvers.fvm.continuation import (
    load_restart, prepare_continuation_state, save_restart, solve_parameter_sequence,
)


class _ConstrainedModel:
    nvar, ntransport, nconstraints, nboundary = 2, 1, 1, 2
    nonconservative_mode = "none"
    equation_names = ("content",)
    constraint_names = ("constitutive relation",)
    constraint_scales = np.ones(1)
    conservative_scales = np.ones(1)
    global_flux_scales = np.ones(1)
    variable_scales = np.ones(2)
    reference_time = 1.
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, cells=3, offset=0., target=1.):
        faces = np.linspace(0., 1., cells + 1) ** 1.3
        self.mesh = Mesh(faces, .5 * (faces[:-1] + faces[1:]), np.diff(faces), np.ones(cells + 1))
        self.offset, self.target = offset, target

    def initial_state(self):
        return jnp.tile(jnp.array([1., 1. + self.offset]), (self.mesh.n_cells + 2, 1))

    def cell_conserved(self, cells):
        return cells[..., :1] ** 2

    face_conserved = cell_conserved

    def cell_constraints(self, cells):
        return cells[..., 1:] - cells[..., :1] ** 2 - self.offset

    def face_physical_flux(self, faces):
        return jnp.zeros(faces.shape[:-1] + (1,))

    def face_speed_bounds(self, faces):
        return -jnp.ones(faces.shape[:-1]), jnp.ones(faces.shape[:-1])

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return jnp.asarray(self.mesh.volumes)[:, None] * (self.target - self.cell_conserved(full[1:-1]))

    def boundary_residual(self, full, time=0.):
        traces = full[jnp.array([0, -1])]
        return jnp.concatenate((traces[:, :1] - jnp.sqrt(self.target), self.cell_constraints(traces)), axis=1)

    def cell_admissible(self, cells):
        return jnp.all(jnp.isfinite(cells)) & jnp.all(cells[..., 0] > 0)

    def admissible(self, full, left, right):
        return self.cell_admissible(full) & self.cell_admissible(left) & self.cell_admissible(right)

    def wave_speed(self, cells):
        return jnp.ones(cells.shape[:-1])


SCHEME = Scheme(flux="hll")


def _source_solution(model):
    operator = Operator(model, SCHEME)
    state = np.asarray(model.initial_state())
    return Solution(state, True, "fixture", [], {}, operator=operator)


def test_mesh_and_parameter_transfer_recovers_content_and_new_constraint():
    source = _ConstrainedModel(cells=3, offset=.2)
    previous = _source_solution(source)
    state = previous.state.copy()
    state[:, 0] = 1 + .2 * source.mesh.points
    state[:, 1] = state[:, 0] ** 2 + source.offset
    previous.state = state
    target = _ConstrainedModel(cells=5, offset=.7)
    prepared, record = prepare_continuation_state(target, previous, SCHEME)
    assert record["success"], record
    q = np.interp(target.mesh.centers, source.mesh.centers, state[1:-1, 0] ** 2)
    np.testing.assert_allclose(target.cell_conserved(prepared[1:-1])[:, 0], q, atol=2e-13)
    np.testing.assert_allclose(target.cell_constraints(prepared[1:-1]), 0, atol=2e-13)
    np.testing.assert_allclose(target.boundary_residual(prepared), 0, atol=2e-13)
    assert record["elapsed_s"] >= record["recovery_s"] > 0
    assert not np.array_equal(record["source_inventory"], record["target_inventory"])


def test_continuation_reuses_target_operator_and_rejects_mismatched_binding():
    from unittest.mock import patch
    previous = _source_solution(_ConstrainedModel(cells=3, offset=.2))
    target = _ConstrainedModel(cells=5, offset=.7)
    operator = Operator(target, SCHEME)
    with patch.object(Operator, "__init__", side_effect=AssertionError("operator rebound")):
        first, record = prepare_continuation_state(target, previous, SCHEME, operator=operator)
        second, repeated = prepare_continuation_state(target, previous, SCHEME, operator=operator)
    assert record["success"] and repeated["success"]
    assert record["operator_reused"] and repeated["operator_reused"]
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(target.cell_constraints(first[1:-1]), 0., atol=2e-13)
    with pytest.raises(ValueError, match="operator must match"):
        prepare_continuation_state(_ConstrainedModel(cells=5), previous, SCHEME, operator=operator)
    with pytest.raises(ValueError, match="operator must match"):
        prepare_continuation_state(target, previous, Scheme(flux="rusanov"), operator=operator)


def test_restart_roundtrip_is_nonpickle_and_needs_recovery(tmp_path):
    previous = _source_solution(_ConstrainedModel())
    path = tmp_path / "restart.npz"
    save_restart(path, previous, metadata={"case": "synthetic constraint audit"})
    loaded = load_restart(path)
    np.testing.assert_array_equal(loaded.state, previous.state)
    assert loaded.metadata["metadata"]["case"] == "synthetic constraint audit"
    prepared, record = prepare_continuation_state(_ConstrainedModel(offset=.6), loaded, SCHEME)
    assert record["success"]
    np.testing.assert_allclose(prepared[:, 1], 1.6, atol=2e-13)
    with pytest.raises(FileExistsError):
        save_restart(path, previous)


def test_parameter_sequence_returns_failed_requested_stage_and_retains_history():
    # First stage is exactly steady; the requested target needs evolution but
    # the declared zero update budget cannot complete it. Earlier success must
    # never be presented as the requested endpoint.
    final, stages = solve_parameter_sequence(
        lambda parameters: _ConstrainedModel(**parameters),
        [dict(target=1.), dict(target=2., offset=.5)], SCHEME,
        SteadySettings(max_steps=0),
    )
    assert stages[0]["success"] and not final.success
    assert stages[-1]["requested_final_stage"]
    assert stages[-1]["preparation"]["success"]
    assert len(stages[-1]["attempts"]) == 2
    assert [attempt["success"] for attempt in stages[-1]["attempts"]] == [False, False]
    for attempt in stages[-1]["attempts"]:
        assert attempt["history"] and attempt["state"].shape == (5, 2)
        assert attempt["solve_elapsed_s"] >= 0
    assert final.operator.model.target == 2.


def test_unsuccessful_source_is_not_reused_and_final_stage_still_runs():
    previous = _source_solution(_ConstrainedModel())
    previous.success = False
    full, record = prepare_continuation_state(_ConstrainedModel(offset=.3), previous, SCHEME)
    assert full is None and record["reason"] == "source_solution_unsuccessful"
    diagnostic, prepared = prepare_continuation_state(
        _ConstrainedModel(offset=.3), previous, SCHEME, allow_unsuccessful=True,
    )
    assert prepared["success"] and not prepared["source_success"]
    assert prepared["allow_unsuccessful"] and not previous.success
    np.testing.assert_allclose(diagnostic[:, 1], 1.3, atol=2e-13)
    final, records = solve_parameter_sequence(
        lambda parameters: _ConstrainedModel(**parameters),
        [dict(target=2.), dict(target=1.)], SCHEME, SteadySettings(max_steps=0),
    )
    assert not records[0]["success"] and final.success
    assert records[1]["preparation"]["reason"] == "source_solution_unsuccessful"
    assert records[1]["attempts"][0]["initialization"] == "cold physical guess"


def test_incompatible_equations_are_rejected():
    previous = _source_solution(_ConstrainedModel())
    target = _ConstrainedModel()
    target.equation_names = ("different content",)
    with pytest.raises(ValueError, match="matching state/content"):
        prepare_continuation_state(target, previous, SCHEME)


def test_public_scalar_physical_speed_continuation_converges():
    from nozzle_hyperbolic.models.scalar import AdvectionModel

    final, records = solve_parameter_sequence(
        lambda parameters: AdvectionModel(4, speed=parameters["speed"]),
        [dict(speed=.5), dict(speed=-.75)], SCHEME, SteadySettings(max_steps=100),
    )
    assert final.success, final.reason
    assert all(record["success"] for record in records)
    assert records[1]["preparation"]["success"]
    assert final.operator.model.speed == -.75


def test_constrained_grid_sequence_recovers_before_carrying_cfl():
    solutions = []
    final, stages = solve_grid_sequence(
        lambda cells: _ConstrainedModel(cells=cells, target=2., offset=.3),
        (3,5,7), SCHEME, SteadySettings(cfl=.2, cfl_growth=2., max_steps=100),
        on_stage=lambda model, solution, metadata: solutions.append(solution),
    )
    assert final.success and all(stage["success"] for stage in stages)
    for previous, stage, solution in zip(stages, stages[1:], solutions[1:]):
        assert stage["preparation"]["success"]
        assert stage["initial_cfl"] == previous["last_accepted_cfl"]
        np.testing.assert_allclose(solution.operator.model.cell_constraints(solution.state[1:-1]), 0., atol=1e-12)
