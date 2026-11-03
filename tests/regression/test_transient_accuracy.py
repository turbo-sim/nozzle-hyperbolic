"""Complete transient solutions, exact wave limits and inventory budgets."""

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar import AdvectionModel, BurgersModel
from nozzle_hyperbolic.solvers.fvm import Scheme, solve_transient

from tests._support.regression import (
    TRANSIENT_SCHEME,
    assert_transient_budget,
    cell_averages,
)
from tests._support.transient_cases import (
    annular_contact,
    perturbed_two_fluid,
    sod_case,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("speed", [-1.0, 1.0], ids=["left-going", "right-going"])
def test_public_advection_converges_to_exact_conserved_cell_averages(speed):
    errors = []
    for count in (32, 64):
        model = AdvectionModel(count, speed=speed)
        initial = (
            model.initial_state()
            .at[1:-1, 0]
            .set(cell_averages(model.mesh.faces, lambda x: model.profile(x)))
        )
        time = 0.1
        solution = solve_transient(
            model, (time,), TRANSIENT_SCHEME, cfl=0.4, initial_state=initial
        )[-1]
        expected = cell_averages(model.mesh.faces, lambda x: model.profile(x, time))
        error = np.average(
            np.abs(solution.state[1:-1, 0] - expected), weights=model.mesh.volumes
        )
        errors.append(error)
        assert_transient_budget(model, initial, solution)
    assert errors[1] < 0.004, errors
    assert errors[1] < 0.4 * errors[0], errors


@pytest.mark.parametrize(
    "left,right",
    [(1.0, -0.5), (-1.0, 1.0)],
    ids=["compressive-shock", "entropy-rarefaction"],
)
def test_burgers_matches_the_entropy_solution(left, right):
    model = BurgersModel(128, left=left, right=right)
    initial, time = model.initial_state(), 0.12
    solution = solve_transient(model, (time,), TRANSIENT_SCHEME, cfl=0.4)[-1]
    fronts = (
        model.location + (left + right) * time / 2,
        model.location + left * time,
        model.location + right * time,
    )
    expected = cell_averages(model.mesh.faces, lambda x: model.profile(x, time), fronts)
    error = np.average(
        np.abs(solution.state[1:-1, 0] - expected), weights=model.mesh.volumes
    )
    assert error < 0.015, error
    assert_transient_budget(model, initial, solution)
    if left > right:
        crossing = np.flatnonzero(solution.state[1:-2, 0] > (left + right) / 2)[-1]
        position = np.mean(model.mesh.centers[crossing : crossing + 2])
        assert abs(position - fronts[0]) <= 2 * max(model.mesh.widths)
    else:
        assert np.min(np.diff(solution.state[:, 0])) > -1e-10


def test_sod_matches_projected_exact_states_before_boundary_interaction():
    model, initial, time, expected, fronts = sod_case()
    solution = solve_transient(model, (time,), TRANSIENT_SCHEME, cfl=0.4)[-1]
    actual = np.asarray(model.cell_conserved(jnp.asarray(solution.state[1:-1])))
    error = np.average(
        np.abs((actual - expected) / model.conservative_scales),
        weights=model.mesh.volumes,
        axis=0,
    )
    assert max(error) < 0.015, error
    assert 0 < min(fronts) < max(fronts) < model.mesh.faces[-1]
    assert_transient_budget(model, initial, solution)


def test_annular_contact_preserves_pressure_and_translates_conserved_contents():
    model, initial, time, expected = annular_contact()
    solution = solve_transient(model, (time,), TRANSIENT_SCHEME, cfl=0.4)[-1]
    actual = np.asarray(model.cell_conserved(jnp.asarray(solution.state[1:-1])))
    error = np.average(
        np.abs((actual - expected) / model.conservative_scales),
        weights=model.mesh.volumes,
        axis=0,
    )
    assert max(error) < 0.005, error
    np.testing.assert_allclose(solution.state[1:-1, 0], 1e5, rtol=3e-8)
    assert_transient_budget(model, initial, solution)


def test_two_fluid_complete_coupled_advance_retains_eos_and_inventory_audits():
    model, initial, time = perturbed_two_fluid()
    scheme = Scheme(
        flux="model_compatibility",
        reconstruction="muscl_van_leer",
        time_integrator="euler-implicit",
        newton_backtracking=True,
    )
    solution = solve_transient(model, (time,), scheme, initial_state=initial)[-1]
    assert_transient_budget(model, initial, solution)
    defects = (
        model.cell_constraints(jnp.asarray(solution.state[1:-1]))
        / model.constraint_scales
    )
    np.testing.assert_allclose(defects, 0, atol=1e-9)
    assert not np.array_equal(solution.state, np.asarray(initial))
