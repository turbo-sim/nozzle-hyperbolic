"""Grid transfers retain a successful controller state without infecting retries."""

from types import SimpleNamespace

import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import Scheme, Solution, SteadySettings, drivers


def _result(model, cfl=0.2, *, success=True, history=None):
    if history is None:
        history = [dict(phase="initial", cfl=cfl), dict(phase="accepted", cfl=3 * cfl)]
    return Solution(
        np.ones((model.mesh.n_cells + 2, 1)),
        success,
        "converged" if success else "retry_limit",
        history,
        {},
        operator=SimpleNamespace(mesh=model.mesh),
    )


@pytest.mark.parametrize("carry", [True, False])
def test_real_grid_sequence_cfl_and_zero_update_transfers(carry):
    settings = SteadySettings(cfl=0.2, cfl_growth=2.0, max_steps=100)
    solutions = []
    final, stages = drivers.solve_grid_sequence(
        AdvectionModel,
        (4, 8, 16),
        Scheme(flux="hll"),
        settings,
        carry_cfl=carry,
        on_stage=lambda model, solution, metadata: solutions.append(solution),
    )
    assert final.success and all(stage["success"] for stage in stages)
    assert stages[0]["initial_cfl"] == 0.2
    assert stages[0]["last_accepted_cfl"] > 0.2
    for previous, stage, solution in zip(stages, stages[1:], solutions[1:]):
        expected = previous["last_accepted_cfl"] if carry else 0.2
        assert stage["initial_cfl"] == expected
        assert solution.history[0]["cfl"] == expected
        assert stage["cfl_policy"] == ("carry" if carry else "reset")
    # A constant steady scalar profile needs no evolution after prolongation;
    # the inherited controller value must survive another zero-update grid.
    assert all(
        not any(h.get("phase") == "accepted" for h in s.history) for s in solutions[1:]
    )
    assert settings.cfl == 0.2


def test_cold_retry_resets_cfl_and_next_grid_uses_successful_retry(monkeypatch):
    source = _result(
        AdvectionModel(4),
        history=[
            dict(phase="initial", cfl=0.2),
            dict(phase="accepted", cfl=40.0),
            dict(phase="rejected", cfl=320.0),
        ],
    )
    settings = SteadySettings(cfl=0.2, cfl_max=10.0)
    calls = []

    def solve(model, scheme, selected, **kwargs):
        warm = kwargs.get("initial_state") is not None
        calls.append((model.mesh.n_cells, selected.cfl, warm))
        result = _result(
            model, selected.cfl, success=not (model.mesh.n_cells == 8 and warm)
        )
        if result.success and model.mesh.n_cells == 8:
            result.history[-1]["cfl"] = 0.8
        return result

    monkeypatch.setattr(drivers, "solve_single_grid", solve)
    final, stages = drivers.solve_grid_sequence(
        AdvectionModel, (8, 16), settings=settings, initial_solution=source
    )
    assert final.success
    assert calls == [(8, 10.0, True), (8, 0.2, False), (16, 0.8, True)]
    assert stages[0]["attempts"][0]["initial_cfl"] == 10.0
    assert not stages[0]["attempts"][0]["success"]
    assert stages[0]["initial_cfl"] == 0.2
    assert stages[1]["initial_cfl"] == 0.8
    assert settings.cfl == 0.2 and settings.cfl_max == 10.0


@pytest.mark.parametrize(
    "success,history,expected,warm",
    [
        (False, [dict(phase="accepted", cfl=8.0)], 0.2, False),
        (True, [], 0.2, True),
        (True, [dict(phase="accepted", cfl=float("nan"))], 0.2, True),
        (True, [dict(phase="initial", cfl=3.0)], 3.0, True),
        (
            True,
            [dict(phase="accepted", cfl=3.0), dict(phase="rejected", cfl=90.0)],
            3.0,
            True,
        ),
        (True, [dict(phase="accepted", cfl=0.05)], 0.05, True),
    ],
)
def test_external_initial_solution_uses_available_accepted_history(
    monkeypatch,
    success,
    history,
    expected,
    warm,
):
    previous = _result(AdvectionModel(4), success=success, history=history)
    calls = []

    def solve(model, scheme, settings, **kwargs):
        calls.append((settings.cfl, kwargs.get("initial_state") is not None))
        return _result(model, settings.cfl)

    monkeypatch.setattr(drivers, "solve_single_grid", solve)
    final, stages = drivers.solve_grid_sequence(
        AdvectionModel, (8,), initial_solution=previous
    )
    assert final.success and calls == [(expected, warm)]
    assert stages[0]["initial_cfl"] == expected


def test_failed_constraint_preparation_starts_cold_and_resets_cfl(monkeypatch):
    from nozzle_hyperbolic.solvers.fvm import continuation

    previous = _result(AdvectionModel(4), cfl=4.0)

    def factory(count):
        model = AdvectionModel(count)
        model.nconstraints = 1
        return model

    monkeypatch.setattr(
        continuation,
        "prepare_continuation_state",
        lambda *args: (None, dict(success=False, reason="cell_recovery_failed")),
    )
    calls = []

    def solve(model, scheme, settings, **kwargs):
        calls.append((settings.cfl, kwargs.get("initial_state")))
        return _result(model, settings.cfl)

    monkeypatch.setattr(drivers, "solve_single_grid", solve)
    final, stages = drivers.solve_grid_sequence(
        factory, (8,), initial_solution=previous
    )
    assert final.success and calls == [(0.2, None)]
    assert stages[0]["preparation"]["reason"] == "cell_recovery_failed"


@pytest.mark.parametrize("value", [1, 0, None, "carry"])
def test_grid_sequence_requires_explicit_boolean_cfl_policy(value):
    with pytest.raises(ValueError, match="carry_cfl must be a boolean"):
        drivers.solve_grid_sequence(AdvectionModel, (4, 8), carry_cfl=value)
