"""Reusable bindings reset solve state and explicit grid policies retain failures."""

from dataclasses import replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import (
    ResidualPolicy,
    Scheme,
    SteadySettings,
    SteadyWorkspace,
    solve_grid_sequence,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.fvm import drivers


def test_workspace_step_reuses_traces_when_state_and_timestep_values_change():
    class CountingAdvection(AdvectionModel):
        def __init__(self):
            super().__init__(4)
            self.flux_traces = 0

        def face_physical_flux(self, state):
            self.flux_traces += 1
            return super().face_physical_flux(state)

    model = CountingAdvection()
    workspace = SteadyWorkspace(model, Scheme(flux="hll"))
    state = jnp.asarray(model.initial_state(), dtype=jnp.float64)
    binding_traces = model.flux_traces
    state, info = workspace.operator.step(state, jnp.float64(0.01), jnp.float64(0.0))
    jax.block_until_ready((state, info))
    compiled_traces = model.flux_traces
    assert compiled_traces > binding_traces
    for dt, time in ((0.02, 0.02), (0.03, 0.05)):
        state, info = workspace.operator.step(state, jnp.float64(dt), jnp.float64(time))
        jax.block_until_ready((state, info))
        assert bool(info.converged and info.admissible)
        assert model.flux_traces == compiled_traces


def test_workspace_replay_reuses_bindings_and_resets_numerical_state():
    model = AdvectionModel(4)
    scheme = Scheme(flux="hll")
    settings = SteadySettings(cfl=0.3, max_steps=100)
    workspace = SteadyWorkspace(model, scheme, settings.residual)
    original = np.asarray(model.initial_state()).copy()
    with patch.object(
        drivers, "Operator", side_effect=AssertionError("operator rebound")
    ), patch.object(drivers, "sampler", side_effect=AssertionError("sampler rebound")):
        first = solve_single_grid(model, scheme, settings, workspace=workspace)
        second = solve_single_grid(model, scheme, settings, workspace=workspace)
    assert first.success and second.success
    assert first.operator is second.operator is workspace.operator
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(model.initial_state(), original)
    assert first.history is not second.history
    for a, b in zip(first.history, second.history, strict=True):
        for key in ("phase", "step", "cfl", "dt"):
            assert a[key] == b[key]
        if "equation_norm" in a:
            np.testing.assert_array_equal(a["equation_norm"], b["equation_norm"])
    assert second.history[0]["cfl"] == settings.cfl
    assert second.history[0]["step"] == 0
    assert not np.array_equal(
        first.history[0]["equation_norm"], first.metrics["equation_norm"]
    )


def test_workspace_cannot_be_reused_with_different_physics_numerics_or_policy():
    model = AdvectionModel(4)
    scheme = Scheme(flux="hll")
    workspace = SteadyWorkspace(model, scheme)
    selections = (
        (AdvectionModel(4), scheme, SteadySettings()),
        (model, replace(scheme, reconstruction="muscl_van_albada"), SteadySettings()),
        (
            model,
            scheme,
            SteadySettings(residual=ResidualPolicy(absolute_tolerance=1e-8)),
        ),
    )
    for selected_model, selected_scheme, settings in selections:
        with pytest.raises(ValueError, match="workspace must match"):
            solve_single_grid(
                selected_model, selected_scheme, settings, workspace=workspace
            )


def test_sequence_selects_scheme_by_grid_and_replays_cached_workspaces():
    models = {n: AdvectionModel(n) for n in (4, 8)}
    schemes = {
        n: Scheme(
            flux="hll", reconstruction=("constant" if n == 4 else "muscl_van_albada")
        )
        for n in models
    }
    settings = SteadySettings(max_steps=100)
    workspaces = {
        n: SteadyWorkspace(models[n], schemes[n], settings.residual) for n in models
    }
    scheme_calls = []

    def choose(n):
        scheme_calls.append(n)
        return schemes[n]

    with patch.object(
        drivers, "Operator", side_effect=AssertionError("operator rebound")
    ), patch.object(drivers, "sampler", side_effect=AssertionError("sampler rebound")):
        results = [
            solve_grid_sequence(
                models.__getitem__,
                tuple(models),
                settings=settings,
                scheme_factory=choose,
                workspace_factory=lambda model, scheme, selected: workspaces[
                    model.mesh.n_cells
                ],
                retry_cold=False,
                stop_on_failure=True,
            )
            for _ in range(2)
        ]
    assert scheme_calls == [4, 8, 4, 8]
    for final, stages in results:
        assert final.success and len(stages) == 2
        assert [s["reconstruction"] for s in stages] == ["constant", "muscl_van_albada"]
        assert stages[1]["initialization"] == "componentwise linear prolongation"
        assert not any(s["attempts"] for s in stages)
    np.testing.assert_array_equal(results[0][0].state, results[1][0].state)
    assert [s["initial_cfl"] for s in results[0][1]] == [
        s["initial_cfl"] for s in results[1][1]
    ]


def test_strict_sequence_stops_at_failed_coarse_solve():
    requested = []

    def factory(n):
        requested.append(n)
        return AdvectionModel(n)

    final, stages = solve_grid_sequence(
        factory,
        (4, 8, 16),
        Scheme(flux="hll"),
        SteadySettings(max_steps=0),
        stop_on_failure=True,
        retry_cold=False,
    )
    assert requested == [4]
    assert not final.success and final.reason == "max_steps"
    assert len(stages) == 1 and not stages[0]["success"]
    assert stages[0]["attempts"] == []


def test_strict_sequence_does_not_retry_failed_transferred_solve(monkeypatch):
    actual_solve = drivers.solve_single_grid
    calls = []

    def solve(model, scheme, settings, **kwargs):
        calls.append((model.mesh.n_cells, kwargs.get("initial_state") is not None))
        if model.mesh.n_cells == 8:
            settings = replace(settings, max_steps=0)
            kwargs["initial_state"] = np.asarray(model.initial_state())
        return actual_solve(model, scheme, settings, **kwargs)

    monkeypatch.setattr(drivers, "solve_single_grid", solve)
    final, stages = solve_grid_sequence(
        AdvectionModel,
        (4, 8, 16),
        Scheme(flux="hll"),
        SteadySettings(max_steps=100),
        retry_cold=False,
        stop_on_failure=True,
    )
    assert calls == [(4, False), (8, True)]
    assert not final.success and len(stages) == 2
    assert stages[1]["attempts"] == []
    assert stages[1]["initialization"] == "componentwise linear prolongation"
