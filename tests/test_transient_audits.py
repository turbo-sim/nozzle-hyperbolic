"""Transient physical audits are independent of final steady branch checks."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.fvm import (
    Scheme,
    SteadySettings,
    solve_single_grid,
    solve_transient,
)


@pytest.mark.parametrize("method", ["euler-explicit", "euler-implicit"])
def test_transient_audit_override_and_recorded_conservative_transfers(method):
    model = AdvectionModel(8, speed=1.0)
    model.solution_audit = lambda state: False
    model.transient_solution_audit = lambda state: True
    (result,) = solve_transient(
        model, (0.01,), Scheme(flux="hll", time_integrator=method), cfl=0.5
    )
    assert result.success, result.reason
    first = np.asarray(result.operator.storage(model.initial_state()[1:-1])).sum(axis=0)
    transfer = sum(
        np.asarray(row["integrated_transfer"])
        for row in result.history
        if row["accepted"]
    )
    final = np.asarray(result.operator.storage(result.state[1:-1])).sum(axis=0)
    np.testing.assert_allclose(final - first, transfer, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        result.history[-1]["conserved_totals"], final, rtol=0.0, atol=1.0e-14
    )


def test_transient_audit_falls_back_to_existing_solution_audit():
    model = AdvectionModel(6)
    model.solution_audit = lambda state: False
    (result,) = solve_transient(
        model, (0.001,), Scheme(flux="hll", time_integrator="euler-explicit")
    )
    assert not result.success and result.reason == "solution_audit_failed"


def test_steady_solver_retains_the_steady_model_audit():
    model = AdvectionModel(6, speed=0.0)
    model.solution_audit = lambda state: False
    model.transient_solution_audit = lambda state: True
    result = solve_single_grid(model, Scheme(flux="hll"), SteadySettings(max_steps=3))
    assert not result.success and result.reason == "solution_audit_failed"


def test_invalid_transient_audit_callback_is_rejected():
    model = AdvectionModel(6)
    model.transient_solution_audit = True
    with pytest.raises(ValueError, match="transient solution audit"):
        solve_transient(model, (0.001,), Scheme(flux="hll"))
