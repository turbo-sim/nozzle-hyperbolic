"""Public solve drivers preserve model contracts and physical outcomes."""

import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import (
    Scheme,
    SteadySettings,
    solve_grid_sequence,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec

from tests._support.generic_models import _UserAdvection


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
