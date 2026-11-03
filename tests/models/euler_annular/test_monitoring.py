"""Physical quantity reductions and shared steady convergence contracts."""

from collections import namedtuple
from dataclasses import replace

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel, AnnularQuantities
from nozzle_hyperbolic.solvers.fvm import (
    Operator,
    Scheme,
    SteadySettings,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.fvm import drivers
from nozzle_hyperbolic.solvers.fvm.drivers import sampler
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy


@pytest.fixture(scope="module")
def setup():
    fluid = jxp.FluidPerfectGas(name="Air")
    static = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    p, h, s = (float(static[key]) for key in ("p", "h", "s"))
    vm, vt, radius = 40.0, 20.0, 0.2
    h0 = h + 0.5 * (vm**2 + vt**2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    inlet = dict(h0=h0, s0=s, p0=float(reservoir["p"]), alpha_in=np.arctan2(vt, vm))
    geometry = conical_channel(0.1, radius, 0.005, 0.0)
    mesh, native = build_annular_mesh(geometry, 4, faces=[0.0, 0.005, 0.02, 0.05, 0.1])
    model = AnnularModel(mesh, native, fluid, inlet_data=inlet, outlet_pressure=p)
    operator = Operator(model, Scheme(newton_iterations=8))
    state = jnp.tile(jnp.asarray([p, vm, radius * vt, h]), (6, 1))
    return operator, state


def _run(operator, state, *, monitors=(), policy=None, **options):
    settings = SteadySettings(residual=policy or ResidualPolicy(), **options)
    evaluator = AnnularQuantities(operator.model, monitors)
    return solve_single_grid(
        operator.model,
        operator.scheme,
        settings,
        initial_state=state,
        monitor_specs=monitors,
        quantity_evaluator=evaluator,
    )


def test_equation_norms_use_original_rows_and_nonuniform_volume_weights(setup):
    operator, exact = setup
    state = exact.at[1:-1, 0].multiply(jnp.asarray([1.01, 1.005, 1.003, 1.002]))
    state = state.at[1:-1, 1].add(jnp.asarray([0.1, 0.3, 0.2, 0.4]))
    metrics = sampler(operator, ResidualPolicy())(state)
    raw = np.asarray(operator.spatial_residual(state))
    model = operator.model
    scales = (
        operator.mesh.volumes[:, None]
        * model.conservative_scales
        / model.reference_time
    )
    scaled = raw / scales
    weights = operator.mesh.volumes / np.sum(operator.mesh.volumes)
    np.testing.assert_allclose(
        metrics["equation_l1"],
        np.sum(weights[:, None] * abs(scaled), axis=0),
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        metrics["equation_l2"],
        np.sqrt(np.sum(weights[:, None] * scaled**2, axis=0)),
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        metrics["equation_linf"], np.max(abs(scaled), axis=0), rtol=2e-13
    )
    np.testing.assert_array_equal(metrics["worst_cell"], np.argmax(abs(scaled), axis=0))
    np.testing.assert_array_equal(metrics["equation_relative"], np.ones(4))
    assert len(model.equation_names) == 4
    # Nonuniform physical-volume weighting must differ from equal-cell RMS.
    assert not np.isclose(
        np.sqrt(np.mean(scaled**2)),
        np.sqrt(np.mean(np.asarray(metrics["equation_l2"]) ** 2)),
    )


def test_cp_and_mass_flow_use_defined_domain_and_actual_face_flux(setup):
    operator, exact = setup
    state = exact.at[1:-1, 0].add(jnp.asarray([10.0, 30.0, 20.0, 40.0]))
    state = state.at[-1, 0].add(50.0).at[1:-1, 1].add(jnp.asarray([0.1, 0.3, 0.2, 0.4]))
    specs = (
        MonitorSpec("cp_mean", "pressure_coefficient", "length_mean"),
        MonitorSpec("cp_exit", "cp", "outlet"),
        MonitorSpec("cp_integral", "cp", "length_integral"),
        MonitorSpec("mdot_in", "mass_flow", "inlet"),
        MonitorSpec("mdot_out", "mdot", "outlet"),
        MonitorSpec("mdot_mean", "mass_flow", "mean"),
    )
    values = AnnularQuantities(operator.model, specs)(state, operator.face_flux(state))
    p0, pin = operator.model.inlet_data["p0"], float(state[0, 0])
    cp = (np.asarray(state[1:-1, 0]) - pin) / (p0 - pin)
    integral = np.sum(operator.mesh.widths * cp)
    assert values["cp_integral"] == pytest.approx(integral)
    assert values["cp_mean"] == pytest.approx(integral / np.sum(operator.mesh.widths))
    assert values["cp_exit"] == pytest.approx(50.0 / (p0 - pin))
    flux = np.asarray(operator.face_flux(state))[:, 0]
    assert values["mdot_in"] == pytest.approx(flux[0])
    assert values["mdot_out"] == pytest.approx(flux[-1])
    assert values["mdot_mean"] == pytest.approx(
        np.sum(operator.mesh.widths * (flux[:-1] + flux[1:]) / 2)
        / sum(operator.mesh.widths)
    )


def test_initially_solved_root_uses_absolute_checks_without_invented_reduction(setup):
    operator, exact = setup
    records = []
    result = solve_single_grid(
        operator.model,
        operator.scheme,
        SteadySettings(max_steps=0),
        initial_state=exact,
        callback=records.append,
    )
    assert result.success, result.reason
    assert result.reason == "converged"
    assert (
        len(records) == 1
        and records[0]["phase"] == "initial"
        and records[0]["step"] == 0
    )
    assert not any(result.metrics["relative_active"])
    assert result.metrics["all_equations_passed"]
    assert all(result.metrics["absolute_satisfied"])


def test_required_monitor_window_needs_real_accepted_updates(setup):
    operator, exact = setup
    spec = MonitorSpec(
        "mdot",
        "mass_flow",
        "outlet",
        convergence=True,
        window=3,
        absolute_tolerance=1e-8,
    )
    result = _run(operator, exact, monitors=(spec,), max_steps=3)
    assert result.success, result.reason
    assert sum(row["phase"] == "accepted" for row in result.history) == 2
    assert [row["step"] for row in result.history] == [0, 1, 2]
    assert result.history[0]["monitors"]["mdot"]["samples"] == 1
    assert not result.history[0]["monitors"]["mdot"]["stable"]
    assert result.metrics["monitors"]["mdot"]["stable"]
    assert result.metrics["monitors"]["mdot"]["samples"] == 3
    assert result.history[-1]["monitors"] == result.metrics["monitors"]
    np.testing.assert_allclose(result.state, exact, rtol=2e-13, atol=1e-8)


def test_unsolved_initial_state_cannot_pass_at_update_limit(setup):
    operator, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    result = _run(operator, initial, max_steps=0)
    assert not result.success and result.reason == "max_steps"
    active = np.asarray(result.metrics["relative_active"])
    np.testing.assert_array_equal(
        np.asarray(result.metrics["equation_relative"])[active], 1.0
    )
    np.testing.assert_array_equal(result.state, initial)


def test_rejected_trials_do_not_reuse_previous_diagnostics_or_monitor_samples(
    setup, monkeypatch
):
    operator, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    Info = namedtuple(
        "Info",
        "converged admissible iterations residual_norm status rejected_trials relative_update",
    )
    info = Info(False, False, 8, np.inf, 1, 2, np.inf)
    monkeypatch.setattr(operator, "step", lambda state, dt, time: (state, info))
    monkeypatch.setattr(drivers, "Operator", lambda model, scheme: operator)
    spec = MonitorSpec("mdot", "mass_flow", "outlet", convergence=True, window=3)
    result = _run(operator, initial, monitors=(spec,), max_retries=1)
    assert not result.success and result.reason == "retry_limit"
    assert len(result.history) == 3
    for row in result.history[1:]:
        assert row["phase"] == "rejected" and not row["accepted"]
        assert "equation_l2" not in row and "monitors" not in row
    assert result.metrics["monitors"]["mdot"]["samples"] == 1
    assert result.history[2]["cfl"] == pytest.approx(result.history[1]["cfl"] * 0.5)


def test_unchanged_state_at_maximum_cfl_is_failure_not_convergence(setup, monkeypatch):
    operator, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    Info = namedtuple(
        "Info",
        "converged admissible iterations residual_norm status rejected_trials relative_update",
    )
    monkeypatch.setattr(
        operator,
        "step",
        lambda state, dt, time: (state, Info(True, True, 0, 0.0, 0, 0, 0.0)),
    )
    monkeypatch.setattr(drivers, "Operator", lambda model, scheme: operator)
    result = _run(
        operator, initial, cfl=1.0, cfl_max=1.0, stagnation_steps=2, max_steps=100
    )
    assert not result.success and result.reason == "stagnation"
    assert sum(row["phase"] == "accepted" for row in result.history) == 2
    assert not result.metrics["all_equations_passed"]
    np.testing.assert_array_equal(result.state, initial)


def test_monitor_setup_rejects_unknown_quantities_and_reductions(setup):
    operator, _ = setup
    for spec in (
        MonitorSpec("x", "unknown", "outlet"),
        MonitorSpec("x", "cp", "mean"),
        MonitorSpec("x", "mass_flow", "length_integral"),
    ):
        with pytest.raises(ValueError):
            AnnularQuantities(operator.model, (spec,))


def test_supersonic_cp_needs_a_reservoir_reference(setup):
    operator, exact = setup
    model = operator.model
    supersonic = AnnularModel(
        model.mesh,
        model.native_mesh,
        model.fluid,
        inlet_data=dict(state=np.asarray(exact[0])),
        outlet_pressure=None,
        inlet_regime="supersonic",
        outlet_regime="supersonic",
    )
    spec = MonitorSpec("cp", "cp", "outlet")
    with pytest.raises(ValueError, match="pressure_total"):
        AnnularQuantities(supersonic, (spec,))
    evaluator = AnnularQuantities(supersonic, (spec,), pressure_total=2e5)
    assert evaluator.metadata[0]["resolved_pressure_total"] == 2e5


def test_invalid_initial_state_does_not_evaluate_monitoring_eos(setup, monkeypatch):
    operator, exact = setup
    monkeypatch.setattr(
        drivers, "sampler", lambda *args: pytest.fail("invalid state sampled")
    )
    result = _run(operator, exact.at[2, 0].set(-1.0))
    assert not result.success and result.reason == "invalid_initial_state"
    assert result.history == []


def test_small_actual_solve_meets_every_relative_and_absolute_target(setup):
    operator, exact = setup
    # The requested nine-order outer target is stricter than production's
    # defaults. Solve each implicit step accurately enough to resolve it.
    operator = Operator(
        operator.model,
        replace(
            operator.scheme,
            newton_residual_tolerance=1e-14,
            newton_update_tolerance=1e-12,
        ),
    )
    initial = exact.at[1:-1, 0].multiply(
        1 + 1e-3 * jnp.asarray([1.0, 0.5, 0.25, 0.125])
    )
    policy = ResidualPolicy(
        relative_tolerance=1e-9, absolute_tolerance=1e-8, criterion="both"
    )
    result = _run(operator, initial, policy=policy, max_steps=250, cfl=0.2)
    assert result.success, (result.reason, result.metrics)
    active = np.asarray(result.metrics["relative_active"])
    relative = np.asarray(result.metrics["equation_relative"])
    assert np.all(relative[active] <= 1e-9)
    assert max(result.metrics["equation_norm"]) <= 1e-8
    initial_norm = sampler(result.operator, policy)(initial)["equation_norm"]
    fresh = sampler(result.operator, policy)(jnp.asarray(result.state), initial_norm)
    np.testing.assert_allclose(
        fresh["equation_l2"], result.metrics["equation_l2"], rtol=1e-12
    )
    assert (
        fresh["all_equations_passed"]
        and fresh["boundary_passed"]
        and fresh["global_passed"]
    )
