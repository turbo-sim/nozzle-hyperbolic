"""Physical annular adapters and opt-in steady convergence contracts."""

from collections import namedtuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel
from nozzle_hyperbolic.models.euler_annular.fvm import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.monitoring import AnnularMonitoring, EQUATIONS
from nozzle_hyperbolic.solvers.monitoring import MonitorSpec
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy


@pytest.fixture(scope="module")
def setup():
    fluid = jxp.FluidPerfectGas(name="Air")
    static = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    p, h, s = (float(static[key]) for key in ("p", "h", "s"))
    vm, vt, radius = 40.0, 20.0, 0.2
    h0 = h+0.5*(vm**2+vt**2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    inlet = dict(h0=h0, s0=s, p0=float(reservoir["p"]), alpha_in=np.arctan2(vt, vm))
    geometry = conical_channel(0.1, radius, 0.005, 0.0)
    mesh = geometry.mesh(m_faces=[0.0, 0.005, 0.02, 0.05, 0.1])
    solver = make_annular_solver(fluid, mesh, inlet_data=inlet, outlet_pressure=p,
                                 newton_iterations=8)
    state = jnp.tile(jnp.asarray([p, vm, radius*vt, h]), (mesh.n_cells+2, 1))
    return solver, state


def test_equation_norms_use_original_rows_and_nonuniform_volume_weights(setup):
    solver, exact = setup
    state = exact.at[1:-1, 0].multiply(jnp.asarray([1.01, 1.005, 1.003, 1.002]))
    state = state.at[1:-1, 1].add(jnp.asarray([0.1, 0.3, 0.2, 0.4]))
    adapter = AnnularMonitoring(solver)
    metrics, _ = adapter.sample(state)
    signed = np.asarray(solver.spatial_residual(state))
    scaled = signed/solver.spatial_flux_scales
    weights = solver.mesh.volumes/np.sum(solver.mesh.volumes)
    np.testing.assert_allclose(metrics["equation_l1"], np.sum(weights[:, None]*abs(scaled), axis=0), rtol=2e-13)
    np.testing.assert_allclose(metrics["equation_l2"], np.sqrt(np.sum(weights[:, None]*scaled**2, axis=0)), rtol=2e-13)
    np.testing.assert_allclose(metrics["equation_linf"], np.max(abs(scaled), axis=0), rtol=2e-13)
    np.testing.assert_array_equal(metrics["equation_worst_cell"], np.argmax(abs(scaled), axis=0))
    np.testing.assert_array_equal(metrics["equation_relative"], np.ones(4))
    assert adapter.metadata["equation_names"] == list(EQUATIONS)
    # Existing all-equation equal-cell RMS retains its exact public meaning.
    np.testing.assert_allclose(metrics["spatial_rms"], np.sqrt(np.mean(scaled**2)), rtol=2e-13)
    assert not np.isclose(metrics["spatial_rms"], np.sqrt(np.mean(np.asarray(metrics["equation_l2"])**2)))


def test_cp_and_mass_flow_use_defined_domain_and_actual_face_flux(setup):
    solver, exact = setup
    state = exact.at[1:-1, 0].add(jnp.asarray([10.0, 30.0, 20.0, 40.0]))
    state = state.at[-1, 0].add(50.0).at[1:-1, 1].add(jnp.asarray([0.1, 0.3, 0.2, 0.4]))
    specs = (MonitorSpec("cp_mean", "pressure_coefficient", "length_mean"),
             MonitorSpec("cp_exit", "cp", "outlet"),
             MonitorSpec("cp_integral", "cp", "length_integral"),
             MonitorSpec("mdot_in", "mass_flow", "inlet"),
             MonitorSpec("mdot_out", "mdot", "outlet"),
             MonitorSpec("mdot_mean", "mass_flow", "mean"))
    adapter = AnnularMonitoring(solver, monitors=specs)
    _, values = adapter.sample(state)
    p0, pin = solver.inlet_data["p0"], float(state[0, 0])
    cp = (np.asarray(state[1:-1, 0])-pin)/(p0-pin)
    expected_integral = np.sum(solver.mesh.widths*cp)
    assert values["cp_integral"] == pytest.approx(expected_integral)
    assert values["cp_mean"] == pytest.approx(expected_integral/np.sum(solver.mesh.widths))
    assert values["cp_exit"] == pytest.approx(50.0/(p0-pin))
    flux = np.asarray(solver.face_flux(state))[:, 0]
    assert values["mdot_in"] == pytest.approx(flux[0])
    assert values["mdot_out"] == pytest.approx(flux[-1])
    assert values["mdot_mean"] == pytest.approx(np.sum(solver.mesh.widths*(flux[:-1]+flux[1:])/2)/sum(solver.mesh.widths))


def test_initially_solved_root_uses_absolute_checks_without_invented_reduction(setup):
    solver, exact = setup
    records = []
    result = solver.solve_steady_monitored(exact, max_steps=0, callback=records.append)
    assert result.success, result.message
    assert result.monitoring["reason"] == "converged"
    assert result.monitoring["accepted_updates"] == 0
    assert len(records) == 1
    assert records[0]["phase"] == "initial" and records[0]["step"] == 0
    assert not any(result.final_metrics["relative_active"])
    assert all(result.final_metrics["equation_passed"])


def test_required_monitor_window_needs_real_accepted_updates(setup):
    solver, exact = setup
    spec = MonitorSpec("mdot", "mass_flow", "outlet", convergence=True, window=3,
                       absolute_tolerance=1e-8)
    result = solver.solve_steady_monitored(exact, monitors=(spec,), max_steps=3)
    assert result.success, result.message
    assert result.monitoring["accepted_updates"] == 2
    assert [row["step"] for row in result.history] == [0, 1, 2]
    assert result.history[0]["monitors"]["mdot"]["samples"] == 1
    assert not result.history[0]["monitors_passed"]
    assert result.final_metrics["monitors"]["mdot"]["stable"]
    assert result.final_metrics["monitors"]["mdot"]["samples"] == 3
    # Fresh audit must not append a fourth, duplicate window sample.
    assert result.history[-1]["monitors"] == result.final_metrics["monitors"]
    np.testing.assert_allclose(result.state, exact, rtol=2e-13, atol=1e-8)


def test_unsolved_initial_state_cannot_pass_at_update_limit(setup):
    solver, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    result = solver.solve_steady_monitored(initial, max_steps=0)
    assert not result.success
    assert result.monitoring["reason"] == "max_iterations"
    active = np.asarray(result.final_metrics["relative_active"])
    np.testing.assert_array_equal(np.asarray(result.final_metrics["equation_relative"])[active], 1.0)
    np.testing.assert_array_equal(result.state, initial)


def test_rejected_trials_do_not_reuse_previous_outer_diagnostics_or_monitor_samples(setup, monkeypatch):
    solver, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    Info = namedtuple("Info", "converged admissible iterations residual_norm status rejected_trials relative_update")
    info = Info(False, False, 8, np.inf, 1, 2, np.inf)
    monkeypatch.setattr(solver, "step", lambda state, dt: (state, info))
    monitor = MonitorSpec("mdot", "mass_flow", "outlet", convergence=True, window=3)
    result = solver.solve_steady_monitored(initial, monitors=(monitor,), max_retries=1)
    assert not result.success
    assert result.monitoring["reason"] == "retry_limit"
    assert len(result.history) == 3
    for row in result.history[1:]:
        assert row["phase"] == "rejected" and not row["accepted"]
        assert not row["diagnostics_evaluated"]
        assert "equation_l2" not in row and "monitors" not in row
    assert result.final_metrics["monitors"]["mdot"]["samples"] == 1
    assert result.history[2]["cfl"] == pytest.approx(result.history[1]["cfl"]*0.5)


def test_unchanged_state_at_maximum_cfl_is_failure_not_convergence(setup, monkeypatch):
    solver, exact = setup
    initial = exact.at[1:-1, 1].add(1.0)
    Info = namedtuple("Info", "converged admissible iterations residual_norm status rejected_trials relative_update")
    monkeypatch.setattr(solver, "step", lambda state, dt: (state, Info(True, True, 0, 0., 0, 0, 0.)))
    result = solver.solve_steady_monitored(initial, cfl=1., cfl_max=1.,
                                          stagnation_steps=2, max_steps=100)
    assert not result.success
    assert result.monitoring["reason"] == "stagnation"
    assert result.monitoring["accepted_updates"] == 2
    assert not result.final_metrics["all_equations_passed"]
    np.testing.assert_array_equal(result.state, initial)


def test_monitor_setup_rejects_unknown_quantities_and_reductions(setup):
    solver, _ = setup
    for spec in (MonitorSpec("x", "unknown", "outlet"),
                 MonitorSpec("x", "cp", "mean"),
                 MonitorSpec("x", "mass_flow", "length_integral")):
        with pytest.raises(ValueError):
            AnnularMonitoring(solver, monitors=(spec,))


def test_supersonic_cp_needs_a_reservoir_reference(setup):
    solver, exact = setup
    supersonic = make_annular_solver(solver.fluid, solver.mesh,
                                    inlet_data=dict(state=np.asarray(exact[0])),
                                    inlet_regime="supersonic", outlet_regime="supersonic")
    spec = MonitorSpec("cp", "cp", "outlet")
    with pytest.raises(ValueError, match="pressure_total"):
        AnnularMonitoring(supersonic, monitors=(spec,))
    adapter = AnnularMonitoring(supersonic, monitors=(spec,), pressure_total=2e5)
    assert adapter.metadata["monitors"][0]["resolved_pressure_total"] == 2e5


def test_invalid_initial_state_does_not_evaluate_monitoring_eos(setup, monkeypatch):
    solver, exact = setup
    monkeypatch.setattr(AnnularMonitoring, "sample", lambda *args: pytest.fail("invalid state sampled"))
    result = solver.solve_steady_monitored(exact.at[2, 0].set(-1.0))
    assert not result.success
    assert result.monitoring["reason"] == "invalid_initial_state"
    assert result.history == []


def test_small_actual_solve_meets_every_relative_and_absolute_target(setup):
    solver, exact = setup
    initial = exact.at[1:-1, 0].multiply(1+1e-3*jnp.asarray([1.0, 0.5, 0.25, 0.125]))
    # A controlled local solve checks the new stop flag against a fresh signed
    # spatial operator. It does not use an analytical profile as a cold benchmark.
    policy = ResidualPolicy(relative_tolerance=1e-9)
    result = solver.solve_steady_monitored(initial, residual_policy=policy,
                                          max_steps=250, cfl=0.2)
    assert result.success, (result.message, result.final_metrics)
    active = np.asarray(result.final_metrics["relative_active"])
    relative = np.asarray(result.final_metrics["equation_relative"])
    assert np.all(relative[active] <= 1e-9)
    assert max(result.final_metrics["equation_norm"]) <= 1e-8
    fresh, _ = AnnularMonitoring(solver, policy).sample(jnp.asarray(result.state),
                                                     result.monitoring["initial_equation_norms"])
    np.testing.assert_allclose(fresh["equation_l2"], result.final_metrics["equation_l2"], rtol=1e-12)
    assert fresh["all_equations_passed"] and fresh["boundary_passed"] and fresh["global_passed"]
