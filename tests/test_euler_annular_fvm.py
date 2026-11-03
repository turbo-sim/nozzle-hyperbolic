"""Integrated annular contracts: full boundary block, AD stencil and time driver."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.annular import conical_channel, radial_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.solvers.fvm import (
    Operator,
    Scheme,
    SteadySettings,
    solve_single_grid,
    solve_transient,
)
from nozzle_hyperbolic.solvers.fvm.drivers import sampler
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from nozzle_hyperbolic.solvers.fvm import drivers
from nozzle_hyperbolic.models.euler_annular.state import admissible


def _operator(
    fluid,
    native,
    *,
    inlet_data,
    outlet_pressure=None,
    reconstruction="constant",
    **options
):
    mesh, native = build_annular_mesh(
        native.geometry, native.n_cells, faces=native.m_faces
    )
    physical = {
        name: options.pop(name)
        for name in tuple(options)
        if name in ("inlet_regime", "outlet_regime")
    }
    model = AnnularModel(
        mesh,
        native,
        fluid,
        inlet_data=inlet_data,
        outlet_pressure=outlet_pressure,
        **physical
    )
    return Operator(model, Scheme(reconstruction=reconstruction, **options))


def _advance(operator, state, time, **options):
    return solve_transient(
        operator.model, [time], operator.scheme, initial_state=state, **options
    )[0]


class HPInterfaceOnly:
    """Actual JAXProp backend behind the declared gamma-free numerical interface."""

    def __init__(self, fluid):
        self._fluid = fluid

    def get_state(self, pair, h, p):
        if pair != jxp.HmassP_INPUTS:
            raise AssertionError(
                "The numerical factory must use its direct HP interface"
            )
        raw = self._fluid.get_state(pair, h, p)
        return {key: raw[key] for key in ("rho", "a", "G", "T", "s", "p", "h")}


@pytest.fixture(scope="module")
def fixture_data():
    fluid = jxp.FluidPerfectGas(name="Air")
    static = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    p, h, s = map(float, (static["p"], static["h"], static["s"]))
    v_m, v_theta, radius = 40.0, 20.0, 0.2
    h0 = h + 0.5 * (v_m**2 + v_theta**2)
    reservoir = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s)
    inlet = dict(
        h0=h0, s0=s, p0=float(reservoir["p"]), alpha_in=np.arctan2(v_theta, v_m)
    )
    axial = conical_channel(0.10, radius, 0.005, 0.0).mesh(6)
    radial = radial_channel(0.10, 0.20, 0.005).mesh(6)
    return dict(
        fluid=HPInterfaceOnly(fluid),
        backend=fluid,
        inlet=inlet,
        p=p,
        h=h,
        v_m=v_m,
        v_theta=v_theta,
        axial=axial,
        radial=radial,
    )


@pytest.fixture(scope="module", params=["constant", "muscl_van_leer"])
def axial_solver(request, fixture_data):
    data = fixture_data
    solver = _operator(
        data["fluid"],
        data["axial"],
        inlet_data=data["inlet"],
        outlet_pressure=data["p"],
        reconstruction=request.param,
    )
    uniform = jnp.tile(
        jnp.asarray([data["p"], data["v_m"], 0.2 * data["v_theta"], data["h"]]), (8, 1)
    )
    return solver, uniform


def test_full_augmented_coloring_matches_independent_dense_ad(axial_solver):
    solver, uniform = axial_solver
    fraction = jnp.linspace(0.0, 1.0, 8)
    z = (
        uniform.at[:, 0]
        .add(1500.0 * fraction)
        .at[:, 1]
        .add(4.0 * fraction + 0.2 * fraction**2)
        .at[:, 2]
        .add(0.3 * fraction)
        .at[:, 3]
        .add(3000.0 * fraction)
    )
    dt = 1e-5
    colored = np.asarray(solver.jacobian(z.ravel(), uniform.ravel(), dt, 0.0))
    dense = np.asarray(solver.dense_jacobian(z.ravel(), uniform.ravel(), dt, 0.0))
    variable_scale = np.tile([1e5, 400.0, 80.0, 1.6e5], 8)
    np.testing.assert_allclose(
        colored * variable_scale, dense * variable_scale, rtol=2e-10, atol=2e-10
    )
    # Boundary compatibility reads the nearest two physical cells, even for
    # constant interior reconstruction. Those full-Jacobian blocks must exist.
    assert np.linalg.norm(dense[:4, 8:12] * variable_scale[8:12]) > 0
    assert np.linalg.norm(dense[-4:, -12:-8] * variable_scale[-12:-8]) > 0


def test_boundary_rows_have_no_old_trace_or_timestep_storage(axial_solver):
    solver, state = axial_solver
    changed_old = state.at[0, :].multiply(1.1).at[-1, :].multiply(0.9)
    first = solver.residual(state.ravel(), state.ravel(), 1e-5, 0.0)
    second = solver.residual(state.ravel(), changed_old.ravel(), 1e-5, 0.0)
    np.testing.assert_array_equal(first, second)
    later = solver.residual(state.ravel(), changed_old.ravel(), 2e-5, 0.0)
    np.testing.assert_array_equal(first[:4], later[:4])
    np.testing.assert_array_equal(first[-4:], later[-4:])
    old_jacobian = np.asarray(
        jax.jacfwd(lambda old: solver.residual(state.ravel(), old, 1e-5, 0.0))(
            state.ravel()
        )
    )
    np.testing.assert_array_equal(old_jacobian[:, :4], 0.0)
    np.testing.assert_array_equal(old_jacobian[:, -4:], 0.0)
    np.testing.assert_array_equal(old_jacobian[:4], 0.0)
    np.testing.assert_array_equal(old_jacobian[-4:], 0.0)


def test_physical_advance_reaches_target_on_its_last_permitted_step(axial_solver):
    solver, state = axial_solver
    np.testing.assert_allclose(solver.boundary_residual(state), 0, atol=2e-13)
    final_time = 1e-7
    result = _advance(solver, state, final_time, max_steps=1)
    assert result.success, result.reason
    assert result.time == final_time
    assert len(result.history) == 1
    assert result.history[0]["accepted"]
    np.testing.assert_allclose(result.state, state, rtol=2e-12, atol=2e-9)


def test_physical_advance_solves_inconsistent_initial_boundary_guesses(axial_solver):
    solver, state = axial_solver
    result = _advance(solver, state.at[0, 0].multiply(1.02), 1e-7, max_steps=1)
    assert result.success, result.reason
    assert np.max(np.abs(solver.boundary_residual(result.state))) < 1e-9


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_varying_annular_geometry_preserves_constant_pressure_rest(
    fixture_data, reconstruction
):
    data = fixture_data
    inlet = dict(
        h0=data["h"],
        s0=float(data["fluid"].get_state(jxp.HmassP_INPUTS, data["h"], data["p"])["s"]),
        p0=data["p"],
        alpha_in=0.0,
    )
    solver = _operator(
        data["fluid"],
        data["radial"],
        inlet_data=inlet,
        outlet_pressure=data["p"],
        reconstruction=reconstruction,
    )
    state = jnp.tile(jnp.asarray([data["p"], 0.0, 0.0, data["h"]]), (8, 1))
    metrics = sampler(solver, ResidualPolicy())(state)
    assert np.max(metrics["equation_norm"]) < 1e-12
    assert float(metrics["boundary_max"]) < 1e-12
    assert np.max(metrics["global_defect"]) < 1e-12


@pytest.mark.parametrize(
    "options",
    [
        dict(reconstruction="unsupported"),
        dict(delta_factor=0.0),
        dict(newton_iterations=-1),
        dict(outlet_regime="unsupported"),
    ],
)
def test_factory_rejects_unsupported_numerical_modes(fixture_data, options):
    data = fixture_data
    with pytest.raises(ValueError):
        _operator(
            data["fluid"],
            data["axial"],
            inlet_data=data["inlet"],
            outlet_pressure=data["p"],
            **options
        )


def test_muscl_trial_admissibility_includes_reconstructed_face_states(fixture_data):
    data = fixture_data
    mesh = data["axial"].geometry.mesh(
        m_faces=[0.0, 0.001, 0.002, 0.095, 0.096, 0.098, 0.1]
    )
    solver = _operator(
        data["fluid"],
        mesh,
        inlet_data=data["inlet"],
        outlet_pressure=data["p"],
        reconstruction="muscl_van_leer",
    )
    state = jnp.tile(jnp.asarray([data["p"], 40.0, 4.0, data["h"]]), (8, 1))
    state = state.at[:, 3].set(jnp.asarray([1e7, 1e7, 1e7, 1e5, 1e4, 1e4, 1e4, 1e4]))
    assert np.all(admissible(state, solver.model._r_full, data["fluid"]))
    left, right = solver.reconstruct(state)
    assert not (
        np.all(admissible(left, mesh.r_face, data["fluid"]))
        and np.all(admissible(right, mesh.r_face, data["fluid"]))
    )
    assert not bool(sampler(solver, ResidualPolicy())(state)["admissible"])


def test_physical_advance_does_not_accept_an_algebraic_root_on_the_wrong_bc_branch(
    fixture_data,
):
    data = fixture_data
    static = data["backend"].get_state(jxp.HmassP_INPUTS, data["h"], data["p"])
    v_m = 2.0 * float(static["a"])
    h0, s0 = data["h"] + 0.5 * v_m**2, float(static["s"])
    reservoir = data["backend"].get_state(jxp.HmassSmass_INPUTS, h0, s0)
    inlet = dict(h0=h0, s0=s0, p0=float(reservoir["p"]), alpha_in=0.0)
    solver = _operator(
        data["fluid"], data["axial"], inlet_data=inlet, outlet_pressure=data["p"]
    )
    state = jnp.tile(jnp.asarray([data["p"], v_m, 0.0, data["h"]]), (8, 1))
    np.testing.assert_allclose(solver.boundary_residual(state), 0, atol=2e-13)
    try:
        result = _advance(solver, state, 1e-7, max_steps=1)
    except ValueError as error:
        assert "regime" in str(error).lower() or "branch" in str(error).lower()
    else:
        assert (
            not result.success
        ), "A wrong characteristic count cannot be a successful physical advance"


class GuardedHPInterface(HPInterfaceOnly):
    """Raise if a driver sends a state outside declared bounds to the EOS."""

    def __init__(self, fluid, reference_h):
        super().__init__(fluid)
        self.h_min, self.h_max = 0.5 * reference_h, 2.0 * reference_h
        self.p_min, self.p_max = 1e4, 1e6

    def get_state(self, pair, h, p):
        def check_domain(enthalpy, pressure):
            if (
                np.any(np.asarray(enthalpy) < self.h_min)
                or np.any(np.asarray(enthalpy) > self.h_max)
                or np.any(np.asarray(pressure) < self.p_min)
                or np.any(np.asarray(pressure) > self.p_max)
            ):
                raise AssertionError(
                    "An invalid initial state reached spatial/flux EOS evaluation"
                )

        jax.debug.callback(check_domain, h, p)
        return super().get_state(pair, h, p)


def guarded_solver(fixture_data, reconstruction):
    data = fixture_data
    fluid = GuardedHPInterface(data["backend"], data["h"])
    solver = _operator(
        fluid,
        data["axial"],
        inlet_data=data["inlet"],
        outlet_pressure=data["p"],
        reconstruction=reconstruction,
    )
    state = jnp.tile(
        jnp.asarray([data["p"], data["v_m"], 0.2 * data["v_theta"], data["h"]]), (8, 1)
    )
    return solver, state, fluid


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_invalid_initial_pressure_is_rejected_before_spatial_metrics(
    fixture_data, reconstruction, monkeypatch
):
    solver, state, _ = guarded_solver(fixture_data, reconstruction)
    invalid = state.at[3, 0].set(-1.0)

    def unexpected_metrics(_):
        raise AssertionError(
            "Invalid initialization must be checked before spatial metrics"
        )

    monkeypatch.setattr(drivers, "sampler", unexpected_metrics)
    result = solve_single_grid(
        solver.model, solver.scheme, SteadySettings(max_steps=0), initial_state=invalid
    )
    assert not result.success
    assert not result.history
    assert result.reason == "invalid_initial_state"
    with pytest.raises(ValueError, match="admissible"):
        _advance(solver, invalid, 0.0, max_steps=0)


@pytest.mark.parametrize("reconstruction", ["constant", "muscl_van_leer"])
def test_declared_initial_domain_is_checked_before_spatial_metrics(
    fixture_data, reconstruction, monkeypatch
):
    solver, state, fluid = guarded_solver(fixture_data, reconstruction)
    invalid = state.at[3, 3].set(0.9 * fluid.h_min)
    assert np.all(np.asarray(invalid[:, 0]) > 0)

    def unexpected_metrics(_):
        raise AssertionError(
            "Out-of-domain initialization must not reach spatial metrics"
        )

    monkeypatch.setattr(drivers, "sampler", unexpected_metrics)
    result = solve_single_grid(
        solver.model, solver.scheme, SteadySettings(max_steps=0), initial_state=invalid
    )
    assert not result.success
    assert not result.history
    assert result.reason == "invalid_initial_state"
    with pytest.raises(ValueError, match="admissible"):
        _advance(solver, invalid, 0.0, max_steps=0)
