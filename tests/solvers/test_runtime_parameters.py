"""Run-time model parameters: inputs of compiled source and boundary kernels."""

import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.models.scalar import AdvectionModel
from nozzle_hyperbolic.solvers.compilation import CompilationReport
from nozzle_hyperbolic.solvers.fvm import (
    Operator,
    Scheme,
    SteadySettings,
    SteadyWorkspace,
    solve_single_grid,
)


class _ParameterAdvection(AdvectionModel):
    """Advection with a per-cell decay field and a scalar inflow parameter.

    The steady solution is approximately ``inflow*exp(-rate*x)`` for a uniform
    rate, so both parameters visibly change sources and boundary rows.
    """

    def parameter_defaults(self):
        return dict(rate=jnp.zeros(self.mesh.n_cells), inflow=jnp.asarray(1.0))

    def validate_parameters(self, values):
        if np.any(values["rate"] < 0):
            raise ValueError("decay rate must be nonnegative")

    def integrated_source(self, full, params):
        volumes = jnp.asarray(self.mesh.volumes)
        return -(volumes * params["rate"])[:, None] * full[1:-1]

    def boundary_residual(self, full, time, params):
        return jnp.stack([full[0] - params["inflow"], full[-1] - full[-2]])


SCHEME = Scheme(flux="hll", reconstruction="muscl_van_albada")


def _state(model):
    x = jnp.asarray(model.mesh.points)
    return (1.0 + 0.3 * x + 0.1 * x**2)[:, None]


def test_parameters_enter_sources_and_boundaries_with_defaults_filled():
    model = _ParameterAdvection(12)
    op = Operator(model, SCHEME)
    full = _state(model)
    flat = full.ravel()
    default = np.asarray(op.steady_residual(flat))
    np.testing.assert_array_equal(default, op.steady_residual(flat, params=op.parameters()))
    np.testing.assert_array_equal(default, op.steady_residual(flat, 0.0, params={}))
    changed = np.asarray(op.steady_residual(flat, params=dict(rate=jnp.full(12, 2.0))))
    assert not np.allclose(changed[1:-1], default[1:-1])
    np.testing.assert_array_equal(changed[[0, -1]], default[[0, -1]])
    boundary = op.boundary_residual(full, 0.0, params=dict(inflow=3.0))
    np.testing.assert_allclose(boundary[0], full[0] - 3.0)
    np.testing.assert_allclose(op.source(full, params=dict(rate=jnp.full(12, 2.0))),
                               -2.0 * jnp.asarray(model.mesh.volumes)[:, None] * full[1:-1])


@pytest.mark.parametrize(
    "values, message",
    [
        (dict(speed=1.0), "Unknown"),
        (dict(rate=jnp.ones(5)), "shape"),
        (dict(inflow=jnp.nan), "finite"),
        (dict(rate=-jnp.ones(12)), "nonnegative"),
    ],
)
def test_unknown_reshaped_nonfinite_and_model_rejected_values(values, message):
    op = Operator(_ParameterAdvection(12), SCHEME)
    with pytest.raises(ValueError, match=message):
        op.steady_residual(_state(op.model).ravel(), params=values)


def test_models_without_parameters_keep_signatures_and_reject_values():
    model = AdvectionModel(12)
    op = Operator(model, SCHEME)
    assert not op.has_parameters and op.parameters() == {}
    with pytest.raises(ValueError, match="no run-time parameters"):
        op.steady_residual(model.initial_state().ravel(), params=dict(rate=1.0))


def test_colored_jacobian_and_parameter_derivatives():
    model = _ParameterAdvection(10)
    op = Operator(model, SCHEME)
    full = _state(model)
    values = dict(rate=jnp.linspace(0.5, 1.5, 10), inflow=jnp.asarray(1.3))
    args = (full.ravel(), full.ravel(), 0.05, 0.0)
    np.testing.assert_allclose(op.jacobian(*args, params=values),
                               op.dense_jacobian(*args, params=values), rtol=2e-14, atol=2e-14)

    def residual(rate, inflow):
        return op.steady_residual(full.ravel(), params=dict(rate=rate, inflow=inflow))

    d_rate, d_inflow = jax.jacfwd(residual, argnums=(0, 1))(values["rate"], values["inflow"])
    eps = 1e-6
    direction = jnp.sin(jnp.arange(10.0))
    fd_rate = (residual(values["rate"] + eps * direction, values["inflow"])
               - residual(values["rate"] - eps * direction, values["inflow"])) / (2 * eps)
    fd_inflow = (residual(values["rate"], values["inflow"] + eps)
                 - residual(values["rate"], values["inflow"] - eps)) / (2 * eps)
    np.testing.assert_allclose(d_rate @ direction, fd_rate, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(d_inflow, fd_inflow, rtol=1e-8, atol=1e-10)


def test_steady_solves_use_parameters_and_reuse_compiled_kernels():
    model = _ParameterAdvection(40)
    workspace = SteadyWorkspace(model, SCHEME)
    settings = SteadySettings(cfl=5.0, max_steps=200)
    first = solve_single_grid(model, SCHEME, settings, workspace=workspace,
                              parameters=dict(rate=jnp.full(40, 2.0), inflow=2.0))
    with CompilationReport(stream=io.StringIO()) as report:
        second = solve_single_grid(model, SCHEME, settings, workspace=workspace,
                                   parameters=dict(rate=jnp.full(40, 1.0), inflow=0.5))
    assert first.success and second.success
    assert report.stats["backend_requests"] == 0
    # The 40-cell discrete solutions follow inflow*exp(-rate*x) within 3%.
    x = np.asarray(model.mesh.centers)
    np.testing.assert_allclose(first.state[1:-1, 0], 2.0 * np.exp(-2.0 * x), rtol=3e-2)
    np.testing.assert_allclose(second.state[1:-1, 0], 0.5 * np.exp(-1.0 * x), rtol=3e-2)
    np.testing.assert_array_equal(second.parameters["rate"], np.ones(40))
    assert float(second.parameters["inflow"]) == 0.5


@pytest.mark.parametrize(
    "scheme",
    [
        Scheme(flux="hll", boundary_treatment="segregated"),
        Scheme(flux="hll", time_integrator="runge-kutta-3"),
    ],
)
def test_segregated_and_explicit_steps_complete_parameter_boundaries(scheme):
    model = _ParameterAdvection(12)
    op = Operator(model, scheme)
    full = _state(model)
    values = dict(rate=jnp.full(12, 0.7), inflow=jnp.asarray(1.8))
    dt = float(op.compute_dt(full, 0.4, params=values))
    new, info = op.step(full, dt, dt, params=values)
    assert bool(info.converged)
    np.testing.assert_allclose(new[0], 1.8, atol=1e-12)
    completed, info = op.complete_boundaries(full, 0.0, params=values)
    assert bool(info.converged)
    np.testing.assert_allclose(completed[0], 1.8, atol=1e-12)
