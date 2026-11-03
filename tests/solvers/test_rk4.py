"""Explicit integration on nonlinear storage with algebraic states and trace forcing."""

import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.time_marching.explicit import make_explicit_step
from nozzle_hyperbolic.solvers.time_marching.implicit import NewtonInfo


def _info(ok=True):
    return NewtonInfo(
        jnp.int32(1),
        jnp.float64(0),
        jnp.float64(0),
        jnp.asarray(ok),
        jnp.asarray(ok),
        jnp.int32(0),
        jnp.int32(0),
    )


@pytest.mark.parametrize(
    "method,order,corrections", [("runge-kutta-3", 3, 7), ("runge-kutta-4", 4, 9)]
)
def test_rk4_order_recovered_constraints_stage_boundary_and_budget(
    method, order, corrections
):
    # Two variables, one transported content q=w0^3, and g=w1-exp(q)=0.
    # Time-dependent trace forcing yields q'=-q+exp(t), q(0)=1: q=cosh(t).
    # A frozen source, primitive RK combination, incorrect stage time or missing
    # constraint recovery reduces the observed order or fails the budget.
    def storage(cells):
        return cells[:, :1] ** 3

    def recover(target, guess):
        return jnp.concatenate((jnp.cbrt(target), jnp.exp(target)), axis=1), _info()

    def complete(full, time):
        q = jnp.exp(time)
        left = jnp.array([jnp.cbrt(q), jnp.exp(q)])
        return full.at[0].set(left).at[-1].set(full[-2]), _info()

    def spatial(full):
        # Depend on the algebraic variable as well, to detect stale constraints.
        return jnp.log(full[1:-1, 1:]) - storage(full[:1])

    def admissible(full):
        return jnp.all(jnp.abs(full[:, 1] - jnp.exp(full[:, 0] ** 3)) < 1e-11)

    step = make_explicit_step(
        storage,
        spatial,
        recover,
        complete,
        lambda full: -jnp.sum(spatial(full), axis=0),
        admissible,
        method=method,
    )
    errors = []
    for count in (2, 4, 8):
        full = jnp.tile(jnp.array([1.0, np.e]), (4, 1))
        integrated = jnp.zeros(1)
        dt = 0.4 / count
        for index in range(count):
            full, info, increment = step(full, dt, (index + 1) * dt)
            assert bool(info.converged & info.admissible)
            assert (
                int(info.iterations) == corrections
            )  # initial BC + stage recoveries/BCs
            assert increment.shape == (1,)
            integrated += increment
        actual = storage(full[1:-1])
        errors.append(float(jnp.max(jnp.abs(actual - jnp.cosh(0.4)))))
        np.testing.assert_allclose(jnp.sum(actual - 1, axis=0), integrated, atol=2e-14)
    assert np.min(np.log2(np.array(errors[:-1]) / errors[1:])) > order - 0.2, errors


def test_rk4_failed_intermediate_recovery_returns_failure_and_zero_budget():
    def recover(target, guess):
        return target, _info(jnp.all(target >= 0.75))

    step = make_explicit_step(
        lambda w: w,
        lambda w: w[1:-1],
        recover,
        lambda w, t: (w, _info()),
        lambda w: -jnp.sum(w[1:-1], axis=0),
        lambda w: jnp.all(w > 0),
        method="runge-kutta-4",
    )
    _, info, increment = step(jnp.ones((4, 1)), 1.0, 1.0)
    assert not bool(info.converged)
    np.testing.assert_array_equal(increment, 0)


@pytest.mark.parametrize(
    "limiter", ["muscl_minmod", "muscl_superbee", "muscl_van_albada"]
)
def test_public_rk4_operator_stage_completion_and_colored_stencil(limiter):
    from nozzle_hyperbolic.models.scalar import AdvectionModel
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme

    model = AdvectionModel(6, speed=-0.7)
    operator = Operator(
        model,
        Scheme(flux="hll", reconstruction=limiter, time_integrator="runge-kutta-4"),
    )
    old = model.initial_state()
    dt = 0.001
    full, info = operator.step(old, dt, dt)
    assert bool(info.converged & info.admissible)
    np.testing.assert_allclose(operator.boundary_residual(full, dt), 0, atol=2e-12)
    assert operator.radius >= 2
    args = (full.ravel(), old.ravel(), dt, dt)
    np.testing.assert_allclose(
        operator.jacobian(*args), operator.dense_jacobian(*args), rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize("method", ["euler-explicit", "runge-kutta-3", "runge-kutta-4"])
def test_initial_constraint_recovery_precedes_first_source_evaluation(method):
    def storage(cells):
        return cells[:, :1]

    def recover(target, guess):
        return jnp.concatenate((target, target**2), axis=1), _info()

    def boundary(full, time):
        return full.at[0].set(full[1]).at[-1].set(full[-2]), _info()

    def spatial(full):
        return full[1:-1, 1:]

    step = make_explicit_step(
        storage,
        spatial,
        recover,
        boundary,
        lambda full: -jnp.sum(spatial(full), axis=0),
        lambda full: jnp.all(jnp.isfinite(full)),
        method=method,
        recover_initial=True,
    )
    correct = jnp.ones((4, 2))
    inconsistent = correct.at[:, 1].set(17.0)
    expected, _, expected_transfer = step(correct, 0.01, 0.01)
    actual, info, transfer = step(inconsistent, 0.01, 0.01)
    assert info.converged
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(transfer, expected_transfer, atol=1e-14)
    np.testing.assert_allclose(actual[:, 1], actual[:, 0] ** 2, atol=1e-14)


def test_failed_initial_recovery_retains_original_state_and_zero_transfer():
    old = jnp.ones((4, 2)).at[:, 1].set(3.0)

    def failed(target, guess):
        return guess * 7.0, _info(False)._replace(status=jnp.int32(3))

    step = make_explicit_step(
        lambda cells: cells[:, :1],
        lambda full: jnp.full((2, 1), jnp.nan),
        failed,
        lambda full, time: (full * 2.0, _info()),
        lambda full: jnp.full((1,), jnp.nan),
        lambda full: jnp.all(jnp.isfinite(full)),
        method="runge-kutta-4",
        recover_initial=True,
    )
    actual, info, transfer = step(old, 0.01, 0.01)
    np.testing.assert_array_equal(actual, old)
    np.testing.assert_array_equal(transfer, 0)
    assert not info.converged and int(info.iterations) == 1 and int(info.status) == 3
