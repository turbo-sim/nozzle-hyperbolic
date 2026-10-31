"""Conservation and temporal order with nonlinear stage boundary constraints."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from nozzle_hyperbolic.solvers.boundaries import make_boundary_solver
from nozzle_hyperbolic.solvers.time_marching.explicit import make_explicit_step
from nozzle_hyperbolic.solvers.time_marching.implicit import NewtonInfo


@pytest.mark.parametrize(
    "method,minimum_order", [("euler-explicit", 0.9), ("runge-kutta-3", 2.8)]
)
def test_conserved_stage_order_and_flux_budget_with_time_dependent_nonlinear_boundary(
    method, minimum_order
):
    # Q=W^3; Q'=-Q+exp(t), Q(0)=1 -> Q(t)=cosh(t). Averaging W instead
    # of Q, or freezing boundary values/times across RK stages, fails this audit.
    def storage(W):
        return W**3

    def boundary(full, time):
        return jnp.stack((full[0] ** 3 - jnp.exp(time), full[-1] - full[-2]))

    complete = make_boundary_solver(boundary, (4, 1), jnp.ones(1))

    def recover(target, guess):
        W = jnp.cbrt(target)
        info = NewtonInfo(
            jnp.int32(0),
            jnp.float64(0),
            jnp.float64(0),
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.int32(0),
            jnp.int32(0),
        )
        return W, info

    def spatial(full):
        return storage(full[1:-1]) - full[0] ** 3

    def balance(full):
        return -jnp.sum(spatial(full), axis=0)

    step = make_explicit_step(
        storage,
        spatial,
        recover,
        complete,
        balance,
        lambda W: jnp.all(jnp.isfinite(W)),
        method=method,
    )
    errors = []
    for count in (4, 8, 16):
        dt = 0.4 / count
        full = jnp.ones((4, 1))
        integrated = jnp.zeros(1)
        for index in range(count):
            full, info, increment = step(full, dt, (index + 1) * dt)
            assert bool(info.converged & info.admissible)
            integrated += increment
            np.testing.assert_allclose(
                boundary(full, (index + 1) * dt), 0.0, atol=2e-12
            )
        contents = storage(full[1:-1])
        errors.append(float(jnp.max(jnp.abs(contents - jnp.cosh(0.4)))))
        np.testing.assert_allclose(
            jnp.sum(contents - 1.0, axis=0), integrated, atol=1e-11
        )
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.min(orders) > minimum_order, (errors, orders)
