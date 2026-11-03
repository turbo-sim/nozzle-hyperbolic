"""Low-level residual assembly agrees with independent balance formulas."""

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.finite_volume import (
    flux_divergence,
    make_coupled_residual,
)


def test_coupled_residual_has_current_boundary_fluxes_and_zero_boundary_storage():
    nphysical, nvar = 3, 4
    physical_source = jnp.arange(nphysical * nvar).reshape(nphysical, nvar) * 0.01

    def storage(physical):
        return physical**2 + 0.3 * physical

    def faces(full):
        return 0.7 * full[:-1] + 0.3 * full[1:]

    def spatial(full):
        return flux_divergence(faces(full)) - physical_source

    def boundaries(full):
        return jnp.stack([full[0] - full[1], full[-1] - 2 * full[-2] + full[-3]])

    scales = jnp.array([1.0, 2.0, 3.0, 4.0])
    operators = make_coupled_residual(
        storage,
        spatial,
        boundaries,
        nphysical=nphysical,
        nvar=nvar,
        content_scale=scales,
        flux_scale=scales * 2,
    )
    full = jnp.linspace(0.8, 1.8, 20).reshape(5, 4)
    previous = full * 0.97
    residual = jax.jit(operators["residual"])
    actual = residual(full.ravel(), previous.ravel(), 0.4).reshape(5, 4)
    changed_old = previous.at[0].set(30.0).at[-1].set(-20.0)
    np.testing.assert_array_equal(
        residual(full.ravel(), changed_old.ravel(), 0.4).reshape(5, 4), actual
    )
    zero_dt = residual(full.ravel(), previous.ravel(), 0.0).reshape(5, 4)
    boundary_indices = jnp.array([0, -1])
    np.testing.assert_array_equal(zero_dt[boundary_indices], actual[boundary_indices])
    np.testing.assert_allclose(
        zero_dt[1:-1], (storage(full[1:-1]) - storage(previous[1:-1])) / scales
    )
    # Current independent boundary values must enter the physical flux rows.
    changed = full.at[0, 0].add(0.2)
    assert not np.array_equal(
        residual(changed.ravel(), previous.ravel(), 0.4).reshape(5, 4)[1], actual[1]
    )
    expected_budget = faces(full)[-1] - faces(full)[0] - physical_source.sum(axis=0)
    np.testing.assert_allclose(
        operators["spatial"](full.ravel()).sum(axis=0), expected_budget
    )
    steady = operators["steady"](full.ravel()).reshape(5, 4)
    np.testing.assert_array_equal(steady[boundary_indices], boundaries(full))
    np.testing.assert_allclose(steady[1:-1], spatial(full) / (2 * scales))
