"""The two nozzle solvers use the same complete Haaland constitutive policy."""

import pytest

pytest.importorskip("nozzlex")
import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from nozzlex.functions.nozzle_walls import (
    Haaland as nozzlex_haaland,
    darcy_friction_factor as nozzlex_factor,
    haaland_wall_shear as nozzlex_shear,
    nozzle_wall_sources,
)

from nozzle_hyperbolic.models.euler_nozzle.closures.wall import (
    Haaland,
    darcy_friction_factor,
    haaland_wall_shear,
    HaalandWall,
)


def test_complete_darcy_factors_match_in_both_regimes_and_at_limits():
    reynolds = jnp.array([-1.0, 0.0, 1.0, 100.0, 2299.0, 2300.0, 1e4, 1e5, 1e7])
    for roughness in (0.0, 1e-6):
        np.testing.assert_allclose(
            nozzlex_factor(reynolds, roughness, 0.004),
            darcy_friction_factor(reynolds, roughness, 0.004),
            rtol=2e-14,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            nozzlex_haaland(reynolds, roughness, 0.004),
            Haaland(reynolds, roughness, 0.004),
            rtol=2e-14,
            equal_nan=True,
        )


def test_signed_stress_and_derivatives_match_including_rest():
    velocity = jnp.array([-600.0, -20.0, -0.5, 0.0, 0.5, 20.0, 600.0])
    nx = lambda u: nozzlex_shear(u, 50.0, 1.5e-5, 0.004, 0.0)
    hp = lambda u: haaland_wall_shear(u, 50.0, 1.5e-5, 0.004, 0.0)
    np.testing.assert_allclose(jax.jit(nx)(velocity), hp(velocity), rtol=2e-14)
    np.testing.assert_allclose(
        jax.jacfwd(nx)(velocity), jax.jacfwd(hp)(velocity), rtol=2e-13
    )
    np.testing.assert_allclose(
        jax.jacrev(nx)(velocity), jax.jacrev(hp)(velocity), rtol=2e-13
    )
    assert np.isfinite(jax.jacfwd(nx)(velocity)).all()
    assert float(jax.grad(nx)(0.0)) == pytest.approx(8 * 1.5e-5 / 0.004)


def test_actual_nozzle_wall_policies_have_matching_signed_loads():
    fluid = jxp.FluidPerfectGas("Air")
    velocity = jnp.array([-100.0, -0.1, 0.0, 0.1, 100.0])
    W = jnp.column_stack((jnp.full(5, 1e5), velocity, jnp.full(5, 3e5)))
    state = fluid.get_state(jxp.HmassP_INPUTS, W[:, 2], W[:, 0])
    loads = nozzle_wall_sources(
        state, velocity, 0.004, roughness=0.0, wall_friction=1.0, heat_transfer=0.0
    )
    wall = HaalandWall(hydraulic_diameter=0.004)
    np.testing.assert_allclose(
        loads["tau_w"], wall.drag_coefficient(W, 0.0, fluid) * velocity, rtol=2e-14
    )
    np.testing.assert_array_equal(loads["q_w"], 0.0)
