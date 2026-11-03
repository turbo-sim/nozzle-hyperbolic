"""Directional stresses preserve meridional force, torque and heat units."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.euler_annular.sources import integrated_source


def test_directional_stresses_are_independent_and_differentiable():
    class Directional:
        def directional_stress(self, W, r, fluid):
            return 2 * W[..., 1], jnp.zeros_like(W[..., 1]), -100.0

    fluid = jxp.FluidPerfectGas("Air")
    W = jnp.array([[1e5, 3.0, 1.6, 3e5], [1e5, 4.0, -0.8, 3e5]])
    source = lambda value: integrated_source(
        value,
        jnp.array([0.2, 0.4]),
        0,
        0,
        jnp.array([1e-4, 2e-4]),
        fluid,
        height=jnp.array([0.005, 0.01]),
        wall_closure=Directional(),
    )
    np.testing.assert_allclose(
        jax.jit(source)(W), [[0, -0.24, 0, -4], [0, -0.32, 0, -4]], atol=1e-14
    )
    jac = jax.jacfwd(source)(W)
    assert np.isfinite(jac).all()
    np.testing.assert_allclose(jac[:, 1, :, 1], np.diag([-0.08, -0.08]), atol=1e-14)
