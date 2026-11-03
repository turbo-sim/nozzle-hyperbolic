"""Production Euler projectors diagonalize an independently differentiated flux."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import pytest

from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel


@pytest.mark.parametrize("mach", [-1.7, -0.3, 0.0, 0.6, 1.0, 1.8])
def test_active_model_projectors_resolve_the_conservative_flux_jacobian(mach):
    fluid = jxp.FluidPerfectGas("Air")
    model = NozzleModel(ConstantAreaGeometry().build_mesh(3), fluid, transmissive=True)
    gas = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    state = jnp.array([1e5, mach * gas["a"], gas["h"]])
    faces = jnp.tile(state, (model.mesh.n_cells + 1, 1))
    projectors, speeds, widths = model.face_characteristics(faces, faces, 1e-4)
    storage_jacobian = jax.jacfwd(model.face_conserved)(state)
    flux_jacobian = jax.jacfwd(model.face_physical_flux)(state)
    # Chain rule dF/dq = (dF/dW) (dq/dW)^-1; this does not use a projector helper.
    conservative_jacobian = np.linalg.solve(storage_jacobian.T, flux_jacobian.T).T
    scale = model.conservative_scales
    normalized = conservative_jacobian * scale[None, :] / scale[:, None]
    p = np.asarray(projectors[0]) * scale[None, None, :] / scale[None, :, None]
    np.testing.assert_allclose(p.sum(axis=0), np.eye(3), atol=2e-14)
    np.testing.assert_allclose(
        np.einsum("k,kij->ij", speeds[0], p), normalized, atol=2e-11, rtol=2e-13
    )
    for index in range(3):
        np.testing.assert_allclose(
            normalized @ p[index], speeds[0, index] * p[index], atol=2e-11, rtol=2e-13
        )
        for other in range(3):
            np.testing.assert_allclose(
                p[index] @ p[other], p[index] if index == other else 0, atol=2e-14
            )
    assert np.all(np.isfinite(widths)) and np.all(widths > 0)
