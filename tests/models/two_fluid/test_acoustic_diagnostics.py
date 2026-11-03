"""Public mixture diagnostics retain the physical model's acoustic definition."""

import jax
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.two_fluid import TwoFluidModel


def test_public_acoustic_diagnostics_are_traceable_and_match_speed_bounds():
    fluid = jxp.FluidPerfectGas("Air")
    model = TwoFluidModel(
        ConstantAreaGeometry().build_mesh(3),
        fluid,
        fluid,
        initial_primitives=[2e5, 25.0, 25.0, 0.4, 3e5, 2.8e5],
    )
    cells = model.initial_state()[1:-1]
    values = jax.jit(model.acoustic_diagnostics)(cells)
    np.testing.assert_allclose(values["velocity"], 25.0, atol=1e-13)
    np.testing.assert_allclose(values["mach"], 25.0 / values["sound_speed"], rtol=2e-15)
    np.testing.assert_allclose(
        model.wave_speed(cells), 25.0 + values["sound_speed"], rtol=2e-15
    )
    assert np.all(np.isfinite(jax.jacfwd(model.acoustic_diagnostics)(cells)["mach"]))
