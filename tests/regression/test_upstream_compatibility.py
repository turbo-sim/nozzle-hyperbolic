"""Two-fluid flux and nonconservative action match the frozen upstream fixture."""

from types import SimpleNamespace
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import from_primitives
from nozzle_hyperbolic.models.two_fluid._alberto_flux import nonconservative_correction

from tests._support.two_fluid import model, physical_data


def test_frozen_upstream_flux_and_nc_fixture_agreement():
    """Fixture was evaluated directly from git objects at its recorded SHA."""
    fixture = json.loads(
        (Path(__file__).parent / "data/two_fluid_alberto_reference.json").read_text()
    )
    m = model()
    assert fixture["revision"] == m.reference_revision
    for case in fixture["cases"]:
        left, right = jnp.array(case["left"]), jnp.array(case["right"])
        WL = from_primitives(left[None], m.fluid_g, m.fluid_l)
        WR = from_primitives(right[None], m.fluid_g, m.fluid_l)
        actual = m.numerical_face_flux(
            WL, WR, SimpleNamespace(flux="model_compatibility")
        )[0]
        np.testing.assert_allclose(actual, case["flux"], rtol=2e-12, atol=1e-8)
        _, eg, el, cq = physical_data(left)
        nc = nonconservative_correction(left, jnp.array(case["df"]), eg, el, cq)
        np.testing.assert_allclose(nc, case["nc"], rtol=2e-12, atol=1e-8)
