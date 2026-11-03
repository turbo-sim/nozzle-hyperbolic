"""Pinned two-fluid identities and independent physical contract checks."""

from types import SimpleNamespace

import jaxprop as jxp

from nozzle_hyperbolic.geometry.nozzle import ConstantAreaGeometry
from nozzle_hyperbolic.models.two_fluid import TwoFluidModel
from nozzle_hyperbolic.models.two_fluid.matrices import (
    phase_props,
    common_quantities,
)


def model(*, velocity=25.0, alpha=0.4, mesh=None, **kwargs):
    fluid = jxp.FluidPerfectGas(name="Air")
    return TwoFluidModel(
        mesh or ConstantAreaGeometry().build_mesh(3),
        fluid,
        fluid,
        initial_primitives=[2e5, velocity, velocity, alpha, 3e5, 2.8e5],
        **kwargs
    )


def physical_data(U):
    fluid = jxp.FluidPerfectGas(name="Air")
    eg, el = phase_props(fluid, U[0], U[4]), phase_props(fluid, U[0], U[5])
    cq = common_quantities(U, eg, el, 0.5)
    return fluid, eg, el, cq


def raw_flux(m, W):
    return m.numerical_face_flux(
        W[:-1], W[1:], SimpleNamespace(flux="model_compatibility")
    )
