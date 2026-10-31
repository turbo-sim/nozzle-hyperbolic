"""
Two-fluid FVS solver for Städtke's six-equation single-pressure model.

Implements quasi-1D compressible two-phase flow using:
  - Enthalpy-primitive state U_h = (p, u_g, u_l, alpha_g, h_g, h_l)
  - Total-energy conservative state V = (alpha_g*rho_g, alpha_l*rho_l,
      alpha_g*rho_g*u_g, alpha_l*rho_l*u_l, alpha_g*rho_g*E_g, alpha_l*rho_l*E_l)
  - Städtke flux-vector-splitting (FVS) with Harten entropy fix
  - Non-conservative correction H^nc = X K^{-1}
  - Implicit Newton time-stepping via jax.jacobian

Two separate FluidBicubic tables are required: fluid_g (metastable vapor) and
fluid_l (metastable liquid). EOS is called directly via
    fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
with st["rho"], st["a"], st["G"], st["T"] access — same pattern as the
single-phase Städtke solver.

Reference: Städtke (Wiley, 2006); derivation in stadtke_enthalpy_frame_derivation_latest.md
"""

from .state import U_h_to_V, V_to_U_h
from .matrices import (
    PhaseProps,
    phase_props,
    common_quantities,
    G_h_matrix,
    Phi_inv_matrix,
    J_matrix,
    K_matrix,
    X_matrix,
    eigenstructure,
    flux_vector,
)
from .flux import fvs_interface_flux, nonconservative_correction
from .sources import D_s_primitive, D_VE_conservative, D_VE_algebric
from .solver import (
    make_two_fluid_step,
    V_from_W, U_h_from_W, U_h_from_W_batch,
    W_from_U_h_batch,
)

__all__ = [
    "U_h_to_V", "V_to_U_h",
    "PhaseProps", "phase_props",
    "common_quantities",
    "G_h_matrix", "Phi_inv_matrix", "J_matrix", "K_matrix", "X_matrix",
    "eigenstructure", "flux_vector",
    "fvs_interface_flux", "nonconservative_correction",
    "D_s_primitive", "D_VE_conservative", "D_VE_algebric",
    "make_two_fluid_step",
    "V_from_W", "U_h_from_W", "U_h_from_W_batch", "W_from_U_h_batch",
]
