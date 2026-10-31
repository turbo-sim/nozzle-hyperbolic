"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
import jaxprop as jxp
from ...thermodynamics.recovery import _get_state_hs


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _bc_inlet_perfect(Q_n, p_stag, T_stag, gamma, R_const):
    u_in = 2.0 * Q_n[1, 1] / Q_n[0, 1] - Q_n[1, 2] / Q_n[0, 2]
    T_s  = T_stag - 0.5 * u_in**2 * (gamma - 1.0) / (gamma * R_const)
    T_s  = max(T_s, 1.0)
    M_in = np.sqrt(max(0.0, 2.0 * (T_stag / T_s - 1.0) / (gamma - 1.0)))
    p_s  = p_stag / (1.0 + 0.5 * (gamma - 1.0) * M_in**2) ** (gamma / (gamma - 1.0))
    rho_s = p_s / (R_const * T_s)
    Q_n[0, 0] = rho_s
    Q_n[1, 0] = rho_s * u_in
    Q_n[2, 0] = p_s / (gamma - 1.0) + 0.5 * rho_s * u_in**2


def _bc_outlet_perfect(Q_n, p_out, A_nodes, gamma):
    A = A_nodes
    Q_n[0, -1] = 2.0 * Q_n[0, -2] * A[-2] / A[-1] - Q_n[0, -3] * A[-3] / A[-1]
    Q_n[1, -1] = 2.0 * Q_n[1, -2] * A[-2] / A[-1] - Q_n[1, -3] * A[-3] / A[-1]
    Q_n[2, -1] = p_out / (gamma - 1.0) + 0.5 * Q_n[1, -1]**2 / Q_n[0, -1]


def _bc_inlet_real(Q_n, h0, s0, fluid):
    u_in = 2.0 * Q_n[1, 1] / Q_n[0, 1] - Q_n[1, 2] / Q_n[0, 2]
    h    = h0 - 0.5 * u_in**2
    state = _get_state_hs(fluid, h, s0)
    rho_s = state.rhomass
    Q_n[0, 0] = rho_s
    Q_n[1, 0] = rho_s * u_in
    Q_n[2, 0] = state.umass * rho_s + 0.5 * rho_s * u_in**2


def _bc_outlet_real(Q_n, p_out, A_nodes, fluid):
    A = A_nodes
    Q_n[0, -1] = 2.0 * Q_n[0, -2] * A[-2] / A[-1] - Q_n[0, -3] * A[-3] / A[-1]
    Q_n[1, -1] = 2.0 * Q_n[1, -2] * A[-2] / A[-1] - Q_n[1, -3] * A[-3] / A[-1]
    rho_out = Q_n[0, -1]
    u_out   = Q_n[1, -1] / rho_out
    state   = fluid.get_state(jxp.DmassP_INPUTS, rho_out, p_out)
    Q_n[2, -1] = state.umass * rho_out + 0.5 * rho_out * u_out**2


def _bc_inlet_real_jax(Q_n, h0, s0, fluid):
    u_in  = float(2.0 * Q_n[1, 1] / Q_n[0, 1] - Q_n[1, 2] / Q_n[0, 2])
    st    = _get_state_hs(fluid, h0 - 0.5 * u_in**2, s0)
    rho_s = float(st.rhomass);  e_s = float(st.umass)
    Q_n   = Q_n.at[0, 0].set(rho_s)
    Q_n   = Q_n.at[1, 0].set(rho_s * u_in)
    Q_n   = Q_n.at[2, 0].set(rho_s * (e_s + 0.5 * u_in**2))
    return Q_n


def _bc_outlet_real_jax(Q_n, p_out, A_nodes, fluid):
    A        = A_nodes
    rho_out  = float(2.0 * Q_n[0, -2] * A[-2] / A[-1] - Q_n[0, -3] * A[-3] / A[-1])
    rhou_out = float(2.0 * Q_n[1, -2] * A[-2] / A[-1] - Q_n[1, -3] * A[-3] / A[-1])
    u_out    = rhou_out / rho_out
    e_out    = float(fluid.get_state(jxp.DmassP_INPUTS, rho_out, p_out).umass)
    Q_n      = Q_n.at[0, -1].set(rho_out)
    Q_n      = Q_n.at[1, -1].set(rhou_out)
    Q_n      = Q_n.at[2, -1].set(rho_out * (e_out + 0.5 * u_out**2))
    return Q_n
