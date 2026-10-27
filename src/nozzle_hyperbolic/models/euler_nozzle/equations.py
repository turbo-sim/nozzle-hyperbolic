"""Unchanged definitions relocated from fvm_solver.fvm."""
import math
import numpy as np
import jaxprop as jxp


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def euler_eigenvalues_perfect(Q, gamma):
    rho  = Q[0]
    rhou = Q[1]
    rhoE = Q[2]
    u = rhou / rho
    p = (gamma - 1.0) * (rhoE - 0.5 * rhou**2 / rho)
    c = np.sqrt(gamma * p / rho)
    return np.array([u - c, u, u + c])


def euler_eigenvalues(Q, fluid, parameters):
    rho = Q[0]
    p   = parameters['p']
    u   = Q[1] / rho
    c   = np.zeros_like(rho)
    for i in range(len(rho)):
        state = fluid.get_state(jxp.DmassP_INPUTS, rho[i], p[i])
        c[i] = state.speed_sound
    return np.array([u - c, u, u + c])


def euler_flux(Q, p):
    rho  = Q[0]
    rhou = Q[1]
    rhoE = Q[2]
    u = rhou / rho
    return np.array([rhou, rhou * u + p, (rhoE + p) * u])


def euler_flux_perfect(Q, gamma):
    rho  = Q[0]
    rhou = Q[1]
    rhoE = Q[2]
    u = rhou / rho
    p = (gamma - 1.0) * (rhoE - 0.5 * rhou**2 / rho)
    return np.array([rhou, rhou * u + p, (rhoE + p) * u])


# ---------------------------------------------------------------------------
# Roe matrix  (kept here; called from flux.py)
# ---------------------------------------------------------------------------

def euler_roe_matrix_perfect(Q_L, Q_R, gamma):
    rho_L, rhou_L, rhoE_L = Q_L
    u_L = rhou_L / rho_L
    p_L = (gamma - 1.0) * (rhoE_L - 0.5 * rhou_L**2 / rho_L)
    H_L = (rhoE_L + p_L) / rho_L
    c_L = math.sqrt(gamma * p_L / rho_L)

    rho_R, rhou_R, rhoE_R = Q_R
    u_R = rhou_R / rho_R
    p_R = (gamma - 1.0) * (rhoE_R - 0.5 * rhou_R**2 / rho_R)
    H_R = (rhoE_R + p_R) / rho_R
    c_R = math.sqrt(gamma * p_R / rho_R)

    sr_L = math.sqrt(rho_L)
    sr_R = math.sqrt(rho_R)
    sr   = sr_L + sr_R
    u    = (sr_L * u_L + sr_R * u_R) / sr
    H    = (sr_L * H_L + sr_R * H_R) / sr
    c    = math.sqrt((gamma - 1.0) * (H - 0.5 * u**2))

    Rv = np.array([[1.0,       1.0,         1.0      ],
                   [u - c,     u,           u + c    ],
                   [H - c * u, 0.5 * u**2,  H + c * u]])

    lam = np.diag([_H_fix(u_L - c_L, u_R - c_R, u - c, c),
                   abs(u),
                   _H_fix(u_L + c_L, u_R + c_R, u + c, c)])

    return Rv @ lam @ np.linalg.inv(Rv)


def euler_roe_matrix(Q_L, Q_R, p_L, p_R, fluid):
    rho_L, rhou_L, rhoE_L = Q_L
    u_L = rhou_L / rho_L
    H_L = (rhoE_L + p_L) / rho_L
    c_L = fluid.get_state(jxp.DmassP_INPUTS, rho_L, p_L).speed_sound

    rho_R, rhou_R, rhoE_R = Q_R
    u_R = rhou_R / rho_R
    H_R = (rhoE_R + p_R) / rho_R
    c_R = fluid.get_state(jxp.DmassP_INPUTS, rho_R, p_R).speed_sound

    sr_L = math.sqrt(rho_L)
    sr_R = math.sqrt(rho_R)
    sr   = sr_L + sr_R
    u    = (sr_L * u_L + sr_R * u_R) / sr
    H    = (sr_L * H_L + sr_R * H_R) / sr

    avg_state = fluid.get_state(jxp.DmassP_INPUTS, 0.5 * (rho_L + rho_R),
                                                    0.5 * (p_L  + p_R))
    c = avg_state.speed_sound

    Rv = np.array([[1.0,       1.0,         1.0      ],
                   [u - c,     u,           u + c    ],
                   [H - c * u, 0.5 * u**2,  H + c * u]])

    lam = np.diag([_H_fix(u_L - c_L, u_R - c_R, u - c, c),
                   abs(u),
                   _H_fix(u_L + c_L, u_R + c_R, u + c, c)])

    return Rv @ lam @ np.linalg.inv(Rv)


def _H_fix(a_L, a_R, a, c):
    a_abs = abs(a)
    delta = max(abs(a_R - a_L), 0.01 * c)
    return a_abs if a_abs >= delta else (a_abs**2 + delta**2) / (2.0 * delta)
