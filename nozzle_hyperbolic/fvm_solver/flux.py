"""
Numerical flux functions for the quasi-1D Euler equations.

All functions return area-weighted fluxes FA at each interface:
    FA[:, i]  ≈  F_{i+1/2} * A_{i+1/2}

so the caller (R in fvm.py) can directly compute the flux divergence
    -(FA_{i+1/2} - FA_{i-1/2}) / (A_i * dx)
without any additional area scaling.
"""

import numpy as np
import math
import jaxprop as jxp
from scipy.optimize import brentq as _brentq

_FluidBicubic = getattr(jxp, 'FluidBicubic', None)


def state_from_rho_u(fluid, rho, u_int):
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        p_lo_raw = float(getattr(fluid, 'p_min', 1e4))
        p_hi_raw = float(getattr(fluid, 'p_max', 300e5))
        h_min    = float(getattr(fluid, 'h_min', 50e3))
        h_max    = float(getattr(fluid, 'h_max', 850e3))
        rho_f    = float(rho)
        u_f      = float(u_int)
        p_lo = max(p_lo_raw, (h_min - u_f) * rho_f)
        p_hi = min(p_hi_raw, (h_max - u_f) * rho_f)
        if p_lo >= p_hi or p_lo <= 0.0:
            p_lo, p_hi = p_lo_raw, p_hi_raw
        def _f(p):
            h = u_f + p / rho_f
            return float(fluid.get_state(jxp.HmassP_INPUTS, h, p).density) - rho_f
        try:
            p_sol = _brentq(_f, p_lo, p_hi, xtol=1.0, maxiter=60)
        except Exception:
            p_sol = (p_lo + p_hi) * 0.5
        h_sol = u_f + p_sol / rho_f
        return fluid.get_state(jxp.HmassP_INPUTS, h_sol, p_sol)
    return fluid.get_state(jxp.DmassUmass_INPUTS, rho, u_int)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def F_numeric(Q_L, Q_R, parameters):
    method = parameters['space_method']
    if parameters['perfect_gas']:
        _map = {
            'rusanov': F_Rusanov_perfect,
            'roe':     F_Roe_perfect,
            'hllc':    F_HLLC_perfect,
            'hll':     F_HLL_perfect,
            'lf':      F_LaxFriedrichs_perfect,
        }
    else:
        _map = {
            'rusanov': F_Rusanov,
            'roe':     F_Roe,
            'hllc':    F_HLLC,
            'hll':     F_HLL,
            'lf':      F_LaxFriedrichs,
        }
    if method not in _map:
        raise ValueError(f"Unknown flux method '{method}'. Choose from {list(_map)}")
    return _map[method](Q_L, Q_R, parameters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wave_speed_bounds(eig_L, eig_R):
    """Minimum and maximum wave speeds from eigenvalue arrays (3, Nfaces)."""
    S_L = np.minimum(eig_L[0], eig_R[0])
    S_R = np.maximum(eig_L[2], eig_R[2])
    return S_L, S_R


def _H_fix(a_L, a_R, a, c):
    a_abs = abs(a)
    delta = max(abs(a_R - a_L), 0.01 * c)
    return a_abs if a_abs >= delta else (a_abs**2 + delta**2) / (2.0 * delta)


def _H_fix_perfect(a_L, a_R, a, c):
    a_abs = abs(a)
    delta = max(abs(a_R - a_L), 0.1 * c)
    return a_abs if a_abs >= delta else (a_abs**2 + delta**2) / (2.0 * delta)


# ---------------------------------------------------------------------------
# Perfect-gas fluxes
# ---------------------------------------------------------------------------

def _prim_perfect(Q, gamma):
    rho = Q[0]
    u   = Q[1] / rho
    p   = (gamma - 1.0) * (Q[2] - 0.5 * Q[1]**2 / rho)
    c   = np.sqrt(gamma * p / rho)
    return rho, u, p, c


def _euler_flux_perfect(Q, gamma):
    rho, u, p, _ = _prim_perfect(Q, gamma)
    rhoE = Q[2]
    return np.array([rho * u, rho * u**2 + p, (rhoE + p) * u])


def F_Rusanov_perfect(Q_L, Q_R, parameters):
    gamma  = parameters['gamma']
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    F_L = _euler_flux_perfect(Q_L, gamma)
    F_R = _euler_flux_perfect(Q_R, gamma)

    _, u_L, _, c_L = _prim_perfect(Q_L, gamma)
    _, u_R, _, c_R = _prim_perfect(Q_R, gamma)
    a = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)

    return 0.5 * (F_L + F_R - a * (Q_R - Q_L)) * A_face


def F_Roe_perfect(Q_L, Q_R, parameters):
    from .fvm import euler_roe_matrix_perfect
    gamma  = parameters['gamma']
    A      = parameters['A_nodes']
    F_L    = _euler_flux_perfect(Q_L, gamma)
    F_R    = _euler_flux_perfect(Q_R, gamma)

    Nfaces = Q_L.shape[1]
    FA = np.empty_like(F_L)
    for i in range(Nfaces):
        A_face = 0.5 * (A[i] + A[i + 1])
        Ar = euler_roe_matrix_perfect(Q_L[:, i], Q_R[:, i], gamma)
        FA[:, i] = 0.5 * (F_L[:, i] + F_R[:, i] - Ar @ (Q_R[:, i] - Q_L[:, i])) * A_face
    return FA


def F_HLL_perfect(Q_L, Q_R, parameters):
    gamma  = parameters['gamma']
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    _, u_L, p_L, c_L = _prim_perfect(Q_L, gamma)
    _, u_R, p_R, c_R = _prim_perfect(Q_R, gamma)

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    F_L   = _euler_flux_perfect(Q_L, gamma)
    F_R   = _euler_flux_perfect(Q_R, gamma)
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / (S_R - S_L)

    return np.where(S_L >= 0.0, F_L, np.where(S_R <= 0.0, F_R, F_hll)) * A_face


def F_HLLC_perfect(Q_L, Q_R, parameters):
    gamma  = parameters['gamma']
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    rho_L, u_L, p_L, c_L = _prim_perfect(Q_L, gamma)
    rho_R, u_R, p_R, c_R = _prim_perfect(Q_R, gamma)

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    S_M = ((p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
           / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R)))

    # star states
    rho_Ls = rho_L * (S_L - u_L) / (S_L - S_M)
    rho_Rs = rho_R * (S_R - u_R) / (S_R - S_M)
    p_Ls   = p_L + rho_L * (S_L - u_L) * (S_M - u_L)
    p_Rs   = p_R + rho_R * (S_R - u_R) * (S_M - u_R)
    Q_Ls   = np.array([rho_Ls,
                        rho_Ls * S_M,
                        p_Ls / (gamma - 1.0) + 0.5 * rho_Ls * S_M**2])
    Q_Rs   = np.array([rho_Rs,
                        rho_Rs * S_M,
                        p_Rs / (gamma - 1.0) + 0.5 * rho_Rs * S_M**2])

    F_L   = _euler_flux_perfect(Q_L, gamma)
    F_R   = _euler_flux_perfect(Q_R, gamma)
    FA_L  = F_L  * A_face
    FA_R  = F_R  * A_face
    FA_Ls = FA_L + S_L * (Q_Ls - Q_L) * A_face
    FA_Rs = FA_R + S_R * (Q_Rs - Q_R) * A_face

    return np.where(S_L > 0.0,  FA_L,
           np.where(S_M >= 0.0, FA_Ls,
           np.where(S_R >= 0.0, FA_Rs,
                                FA_R)))


def F_LaxFriedrichs_perfect(Q_L, Q_R, parameters):
    gamma  = parameters['gamma']
    A      = parameters['A_nodes']
    dx     = parameters['dx']
    dt     = parameters['dt']
    A_face = 0.5 * (A[:-1] + A[1:])

    F_L = _euler_flux_perfect(Q_L, gamma)
    F_R = _euler_flux_perfect(Q_R, gamma)
    return 0.5 * (F_L + F_R - (dx / dt) * (Q_R - Q_L)) * A_face


# ---------------------------------------------------------------------------
# Real-gas fluxes
# ---------------------------------------------------------------------------

def _euler_flux_real(Q, p):
    rho  = Q[0]
    rhou = Q[1]
    rhoE = Q[2]
    u = rhou / rho
    return np.array([rhou, rhou * u + p, (rhoE + p) * u])


def _prim_real(Q, fluid, p_pre=None, c_pre=None):
    """
    Extract (p, c) from conservative array Q (shape 3×N).

    When p_pre/c_pre are supplied (primitive-variable reconstruction path)
    the brentq inversion is skipped entirely — direct return.
    """
    if p_pre is not None:
        return p_pre, c_pre
    N   = Q.shape[1]
    p_  = np.empty(N)
    c_  = np.empty(N)
    rho = Q[0]
    e   = Q[2] / rho - 0.5 * (Q[1] / rho) ** 2
    for i in range(N):
        try:
            st    = state_from_rho_u(fluid, rho[i], e[i])
            p_[i] = st.pressure
            c_[i] = st.speed_sound
        except Exception:
            p_[i] = p_[i - 1] if i > 0 else 1e5
            c_[i] = c_[i - 1] if i > 0 else 100.0
    return p_, c_


def F_Rusanov(Q_L, Q_R, parameters):
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    p_L, c_L = _prim_real(Q_L, fluid, parameters.get('p_L'), parameters.get('c_L'))
    p_R, c_R = _prim_real(Q_R, fluid, parameters.get('p_R'), parameters.get('c_R'))
    u_L = Q_L[1] / Q_L[0]
    u_R = Q_R[1] / Q_R[0]

    F_L = _euler_flux_real(Q_L, p_L)
    F_R = _euler_flux_real(Q_R, p_R)

    a = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)
    return 0.5 * (F_L + F_R - a * (Q_R - Q_L)) * A_face


def F_Roe(Q_L, Q_R, parameters):
    from .fvm import euler_roe_matrix
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    A      = parameters['A_nodes']

    p_L, _ = _prim_real(Q_L, fluid, parameters.get('p_L'), parameters.get('c_L'))
    p_R, _ = _prim_real(Q_R, fluid, parameters.get('p_R'), parameters.get('c_R'))
    F_L = _euler_flux_real(Q_L, p_L)
    F_R = _euler_flux_real(Q_R, p_R)

    Nfaces = Q_L.shape[1]
    FA = np.empty_like(F_L)
    for i in range(Nfaces):
        A_face = 0.5 * (A[i] + A[i + 1])
        Ar = euler_roe_matrix(Q_L[:, i], Q_R[:, i], p_L[i], p_R[i], fluid)
        FA[:, i] = 0.5 * (F_L[:, i] + F_R[:, i] - Ar @ (Q_R[:, i] - Q_L[:, i])) * A_face
    return FA


def F_HLL(Q_L, Q_R, parameters):
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    p_L, c_L = _prim_real(Q_L, fluid, parameters.get('p_L'), parameters.get('c_L'))
    p_R, c_R = _prim_real(Q_R, fluid, parameters.get('p_R'), parameters.get('c_R'))
    u_L = Q_L[1] / Q_L[0]
    u_R = Q_R[1] / Q_R[0]

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    F_L   = _euler_flux_real(Q_L, p_L)
    F_R   = _euler_flux_real(Q_R, p_R)
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / (S_R - S_L)

    return np.where(S_L >= 0.0, F_L, np.where(S_R <= 0.0, F_R, F_hll)) * A_face


def F_HLLC(Q_L, Q_R, parameters):
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    A      = parameters['A_nodes']
    A_face = 0.5 * (A[:-1] + A[1:])

    p_L, c_L = _prim_real(Q_L, fluid, parameters.get('p_L'), parameters.get('c_L'))
    p_R, c_R = _prim_real(Q_R, fluid, parameters.get('p_R'), parameters.get('c_R'))
    rho_L = Q_L[0];  rho_R = Q_R[0]
    u_L   = Q_L[1] / rho_L
    u_R   = Q_R[1] / rho_R

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    S_M = ((p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
           / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R)))

    # star states: density + momentum; energy from internal energy at (rho, p_star)
    rho_Ls = rho_L * (S_L - u_L) / (S_L - S_M)
    rho_Rs = rho_R * (S_R - u_R) / (S_R - S_M)
    p_Ls   = p_L + rho_L * (S_L - u_L) * (S_M - u_L)
    p_Rs   = p_R + rho_R * (S_R - u_R) * (S_M - u_R)

    _st_Ls = fluid.get_state(jxp.DmassP_INPUTS, rho_Ls, p_Ls)
    _st_Rs = fluid.get_state(jxp.DmassP_INPUTS, rho_Rs, p_Rs)
    e_Ls   = np.asarray(_st_Ls.umass, dtype=float)
    e_Rs   = np.asarray(_st_Rs.umass, dtype=float)

    Q_Ls = np.array([rho_Ls, rho_Ls * S_M, rho_Ls * (e_Ls + 0.5 * S_M**2)])
    Q_Rs = np.array([rho_Rs, rho_Rs * S_M, rho_Rs * (e_Rs + 0.5 * S_M**2)])

    F_L   = _euler_flux_real(Q_L, p_L)
    F_R   = _euler_flux_real(Q_R, p_R)
    FA_L  = F_L  * A_face
    FA_R  = F_R  * A_face
    FA_Ls = FA_L + S_L * (Q_Ls - Q_L) * A_face
    FA_Rs = FA_R + S_R * (Q_Rs - Q_R) * A_face

    return np.where(S_L > 0.0,  FA_L,
           np.where(S_M >= 0.0, FA_Ls,
           np.where(S_R >= 0.0, FA_Rs,
                                FA_R)))


def F_LaxFriedrichs(Q_L, Q_R, parameters):
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    A      = parameters['A_nodes']
    dx     = parameters['dx']
    dt     = parameters['dt']
    A_face = 0.5 * (A[:-1] + A[1:])

    p_L, _ = _prim_real(Q_L, fluid, parameters.get('p_L'), parameters.get('c_L'))
    p_R, _ = _prim_real(Q_R, fluid, parameters.get('p_R'), parameters.get('c_R'))
    F_L = _euler_flux_real(Q_L, p_L)
    F_R = _euler_flux_real(Q_R, p_R)
    return 0.5 * (F_L + F_R - (dx / dt) * (Q_R - Q_L)) * A_face
