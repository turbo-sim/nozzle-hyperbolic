"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
import jaxprop as jxp
from ...numerics.reconstruction import _LIMITERS, _muscl_reconstruct


def _muscl_reconstruct_prim(Q, parameters, method):
    """
    MUSCL reconstruction in primitive variables (rho, u, p).

    Reconstructing in primitives avoids the expensive _state_rho_u brentq for
    the interface states.  After limiting, DmassP_INPUTS (FluidBicubic's
    JIT-compiled 1D Newton — fast) is called once per interface to recover
    speed_sound and umass, then Q_L/Q_R are assembled.

    p_L, p_R, c_L, c_R are stored in parameters so the flux functions can
    bypass _prim_real entirely.

    Falls back to conservative reconstruction when p_cells is unavailable
    (e.g. at the very first time step before _real_gas_state_and_friction runs).
    """
    fluid   = parameters.get('fluid')
    p_cells = parameters.get('p')
    if fluid is None or p_cells is None:
        return _muscl_reconstruct(Q, method)

    limiter = _LIMITERS[method]

    rho = Q[0]
    u   = Q[1] / rho

    W     = np.array([rho, u, p_cells])     # shape (3, Nj)
    dW    = W[:, 1:] - W[:, :-1]
    sigma = np.zeros_like(W)
    sigma[:, 1:-1] = limiter(dW[:, :-1], dW[:, 1:])

    W_L = W[:, :-1] + 0.5 * sigma[:, :-1]
    W_R = W[:, 1:]  - 0.5 * sigma[:, 1:]

    # Positivity guards
    W_L[0] = np.maximum(W_L[0], 1e-6)
    W_R[0] = np.maximum(W_R[0], 1e-6)
    W_L[2] = np.maximum(W_L[2], 1.0)
    W_R[2] = np.maximum(W_R[2], 1.0)

    rho_L, u_L, p_L = W_L[0], W_L[1], W_L[2]
    rho_R, u_R, p_R = W_R[0], W_R[1], W_R[2]

    c_cells = parameters['c']

    # Single batch call for each side — jax.vmap handles the N faces at once
    try:
        st_L   = fluid.get_state(jxp.DmassP_INPUTS, rho_L, p_L)
        c_L    = np.asarray(st_L.speed_sound, dtype=float)
        rhoE_L = rho_L * (np.asarray(st_L.umass, dtype=float) + 0.5 * u_L**2)
    except Exception:
        c_L    = c_cells[:-1]
        rhoE_L = Q[2, :-1]

    try:
        st_R   = fluid.get_state(jxp.DmassP_INPUTS, rho_R, p_R)
        c_R    = np.asarray(st_R.speed_sound, dtype=float)
        rhoE_R = rho_R * (np.asarray(st_R.umass, dtype=float) + 0.5 * u_R**2)
    except Exception:
        c_R    = c_cells[1:]
        rhoE_R = Q[2, 1:]

    Q_L = np.array([rho_L, rho_L * u_L, rhoE_L])
    Q_R = np.array([rho_R, rho_R * u_R, rhoE_R])

    parameters['p_L'] = p_L
    parameters['p_R'] = p_R
    parameters['c_L'] = c_L
    parameters['c_R'] = c_R

    return Q_L, Q_R
