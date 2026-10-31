"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
import jaxprop as jxp
from ...thermodynamics.recovery import state_from_rho_u_batch


# ---------------------------------------------------------------------------
# Städtke Newton implicit solver  — (p, u, h) primitive variables
# ---------------------------------------------------------------------------

def _conservative_to_puh(Q, fluid):
    """Convert conservative Q (3, N) → primitive U_ph (N, 3) of (p, u, h)."""
    rho    = Q[0]
    e_int  = Q[2] / rho - 0.5 * (Q[1] / rho)**2
    states = state_from_rho_u_batch(fluid, rho, e_int)
    if isinstance(states, list):
        p = np.array([float(s.pressure) for s in states])
        h = np.array([float(s.hmass)    for s in states])
    else:
        p = np.asarray(states.pressure, dtype=float)
        h = np.asarray(states.hmass,    dtype=float)
    return np.stack([p, Q[1] / rho, h], axis=1)  # (N, 3)


def _puh_to_conservative(U, fluid):
    """Convert primitive U_ph (N, 3) → conservative Q (3, N)."""
    p, u, h = U[:, 0], U[:, 1], U[:, 2]
    st  = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    rho = np.asarray(st.rhomass, dtype=float)
    e   = h - p / rho          # h = e + p/ρ  →  e = h - p/ρ
    return np.array([rho, rho * u, rho * (e + 0.5 * u**2)])  # (3, N)
