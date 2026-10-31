"""Unchanged definitions relocated from fvm_solver.fvm."""
import numpy as np
from ...thermodynamics.recovery import state_from_rho_u_batch
from .closures.wall import Haaland


# ---------------------------------------------------------------------------
# Real-gas properties + friction  (single combined loop)
# ---------------------------------------------------------------------------

def _real_gas_state_and_friction(Q_n, D_hydraulic, roughness, fluid, wall_friction=True):
    """
    Compute p, c, tau_w for every cell.

    EOS inversion uses a single batch bisection (_state_rho_u_batch) which
    makes ~35 vmap'ed JIT calls instead of N×30 scalar Python→JAX calls.
    """
    Nj  = Q_n.shape[1]
    rho = Q_n[0]
    u   = Q_n[1] / rho
    e   = Q_n[2] / rho - 0.5 * u**2

    try:
        states  = state_from_rho_u_batch(fluid, rho, e)
        p       = np.array(states.pressure,     dtype=float)
        c       = np.array(states.speed_sound,  dtype=float)
        mu_arr  = np.array(states.viscosity,    dtype=float)
    except Exception as exc:
        print(f"Batch EOS error: {exc}", flush=True)
        p      = np.full(Nj, 1e5)
        c      = np.full(Nj, 100.0)
        mu_arr = np.full(Nj, 1e-5)

    tau_w = np.zeros(Nj)
    if wall_friction:
        for j in range(Nj):
            Re  = rho[j] * abs(u[j]) * D_hydraulic[j] / max(mu_arr[j], 1e-12)
            Cf  = Haaland(Re, roughness, D_hydraulic[j])
            tau_w[j] = Cf * rho[j] * u[j]**2 / 8.0

    return p, c, tau_w


# ---------------------------------------------------------------------------
# Source term  (per unit volume: S_A / A_i)
# ---------------------------------------------------------------------------

def _source_term(p, tau_w, A_nodes, D_hydraulic, dx):
    """
    Returns source/volume for all cells.

    Momentum: (p/A)*dA/dx - 4*tau_w/D_h
    Energy:   0  (adiabatic no-slip: total enthalpy flux is conserved)
    """
    Nj = len(p)
    S  = np.zeros((3, Nj))

    # interior — central difference for dA/dx
    S[1, 1:-1] = (p[1:-1] * (A_nodes[2:] - A_nodes[:-2]) / (2.0 * A_nodes[1:-1] * dx)
                  - tau_w[1:-1] * 4.0 / D_hydraulic[1:-1])
    # boundary cells — one-sided differences
    S[1,  0] = p[0]  * (A_nodes[1]  - A_nodes[0])  / (A_nodes[0]  * dx) - tau_w[0]  * 4.0 / D_hydraulic[0]
    S[1, -1] = p[-1] * (A_nodes[-1] - A_nodes[-2]) / (A_nodes[-1] * dx) - tau_w[-1] * 4.0 / D_hydraulic[-1]
    return S
