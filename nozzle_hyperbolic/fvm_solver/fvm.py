"""
1D finite-volume solver for quasi-1D compressible Euler equations.

Governing equations (area-weighted conservative form):
    d(QA)/dt + d(FA)/dx = S_A

    Q = [rho, rho*u, rho*E]^T
    F = [rho*u,  rho*u^2+p,  (rho*E+p)*u]^T
    S_A = [0,  p*dA/dx - tau_w*4A/D_h,  0]^T

    The energy source is zero: at a no-slip adiabatic wall u_wall=0 so wall
    friction does no work on the fluid. Total enthalpy H = h + u^2/2 is
    conserved (consistent with Fanno-flow theory).

Semi-discrete FVM (dividing by A_i*dx):
    dQ_i/dt = -(FA)_{i+1/2} - (FA)_{i-1/2}) / (A_i*dx) + S_A,i / A_i
"""

import time as _time
import numpy as np
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from .flux import F_numeric
from scipy.optimize import brentq as _brentq

_FluidBicubic = getattr(jxp, 'FluidBicubic', None)


def _get_state_ps(fluid, p, s):
    """
    Get thermodynamic state from (p, s).
    FluidBicubic's PSmass_INPUTS handler uses a 2D Newton solver that fails when
    the table has NaN entries.  Instead, do a 1D brentq on h:
        find h s.t.  entropy(h, p) = s
    using HmassP_INPUTS (direct bicubic lookup, no internal solver).
    Entropy is monotone in h at constant p, so the bracket always works.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        h_min = float(getattr(fluid, 'h_min', 50e3))
        h_max = float(getattr(fluid, 'h_max', 850e3))
        p_f   = float(p)
        s_f   = float(s)
        def _f(h):
            return float(fluid.get_state(jxp.HmassP_INPUTS, h, p_f).entropy) - s_f
        try:
            h_sol = _brentq(_f, h_min, h_max, xtol=1.0, maxiter=60)
        except Exception:
            h_sol = (h_min + h_max) * 0.5
        return fluid.get_state(jxp.HmassP_INPUTS, h_sol, p_f)
    return fluid.get_state(jxp.PSmass_INPUTS, p, s)


def _get_state_hs(fluid, h, s):
    """
    Get thermodynamic state from (h, s).
    FluidBicubic's HmassSmass_INPUTS handler uses a 1D Newton on p that can fail
    when the table has NaN entries near the saturation boundary.  Instead, do a
    1D brentq on p:
        find p s.t.  entropy(h, p) = s
    using HmassP_INPUTS (direct bicubic lookup).
    Entropy decreases with pressure at constant h, so the bracket always works.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        p_min = float(getattr(fluid, 'p_min', 1e4))
        p_max = float(getattr(fluid, 'p_max', 300e5))
        h_f   = float(h)
        s_f   = float(s)
        def _f(p):
            return float(fluid.get_state(jxp.HmassP_INPUTS, h_f, p).entropy) - s_f
        try:
            p_sol = _brentq(_f, p_min, p_max, xtol=1.0, maxiter=60)
        except Exception:
            p_sol = (p_min + p_max) * 0.5
        return fluid.get_state(jxp.HmassP_INPUTS, h_f, p_sol)
    return fluid.get_state(jxp.HmassSmass_INPUTS, h, s)


def state_from_rho_u_batch(fluid, rho_arr, u_int_arr):
    """
    Vectorized (rho, u_int) → state for an entire array of cells.

    Replaces N individual brentq calls with ~32 batch EOS calls via bisection.
    Each batch call goes through FluidBicubic.get_state which uses jax.vmap
    internally, so all N values are processed in a single JIT-compiled pass.
    Falls back to the scalar loop for non-FluidBicubic fluids.
    """
    if not (_FluidBicubic and isinstance(fluid, _FluidBicubic)):
        states = [fluid.get_state(jxp.DmassUmass_INPUTS, float(r), float(u))
                  for r, u in zip(rho_arr, u_int_arr)]
        return states  # list of scalar states

    p_lo_raw = float(getattr(fluid, 'p_min', 1e4))
    p_hi_raw = float(getattr(fluid, 'p_max', 300e5))
    h_lo_raw = float(getattr(fluid, 'h_min', 50e3))
    h_hi_raw = float(getattr(fluid, 'h_max', 850e3))

    rho   = np.asarray(rho_arr,   dtype=float)
    u_int = np.asarray(u_int_arr, dtype=float)

    # Per-cell h-range constraint on the pressure bracket
    p_lo = np.full(len(rho), p_lo_raw)
    p_hi = np.full(len(rho), p_hi_raw)
    p_from_hmin = (h_lo_raw - u_int) * rho
    p_from_hmax = (h_hi_raw - u_int) * rho
    valid_lo = (p_from_hmin > p_lo_raw) & (p_from_hmin < p_hi_raw)
    valid_hi = (p_from_hmax > p_lo_raw) & (p_from_hmax < p_hi_raw)
    p_lo = np.where(valid_lo, np.maximum(p_lo, p_from_hmin), p_lo)
    p_hi = np.where(valid_hi, np.minimum(p_hi, p_from_hmax), p_hi)
    bad  = p_lo >= p_hi
    p_lo = np.where(bad, p_lo_raw, p_lo)
    p_hi = np.where(bad, p_hi_raw, p_hi)
    p = 0.5 * (p_lo + p_hi)

    # Bisection: 35 batch calls instead of N×30 scalar calls
    for _ in range(35):
        h = np.clip(u_int + p / rho, h_lo_raw + 1.0, h_hi_raw - 1.0)
        try:
            rho_pred = np.asarray(
                fluid.get_state(jxp.HmassP_INPUTS, h, p).density)
        except Exception:
            break
        residual = rho_pred - rho
        p_lo = np.where(residual < 0, p, p_lo)
        p_hi = np.where(residual > 0, p, p_hi)
        p = 0.5 * (p_lo + p_hi)

    h_final = np.clip(u_int + p / rho, h_lo_raw + 1.0, h_hi_raw - 1.0)
    return fluid.get_state(jxp.HmassP_INPUTS, h_final, p)


def _state_rho_u(fluid, rho, u_int):
    """
    Get thermodynamic state from (density, internal_energy).

    FluidBicubic table is indexed by (h, logP).  HmassP_INPUTS is a direct
    bicubic lookup (no internal solver), so we use it inside a brentq over p:
        residual(p) = density(h = u_int + p/rho,  p) - rho = 0
    The p bracket is constrained so that h = u_int + p/rho stays inside the
    table's h range, avoiding extrapolation at the bracket endpoints.
    For jxp.Fluid the direct DmassUmass_INPUTS EOS call is used.
    """
    if _FluidBicubic and isinstance(fluid, _FluidBicubic):
        p_lo_raw = float(getattr(fluid, 'p_min', 1e4))
        p_hi_raw = float(getattr(fluid, 'p_max', 300e5))
        h_min    = float(getattr(fluid, 'h_min', 50e3))
        h_max    = float(getattr(fluid, 'h_max', 850e3))
        rho_f    = float(rho)
        u_f      = float(u_int)
        # Clamp bracket so h = u+p/rho stays within the table h range
        p_lo = max(p_lo_raw, (h_min - u_f) * rho_f)
        p_hi = min(p_hi_raw, (h_max - u_f) * rho_f)
        if p_lo >= p_hi or p_lo <= 0.0:   # fallback: use full table p range
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
# Isentropic initialization
# ---------------------------------------------------------------------------

def isentropic_initialization(x, A_nodes, parameters):
    """
    Initial condition for the FVM solver.

    Perfect gas: analytical isentropic solution with shock detection.
    Real gas: linear interpolation between inlet and outlet PT states (fast,
              no per-cell EOS calls — the solver corrects the field quickly).
    """
    perfect_gas = parameters['perfect_gas']
    p_stag      = parameters['pT_stagnation'][0]
    T_stag      = parameters['pT_stagnation'][1]
    p_out       = parameters['p_out']
    A_throat    = np.min(A_nodes)
    i_throat    = int(np.argmin(A_nodes))
    Nj          = len(x)
    Q0          = np.zeros((3, Nj))

    if perfect_gas:
        cp, cv  = hyp.fvm_solver.specific_heat2()
        gamma   = cp / cv
        R_const = cp - cv
        rho_stag = p_stag / (R_const * T_stag)

        # Analytical throat (critical) state
        T_thr   = T_stag * 2.0 / (gamma + 1.0)
        rho_thr = rho_stag * (T_thr / T_stag) ** (1.0 / (gamma - 1.0))
        u_thr   = math.sqrt(gamma * R_const * T_thr)   # u_throat = c_throat
        m_dot   = rho_thr * u_thr * A_throat

        def _rho_u(u):
            T = T_stag - u**2 / (2.0 * cp)
            if T <= 0.0:
                return 0.0
            return rho_stag * (T / T_stag) ** (1.0 / (gamma - 1.0)) * u

        for i in range(Nj):
            Ai = A_nodes[i]
            f  = lambda u, Ai=Ai: _rho_u(u) * Ai - m_dot
            try:
                u_i = _brentq(f, 1e-6, u_thr, xtol=0.01, maxiter=60)
            except ValueError:
                u_i = m_dot / (rho_stag * Ai)   # fallback: stagnation density

            u_i   = max(u_i, 0.0)
            T_i   = T_stag - u_i**2 / (2.0 * cp)
            p_i   = p_stag * (T_i / T_stag) ** (gamma / (gamma - 1.0))
            rho_i = p_i / (R_const * T_i)
            e_i   = p_i / (rho_i * (gamma - 1.0))
            Q0[0, i] = rho_i
            Q0[1, i] = rho_i * u_i
            Q0[2, i] = rho_i * (e_i + 0.5 * u_i**2)

    else:
        fluid   = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
        st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
        h0      = float(st_stag.hmass)
        s0      = float(st_stag.smass)

        h_lo = float(getattr(fluid, 'h_min', 50e3)) + 1.0
        h_hi = min(float(getattr(fluid, 'h_max', 850e3)) - 1.0, h0 - 1.0)
        p_lo = float(getattr(fluid, 'p_min', 1e4))   * 1.01
        p_hi = float(getattr(fluid, 'p_max', 300e5)) * 0.99

        # Build isentropic table using only HmassP_INPUTS (direct bicubic lookup).
        # Sweep h from h_lo to h0; for each h find p s.t. entropy(h,p) = s0 via
        # vectorized bisection.  Entropy decreases with p at constant h, so:
        #   s(h, p_lo) > s0  and  s(h, p_hi_bracket) < s0.
        N_tab  = 300
        h_tab  = np.linspace(h_lo, h_hi, N_tab)
        p_lo_v = np.full(N_tab, p_lo)
        p_hi_v = np.full(N_tab, min(p_stag * 1.05, p_hi))

        for _ in range(50):
            p_mid  = 0.5 * (p_lo_v + p_hi_v)
            s_pred = np.asarray(
                fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_mid).entropy, dtype=float)
            p_lo_v = np.where(s_pred > s0, p_mid, p_lo_v)
            p_hi_v = np.where(s_pred < s0, p_mid, p_hi_v)

        p_iso  = 0.5 * (p_lo_v + p_hi_v)
        _st    = fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_iso)
        rho_tab = np.asarray(_st.rhomass, dtype=float)
        e_tab   = np.asarray(_st.umass,   dtype=float)
        u_tab   = np.sqrt(np.maximum(2.0 * (h0 - h_tab), 0.0))
        G_tab   = rho_tab * u_tab   # mass flux per unit area along the isentrope

        # Throat: maximum G (Ma = 1)
        k_thr = int(np.argmax(G_tab))
        G_thr = G_tab[k_thr]
        m_dot = G_thr * A_throat

        # Subsonic branch: h in [h_thr, h0) → G decreasing; reverse for searchsorted
        # Supersonic branch: h in [h_lo, h_thr] → G increasing
        sub_G = G_tab[k_thr:][::-1];  sub_h = h_tab[k_thr:][::-1];  sub_p = p_iso[k_thr:][::-1]
        sup_G = G_tab[:k_thr + 1];    sup_h = h_tab[:k_thr + 1];    sup_p = p_iso[:k_thr + 1]

        h_cells = np.empty(Nj)
        p_cells = np.empty(Nj)
        for i in range(Nj):
            G_target          = m_dot / A_nodes[i]
            seg_G, seg_h, seg_p = (sub_G, sub_h, sub_p) if i <= i_throat else (sup_G, sup_h, sup_p)
            idx               = np.clip(np.searchsorted(seg_G, G_target), 1, len(seg_G) - 1)
            t                 = (G_target - seg_G[idx-1]) / max(seg_G[idx] - seg_G[idx-1], 1e-300)
            h_cells[i]        = seg_h[idx-1] + t * (seg_h[idx] - seg_h[idx-1])
            p_cells[i]        = seg_p[idx-1] + t * (seg_p[idx] - seg_p[idx-1])

        # One vectorized direct table lookup for all cells
        _st_c = fluid.get_state(jxp.HmassP_INPUTS, h_cells, p_cells)
        rho_c = np.asarray(_st_c.rhomass, dtype=float)
        e_c   = np.asarray(_st_c.umass,   dtype=float)
        u_c   = np.sqrt(np.maximum(2.0 * (h0 - h_cells), 0.0))
        Q0[0] = rho_c
        Q0[1] = rho_c * u_c
        Q0[2] = rho_c * (e_c + 0.5 * u_c**2)

    return Q0


def shock_tube_Q_initialization(x, A_nodes, parameters):
    """
    Step-function initial condition for shock tube tests.
    Returns Q (3, N) in conservative form.

    Reads 'shock_tube_cfg' from parameters:
        x_diaphragm, p_L, T_L, u_L, p_R, T_R, u_R
    """
    fluid = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    cfg   = parameters['shock_tube_cfg']
    x_d   = float(cfg.get('x_diaphragm', 0.5 * (x[0] + x[-1])))
    p_L, T_L, u_L = float(cfg['p_L']), float(cfg['T_L']), float(cfg.get('u_L', 1.0))
    p_R, T_R, u_R = float(cfg['p_R']), float(cfg['T_R']), float(cfg.get('u_R', 1.0))

    st_L  = fluid.get_state(jxp.PT_INPUTS, p_L, T_L)
    st_R  = fluid.get_state(jxp.PT_INPUTS, p_R, T_R)
    rho_L, e_L = float(st_L.rhomass), float(st_L.umass)
    rho_R, e_R = float(st_R.rhomass), float(st_R.umass)

    Q0           = np.zeros((3, len(x)))
    left         = x <= x_d
    Q0[0, left]  = rho_L;           Q0[0, ~left] = rho_R
    Q0[1, left]  = rho_L * u_L;     Q0[1, ~left] = rho_R * u_R
    Q0[2, left]  = rho_L * (e_L + 0.5 * u_L**2)
    Q0[2, ~left] = rho_R * (e_R + 0.5 * u_R**2)
    return Q0


def shock_tube_puh_initialization(x, A_nodes, parameters):
    """
    Step-function initial condition for the FVS Newton solver.
    Returns U_ph (N, 3) in primitive (p, u, h) form.
    """
    fluid = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    cfg   = parameters['shock_tube_cfg']
    x_d   = float(cfg.get('x_diaphragm', 0.5 * (x[0] + x[-1])))
    p_L, T_L, u_L = float(cfg['p_L']), float(cfg['T_L']), float(cfg.get('u_L', 1.0))
    p_R, T_R, u_R = float(cfg['p_R']), float(cfg['T_R']), float(cfg.get('u_R', 1.0))

    st_L = fluid.get_state(jxp.PT_INPUTS, p_L, T_L)
    st_R = fluid.get_state(jxp.PT_INPUTS, p_R, T_R)
    h_L  = float(st_L.hmass);  h_R = float(st_R.hmass)

    U0          = np.zeros((len(x), 3))
    left        = x <= x_d
    U0[left]    = [p_L, u_L, h_L]
    U0[~left]   = [p_R, u_R, h_R]
    return U0


def stadtke_puh_initialization(x, A_nodes, parameters):
    """
    Initial condition for the Städtke Newton solver in primitive (p, u, h) form.

    Uses the same isentropic-table algorithm as isentropic_initialization for
    real gas, but returns U_ph (N, 3) directly — no conservative-to-primitive
    round-trip, so h and p are exact and self-consistent with HmassP_INPUTS.

    Returns U_ph : (N, 3)  — columns are (p, u, h).
    """
    fluid       = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    p_stag      = parameters['pT_stagnation'][0]
    T_stag      = parameters['pT_stagnation'][1]
    p_out       = parameters['p_out']
    A_throat    = float(np.min(A_nodes))
    i_throat    = int(np.argmin(A_nodes))
    Nj          = len(x)

    st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
    h0      = float(st_stag.hmass)
    s0      = float(st_stag.smass)

    h_lo = float(getattr(fluid, 'h_min', 50e3))  + 10.0
    h_hi = min(float(getattr(fluid, 'h_max', 850e3)) - 10.0, h0 - 10.0)
    p_lo = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi = float(getattr(fluid, 'p_max', 300e5)) * 0.99

    # Build isentropic table: for each h, find p s.t. s(h,p) = s0
    N_tab  = 300
    h_tab  = np.linspace(h_lo, h_hi, N_tab)
    p_lo_v = np.full(N_tab, p_lo)
    p_hi_v = np.full(N_tab, min(p_stag * 1.05, p_hi))
    for _ in range(50):
        p_mid  = 0.5 * (p_lo_v + p_hi_v)
        s_pred = np.asarray(
            fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_mid).entropy, dtype=float)
        p_lo_v = np.where(s_pred > s0, p_mid, p_lo_v)
        p_hi_v = np.where(s_pred < s0, p_mid, p_hi_v)
    p_iso = 0.5 * (p_lo_v + p_hi_v)

    # Remove entries where bisection converged to the lower bound — these are states
    # outside the physical region (solid, two-phase, or below EOS table range) where
    # the EOS returns zero entropy so the bisection always shrinks p_hi → p_lo.
    valid  = p_iso > p_lo * 1.05
    if not np.any(valid):
        raise ValueError(
            "stadtke_puh_initialization: isentropic bisection found no valid states. "
            "Check that stagnation conditions (p_stag, T_stag) are within the EOS "
            "table bounds (h_min, h_max, p_min, p_max).")
    h_tab  = h_tab[valid]
    p_iso  = p_iso[valid]

    rho_iso = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_iso).rhomass, dtype=float)
    u_tab  = np.sqrt(np.maximum(2.0 * (h0 - h_tab), 0.0))
    G_tab  = rho_iso * u_tab                     # mass flux per unit area

    # Throat at G_max
    k_thr = int(np.argmax(G_tab))
    G_thr = G_tab[k_thr]
    m_dot = G_thr * A_throat

    # Subsonic branch (h decreasing from h_thr to h0, G decreasing)
    sub_G = G_tab[k_thr:][::-1];  sub_h = h_tab[k_thr:][::-1];  sub_p = p_iso[k_thr:][::-1]
    # Supersonic branch (h decreasing toward h_lo, G increasing toward G_thr)
    sup_G = G_tab[:k_thr + 1];    sup_h = h_tab[:k_thr + 1];    sup_p = p_iso[:k_thr + 1]

    h_cells = np.empty(Nj)
    p_cells = np.empty(Nj)
    for i in range(Nj):
        G_target              = m_dot / A_nodes[i]
        seg_G, seg_h, seg_p   = (sub_G, sub_h, sub_p) if i <= i_throat else (sup_G, sup_h, sup_p)
        idx                   = np.clip(np.searchsorted(seg_G, G_target), 1, len(seg_G) - 1)
        t                     = (G_target - seg_G[idx-1]) / max(seg_G[idx] - seg_G[idx-1], 1e-300)
        h_cells[i]            = seg_h[idx-1] + t * (seg_h[idx] - seg_h[idx-1])
        p_cells[i]            = seg_p[idx-1] + t * (seg_p[idx] - seg_p[idx-1])

    u_cells = np.sqrt(np.maximum(2.0 * (h0 - h_cells), 0.0))
    # jax.debug.print("initial p:{uo}", uo=p_cells)
    # jax.debug.print("initial u:{uo}", uo=u_cells)
    # jax.debug.print("initial h:{uo}", uo=h_cells)

    return np.stack([p_cells, u_cells, h_cells], axis=1)  # (N, 3)


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


# ---------------------------------------------------------------------------
# Real-gas properties + friction  (single combined loop)
# ---------------------------------------------------------------------------

def _real_gas_state_and_friction(Q_n, D_hydraulic, roughness, fluid):
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


# ---------------------------------------------------------------------------
# MUSCL reconstruction  (2nd-order spatial accuracy)
# ---------------------------------------------------------------------------

def _minmod(a, b):
    return np.where(np.sign(a) == np.sign(b),
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _van_leer(a, b):
    ab = a * b
    return np.where(ab > 0.0, 2.0 * ab / (a + b + 1e-300), 0.0)


def _superbee(a, b):
    s1 = np.where(np.sign(a) == np.sign(b),
                  np.sign(a) * np.minimum(2.0 * np.abs(a), np.abs(b)), 0.0)
    s2 = np.where(np.sign(a) == np.sign(b),
                  np.sign(a) * np.minimum(np.abs(a), 2.0 * np.abs(b)), 0.0)
    return np.where(np.abs(s1) > np.abs(s2), s1, s2)


_LIMITERS = {
    'muscl_minmod':   _minmod,
    'muscl_van_leer': _van_leer,
    'muscl_superbee': _superbee,
}


def _muscl_reconstruct(Q, method):
    """
    MUSCL piecewise-linear reconstruction in conservative variables.

    Returns Q_L, Q_R at each of the Nj-1 interfaces (shape (3, Nj-1)).
    Ghost cells (i=0 and i=Nj-1) use zero slope so the BC values are
    passed through unchanged.
    """
    limiter = _LIMITERS[method]

    dQ = Q[:, 1:] - Q[:, :-1]          # (3, Nj-1): dQ[:, i] = Q_{i+1} - Q_i

    # Slopes for interior cells 1 … Nj-2; zero at ghost cells
    sigma = np.zeros_like(Q)            # (3, Nj)
    sigma[:, 1:-1] = limiter(dQ[:, :-1], dQ[:, 1:])   # (3, Nj-2)

    Q_L = Q[:, :-1] + 0.5 * sigma[:, :-1]   # left  state at each interface
    Q_R = Q[:, 1:]  - 0.5 * sigma[:, 1:]    # right state at each interface
    return Q_L, Q_R


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


# ---------------------------------------------------------------------------
# Spatial operator:  R(Q) = -d(FA)/dx / A
# ---------------------------------------------------------------------------

def R(Q, parameters):
    """
    Flux divergence for all cells.

        RHS[:, i] = -(FA_{i+1/2} - FA_{i-1/2}) / (A_i * dx)

    Supports piecewise-constant (1st order) and MUSCL (2nd order) reconstruction
    controlled by parameters['reconstruction']:
        'constant'        — 1st-order Godunov
        'muscl_minmod'    — 2nd-order MUSCL with minmod limiter
        'muscl_van_leer'  — 2nd-order MUSCL with van Leer limiter
        'muscl_superbee'  — 2nd-order MUSCL with superbee limiter
    """
    _, Nj = Q.shape
    A     = parameters['A_nodes']
    dx    = parameters['dx']
    recon = parameters.get('reconstruction', 'constant')
    real  = not parameters['perfect_gas']

    if recon == 'constant':
        Q_L, Q_R = Q[:, :-1], Q[:, 1:]
        if real and 'p' in parameters and 'c' in parameters:
            parameters['p_L'] = parameters['p'][:-1]
            parameters['p_R'] = parameters['p'][1:]
            parameters['c_L'] = parameters['c'][:-1]
            parameters['c_R'] = parameters['c'][1:]
    elif recon in _LIMITERS:
        if real:
            Q_L, Q_R = _muscl_reconstruct_prim(Q, parameters, recon)
        else:
            Q_L, Q_R = _muscl_reconstruct(Q, recon)
    else:
        raise ValueError(f"Unknown reconstruction '{recon}'. "
                         f"Choose from: constant, {', '.join(_LIMITERS)}")

    FA  = F_numeric(Q_L, Q_R, parameters)   # shape (3, Nj-1)
    RHS = np.zeros((3, Nj))
    RHS[:, 1:-1] = -(FA[:, 1:] - FA[:, :-1]) / (A[1:-1] * dx)
    return RHS


# ---------------------------------------------------------------------------
# CFL time step
# ---------------------------------------------------------------------------

def _compute_dt(u, c, dx, CFL):
    max_speed = np.max(np.abs(u) + c)
    return CFL * dx / max_speed if max_speed > 0.0 else CFL * dx


# ---------------------------------------------------------------------------
# JAX-compiled real-gas step
# ---------------------------------------------------------------------------

def _haaland_jnp(Re, roughness, D_h):
    eps_D = roughness / D_h
    Re_s  = jnp.maximum(Re, 1.0)
    term  = (eps_D / 3.7) ** 1.11 + 6.9 / Re_s
    f_t   = (-1.81 * jnp.log10(jnp.maximum(term, 1e-300))) ** (-2)
    return jnp.where(Re < 2300.0, 64.0 / Re_s, f_t)


def _jax_euler_flux(Q, p):
    u = Q[1] / Q[0]
    return jnp.stack([Q[1], Q[1] * u + p, (Q[2] + p) * u])


def _jax_F_rusanov(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R):
    u_L = Q_L[1] / Q_L[0];  u_R = Q_R[1] / Q_R[0]
    F_L = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    a      = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)
    A_face = 0.5 * (A_L + A_R)
    return 0.5 * (F_L + F_R - a * (Q_R - Q_L)) * A_face


def _jax_F_hll(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R):
    u_L = Q_L[1] / Q_L[0];  u_R = Q_R[1] / Q_R[0]
    S_L = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R = jnp.maximum(u_L + c_L, u_R + c_R)
    F_L   = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / jnp.maximum(S_R - S_L, 1e-300)
    A_face = 0.5 * (A_L + A_R)
    return jnp.where(S_L >= 0.0, F_L, jnp.where(S_R <= 0.0, F_R, F_hll)) * A_face


def _jax_F_lf(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R, dx, dt):
    F_L = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    A_face = 0.5 * (A_L + A_R)
    return 0.5 * (F_L + F_R - (dx / dt) * (Q_R - Q_L)) * A_face


def _jax_F_hllc(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R, fluid):
    rho_L = Q_L[0];  rho_R = Q_R[0]
    u_L   = Q_L[1] / rho_L;  u_R = Q_R[1] / rho_R
    E_L   = Q_L[2] / rho_L;  E_R = Q_R[2] / rho_R   # specific total energy
    S_L   = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R   = jnp.maximum(u_L + c_L, u_R + c_R)
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_M   = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / jnp.where(jnp.abs(denom) > 1e-300, denom, 1e-300)
    dL    = jnp.where(jnp.abs(S_L - S_M) > 1e-300, S_L - S_M, 1e-300)
    dR    = jnp.where(jnp.abs(S_R - S_M) > 1e-300, S_R - S_M, 1e-300)
    rho_Ls = rho_L * (S_L - u_L) / dL
    rho_Rs = rho_R * (S_R - u_R) / dR
    # Star total energy via Toro (2009) eq. 10.73 — no EOS call required.
    # rhoE* = rho* * (E + (S* - u) * (S* + p / (rho*(S - u))))
    rhoE_Ls = rho_Ls * (E_L + (S_M - u_L) * (S_M + p_L / jnp.where(jnp.abs(rho_L * (S_L - u_L)) > 1e-300, rho_L * (S_L - u_L), 1e-300)))
    rhoE_Rs = rho_Rs * (E_R + (S_M - u_R) * (S_M + p_R / jnp.where(jnp.abs(rho_R * (S_R - u_R)) > 1e-300, rho_R * (S_R - u_R), 1e-300)))
    Q_Ls   = jnp.stack([rho_Ls, rho_Ls * S_M, rhoE_Ls])
    Q_Rs   = jnp.stack([rho_Rs, rho_Rs * S_M, rhoE_Rs])
    F_L    = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    A_face = 0.5 * (A_L + A_R)
    FA_L   = F_L * A_face;  FA_R = F_R * A_face
    FA_Ls  = FA_L + S_L * (Q_Ls - Q_L) * A_face
    FA_Rs  = FA_R + S_R * (Q_Rs - Q_R) * A_face
    return jnp.where(S_L > 0.0, FA_L, jnp.where(S_M >= 0.0, FA_Ls, jnp.where(S_R >= 0.0, FA_Rs, FA_R)))


# ---------------------------------------------------------------------------
# Städtke FVS flux  (JAX, operates in (p, u, h) primitive space)
# ---------------------------------------------------------------------------

def _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor=0.1):
    """
    Städtke flux-vector-splitting interface flux for all faces simultaneously.

    Primitive states U_L, U_R : (Nfaces, 3)  — columns are (p, u, h).
    Returns conservative flux  F̂ : (Nfaces, 3)  — (ρu, ρu²+p, uρH).

    Algorithm (theory doc §5):
      1. Average state → EOS → ρ, a, G → rho_p, rho_h.
      2. Build split matrices G_k = λ_k v_k ⊗ l_k  (each (Nfaces,3,3)).
      3. Build flux Jacobian K = ∂F/∂U  (Nfaces,3,3).
      4. Projection: R̃_k = K G_k K⁻¹ / λ_k^safe  via batched solve.
      5. Upwind sum:  R̃⁺ = Σ_{λ_k≥0} R̃_k,  R̃⁻ = Σ_{λ_k<0} R̃_k.
      6. F̂ = R̃⁺ F_L + R̃⁻ F_R  (physical fluxes from actual L/R states).
    """
    # ── Averaged state ────────────────────────────────────────────────────
    U_avg  = 0.5 * (U_L + U_R)
    p_a    = U_avg[:, 0];  u_a = U_avg[:, 1];  h_a = U_avg[:, 2]

    st_a   = fluid.get_state(jxp.HmassP_INPUTS, h_a, p_a)
    rho_a  = st_a["rho"]
    a_a    = st_a["a"]
    G_a    = st_a["G"]
    rho_p  = (1.0 + G_a) / a_a**2        # (∂ρ/∂p)_h  — theory §1.3
    rho_h  = -(rho_a * G_a) / a_a**2     # (∂ρ/∂h)_p

    # ── Eigenvalues ───────────────────────────────────────────────────────
    lam1 = u_a + a_a;  lam2 = u_a - a_a;  lam3 = u_a

    # ── Split matrices  G_k = λ_k (v_k ⊗ l_k)  (each (Nfaces,3,3)) ──────
    # Right eigenvectors:  v1=(ρa,1,a),  v2=(-ρa,1,-a),  v3=(0,0,1)
    # Left eigenvectors:   l1=(1/2ρa,1/2,0),  l2=(-1/2ρa,1/2,0),  l3=(-1/ρ,0,1)
    z = jnp.zeros_like(u_a);  o = jnp.ones_like(u_a)

    v1 = jnp.stack([ rho_a * a_a,  o,          a_a  ], axis=-1)  # (N,3)
    v2 = jnp.stack([-rho_a * a_a,  o,         -a_a  ], axis=-1)
    v3 = jnp.stack([ z,             z,          o    ], axis=-1)
    l1 = jnp.stack([ 0.5 / (rho_a * a_a),  0.5 * o,  z], axis=-1)
    l2 = jnp.stack([-0.5 / (rho_a * a_a),  0.5 * o,  z], axis=-1)
    l3 = jnp.stack([-1.0 / rho_a,           z,        o], axis=-1)

    def _outer_scaled(v, l, lam):
        return lam[:, None, None] * jnp.einsum('...i,...j->...ij', v, l)

    G1 = _outer_scaled(v1, l1, lam1)
    G2 = _outer_scaled(v2, l2, lam2)
    G3 = _outer_scaled(v3, l3, lam3)

    # ── Flux Jacobian  K = ∂F/∂U  (Nfaces,3,3) ───────────────────────────
    H_a  = h_a + 0.5 * u_a**2
    r0   = jnp.stack([u_a * rho_p,            rho_a,               u_a * rho_h           ], axis=-1)
    r1   = jnp.stack([u_a**2 * rho_p + 1.0,   2.0 * rho_a * u_a,  u_a**2 * rho_h        ], axis=-1)
    r2   = jnp.stack([u_a * rho_p * H_a,       rho_a * (H_a + u_a**2), u_a * (rho_h * H_a + rho_a)], axis=-1)
    K    = jnp.stack([r0, r1, r2], axis=1)   # (Nfaces, 3, 3)

    # ── Projection matrices  R̃_k = K G_k K⁻¹ / λ_k^safe  ─────────────
    # Solve Kᵀ Xᵀ = (K G_k)ᵀ  →  X = K G_k K⁻¹ = R_k; then divide by λ^safe.
    delta  = delta_factor * (jnp.abs(u_a) + a_a)  # (Nfaces,)

    def _safe(lam):
        # jnp.sign(0) = 0 in JAX → would give 0/0; use where to always return ±1.
        sign = jnp.where(lam >= 0.0, 1.0, -1.0)
        return sign * jnp.maximum(jnp.abs(lam), delta)

    def _Rt(Gk, lam):
        KGk  = K @ Gk                                               # (N,3,3) batched matmul
        KT   = K.swapaxes(-1, -2)
        Rk_T = jnp.linalg.solve(KT, KGk.swapaxes(-1, -2))         # solve KᵀXᵀ = (KGk)ᵀ
        return Rk_T.swapaxes(-1, -2) / _safe(lam)[:, None, None]   # R̃_k

    Rt1 = _Rt(G1, lam1)
    Rt2 = _Rt(G2, lam2)
    Rt3 = _Rt(G3, lam3)

    # ── Sum by TRUE eigenvalue sign ───────────────────────────────────────
    s1p = jnp.where(lam1[:, None, None] >= 0.0, 1.0, 0.0)
    s2p = jnp.where(lam2[:, None, None] >= 0.0, 1.0, 0.0)
    s3p = jnp.where(lam3[:, None, None] >= 0.0, 1.0, 0.0)

    R_plus  = s1p * Rt1 + s2p * Rt2 + s3p * Rt3
    R_minus = (1.0 - s1p) * Rt1 + (1.0 - s2p) * Rt2 + (1.0 - s3p) * Rt3

    # ── Physical fluxes from ACTUAL L/R states ────────────────────────────
    st_L  = fluid.get_state(jxp.HmassP_INPUTS, U_L[:, 2], U_L[:, 0])
    st_R  = fluid.get_state(jxp.HmassP_INPUTS, U_R[:, 2], U_R[:, 0])
    rho_L = st_L["rho"];  rho_R = st_R["rho"]
    H_L   = U_L[:, 2] + 0.5 * U_L[:, 1]**2
    H_R   = U_R[:, 2] + 0.5 * U_R[:, 1]**2
    F_L   = jnp.stack([rho_L * U_L[:, 1],
                        rho_L * U_L[:, 1]**2 + U_L[:, 0],
                        U_L[:, 1] * rho_L * H_L], axis=-1)
    F_R   = jnp.stack([rho_R * U_R[:, 1],
                        rho_R * U_R[:, 1]**2 + U_R[:, 0],
                        U_R[:, 1] * rho_R * H_R], axis=-1)

    # F̂ = R̃⁺ F_L + R̃⁻ F_R  (batched matvec)
    return (jnp.einsum('...ij,...j->...i', R_plus,  F_L)
          + jnp.einsum('...ij,...j->...i', R_minus, F_R))


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


def make_jax_step(fluid, A, D_h, roughness, dx, CFL, p_out, h0, s0,
                  recon, flux_method, time_method):
    """
    Returns a @jax.jit compiled step function for the real-gas Euler solver.
    BCs are applied inside step() — no Python calls needed between iterations.
    Returns (Q_next, p, c, dt) — all JAX arrays.
    """
    A_j   = jnp.asarray(A,   dtype=jnp.float64)
    D_h_j = jnp.asarray(D_h, dtype=jnp.float64)
    A_L   = A_j[:-1];  A_R = A_j[1:]

    # Precompute dA_dx / A once — geometry is constant, no need to recompute every step.
    # Central differences for interior, one-sided at boundaries.
    _dA_dx_j = jnp.concatenate([
        (A_j[1:2]   - A_j[:1])    / (A_j[:1]    * dx),
        (A_j[2:]    - A_j[:-2])   / (A_j[1:-1]  * 2.0 * dx),
        (A_j[-1:]   - A_j[-2:-1]) / (A_j[-1:]   * dx),
    ])

    h_lo    = float(getattr(fluid, 'h_min', 50e3))  + 1.0
    h_hi    = float(getattr(fluid, 'h_max', 850e3)) - 1.0
    p_lo_r  = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi_r  = float(getattr(fluid, 'p_max', 300e5)) * 0.99
    h0_j    = jnp.float64(h0)
    s0_j    = jnp.float64(s0)

    _LIMS = {
        'muscl_minmod':   lambda a, b: jnp.where(jnp.sign(a) == jnp.sign(b),
                              jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0),
        'muscl_van_leer': lambda a, b: jnp.where(a * b > 0, 2 * a * b / (a + b + 1e-300), 0.0),
        'muscl_superbee': lambda a, b: jnp.maximum(
            jnp.where(jnp.sign(a) == jnp.sign(b), jnp.sign(a) * jnp.minimum(2*jnp.abs(a), jnp.abs(b)), 0.0),
            jnp.where(jnp.sign(a) == jnp.sign(b), jnp.sign(a) * jnp.minimum(jnp.abs(a), 2*jnp.abs(b)), 0.0)),
    }
    _limiter = _LIMS.get(recon)  # None → constant reconstruction

    _FLUX = {
        'hllc':    lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_hllc(QL, QR, pL, cL, pR, cR, A_L, A_R, fluid),
        'hll':     lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_hll(QL, QR, pL, cL, pR, cR, A_L, A_R),
        'rusanov': lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_rusanov(QL, QR, pL, cL, pR, cR, A_L, A_R),
        'lf':      lambda QL, QR, pL, cL, pR, cR, dt: _jax_F_lf(QL, QR, pL, cL, pR, cR, A_L, A_R, dx, dt),
    }
    if flux_method not in _FLUX:
        raise ValueError(f"JAX step not supported for '{flux_method}' (Roe uses scalar loops). "
                         f"Choose from: {list(_FLUX)}")
    _flux_fn = _FLUX[flux_method]

    def _eos_batch(rho, e):
        p_from_hmin = (h_lo - e) * rho
        p_from_hmax = (h_hi - e) * rho
        p_lo = jnp.where((p_from_hmin > p_lo_r) & (p_from_hmin < p_hi_r),
                         jnp.maximum(p_lo_r, p_from_hmin), p_lo_r)
        p_hi = jnp.where((p_from_hmax > p_lo_r) & (p_from_hmax < p_hi_r),
                         jnp.minimum(p_hi_r, p_from_hmax), p_hi_r)
        bad  = p_lo >= p_hi
        p_lo = jnp.where(bad, p_lo_r, p_lo)
        p_hi = jnp.where(bad, p_hi_r, p_hi)
        p    = 0.5 * (p_lo + p_hi)
        def _bisect_eos(_, carry):
            p_lo, p_hi, p = carry
            h        = jnp.clip(e + p / rho, h_lo + 1.0, h_hi - 1.0)
            rho_pred = fluid.get_state(jxp.HmassP_INPUTS, h, p).density
            residual = rho_pred - rho
            p_lo     = jnp.where(residual < 0, p, p_lo)
            p_hi     = jnp.where(residual > 0, p, p_hi)
            return p_lo, p_hi, 0.5 * (p_lo + p_hi)
        p_lo, p_hi, p = jax.lax.fori_loop(0, 15, _bisect_eos, (p_lo, p_hi, p))
        h_f = jnp.clip(e + p / rho, h_lo + 1.0, h_hi - 1.0)
        _st = fluid.get_state(jxp.HmassP_INPUTS, h_f, p)
        return _st.pressure, _st.speed_sound, _st.viscosity

    def _recon(Q_n, p_c, c_c):
        if _limiter is None:
            return Q_n[:, :-1], Q_n[:, 1:], p_c[:-1], p_c[1:], c_c[:-1], c_c[1:]
        rho   = Q_n[0];  u = Q_n[1] / rho
        W     = jnp.stack([rho, u, p_c])
        dW    = W[:, 1:] - W[:, :-1]
        sigma = jnp.zeros_like(W).at[:, 1:-1].set(_limiter(dW[:, :-1], dW[:, 1:]))
        W_L_raw = W[:, :-1] + 0.5 * sigma[:, :-1]
        W_R_raw = W[:, 1:]  - 0.5 * sigma[:, 1:]
        W_L = W_L_raw.at[0].set(jnp.maximum(W_L_raw[0], 1e-6)).at[2].set(jnp.maximum(W_L_raw[2], 1.0))
        W_R = W_R_raw.at[0].set(jnp.maximum(W_R_raw[0], 1e-6)).at[2].set(jnp.maximum(W_R_raw[2], 1.0))
        rL, uL, pL = W_L[0], W_L[1], W_L[2]
        rR, uR, pR = W_R[0], W_R[1], W_R[2]
        stL  = fluid.get_state(jxp.DmassP_INPUTS, rL, pL)
        stR  = fluid.get_state(jxp.DmassP_INPUTS, rR, pR)
        Q_L  = jnp.stack([rL, rL * uL, rL * (stL.umass + 0.5 * uL**2)])
        Q_R  = jnp.stack([rR, rR * uR, rR * (stR.umass + 0.5 * uR**2)])
        return Q_L, Q_R, pL, pR, stL.speed_sound, stR.speed_sound

    def _R(Q_n, p_c, c_c, dt):
        Q_L, Q_R, p_L, p_R, c_L, c_R = _recon(Q_n, p_c, c_c)
        FA   = _flux_fn(Q_L, Q_R, p_L, c_L, p_R, c_R, dt)
        RHS  = jnp.zeros_like(Q_n).at[:, 1:-1].set(
                   -(FA[:, 1:] - FA[:, :-1]) / (A_j[1:-1] * dx))
        return RHS

    def _bc_inlet(Q_n):
        # Extrapolate velocity; find isentropic static state via 12-iter bisection.
        # Uses 1-element arrays so FluidBicubic's vmap path is exercised, not scalar.
        u_in  = 2.0 * Q_n[1, 1] / Q_n[0, 1] - Q_n[1, 2] / Q_n[0, 2]
        h_s   = jnp.expand_dims(h0_j - 0.5 * u_in**2, 0)   # shape (1,)
        plo   = jnp.array([p_lo_r], dtype=jnp.float64)
        phi   = jnp.array([p_hi_r], dtype=jnp.float64)
        def _bisect_inlet(_, carry):
            plo, phi = carry
            pm    = 0.5 * (plo + phi)
            s_mid = fluid.get_state(jxp.HmassP_INPUTS, h_s, pm).entropy
            plo   = jnp.where(s_mid > s0_j, pm, plo)
            phi   = jnp.where(s_mid < s0_j, pm, phi)
            return plo, phi
        plo, phi = jax.lax.fori_loop(0, 12, _bisect_inlet, (plo, phi))
        p_s   = 0.5 * (plo + phi)
        st    = fluid.get_state(jxp.HmassP_INPUTS, h_s, p_s)
        rho_s = st.rhomass[0]
        e_s   = st.umass[0]
        Q_n   = Q_n.at[0, 0].set(rho_s)
        Q_n   = Q_n.at[1, 0].set(rho_s * u_in)
        Q_n   = Q_n.at[2, 0].set(rho_s * (e_s + 0.5 * u_in**2))
        return Q_n

    def _bc_outlet(Q_n):
        rho_out  = 2.0 * Q_n[0, -2] * A_j[-2] / A_j[-1] - Q_n[0, -3] * A_j[-3] / A_j[-1]
        rhou_out = 2.0 * Q_n[1, -2] * A_j[-2] / A_j[-1] - Q_n[1, -3] * A_j[-3] / A_j[-1]
        # Clamp rho to physical range before EOS call (high-CFL extrapolation can go negative)
        rho_out  = jnp.clip(rho_out, 1.0, 1e5)
        u_out    = rhou_out / rho_out
        # Find h_out s.t. rho(h_out, p_out) = rho_out via bisection on HmassP_INPUTS
        # (avoids DmassP_INPUTS which uses an internal Newton solver that can fail).
        # rho decreases with h at constant p, so the bracket [h_lo, h_hi] always works.
        rho_t  = jnp.expand_dims(rho_out, 0)
        p_arr  = jnp.array([p_out], dtype=jnp.float64)
        hlo    = jnp.array([h_lo],  dtype=jnp.float64)
        hhi    = jnp.array([h_hi],  dtype=jnp.float64)
        def _bisect_outlet(_, carry):
            hlo, hhi = carry
            hm  = 0.5 * (hlo + hhi)
            rhm = fluid.get_state(jxp.HmassP_INPUTS, hm, p_arr).rhomass
            hlo = jnp.where(rhm > rho_t, hm, hlo)
            hhi = jnp.where(rhm < rho_t, hm, hhi)
            return hlo, hhi
        hlo, hhi = jax.lax.fori_loop(0, 15, _bisect_outlet, (hlo, hhi))
        h_out  = 0.5 * (hlo + hhi)
        st_out = fluid.get_state(jxp.HmassP_INPUTS, h_out, p_arr)
        e_out  = st_out.umass[0]
        Q_n    = Q_n.at[0, -1].set(rho_out)
        Q_n    = Q_n.at[1, -1].set(rhou_out)
        Q_n    = Q_n.at[2, -1].set(rho_out * (e_out + 0.5 * u_out**2))
        return Q_n

    # @jax.jit  # disabled: skip XLA compilation, run eagerly
    def step(Q_n):
        Q_n = _bc_inlet(Q_n)
        Q_n = _bc_outlet(Q_n)

        rho = Q_n[0];  u = Q_n[1] / rho
        e   = Q_n[2] / rho - 0.5 * u**2

        p, c, mu = _eos_batch(rho, e)
        p = p.at[-1].set(p_out)

        Re    = rho * jnp.abs(u) * D_h_j / jnp.maximum(mu, 1e-12)
        tau_w = _haaland_jnp(Re, roughness, D_h_j) * rho * u**2 / 8.0

        S = jnp.stack([jnp.zeros_like(p), p * _dA_dx_j - tau_w * 4.0 / D_h_j, jnp.zeros_like(p)])

        # ── Local Time Stepping (LTS) ─────────────────────────────────────────
        # Each cell advances at dt_i = CFL * dx / (|u_i| + c_i) instead of the
        # global minimum.  Subsonic cells away from the throat use a much larger
        # dt_i, removing the throat bottleneck and accelerating convergence to
        # steady state without affecting the final solution.
        lam    = jnp.abs(u) + c
        dt_vec = CFL * dx / jnp.maximum(lam, 1e-10)   # (Nj,) per-cell time step
        dt_min = jnp.min(dt_vec)                        # scalar: throat-limited dt
        dt_b   = dt_vec[jnp.newaxis, :]                 # (1, Nj) for broadcasting

        # Evaluate the steady-state residual R(Q)+S once at Q_n.
        # This is the true convergence metric: zero at steady state regardless of CFL.
        # Reusing it in the time update also saves one flux call for RK methods.
        # dt_min passed to _R only for the Lax-Friedrichs flux (ignored by others).
        RHS_n = _R(Q_n, p, c, dt_min) + S

        # Normalised steady-state residual (SSR): dimensionless, CFL-independent.
        # t_min = dt_min/CFL = dx/max(lam): physical advection time at the throat.
        t_min = dt_min / CFL
        ssr = jnp.sqrt(jnp.mean(
            (t_min * RHS_n[:, 1:-1] / (jnp.abs(Q_n[:, 1:-1]) + 1e-300))**2))

        if time_method == 'euler':
            Q_new = Q_n + dt_b * RHS_n
        elif time_method == 'RK3':
            Q1    = Q_n + dt_b * RHS_n                     # stage 1: reuse RHS_n
            R2    = _R(Q1, p, c, dt_min)
            Q2    = 0.75 * Q_n + 0.25 * (Q1 + dt_b * (R2 + S))
            R3    = _R(Q2, p, c, dt_min)
            Q_new = (Q_n + 2.0 * (Q2 + dt_b * (R3 + S))) / 3.0
        elif time_method == 'RK4':
            k1    = RHS_n                                   # reuse RHS_n
            k2    = _R(Q_n + 0.5 * dt_b * k1, p, c, dt_min) + S
            k3    = _R(Q_n + 0.5 * dt_b * k2, p, c, dt_min) + S
            k4    = _R(Q_n +       dt_b * k3,  p, c, dt_min) + S
            Q_new = Q_n + (dt_b / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        elif time_method == 'implicit':
            # Per-cell denom: each cell uses its own dt_vec[i], giving uniformly
            # strong damping across the domain — not just at the throat.
            lam_face = jnp.maximum(lam[:-1], lam[1:])
            lam_L    = jnp.concatenate([lam_face[:1],  lam_face])
            lam_R    = jnp.concatenate([lam_face, lam_face[-1:]])
            denom    = 1.0 + dt_vec * (lam_L + lam_R) / dx   # (Nj,) per-cell
            Q_new    = Q_n + dt_b * RHS_n / denom[jnp.newaxis, :]
        else:
            Q_new = Q_n + dt_b * RHS_n  # fallback to Euler

        return Q_new, p, c, dt_min, ssr

    return step


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


def make_jax_stadtke_step(fluid, A, D_h, roughness, dx, CFL, p_out, h0, s0,
                           delta_factor=0.1, newton_iter=20):
    """
    Build a @jax.jit step function implementing the Städtke Newton implicit scheme.

    The solver works in primitive (p, u, h) variables:
      - MUSCL reconstruction with Van Leer limiter in (p, u, h).
      - Städtke FVS interface flux  (_jax_F_fvs_stadtke).
      - Fully implicit residual assembled in conservative form.
      - Newton-Raphson: Jacobian via jax.jacobian, solve via jnp.linalg.solve.
      - `newton_iter` iterations are unrolled inside @jax.jit.

    step(U_n) → (U_new, dt, ssr)
      U_n   : (N, 3) — primitive state at time n
      U_new : (N, 3) — primitive state at time n+1
      dt    : scalar — time step used
      ssr   : scalar — max relative change (convergence metric)
    """
    N      = len(A)
    A_j    = jnp.asarray(A,   dtype=jnp.float64)
    D_h_j  = jnp.asarray(D_h, dtype=jnp.float64)
    A_face = 0.5 * (A_j[:-1] + A_j[1:])   # (N-1,) — face areas
    V_cell = A_j * dx                       # (N,)   — cell volumes

    # dA/dx / A  (precomputed, geometry is constant)
    dAdx_A = jnp.concatenate([
        (A_j[1:2]   - A_j[:1])    / (A_j[:1]   * dx),
        (A_j[2:]    - A_j[:-2])   / (A_j[1:-1] * 2.0 * dx),
        (A_j[-1:]   - A_j[-2:-1]) / (A_j[-1:]  * dx),
    ])  # (N,)

    h_lo  = float(getattr(fluid, 'h_min', 50e3))  + 1.0
    h_hi  = float(getattr(fluid, 'h_max', 850e3)) - 1.0
    p_lo  = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi  = float(getattr(fluid, 'p_max', 300e5)) * 0.99
    h0_j  = jnp.float64(h0)
    s0_j  = jnp.float64(s0)
    pout_j = jnp.float64(p_out)

    # ── Inner functions (all JAX-traceable) ──────────────────────────────

    def _V_from_U(U):
        """Primitive (N,3) → conservative V=(ρ, ρu, ρH−p) (N,3)."""
        p_, u_, h_ = U[:, 0], U[:, 1], U[:, 2]
        st   = fluid.get_state(jxp.HmassP_INPUTS, h_, p_)
        rho_ = st["rho"]
        H_   = h_ + 0.5 * u_**2
        return jnp.stack([rho_, rho_ * u_, rho_ * H_ - p_], axis=1)

    def _muscl_ph(U):
        """MUSCL Van Leer in (p,u,h). (N,3) → U_L, U_R each (N-1,3)."""
        dU    = U[1:] - U[:-1]                           # (N-1, 3)
        vl    = lambda a, b: jnp.where(a * b > 0, 2.0 * a * b / (a + b + 1e-300), 0.0)
        sigma = jnp.zeros_like(U).at[1:-1].set(vl(dU[:-1], dU[1:]))
        U_L   = U[:-1] + 0.5 * sigma[:-1]
        U_R   = U[1:]  - 0.5 * sigma[1:]
        # positivity guard on pressure
        U_L   = U_L.at[:, 0].set(jnp.maximum(U_L[:, 0], 1.0))
        U_R   = U_R.at[:, 0].set(jnp.maximum(U_R[:, 0], 1.0))
        return U_L, U_R

    def _bc_inlet(U):
        """Inlet BC: extrapolate u, isentropic static state from (h0, s0)."""
        u_in = 2.0 * U[1, 1] - U[2, 1]
        h_s  = jnp.expand_dims(
                   jnp.clip(h0_j - 0.5 * u_in**2,
                             jnp.float64(h_lo + 1.0),
                             jnp.float64(h_hi - 1.0)), 0)
        plo  = jnp.array([p_lo], dtype=jnp.float64)
        phi  = jnp.array([p_hi], dtype=jnp.float64)
        def _bisect_inlet(_, carry):
            plo, phi = carry
            pm    = 0.5 * (plo + phi)
            s_mid = fluid.get_state(jxp.HmassP_INPUTS, h_s, pm)["s"]
            plo   = jnp.where(s_mid > s0_j, pm, plo)
            phi   = jnp.where(s_mid < s0_j, pm, phi)
            return plo, phi
        plo, phi = jax.lax.fori_loop(0, 12, _bisect_inlet, (plo, phi))
        p_s = 0.5 * (plo + phi)

        # jax.debug.print("inlet values - p_s:{ps}, h_s:{hs}", ps=p_s, hs=h_s)
        return U.at[0].set(jnp.array([p_s[0], u_in, h_s[0]]))

    def _bc_outlet(U):
        """Outlet BC: fix p_out (subsonic) or zero-gradient (supersonic)."""
        u_out  = U[-2, 1];  h_out = U[-2, 2];  p_prev = U[-2, 0]
        a_out  = fluid.get_state(jxp.HmassP_INPUTS,
                     jnp.expand_dims(h_out,  0),
                     jnp.expand_dims(p_prev, 0))["a"][0]
        p_ghost = jnp.where(jnp.abs(u_out) < a_out, pout_j, p_prev)
        # jax.debug.print("outlet values - a_out:{ao}", ao=a_out)

        return U.at[-1].set(jnp.array([p_ghost, u_out, h_out]))

    def _residual(U_int_flat, U_n_flat, dt):
        """
        Conservative residual for interior cells 1..N-2.

        U_int_flat : (3*(N-2),) — Newton unknowns (interior cells only)
        U_n_flat   : (3*N,)     — full state at time n (fixed)
        Returns    : (3*(N-2),) — residual R (zero at steady state)
        """
        U_int = U_int_flat.reshape(N - 2, 3)
        U_n   = U_n_flat.reshape(N, 3)

        # Assemble full domain; BCs overwrite ghost cells 0 and N-1
        U   = jnp.concatenate([U_n[:1], U_int, U_n[-1:]], axis=0)
        U   = _bc_inlet(U)
        U   = _bc_outlet(U)

        V_new = _V_from_U(U)
        V_old = _V_from_U(U_n)

        U_L, U_R = _muscl_ph(U)
        F_hat    = _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor)  # (N-1, 3)

        # Source: area variation + wall friction — momentum only (theory §7.3)
        st_U  = fluid.get_state(jxp.HmassP_INPUTS, U[:, 2], U[:, 0])
        rho_U = st_U["rho"]
        mu_U  = st_U["mu"]
        Re    = rho_U * jnp.abs(U[:, 1]) * D_h_j / jnp.maximum(mu_U, 1e-12)
        tau_w = _haaland_jnp(Re, roughness, D_h_j) * rho_U * U[:, 1]**2 / 8.0
        src   = jnp.zeros((N, 3)).at[:, 1].set(
                    U[:, 0] * dAdx_A - tau_w * 4.0 / D_h_j)

        # Residual for interior cells 1..N-2  (indices in arrays: [1:-1])
        # Right face of cell i is face i; left face is face i-1.
        # For cells 1..N-2: right faces = F_hat[1:], left faces = F_hat[:-1]
        Vc    = V_cell[1:-1, None]                                         # (N-2, 1)
        dV    = V_new[1:-1] - V_old[1:-1]                                  # (N-2, 3)
        fdiv  = A_face[1:, None] * F_hat[1:] - A_face[:-1, None] * F_hat[:-1]  # (N-2, 3)
        R     = Vc * dV + dt * fdiv - dt * Vc * src[1:-1]
        return R.flatten()

    # Jacobian function — created once, compiled into the JIT kernel
    _jac_fn = jax.jacobian(_residual, argnums=0)

    # @jax.jit  # disabled: skip XLA compilation, run eagerly
    def step(U_n):
        # ── dt from CFL at current primitive state ────────────────────────
        st_c = fluid.get_state(jxp.HmassP_INPUTS, U_n[:, 2], U_n[:, 0])
        a_c  = st_c["a"]
        dt   = CFL * dx / jnp.max(jnp.abs(U_n[:, 1]) + a_c)

        U_n_flat  = U_n.reshape(-1)                    # (3*N,)
        U_k       = U_n[1:-1].reshape(-1)              # (3*(N-2),) — initial Newton iterate

        # ── Newton iterations (Python loop → unrolled by JAX trace) ──────
        for _ in range(newton_iter):
            R_k = _residual(U_k, U_n_flat, dt)
            J_k = _jac_fn(U_k, U_n_flat, dt)
            U_k = U_k + jnp.linalg.solve(J_k, -R_k)

        # ── Reconstruct full solution with updated BCs ────────────────────
        U_new = jnp.concatenate([U_n[:1], U_k.reshape(N - 2, 3), U_n[-1:]], axis=0)
        U_new = _bc_inlet(U_new)
        U_new = _bc_outlet(U_new)

        # ── Convergence metric: max relative change in (p, u, h) ─────────
        ssr = jnp.max(jnp.abs(U_new[1:-1] - U_n[1:-1])
                      / (jnp.abs(U_n[1:-1]) + 1e-300))

        return U_new, dt, ssr

    return step


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def fvm_euler_1d(x, t, parameters):
    dx          = parameters['dx']
    A_nodes     = parameters['A_nodes']
    roughness   = parameters['roughness']
    D_hydraulic = parameters['D_hydraulic']
    Nj          = len(x)
    N           = parameters['N_eq']
    CFL         = parameters['CFL']
    p_stag      = parameters['pT_stagnation'][0]
    T_stag      = parameters['pT_stagnation'][1]
    p_out       = parameters['p_out']
    perfect_gas = parameters['perfect_gas']
    t_end       = t[-1]
    warmup      = 50

    # Gas-specific setup
    flux_method  = parameters['space_method']
    method       = parameters['time_method']
    recon        = parameters.get('reconstruction', 'constant')
    warmup_cfg   = parameters.get('warmup_cfg', None)

    if perfect_gas:
        cp, cv  = hyp.fvm_solver.specific_heat2()
        R_const = cp - cv
        gamma   = cp / cv
    else:
        fluid      = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
        stag_state = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
        s0         = float(stag_state.smass)
        h0         = float(stag_state.hmass)
        parameters['s0'] = s0

    # Allocate storage
    chunk = 1000
    Q      = np.zeros((N, Nj, chunk))
    t_vec  = np.zeros(chunk)
    Q[:, :, 0] = parameters['Q_o'](x)

    steady_thr = parameters['residuals']
    max_iter   = parameters['max_iterations']
    n = 0

    print("Starting time integration "
          f"(max_iter={max_iter}, tol={steady_thr:.1e}) ...", flush=True)

    # ── Städtke Newton implicit path (real gas, (p,u,h) primitives) ─────
    use_stadtke = (not perfect_gas) and (method == 'fvs_newton')
    if use_stadtke:
        Q_init = Q[:, :, 0].copy()
        if parameters.get('nozzle_type', 'nozzle') == 'shock_tube':
            U0 = shock_tube_puh_initialization(x, A_nodes, parameters)
        else:
            U0 = stadtke_puh_initialization(x, A_nodes, parameters)  # (N, 3)

        _stadtke_step = make_jax_stadtke_step(
            fluid, A_nodes, D_hydraulic, roughness,
            dx, CFL, p_out, h0, s0)

        U_jax     = jnp.asarray(U0, dtype=jnp.float64)
        _max_iter = jnp.int32(max_iter)
        _tol      = jnp.float64(steady_thr)
        _warmup_j = jnp.int32(warmup)

        # @jax.jit  # disabled: skip XLA compilation, lax.while_loop runs eagerly
        def _stadtke_run(U0_j):
            def cond_fn(carry):
                _, n, res, res_hist = carry
                below_thr = res < _tol
                hist_max  = jnp.max(res_hist)
                hist_min  = jnp.min(res_hist) + 1e-300
                plateau   = (n >= _warmup_j) & (hist_max / hist_min < 1.5) & (res < 3.0 * _tol)
                return (n < _max_iter) & ~(below_thr | plateau)

            def body_fn(carry):
                U, n, _, res_hist = carry
                U_new, _, res     = _stadtke_step(U)
                res_hist_new      = jnp.roll(res_hist, -1).at[-1].set(res)
                should_print = jnp.where(
                    n < 10,  jnp.bool_(True),
                    jnp.where(n < 100, n % 10 == 0, n % 50 == 0))
                jax.lax.cond(
                    should_print,
                    lambda args: jax.debug.print(
                        "Iter {n:5d}  SSR = {res:.3e}", n=args[0], res=args[1]),
                    lambda args: None,
                    (n, res),
                )
                return U_new, n + 1, res, res_hist_new

            res_hist0 = jnp.ones(5, dtype=jnp.float64)
            init      = (U0_j, jnp.int32(0), jnp.float64(1.0), res_hist0)
            U_f, n_f, res_f, _ = jax.lax.while_loop(cond_fn, body_fn, init)
            return U_f, n_f, res_f

        print("Compiling and running Städtke Newton solver ...", flush=True)
        t0 = _time.perf_counter()
        try:
            U_final, n_final, res_final = _stadtke_run(U_jax)
            jax.block_until_ready(U_final)
        except KeyboardInterrupt:
            print("Interrupted — returning initial solution.")
            Q_out = np.zeros((N, Nj, 2))
            Q_out[:, :, 0] = Q_init
            Q_out[:, :, 1] = Q_init
            return Q_out, np.array([0.0, 0.0])

        elapsed  = _time.perf_counter() - t0
        n_done   = int(n_final)
        res_done = float(res_final)
        reason   = "below threshold" if res_done < float(steady_thr) else "plateau / max_iter"
        print(f"Done: {n_done} iterations ({reason}). "
              f"SSR = {res_done:.3e}.  Computational time = {elapsed:.1f} s", flush=True)

        U_np    = np.asarray(U_final)
        Q_final = _puh_to_conservative(U_np, fluid)

        Q_out = np.zeros((N, Nj, 2))
        Q_out[:, :, 0] = Q_init
        Q_out[:, :, 1] = Q_final
        return Q_out, np.array([0.0, t_end])

    # ── JAX path (real gas, non-Roe) ─────────────────────────────────────
    use_jax = (not perfect_gas) and (flux_method != 'roe') and (method != 'fvs_newton')
    if use_jax:
        _jax_step = make_jax_step(fluid, A_nodes, D_hydraulic, roughness,
                                  dx, CFL, p_out, h0, s0, recon, flux_method, method)
        Q_jax  = jnp.asarray(Q[:, :, 0])
        Q_init = Q[:, :, 0].copy()

        # ── Optional warmup phase ─────────────────────────────────────────────
        # Run N_wu iterations with cheap settings (e.g. Rusanov, constant recon)
        # to establish a decent initial field before the main solver takes over.
        # Compiled as a single fori_loop — no Python overhead between warmup iters.
        if warmup_cfg is not None:
            wu_n     = int(warmup_cfg.get('n_iterations',  200))
            wu_flux  = str(warmup_cfg.get('flux_method',   'rusanov'))
            wu_recon = str(warmup_cfg.get('reconstruction','constant'))
            wu_cfl   = float(warmup_cfg.get('CFL',          CFL))
            # Only build warmup step when settings differ from main settings
            wu_differs = (wu_flux != flux_method or wu_recon != recon or wu_cfl != CFL)
            if wu_differs:
                _wu_step = make_jax_step(fluid, A_nodes, D_hydraulic, roughness,
                                         dx, wu_cfl, p_out, h0, s0,
                                         wu_recon, wu_flux, method)
            else:
                _wu_step = _jax_step   # identical settings: reuse compiled kernel

            @jax.jit
            def _wu_run(Q0):
                def body(i, Q):
                    Q_new, _, _, _, _ = _wu_step(Q)
                    return Q_new
                return jax.lax.fori_loop(0, wu_n, body, Q0)

            print(f"Warmup: {wu_n} iters with flux={wu_flux}, recon={wu_recon}, "
                  f"CFL={wu_cfl} ...", flush=True)
            t_wu = _time.perf_counter()
            Q_jax = _wu_run(Q_jax)
            jax.block_until_ready(Q_jax)
            print(f"Warmup done in {_time.perf_counter()-t_wu:.1f} s", flush=True)

        # ── Fully compiled solver: jax.lax.while_loop stays on-device ────────
        # No Python↔XLA sync between iterations → maximum throughput.
        # Convergence checks (threshold + plateau) compiled as JAX ops inside
        # the loop — zero Python overhead per iteration.
        _max_iter = jnp.int32(max_iter)
        _tol      = jnp.float64(steady_thr)
        _warmup_j = jnp.int32(warmup)
        _t_end_j  = jnp.float64(t_end)   # physical end time — used for transient cases

        # @jax.jit  # disabled: skip XLA compilation, lax.while_loop runs eagerly
        def _jax_run(Q0):

            def cond_fn(carry):
                _, n, t_phys, res, res_hist = carry
                below_thr = res < _tol
                hist_max  = jnp.max(res_hist)
                hist_min  = jnp.min(res_hist) + 1e-300
                plateau   = (n >= _warmup_j) & (hist_max / hist_min < 1.5) & (res < 3.0 * _tol)
                time_done = t_phys >= _t_end_j   # transient stop: physical time reached
                return (n < _max_iter) & ~(below_thr | plateau | time_done)

            def body_fn(carry):
                Q, n, t_phys, _, res_hist = carry
                # ssr (steady-state residual) is returned free from step() —
                # it is computed there anyway as part of RHS_n, at no extra cost.
                Q_new, _, _, dt, res = _jax_step(Q)
                res_hist_new = jnp.roll(res_hist, -1).at[-1].set(res)

                # Adaptive print frequency: every 1 / 10 / 50 iters.
                should_print = jnp.where(
                    n < 10,  jnp.bool_(True),
                    jnp.where(n < 100, n % 10 == 0, n % 50 == 0))
                jax.lax.cond(
                    should_print,
                    lambda args: jax.debug.print(
                        "Iter {n:5d}  SSR = {res:.3e}", n=args[0], res=args[1]),
                    lambda args: None,
                    (n, res),
                )

                return Q_new, n + 1, t_phys + dt, res, res_hist_new

            res_hist0 = jnp.ones(5, dtype=jnp.float64)
            init      = (Q0, jnp.int32(0), jnp.float64(0.0), jnp.float64(1.0), res_hist0)
            Q_f, n_f, t_f, res_f, _ = jax.lax.while_loop(cond_fn, body_fn, init)
            return Q_f, n_f, t_f, res_f

        print("Running solver (eager mode, no XLA compilation) ...", flush=True)
        t0 = _time.perf_counter()
        try:
            Q_final, n_final, t_final, res_final = _jax_run(Q_jax)
            jax.block_until_ready(Q_final)
        except KeyboardInterrupt:
            print("Interrupted — returning initial solution.")
            Q_out = np.zeros((N, Nj, 2))
            Q_out[:, :, 0] = Q_init
            Q_out[:, :, 1] = Q_init
            return Q_out, np.array([0.0, 0.0])

        elapsed  = _time.perf_counter() - t0
        n_done   = int(n_final)
        t_done   = float(t_final)
        res_done = float(res_final)
        if res_done < float(steady_thr):
            reason = "below threshold"
        elif t_done >= float(t_end):
            reason = "physical time reached"
        elif n_done >= max_iter:
            reason = "max iterations"
        else:
            reason = "plateau"
        print(f"Done: {n_done} iterations ({reason}). "
              f"t_phys = {t_done:.3e} s,  NRMS = {res_done:.3e}.  "
              f"Computational time = {elapsed:.1f} s", flush=True)

        Q_out = np.zeros((N, Nj, 2))
        Q_out[:, :, 0] = Q_init
        Q_out[:, :, 1] = np.asarray(Q_final)
        return Q_out, np.array([0.0, t_done])

    # ── NumPy path (perfect gas or Roe) ──────────────────────────────────
    try:
        while t_vec[n] <= t_end and n < max_iter:
            if n + 1 >= Q.shape[2]:
                Q     = np.concatenate([Q,     np.zeros((N, Nj, 100))], axis=2)
                t_vec = np.concatenate([t_vec, np.zeros(100)])

            Q_n = Q[:, :, n]
            _t0 = _time.perf_counter() if n == 0 else None

            if perfect_gas:
                _bc_inlet_perfect(Q_n, p_stag, T_stag, gamma, R_const)
                _bc_outlet_perfect(Q_n, p_out, A_nodes, gamma)
            else:
                _bc_inlet_real(Q_n, h0, s0, fluid)
                _bc_outlet_real(Q_n, p_out, A_nodes, fluid)

            if n == 0:
                print(f"  BCs:     {_time.perf_counter()-_t0:.2f}s", flush=True)
                _t0 = _time.perf_counter()

            rho = Q_n[0];  u = Q_n[1] / rho
            if perfect_gas:
                p     = (gamma - 1.0) * (Q_n[2] - 0.5 * Q_n[1]**2 / rho)
                c     = np.sqrt(gamma * p / rho)
                tau_w = np.zeros(Nj)
            else:
                p, c, tau_w = _real_gas_state_and_friction(Q_n, D_hydraulic, roughness, fluid)
                p[-1] = p_out

            if n == 0:
                print(f"  EOS loop:{_time.perf_counter()-_t0:.2f}s", flush=True)
                _t0 = _time.perf_counter()

            parameters['p'] = p;  parameters['c'] = c
            S  = _source_term(p, tau_w, A_nodes, D_hydraulic, dx)
            dt = _compute_dt(u, c, dx, CFL)
            parameters['dt'] = dt

            if method == 'euler':
                RHS = R(Q_n, parameters)
                if n == 0:
                    print(f"  Flux:    {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q[:, :, n + 1] = Q_n + dt * (RHS + S)
            elif method == 'RK3':
                R1 = R(Q_n, parameters)
                if n == 0:
                    print(f"  Flux R1: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q1 = Q_n + dt * (R1 + S)
                R2 = R(Q1, parameters)
                if n == 0:
                    print(f"  Flux R2: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q2 = 0.75 * Q_n + 0.25 * (Q1 + dt * (R2 + S))
                R3 = R(Q2, parameters)
                if n == 0:
                    print(f"  Flux R3: {_time.perf_counter()-_t0:.2f}s", flush=True)
                Q[:, :, n + 1] = (Q_n + 2.0 * (Q2 + dt * (R3 + S))) / 3.0
            elif method == 'RK4':
                k1 = R(Q_n, parameters) + S
                k2 = R(Q_n + 0.5 * dt * k1, parameters) + S
                k3 = R(Q_n + 0.5 * dt * k2, parameters) + S
                k4 = R(Q_n +       dt * k3,  parameters) + S
                Q[:, :, n + 1] = Q_n + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            elif method == 'implicit':
                lam      = np.abs(u) + c
                lam_face = np.maximum(lam[:-1], lam[1:])
                lam_L    = np.concatenate([lam_face[:1],  lam_face])
                lam_R    = np.concatenate([lam_face, lam_face[-1:]])
                denom    = 1.0 + dt * (lam_L + lam_R) / dx
                RHS = R(Q_n, parameters)
                Q[:, :, n + 1] = Q_n + dt * (RHS + S) / denom[np.newaxis, :]

            if n > 0:
                dQ       = (np.abs(Q[:, 1:-1, n+1] - Q[:, 1:-1, n])
                            / (np.abs(Q[:, 1:-1, n]) + 1e-300))
                L2       = np.sqrt(np.mean(dQ**2, axis=1))
                residual = float(np.max(L2))
                freq = 1 if n < 10 else (10 if n < 100 else 50)
                if n % freq == 0:
                    print(f"Iter {n:5d}  residual = {residual:.4e}"
                          f"  dt = {dt:.3e}  p_min = {p.min():.4e}", flush=True)
                if n > warmup and residual < steady_thr:
                    print(f"Converged after {n} iterations, t = {t_vec[n]:.4e} s", flush=True)
                    n += 1
                    break

            t_vec[n + 1] = t_vec[n] + dt
            n += 1

    except KeyboardInterrupt:
        print("Interrupted — returning current solution.")

    return Q[:, :, :n], t_vec[:n]


# ---------------------------------------------------------------------------
# Friction and two-phase correlations
# ---------------------------------------------------------------------------

def Haaland(Re, epsilon, D):
    if Re <= 0.0:
        return 0.0
    term = (epsilon / (3.7 * D))**1.11 + 6.9 / Re
    if term <= 0.0:
        return 0.0
    return (-1.81 * math.log10(term))**-2


def viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, x):
    y   = rho_L * x / (rho_L * x + rho_v * (1.0 - x))
    return mu_L * (1.0 - y) * (1.0 + 2.5 * y) + mu_v * y


def viscosity_MacAdams(mu_L, mu_v, x):
    return (x / mu_v + (1.0 - x) / mu_L) ** -1


def LM_gronnerud(rho_l, rho_v, mu_l, mu_v, x, G_l, d, g=9.81):
    Fr_l = G_l**2 / (g * d * rho_l**2)
    f_fr = 1.0 if Fr_l >= 1.0 else Fr_l**0.3 + 0.0055 * (math.log(1.0 / Fr_l))**2
    dp_dz_fr = f_fr * (x + 4.0 * (x**1.8 - x**10 * f_fr**0.5))
    return 1.0 + dp_dz_fr * ((rho_l / rho_v) / (mu_l / mu_v)**0.25 - 1.0)


def LM_condensation(Re, rho_L, mu_L, rho_V, mu_V, D_h, u, x, G):
    Re_L = rho_L * u * D_h / mu_L
    Re_V = rho_V * u * D_h / mu_V
    if Re < 2300:
        f_L = 64.0 / Re_L
        f_V = 64.0 / Re_V
    else:
        f_L = 0.316 / Re_L**0.25
        f_V = 0.184 / Re_V**0.2
    X     = (f_L * rho_V * (1.0 - x)**2) / (rho_L * f_V * x**2)
    phi_L = 1.0 + 20.0 / X + 1.0 / X**2
    phi_V = 1.0 + 20.0 * X + X**2
    tau_L = f_L * G**2 * (1.0 - x)**2 / (8.0 * rho_L)
    tau_V = f_V * G**2 * x**2 / (8.0 * rho_V)
    return phi_L, phi_V, tau_L, tau_V


def f_HEM_purdue(Re):
    if Re < 2000:
        return 16.0 / Re
    elif Re < 20000:
        return 0.079 * Re**-0.25
    return 0.046 * Re**-0.2


def richardson(x, rho_L, rho_v):
    gamma = rho_L * x / (rho_L * x + rho_v * (1.0 - x))
    return (1.0 - gamma)**-1.75


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
