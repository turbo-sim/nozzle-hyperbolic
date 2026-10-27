"""Unchanged definitions relocated from fvm_solver.fvm."""
import math
import numpy as np
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from scipy.optimize import brentq as _brentq


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
        p_thr = p_iso[k_thr]   # critical (throat) static pressure on the isentrope

        # Subsonic branch: h in [h_thr, h0) → G decreasing; reverse for searchsorted
        # Supersonic branch: h in [h_lo, h_thr] → G increasing
        sub_G = G_tab[k_thr:][::-1];  sub_h = h_tab[k_thr:][::-1];  sub_p = p_iso[k_thr:][::-1]
        sup_G = G_tab[:k_thr + 1];    sup_h = h_tab[:k_thr + 1];    sup_p = p_iso[:k_thr + 1]

        # Determine if the nozzle is choked (p_out ≤ p_critical) or subsonic.
        # For a pipe (constant area), argmin returns index 0 so every cell
        # downstream would incorrectly use the supersonic branch if we used
        # m_dot from the throat.  For unchoked flow we instead compute m_dot
        # from the actual outlet condition on the subsonic isentrope.
        is_choked = (p_out <= p_thr)

        if is_choked:
            m_dot = G_thr * A_throat
        else:
            # Find outlet state on subsonic isentrope (p_iso on sub-branch is increasing)
            sub_p_asc = sub_p[::-1]   # ascending p order
            sub_G_asc = sub_G[::-1]
            sub_h_asc = sub_h[::-1]
            idx_out = np.clip(np.searchsorted(sub_p_asc, p_out), 1, len(sub_p_asc) - 1)
            t_out   = ((p_out - sub_p_asc[idx_out-1])
                       / max(sub_p_asc[idx_out] - sub_p_asc[idx_out-1], 1e-300))
            h_out   = sub_h_asc[idx_out-1] + t_out * (sub_h_asc[idx_out] - sub_h_asc[idx_out-1])
            G_out   = sub_G_asc[idx_out-1] + t_out * (sub_G_asc[idx_out] - sub_G_asc[idx_out-1])
            m_dot   = G_out * A_nodes[-1]

        h_cells = np.empty(Nj)
        p_cells = np.empty(Nj)
        for i in range(Nj):
            G_target = m_dot / A_nodes[i]
            # Unchoked flow: all cells on subsonic branch regardless of throat index
            if (not is_choked) or (i <= i_throat):
                seg_G, seg_h, seg_p = sub_G, sub_h, sub_p
            else:
                seg_G, seg_h, seg_p = sup_G, sup_h, sup_p
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


def _shock_tube_velocities(cfg, rho_L, rho_R):
    """
    Return (u_L, u_R) for a shock tube IC that is mass-flux continuous across
    the diaphragm.  The physical IC has u=0 everywhere, but FVS has a
    singularity at u=0 (K-matrix singular).  We use a small common mass flux
    G_0 = rho_L * u_L = rho_R * u_R so the interface has no spurious mass source.

    YAML keys (shock_tube block):
      u_L  : velocity on the left side [m/s]  (default 0.1)
      u_R  : if provided, overrides the mass-flux calculation (not recommended)
    """
    u_L = float(cfg.get('u_L', 0.1))
    if 'u_R' in cfg:
        u_R = float(cfg['u_R'])
    else:
        # enforce rho_L * u_L = rho_R * u_R
        u_R = u_L * rho_L / max(rho_R, 1e-300)
    return u_L, u_R


def shock_tube_Q_initialization(x, A_nodes, parameters):
    """
    Step-function initial condition for shock tube tests.
    Returns Q (3, N) in conservative form.

    Reads 'shock_tube_cfg' from parameters:
        x_diaphragm, p_L, T_L, u_L, p_R, T_R
    u_R is computed from mass-flux continuity: rho_L*u_L = rho_R*u_R.
    """
    fluid = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    cfg   = parameters['shock_tube_cfg']
    x_d   = float(cfg.get('x_diaphragm', 0.5 * (x[0] + x[-1])))
    p_L, T_L = float(cfg['p_L']), float(cfg['T_L'])
    p_R, T_R = float(cfg['p_R']), float(cfg['T_R'])

    st_L  = fluid.get_state(jxp.PT_INPUTS, p_L, T_L)
    st_R  = fluid.get_state(jxp.PT_INPUTS, p_R, T_R)
    rho_L, e_L = float(st_L.rhomass), float(st_L.umass)
    rho_R, e_R = float(st_R.rhomass), float(st_R.umass)
    u_L, u_R   = _shock_tube_velocities(cfg, rho_L, rho_R)

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
    u_R is computed from mass-flux continuity: rho_L*u_L = rho_R*u_R.
    """
    fluid = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    cfg   = parameters['shock_tube_cfg']
    x_d   = float(cfg.get('x_diaphragm', 0.5 * (x[0] + x[-1])))
    p_L, T_L = float(cfg['p_L']), float(cfg['T_L'])
    p_R, T_R = float(cfg['p_R']), float(cfg['T_R'])

    st_L  = fluid.get_state(jxp.PT_INPUTS, p_L, T_L)
    st_R  = fluid.get_state(jxp.PT_INPUTS, p_R, T_R)
    rho_L = float(st_L.rhomass);  h_L = float(st_L.hmass)
    rho_R = float(st_R.rhomass);  h_R = float(st_R.hmass)
    u_L, u_R = _shock_tube_velocities(cfg, rho_L, rho_R)

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


def stadtke_puh_initialization_subsonic(x, A_nodes, parameters):
    """
    Fully-subsonic isentropic initialization for the Städtke Newton solver.

    Identical to `stadtke_puh_initialization` except that the subsonic branch
    is used for ALL cells — including those downstream of the throat.  This
    gives a physically consistent (isentropic, mass-conserving) starting point
    where M < 1 everywhere, which is the correct initial guess when the target
    solution contains a normal shock in the diverging section.

    The choked mass-flow rate (G_max * A_throat) is used, so the throat will be
    at M ≈ 1 initially and the diverging section acts as a subsonic diffuser.
    The solver then develops the shock from this state rather than jumping to
    the supersonic branch directly.
    """
    fluid       = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    p_stag      = parameters['pT_stagnation'][0]
    T_stag      = parameters['pT_stagnation'][1]
    A_throat    = float(np.min(A_nodes))
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

    valid  = p_iso > p_lo * 1.05
    h_tab  = h_tab[valid];  p_iso = p_iso[valid]

    rho_iso = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_iso).rhomass, dtype=float)
    u_tab  = np.sqrt(np.maximum(2.0 * (h0 - h_tab), 0.0))
    G_tab  = rho_iso * u_tab

    k_thr = int(np.argmax(G_tab))
    G_thr = G_tab[k_thr]
    m_dot = G_thr * A_throat

    # Subsonic branch only: h from h_thr up to h0, G decreasing from G_max to 0
    sub_G = G_tab[k_thr:][::-1]
    sub_h = h_tab[k_thr:][::-1]
    sub_p = p_iso[k_thr:][::-1]

    h_cells = np.empty(Nj)
    p_cells = np.empty(Nj)
    for i in range(Nj):
        G_target = m_dot / A_nodes[i]
        # Clamp to the subsonic range (inlet/outlet may be slightly above G_max
        # due to area ratio; cap at throat state)
        G_target = min(G_target, sub_G[-1])
        idx      = np.clip(np.searchsorted(sub_G, G_target), 1, len(sub_G) - 1)
        t        = (G_target - sub_G[idx-1]) / max(sub_G[idx] - sub_G[idx-1], 1e-300)
        h_cells[i] = sub_h[idx-1] + t * (sub_h[idx] - sub_h[idx-1])
        p_cells[i] = sub_p[idx-1] + t * (sub_p[idx] - sub_p[idx-1])

    u_cells = np.sqrt(np.maximum(2.0 * (h0 - h_cells), 0.0))
    return np.stack([p_cells, u_cells, h_cells], axis=1)  # (N, 3)


def stadtke_puh_initialization_shocked(x, A_nodes, parameters):
    """
    Shocked-nozzle initialisation for the Städtke Newton solver.

    Places a normal shock at x_shock_fraction * L (default 0.75) and seeds:
      - inlet → throat:        subsonic isentropic  (M < 1)
      - throat → i_shock-1:    supersonic isentropic (M > 1)
      - i_shock:               Rankine-Hugoniot jump (real-gas, iterative)
      - i_shock+1 → outlet:    subsonic isentropic on post-shock isentrope

    Newton only needs to *refine* the shock position rather than discover it,
    which avoids the transient-capture convergence failure.

    YAML keys consumed (all optional):
      initialization: shocked
      x_shock_fraction: 0.75   # fraction of nozzle length for shock placement
    """
    fluid        = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    p_stag       = parameters['pT_stagnation'][0]
    T_stag       = parameters['pT_stagnation'][1]
    A_throat     = float(np.min(A_nodes))
    i_throat     = int(np.argmin(A_nodes))
    Nj           = len(x)
    x_frac       = float(parameters.get('x_shock_fraction', 0.75))
    x_shock      = x[0] + x_frac * (x[-1] - x[0])
    i_shock      = int(np.argmin(np.abs(x - x_shock)))

    st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
    h0      = float(st_stag.hmass)
    s0      = float(st_stag.smass)

    h_lo = float(getattr(fluid, 'h_min', 50e3))  + 10.0
    h_hi = min(float(getattr(fluid, 'h_max', 850e3)) - 10.0, h0 - 10.0)
    p_lo = float(getattr(fluid, 'p_min', 1e4))   * 1.01
    p_hi = float(getattr(fluid, 'p_max', 300e5)) * 0.99

    # ── isentropic table on the pre-shock isentrope (s = s0) ────────────────
    N_tab  = 400
    h_tab  = np.linspace(h_lo, h_hi, N_tab)
    p_lo_v = np.full(N_tab, p_lo)
    p_hi_v = np.full(N_tab, min(p_stag * 1.05, p_hi))
    for _ in range(60):
        p_mid  = 0.5 * (p_lo_v + p_hi_v)
        s_pred = np.asarray(
            fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_mid).entropy, dtype=float)
        p_lo_v = np.where(s_pred > s0, p_mid, p_lo_v)
        p_hi_v = np.where(s_pred < s0, p_mid, p_hi_v)
    p_iso = 0.5 * (p_lo_v + p_hi_v)

    valid = p_iso > p_lo * 1.05
    h_tab = h_tab[valid];  p_iso = p_iso[valid]

    rho_iso = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_iso).rhomass, dtype=float)
    a_iso   = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_tab, p_iso).speed_of_sound, dtype=float)
    u_tab   = np.sqrt(np.maximum(2.0 * (h0 - h_tab), 0.0))
    G_tab   = rho_iso * u_tab

    k_thr = int(np.argmax(G_tab))
    G_thr = G_tab[k_thr]
    m_dot = G_thr * A_throat

    sub_G = G_tab[k_thr:][::-1];  sub_h = h_tab[k_thr:][::-1];  sub_p = p_iso[k_thr:][::-1]
    sup_G = G_tab[:k_thr + 1];    sup_h = h_tab[:k_thr + 1];    sup_p = p_iso[:k_thr + 1]

    def _interp(seg_G, seg_h, seg_p, G_target):
        G_target = np.clip(G_target, seg_G[0], seg_G[-1])
        idx = np.clip(np.searchsorted(seg_G, G_target), 1, len(seg_G) - 1)
        t   = (G_target - seg_G[idx-1]) / max(seg_G[idx] - seg_G[idx-1], 1e-300)
        return (seg_h[idx-1] + t*(seg_h[idx] - seg_h[idx-1]),
                seg_p[idx-1] + t*(seg_p[idx] - seg_p[idx-1]))

    # ── fill pre-shock cells ─────────────────────────────────────────────────
    h_cells = np.empty(Nj)
    p_cells = np.empty(Nj)

    for i in range(i_shock):
        G_t = m_dot / A_nodes[i]
        branch = (sub_G, sub_h, sub_p) if i <= i_throat else (sup_G, sup_h, sup_p)
        h_cells[i], p_cells[i] = _interp(*branch, G_t)

    # ── Rankine-Hugoniot jump at i_shock (real-gas, iterative) ──────────────
    # Pre-shock state (supersonic)
    h1 = h_cells[i_shock - 1]
    p1 = p_cells[i_shock - 1]
    st1  = fluid.get_state(jxp.HmassP_INPUTS, np.array([h1]), np.array([p1]))
    rho1 = float(st1.rhomass[0])
    u1   = float(np.sqrt(max(2.0 * (h0 - h1), 0.0)))
    # a1   = float(st1.speed_of_sound)  # not used directly

    # RH mass flux: rho1*u1 = rho2*u2
    # RH momentum: p1 + rho1*u1^2 = p2 + rho2*u2^2  →  p2 = p1 + rho1*u1*(u1-u2)
    # RH energy:   h1 + 0.5*u1^2 = h2 + 0.5*u2^2   →  h2 = h0 - 0.5*u2^2
    # Given u2, h2 = h0 - 0.5*u2^2 and p2 from EOS s.t. RH momentum holds.
    # Iterate on u2 in (0, u1) using bisection on momentum residual.

    G1 = rho1 * u1   # conserved mass flux across shock

    def _rh_residual(u2):
        h2   = h0 - 0.5 * u2**2
        h2c  = np.clip(h2, h_lo + 1.0, h_hi - 1.0)
        # p2 from momentum: p2 = p1 + G1*(u1 - u2)
        p2_mom = p1 + G1 * (u1 - u2)
        # p2 from energy + EOS: find p s.t. s(h2, p) gives rho2 = G1/u2
        rho2_need = G1 / max(u2, 1e-6)
        # Use EOS at (h2, p2_mom) to check density
        st2  = fluid.get_state(jxp.HmassP_INPUTS,
                                np.array([h2c]), np.array([p2_mom]))
        rho2_eos = float(st2.rhomass[0])
        return rho2_eos - rho2_need

    # Bracket: u2 must satisfy 0 < u2 < u1 (subsonic post-shock)
    u2_lo, u2_hi = 1.0, u1 * 0.999
    r_lo = _rh_residual(u2_lo)
    r_hi = _rh_residual(u2_hi)

    # Bisect on the RH density residual
    if r_lo * r_hi < 0:
        for _ in range(60):
            u2_mid = 0.5 * (u2_lo + u2_hi)
            r_mid  = _rh_residual(u2_mid)
            if r_mid * r_lo > 0:
                u2_lo = u2_mid;  r_lo = r_mid
            else:
                u2_hi = u2_mid
        u2_shock = 0.5 * (u2_lo + u2_hi)
    else:
        # Fallback: use perfect-gas RH (γ≈1.4) as approximation
        # M1^2 from u1 and a1
        st1_a = fluid.get_state(jxp.HmassP_INPUTS, np.array([h1]), np.array([p1]))
        a1    = float(st1_a.speed_of_sound[0])
        M1    = u1 / max(a1, 1e-6)
        gamma = 1.4
        M2_sq = (1.0 + (gamma-1)/2 * M1**2) / (gamma * M1**2 - (gamma-1)/2)
        p2_rh = p1 * (2*gamma*M1**2 - (gamma-1)) / (gamma+1)
        a2_approx = a1 * np.sqrt(M2_sq / M1**2)  # rough
        u2_shock  = np.sqrt(max(M2_sq, 0.0)) * a2_approx
        print(f"  [shocked init] RH bisection failed at i={i_shock}, using perfect-gas fallback "
              f"M1={M1:.3f} → M2={np.sqrt(M2_sq):.3f}")

    h2_shock = h0 - 0.5 * u2_shock**2
    p2_shock = p1 + G1 * (u1 - u2_shock)
    h_cells[i_shock] = h2_shock
    p_cells[i_shock] = p2_shock
    print(f"  [shocked init] shock at i={i_shock} (x={x[i_shock]:.4f} m)  "
          f"p: {p1/1e3:.1f}→{p2_shock/1e3:.1f} kPa  "
          f"u: {u1:.1f}→{u2_shock:.1f} m/s")

    # ── post-shock isentrope (s = s2 > s0, entropy increased by shock) ───────
    h2c = np.clip(h2_shock, h_lo + 1.0, h_hi - 1.0)
    p2c = np.clip(p2_shock, p_lo * 1.05, p_hi * 0.95)
    st2  = fluid.get_state(jxp.HmassP_INPUTS, np.array([h2c]), np.array([p2c]))
    s2   = float(st2.entropy[0])
    rho2 = float(st2.rhomass[0])

    # Stagnation enthalpy is conserved (h0 unchanged across shock in frame where
    # total enthalpy is preserved): h02 = h2 + 0.5*u2^2 = h0
    # Build post-shock isentrope table at s = s2
    p_lo2_v = np.full(N_tab, p_lo)
    p_hi2_v = np.full(N_tab, min(p2_shock * 2.0, p_hi))
    h_tab2  = np.linspace(h2c, h_hi - 10.0, N_tab)  # post-shock: h ≥ h2 (subsonic)
    for _ in range(60):
        p_mid2  = 0.5 * (p_lo2_v + p_hi2_v)
        s_pred2 = np.asarray(
            fluid.get_state(jxp.HmassP_INPUTS, h_tab2, p_mid2).entropy, dtype=float)
        p_lo2_v = np.where(s_pred2 > s2, p_mid2, p_lo2_v)
        p_hi2_v = np.where(s_pred2 < s2, p_mid2, p_hi2_v)
    p_iso2 = 0.5 * (p_lo2_v + p_hi2_v)

    valid2  = p_iso2 > p_lo * 1.05
    h_tab2  = h_tab2[valid2];  p_iso2 = p_iso2[valid2]

    rho_iso2 = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_tab2, p_iso2).rhomass, dtype=float)
    u_tab2   = np.sqrt(np.maximum(2.0 * (h0 - h_tab2), 0.0))
    G_tab2   = rho_iso2 * u_tab2

    # Post-shock is subsonic: G increases with h (low-h end is subsonic far from throat)
    # Sort by increasing G for searchsorted
    sort2   = np.argsort(G_tab2)
    G_tab2s = G_tab2[sort2];  h_tab2s = h_tab2[sort2];  p_tab2s = p_iso2[sort2]

    # Mass flux post-shock is the same m_dot (mass conservation)
    for i in range(i_shock + 1, Nj):
        G_t = m_dot / A_nodes[i]
        h_cells[i], p_cells[i] = _interp(G_tab2s, h_tab2s, p_tab2s, G_t)

    u_cells = np.sqrt(np.maximum(2.0 * (h0 - h_cells), 0.0))
    return np.stack([p_cells, u_cells, h_cells], axis=1)  # (N, 3)


def stadtke_puh_initialization_simple(x, A_nodes, parameters):
    """
    Simple initialization for the Städtke Newton solver in primitive (p, u, h).

    Strategy:
      p  — linear from p_stag to p_out
      h  — h0 - u²/2  (total enthalpy conservation)
      u  — from continuity with m_dot estimated at the outlet via Bernoulli

    No isentrope table, no branch selection.  Works for any geometry and is
    the preferred starting point when the isentropic init gives the wrong branch.
    """
    fluid  = parameters.get('fluid') or jxp.Fluid(name=parameters['fluid_name'])
    p_stag = float(parameters['pT_stagnation'][0])
    T_stag = float(parameters['pT_stagnation'][1])
    p_out  = float(parameters['p_out'])
    Nj     = len(x)

    st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
    h0      = float(st_stag.hmass)

    # Linear pressure; density at rest-state h=h0
    p_cells   = np.linspace(p_stag, p_out, Nj)
    rho_cells = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, np.full(Nj, h0), p_cells).rhomass,
        dtype=float)

    # Global velocity ceiling: h = h0 - u²/2 must stay above the EOS lower bound
    h_lo    = float(getattr(fluid, 'h_min', 50e3)) + 10.0
    u_max   = np.sqrt(max(2.0 * (h0 - h_lo), 0.0)) * 0.9

    # Outlet velocity from Bernoulli Δh ≈ Δp/ρ, capped so h stays valid
    rho_out = rho_cells[-1]
    u_out   = min(np.sqrt(max(2.0 * (p_stag - p_out) / (rho_out + 1e-300), 0.0)), u_max)
    m_dot   = rho_out * u_out * A_nodes[-1]

    # Subsonic cap: ensure M < 0.9 everywhere.  The first density estimate
    # rho_cells uses h=h0 (stagnation), which underestimates actual density
    # (actual h < h0 → actual T < T0 → actual rho > rho_est for ideal gas).
    # Use a conservative 0.5 multiplier to stay well below sonic.
    a_est       = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, np.full(Nj, h0), p_cells)["a"],
        dtype=float)
    m_dot_sonic = float(np.min(0.5 * rho_cells * a_est * A_nodes))
    m_dot       = min(m_dot, m_dot_sonic)

    # First-pass velocity and enthalpy (density still uses h0 estimate)
    u_cells = np.minimum(m_dot / np.maximum(rho_cells * A_nodes, 1e-300), u_max)
    h_cells = np.maximum(h0 - 0.5 * u_cells**2, h_lo)

    # One density-correction iteration: recompute rho with the actual h,
    # which gives higher density in the divergent (lower T) → lower u →
    # physically correct decelerating profile in the divergent section.
    rho_corr = np.asarray(
        fluid.get_state(jxp.HmassP_INPUTS, h_cells, p_cells).rhomass, dtype=float)
    u_cells = np.minimum(m_dot / np.maximum(rho_corr * A_nodes, 1e-300), u_max)
    h_cells = np.maximum(h0 - 0.5 * u_cells**2, h_lo)

    return np.stack([p_cells, u_cells, h_cells], axis=1)  # (N, 3)
