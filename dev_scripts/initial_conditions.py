"""
Initial conditions for the two-fluid driver (main_two_fluid.py).

`build_initial_condition` (bottom of this file) turns the YAML `ic:` block into
U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l], shape (Nx, 6):

    ic.type                          builder                   built from
    -------------------------------  ------------------------  ----------------------
    uniform                          uniform_ic                inlet-static estimate
    stagnation                       stagnation_ic             reservoir state, at rest
    simple                           simple_ic                 p ramp p0 → p_out
    isentropic                       isentropic_ic             area–Mach (choked)
    isentropic_shock                 isentropic_shock_ic       area–Mach + normal shock
    isentropic_subsonic | subsonic   isentropic_subsonic_ic    area–Mach (unchoked)
    bernoulli                        bernoulli_ic              incompressible Bernoulli
    restart                          restart_ic                previous run's solution.csv
    nozzlex_csv                      nozzlex_csv_ic            nozzlex space-marching CSV
    nozzlex                          ic_from_nozzlex_pickle    saved nozzlex pickle

Restart options (ic.type: restart):
    ic.smooth_passes: N    N passes of a 1-2-1 filter on the source profile before
                           interpolation (kills 2Δx content, barely touches the
                           smooth profile — the "basin-escape" restart)
    ic.slave_ghost: true   replace the ghost-gas fields by their drag/heat-slaved
                           values: u_g = u_l, T_g = T_l (h_g = cp·T_l with cp fitted
                           from the run), α_g = its median (perfect-gas ghost)

nozzlex_initial_condition (used by debug scripts) runs the nozzlex space-marching
solve live; the nozzlex package is imported only then.

History: merged 2026-10-06 from isentropic_ic.py, bernoulli_ic.py, nozzlex_ic.py,
make_smoothed_restart.py and the inline IC blocks of main_two_fluid.py; the physics
code is unchanged (the driver builds bit-identical U_h_0 for every ic.type).
"""

import pathlib

import numpy as np
import jax.numpy as jnp
import jaxprop as jxp


# ═════════════════════════════════════════════════════════════════════════════
# Simple ICs (formerly inline in main_two_fluid.py)
# ═════════════════════════════════════════════════════════════════════════════

def uniform_ic(Nx, inlet_static, alpha_g):
    """Every cell at the driver's isentropic inlet-static estimate.

    inlet_static = (p_in, u_g_in, u_l_in, h_g_in, h_l_in).
    """
    p_in, u_g_in, u_l_in, h_g_in, h_l_in = inlet_static
    return np.tile(
        [p_in, u_g_in, u_l_in, alpha_g, h_g_in, h_l_in], (Nx, 1),
    ).astype(float)


def stagnation_ic(Nx, p0, alpha_g, H_g0, H_l0):
    """Fluid at rest at the reservoir state (u = 1e-6 m/s, h = H0).

    For a nozzle start-up, pair it with a ramped back-pressure
    (bc.p_out_ramp_steps): with p_out = p0 this state is already steady.
    """
    return np.tile(
        [p0, 1e-6, 1e-6, alpha_g, H_g0, H_l0], (Nx, 1),
    ).astype(float)


def simple_ic(Nx, p0, p_out, alpha_g, slip, H_g0, H_l0, u_seed_l=5.0):
    """Robust "cold-start" IC (no nozzlex needed).

    A GENTLE, SUBSONIC, slip-consistent guess that lets the time-marching build
    the (transonic) flow itself.  Converges in more iterations than the nozzlex
    seed, but is robust and self-contained.

    Why NOT per-phase continuity (the previous attempt): computing u_g from GAS
    continuity (ρ_g drops ~10× along the nozzle) and u_l from LIQUID continuity
    (ρ_l≈const) gives two UNRELATED velocity fields — u_g→585, u_l→264 at the
    throat, du swinging −2→+320 m/s — which (a) destroys the imposed slip and
    (b) is wildly mixture-SUPERSONIC (Wood a_mix ~tens of m/s at α≈0.39), with a
    velocity CAP kink.  The first implicit step then blows up (NaN).  See probe.

    Robust recipe instead:
      * pressure: mild linear ramp p0 → p_out (matches the stagnation inlet and
        the back-pressure outlet at the two ends → minimal startup shock),
      * composition: uniform α_g,
      * velocities: SMALL and SLIP-CONSISTENT (u_g = S·u_l, so du≈0) — do not
        pre-build the throat spike; the transient accelerates the flow,
      * enthalpies: stagnation H_g0/H_l0 minus the (tiny) dynamic head → always
        valid EOS states.
    """
    p_cells   = np.linspace(p0, p_out, Nx)            # inlet→outlet pressure ramp
    ag        = np.full(Nx, alpha_g)
    u_cells_l = np.full(Nx, u_seed_l)                 # m/s — gentle, safely subsonic
    u_cells_g = np.full(Nx, u_seed_l * slip)          # impose slip → du starts ~0
    h_cells_g = np.full(Nx, H_g0 - 0.5 * u_cells_g**2)
    h_cells_l = np.full(Nx, H_l0 - 0.5 * u_cells_l**2)
    return np.stack([p_cells, u_cells_g, u_cells_l, ag, h_cells_g, h_cells_l], axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# Isentropic (area-Mach) ICs (formerly isentropic_ic.py)
# ═════════════════════════════════════════════════════════════════════════════
# Analytic isentropic (area-Mach) initial condition for the two-fluid solver.
#
# Purpose
# -------
# GAS-DOMINANT mirror of bernoulli_ic: a self-contained cold-start IC for the
# compressible perfect-expansion regime (alpha_g ~ 1 - 1e-4, choked C-D nozzle,
# shock-free supersonic diverging section).  For a perfect-gas dominant phase the
# exact steady quasi-1D solution is the area-Mach relation
#
#     A/A* = (1/M) [ (2/(g+1)) (1 + (g-1)/2 M^2) ]^((g+1)/(2(g-1)))
#
# with the SUBSONIC branch upstream of the throat and the SUPERSONIC branch
# downstream (perfect/adapted expansion), and
#
#     T = T0 / (1 + (g-1)/2 M^2),   p = p0 (T/T0)^(g/(g-1)),   u = M sqrt(g R T).
#
# This profile doubles as the ANALYTIC REFERENCE the converged two-fluid solution
# is validated against (check_isentropic.py), including the choked mass flux
#
#     mdot* = A* p0 sqrt(g/(R T0)) (2/(g+1))^((g+1)/(2(g-1))) .
#
# Ghost (passive) phase
# ---------------------
# Near-massless second phase (alpha_l ~ 1e-4): constant void fraction, co-flowing
# (slip S = u_g/u_l), enthalpy from (p, T_gas) per its own EOS — with SRC on the
# drag+heat slave it to the gas along the whole expansion.  Use an UNBOUNDED
# (perfect-gas) backend for the ghost: the gas T drops to ~0.5 T0 at the exit,
# far off any liquid table.
#
# gamma and R are DERIVED from fluid_g at (p0, T0) (R = p0/(rho0 T0),
# gamma = a0^2/(R T0)) so the reference is self-consistent with the solver's EOS.
#
# Returns  U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]  shape (Nx, 6).

def _area_mach(ar, gamma, branch):
    """Solve A/A* = f(M) by bisection.  branch: 'sub' | 'sup'."""
    g = gamma

    def f(M):
        return (1.0 / M) * ((2.0 / (g + 1.0))
                            * (1.0 + 0.5 * (g - 1.0) * M * M)
                            ) ** ((g + 1.0) / (2.0 * (g - 1.0))) - ar

    if ar <= 1.0 + 1e-12:
        return 1.0
    lo, hi = (1e-8, 1.0) if branch == "sub" else (1.0, 100.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _am_ratio(M, gamma):
    """Forward area-Mach ratio A/A* as a function of Mach M."""
    g = gamma
    return (1.0 / M) * ((2.0 / (g + 1.0))
                        * (1.0 + 0.5 * (g - 1.0) * M * M)
                        ) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def isentropic_ic(
    x_grid, A, *,
    p0, T0_g, T0_l, alpha_g, slip,
    fluid_g, fluid_l,
):
    """
    Build a choked perfect-expansion two-fluid IC (and its analytic reference).

    Parameters
    ----------
    x_grid : (Nx,) cell-centre positions [m]
    A      : (Nx,) cross-sectional areas [m^2] (throat = argmin A)
    p0     : reservoir (stagnation) pressure [Pa]
    T0_g, T0_l : gas / liquid stagnation temperatures [K] (T0_l only used for
             the ghost enthalpy reference; the ghost is built at the GAS T(x))
    alpha_g: (constant) gas void fraction (~1 - 1e-4 for gas-dominant)
    slip   : S = u_g / u_l
    fluid_g, fluid_l : EOS objects (PT_INPUTS interface)

    Returns
    -------
    U_h_0 : (Nx, 6) ndarray — [p, u_g, u_l, alpha_g, h_g, h_l]
    prof  : dict — analytic reference (M, p, T, u, mdot_choked, gamma, R, ...)
    """
    x_grid = np.asarray(x_grid, dtype=float)
    A      = np.asarray(A,      dtype=float)
    Nx     = x_grid.size

    # gamma, R from the dominant gas EOS at the reservoir state
    st0   = fluid_g.get_state(jxp.PT_INPUTS, jnp.array([p0]), jnp.array([T0_g]))
    rho0  = float(st0["density"][0])
    a0    = float(st0["speed_of_sound"][0])
    R     = p0 / (rho0 * T0_g)
    gamma = a0 * a0 / (R * T0_g)

    i_th  = int(np.argmin(A))
    A_st  = float(A[i_th])                     # discrete throat area A*
    ar    = A / A_st

    M = np.empty(Nx)
    for i in range(Nx):
        M[i] = _area_mach(ar[i], gamma, "sub" if i <= i_th else "sup")

    T = T0_g / (1.0 + 0.5 * (gamma - 1.0) * M * M)
    p = p0 * (T / T0_g) ** (gamma / (gamma - 1.0))
    u = M * np.sqrt(gamma * R * T)

    u_g = u
    u_l = u / slip

    # enthalpies at the LOCAL gas temperature (SRC slaves the ghost thermally)
    h_g = np.asarray(fluid_g.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])
    h_l = np.asarray(fluid_l.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])

    ag = np.full(Nx, float(alpha_g))
    U_h_0 = np.column_stack([p, u_g, u_l, ag, h_g, h_l])

    gp1, gm1 = gamma + 1.0, gamma - 1.0
    mdot_choked = A_st * p0 * np.sqrt(gamma / (R * T0_g)) \
        * (2.0 / gp1) ** (gp1 / (2.0 * gm1))

    prof = dict(
        x=x_grid, A=A, M=M, p=p, T=T, u=u, alpha_g=ag,
        gamma=gamma, R=R, rho0=rho0, a0=a0, i_throat=i_th, A_star=A_st,
        mdot_choked=mdot_choked,
        M_exit=float(M[-1]), p_exit=float(p[-1]), T_exit=float(T[-1]),
    )
    return U_h_0, prof


def isentropic_shock_ic(
    x_grid, A, *,
    p0, T0_g, T0_l, alpha_g, slip,
    fluid_g, fluid_l, x_shock,
):
    """
    Choked nozzle with a NORMAL SHOCK standing at x_shock in the diverging
    section: isentropic supersonic branch up to the shock, Rankine-Hugoniot
    jump, then the subsonic branch to the exit on the post-shock isentrope
    (new sonic area A2* from M2, stagnation pressure p02 < p0).

    The construction OUTPUTS the consistent exit pressure prof['p_exit'] —
    set bc.p_out to it: downstream of the shock the exit is subsonic, so the
    characteristic outlet is active and the back-pressure HOLDS the shock.

    Same return convention as isentropic_ic.  Extra prof keys:
      i_shock, M1, M2, p_ratio (p2/p1), p02_ratio (p02/p0), A2_star.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    A      = np.asarray(A,      dtype=float)
    Nx     = x_grid.size

    st0   = fluid_g.get_state(jxp.PT_INPUTS, jnp.array([p0]), jnp.array([T0_g]))
    rho0  = float(st0["density"][0])
    a0    = float(st0["speed_of_sound"][0])
    R     = p0 / (rho0 * T0_g)
    gamma = a0 * a0 / (R * T0_g)
    g     = gamma

    i_th = int(np.argmin(A))
    i_sh = int(np.argmin(np.abs(x_grid - x_shock)))
    if i_sh <= i_th:
        raise ValueError(f"x_shock={x_shock} must lie in the diverging section "
                         f"(throat at x={x_grid[i_th]:.4f})")
    A_st = float(A[i_th])

    # pre-shock: standard perfect expansion up to (and including) cell i_sh
    M = np.empty(Nx)
    for i in range(i_sh + 1):
        M[i] = _area_mach(A[i] / A_st, g, "sub" if i <= i_th else "sup")
    M1 = M[i_sh]

    # Rankine-Hugoniot at M1
    M2sq      = (1.0 + 0.5 * (g - 1.0) * M1 * M1) / (g * M1 * M1 - 0.5 * (g - 1.0))
    M2        = float(np.sqrt(M2sq))
    p_ratio   = 1.0 + 2.0 * g / (g + 1.0) * (M1 * M1 - 1.0)
    p02_ratio = (((g + 1.0) * M1 * M1 / ((g - 1.0) * M1 * M1 + 2.0)) ** (g / (g - 1.0))
                 * ((g + 1.0) / (2.0 * g * M1 * M1 - (g - 1.0))) ** (1.0 / (g - 1.0)))
    p02  = p0 * p02_ratio                       # post-shock stagnation pressure
    # post-shock sonic area from M2 at A[i_sh]
    ar2  = (1.0 / M2) * ((2.0 / (g + 1.0))
                         * (1.0 + 0.5 * (g - 1.0) * M2 * M2)
                         ) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    A2_st = float(A[i_sh]) / ar2

    # post-shock subsonic branch to the exit (T0 unchanged across the shock)
    for i in range(i_sh + 1, Nx):
        M[i] = _area_mach(A[i] / A2_st, g, "sub")

    T = T0_g / (1.0 + 0.5 * (g - 1.0) * M * M)
    p = np.where(np.arange(Nx) <= i_sh,
                 p0  * (T / T0_g) ** (g / (g - 1.0)),
                 p02 * (T / T0_g) ** (g / (g - 1.0)))
    u = M * np.sqrt(g * R * T)

    u_g = u
    u_l = u / slip
    h_g = np.asarray(fluid_g.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])
    h_l = np.asarray(fluid_l.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])
    ag  = np.full(Nx, float(alpha_g))
    U_h_0 = np.column_stack([p, u_g, u_l, ag, h_g, h_l])

    gp1, gm1 = g + 1.0, g - 1.0
    mdot_choked = A_st * p0 * np.sqrt(g / (R * T0_g)) \
        * (2.0 / gp1) ** (gp1 / (2.0 * gm1))

    prof = dict(
        x=x_grid, A=A, M=M, p=p, T=T, u=u, alpha_g=ag,
        gamma=g, R=R, rho0=rho0, a0=a0, i_throat=i_th, A_star=A_st,
        mdot_choked=mdot_choked,
        i_shock=i_sh, M1=float(M1), M2=M2, p_ratio=float(p_ratio),
        p02_ratio=float(p02_ratio), A2_star=A2_st,
        M_exit=float(M[-1]), p_exit=float(p[-1]), T_exit=float(T[-1]),
    )
    return U_h_0, prof


def isentropic_subsonic_ic(
    x_grid, A, *,
    p0, T0_g, T0_l, alpha_g, slip,
    fluid_g, fluid_l, p_out,
):
    """
    Analytic SUBSONIC (unchoked) isentropic venturi IC — the dip-and-recover
    steady state of a converging-diverging nozzle that NEVER reaches sonic.

    Unlike isentropic_ic (choked: subsonic upstream, supersonic downstream of a
    sonic throat), here the SUBSONIC branch holds EVERYWHERE and the throat is
    the Mach MAXIMUM (< 1).  This IS the steady solution up to discretization
    error, so it seeds the time-march at (near) convergence — no slow low-Mach
    transient, which is why a choked isentropic seed or a cold start stalls.

    Anchored on the imposed back pressure p_out:
      * exit Mach from the reservoir isentrope,
          M_exit = sqrt( (2/(g-1)) [ (p0/p_out)^((g-1)/g) - 1 ] ),
      * VIRTUAL sonic area  A* = A_exit / (A/A*)(M_exit)   (< A_throat, never
        physically reached),
      * M(x) = subsonic root of  A(x)/A* = (A/A*)(M)  per cell,
      * T, p, u from the isentropic relations.
    If the resulting throat Mach hits 1 the case is actually CHOKED (p_out below
    the choking back pressure): a warning is printed and M is clamped subsonic —
    raise p_out (toward p0) for a genuine venturi.

    Same return convention as isentropic_ic.  prof adds M_throat, mdot.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    A      = np.asarray(A,      dtype=float)
    Nx     = x_grid.size

    st0   = fluid_g.get_state(jxp.PT_INPUTS, jnp.array([p0]), jnp.array([T0_g]))
    rho0  = float(st0["density"][0])
    a0    = float(st0["speed_of_sound"][0])
    R     = p0 / (rho0 * T0_g)
    gamma = a0 * a0 / (R * T0_g)
    g     = gamma

    i_th   = int(np.argmin(A))
    A_exit = float(A[-1])

    # Exit Mach from the back pressure on the reservoir isentrope
    pr = float(p_out) / float(p0)
    M_exit = float(np.sqrt(max((pr ** (-(g - 1.0) / g) - 1.0) * 2.0 / (g - 1.0), 0.0)))
    M_exit = max(M_exit, 1e-6)                      # avoid /0 in the ratio

    # Virtual sonic area A* (< A_throat, never reached); M(x) on the sub branch
    A_star = A_exit / _am_ratio(M_exit, g)
    M = np.array([_area_mach(A[i] / A_star, g, "sub") for i in range(Nx)])
    M_throat = float(M[i_th])
    if M_throat >= 0.999:
        print(f"  [subsonic IC WARNING] throat M={M_throat:.4f} ≥ 1 → p_out="
              f"{p_out:.1f} Pa is at/below the choking back pressure; clamping "
              f"subsonic. Raise p_out toward p0 for a genuine venturi.")
        M = np.minimum(M, 0.999)

    T = T0_g / (1.0 + 0.5 * (g - 1.0) * M * M)
    p = p0 * (T / T0_g) ** (g / (g - 1.0))
    u = M * np.sqrt(g * R * T)

    u_g = u
    u_l = u / slip
    h_g = np.asarray(fluid_g.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])
    h_l = np.asarray(fluid_l.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.array(T))["enthalpy"])
    ag  = np.full(Nx, float(alpha_g))
    U_h_0 = np.column_stack([p, u_g, u_l, ag, h_g, h_l])

    rho_th = p[i_th] / (R * T[i_th])
    mdot   = float(A[i_th] * rho_th * u[i_th])       # uniform along x

    prof = dict(
        x=x_grid, A=A, M=M, p=p, T=T, u=u, alpha_g=ag,
        gamma=g, R=R, rho0=rho0, a0=a0, i_throat=i_th, A_star=A_star,
        M_throat=M_throat, mdot=mdot,
        M_exit=float(M[-1]), p_exit=float(p[-1]), T_exit=float(T[-1]),
    )
    return U_h_0, prof


# ═════════════════════════════════════════════════════════════════════════════
# Bernoulli IC (formerly bernoulli_ic.py)
# ═════════════════════════════════════════════════════════════════════════════
# Analytic incompressible (Bernoulli) initial condition for the two-fluid solver.
#
# Purpose
# -------
# A self-contained, geometry-agnostic cold-start IC for the LOW-SPEED LIQUID regime
# (the flashing-inlet regime) that needs NO nozzlex solve.  For a nearly
# incompressible dominant phase the exact steady quasi-1D solution is
#
#     mass    :  u_dom(x) = u_in * A_in / A(x)                    (rho ~ const)
#     Bernoulli: p(x)     = p0 - 1/2 * rho_dom * u_dom(x)^2       (reservoir p0)
#
# which is exact to O(Ma^2).  This same profile doubles as the ANALYTIC REFERENCE
# the converged two-fluid solution is validated against (the ghost limit must
# reproduce single-phase incompressible nozzle flow).
#
# Ghost (passive) phase
# ---------------------
# The near-massless second phase (alpha ~ 1e-4) is given, per the physical picture:
#   * constant void fraction  alpha_g = alpha_g_in,
#   * NO SLIP by default (slip=1 -> u_ghost = u_dom); general slip S = u_g/u_l,
#   * SAME temperature T0 as the dominant phase, ISOTHERMAL along x, so its
#     enthalpy h_ghost(x) = h(p(x), T0) via PT_INPUTS is always a valid in-table
#     EOS state (no risk of driving the ghost off its table on step 0).
#
# Both phases are built ISOTHERMAL at T0 (the flow is ~isothermal at low Ma), so
# h_i(x) = h_i(p(x), T0_i) via PT_INPUTS -> every cell is a valid EOS state.  The
# tiny H0 = h + u^2/2 drift this implies (order 1/2 u^2, a few J/kg at low speed)
# relaxes away in the first transient.
#
# Returns  U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]  shape (Nx, 6).

def bernoulli_ic(
    x_grid, A, *,
    p0, T0_g, T0_l, u_in, alpha_g, slip,
    fluid_g, fluid_l, dom="liquid", p_floor=1.0e3,
):
    """
    Build an incompressible-Bernoulli two-fluid IC (and its analytic reference).

    Parameters
    ----------
    x_grid : (Nx,) cell-centre positions [m]                (only for prof / sizing)
    A      : (Nx,) cross-sectional area at each cell [m^2]   (geometry-agnostic input)
    p0     : reservoir (stagnation) pressure [Pa], shared by both phases
    T0_g, T0_l : gas / liquid stagnation temperature [K]
    u_in   : inlet velocity of the DOMINANT phase [m/s]
    alpha_g: (constant) gas void fraction [-]  (~1 for gas-dom, ~1e-4 for liquid-dom)
    slip   : S = u_g / u_l  (1 = no slip)
    fluid_g, fluid_l : EOS objects with get_state(PT_INPUTS, p, T) -> FluidState
                       (FluidBicubic and/or FluidPerfectGas — same interface)
    dom    : 'liquid' | 'gas' — which phase carries the incompressible mass balance
    p_floor: pressure clamp [Pa] so a strong Bernoulli dip never drives p<=0
             (invalid EOS).  A hit is reported (raise u_in or widen the throat).

    Returns
    -------
    U_h_0 : (Nx, 6) ndarray — [p, u_g, u_l, alpha_g, h_g, h_l]
    prof  : dict — analytic reference (x, A, p, u_g, u_l, alpha_g, T0, rho_dom,
            u_dom, and the inlet Mach) for overlay / validation
    """
    x_grid = np.asarray(x_grid, dtype=float)
    A      = np.asarray(A,      dtype=float)
    Nx     = x_grid.size
    A_in   = float(A[0])

    # Dominant-phase incompressible density from the reservoir state (rho ~ const).
    if dom == "liquid":
        fl_dom, T0_dom = fluid_l, T0_l
    elif dom == "gas":
        fl_dom, T0_dom = fluid_g, T0_g
    else:
        raise ValueError(f"dom must be 'liquid'|'gas', got {dom!r}")

    _st0    = fl_dom.get_state(jxp.PT_INPUTS, jnp.array([p0]), jnp.array([T0_dom]))
    rho_dom = float(_st0["density"][0])
    a_dom   = float(_st0["speed_of_sound"][0])

    # Mass conservation (incompressible) + Bernoulli from the reservoir.
    u_dom = u_in * A_in / A                               # u peaks at the throat
    p     = p0 - 0.5 * rho_dom * u_dom**2                 # static-pressure dip
    p_hit = bool(np.any(p <= p_floor))
    p     = np.maximum(p, p_floor)

    # Phase velocities from the slip ratio S = u_g/u_l.
    if dom == "liquid":
        u_l = u_dom
        u_g = slip * u_l
    else:                                                 # dom == "gas"
        u_g = u_dom
        u_l = u_g / slip

    # Isothermal enthalpies at T0 -> guaranteed in-table EOS states for both phases.
    h_g = np.asarray(fluid_g.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.full(Nx, T0_g))["enthalpy"])
    h_l = np.asarray(fluid_l.get_state(
        jxp.PT_INPUTS, jnp.array(p), jnp.full(Nx, T0_l))["enthalpy"])

    ag = np.full(Nx, float(alpha_g))

    U_h_0 = np.column_stack([p, u_g, u_l, ag, h_g, h_l])

    prof = dict(
        x=x_grid, A=A, p=p, u_g=u_g, u_l=u_l, alpha_g=ag,
        rho_dom=rho_dom, u_dom=u_dom, dom=dom,
        T0_g=T0_g, T0_l=T0_l, u_in=u_in, a_dom=a_dom,
        Ma_in=u_in / a_dom if a_dom > 0 else np.nan,
        p_floor_hit=p_hit,
    )
    return U_h_0, prof


# ═════════════════════════════════════════════════════════════════════════════
# Initial conditions from an EXISTING solution (restart, nozzlex CSV / pickle, live nozzlex solve)
# ═════════════════════════════════════════════════════════════════════════════

def resolve_path(path, base_dir):
    """ic.run / ic.pickle: relative to base_dir (dev_scripts/) if that exists,
    otherwise as given (absolute, or relative to the working directory)."""
    p = pathlib.Path(base_dir) / path
    return p if p.exists() else pathlib.Path(path)


def ic_source_file(ic, base_dir):
    """The data file an ic block reads (restart solution.csv, nozzlex CSV or
    pickle), or None for the analytic ICs — for the run record."""
    kind = str(ic.get("type", "")).strip().lower()
    if kind in ("restart", "nozzlex_csv") and ic.get("run") is not None:
        p = resolve_path(ic["run"], base_dir)
        if p.is_dir():
            p = p / ("solution.csv" if kind == "restart" else "results_tfm_120a.csv")
        return p
    if kind == "nozzlex" and ic.get("pickle") is not None:
        return resolve_path(ic["pickle"], base_dir)
    return None


def smooth_121(f, passes):
    """`passes` sweeps of the 1-2-1 filter on the interior (end values kept)."""
    g = np.asarray(f, dtype=float).copy()
    for _ in range(passes):
        g[1:-1] = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
    return g


# ═════════════════════════════════════════════════════════════════════════════
# Restart from a previous run (solution.csv)
# ═════════════════════════════════════════════════════════════════════════════

_U_H_COLS = ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")


def restart_ic(run, x, *, base_dir, smooth_passes=0, slave_ghost=False):
    """Seed from a PREVIOUS run's saved solution (solution.csv).

    The cheap way to sweep boundary conditions: converge once, then change
    bc.p_out (or anything else) and restart — the transient is only the small
    repricing between neighbouring steady states, never the full cold-start
    column drain.  `run`: run folder (or solution.csv path), relative to
    base_dir or absolute.  A different Nx is fine: linear interp in x.

    Returns (U_h_0, info) with info = {path, n_source, smooth_passes, slave_ghost}.
    """
    path = resolve_path(run, base_dir)
    if path.is_dir():
        path = path / "solution.csv"
    D = np.genfromtxt(path, delimiter=",", names=True)
    xs = np.asarray(D["x"])

    if smooth_passes or slave_ghost:
        fields = {c: smooth_121(D[c], smooth_passes) for c in _U_H_COLS}
        if slave_ghost:
            # Ghost-limit restarts: replace the ghost-gas fields with their
            # drag/heat-slaved values instead of restarting the transient's
            # garbage (a smoothed h_g once carried a 60 K hump → stiff heat
            # source → NaN at t=0).  cp fitted from the run's own data.
            cp_est = float(np.median(D["h_g"] / D["T_g"]))
            fields["u_g"]     = fields["u_l"].copy()
            fields["h_g"]     = cp_est * smooth_121(D["T_l"], smooth_passes)
            fields["alpha_g"] = np.full_like(xs, float(np.median(D["alpha_g"])), dtype=float)
        U_h_0 = np.column_stack([np.interp(x, xs, fields[c]) for c in _U_H_COLS])
    else:
        U_h_0 = np.column_stack([np.interp(x, xs, D[c]) for c in _U_H_COLS])

    if not np.isfinite(U_h_0).all():
        raise ValueError(f"restart solution contains NaN/inf: {path}")
    return U_h_0, dict(path=path, n_source=xs.size,
                       smooth_passes=int(smooth_passes), slave_ghost=bool(slave_ghost))


# ═════════════════════════════════════════════════════════════════════════════
# nozzlex space-marching CSV export
# ═════════════════════════════════════════════════════════════════════════════

def nozzlex_csv_ic(run, x, *, base_dir, alpha_floor=1.0e-4,
                   csv_name="results_tfm_120a.csv"):
    """Seed from a nozzlex space-marching CSV (columns distance, pressure,
    velocity_gas, velocity_liquid, void_fraction, enthalpy_gas, enthalpy_liquid).

    Ghost-α floor: the nozzlex reference CSV is built with α_v,in = 1e-6 (fine
    for the space-marching ODE, which has no ghost checkerboard), so its
    pre-flashing region carries α_g ≈ 1e-6 — two decades below the FVM's
    certified-safe ghost thickness (α≥1e-4; α≲1e-5 triggers the ghost-h
    instability: h_g = E/(α ρ_g) divides the ghost energy by ~0, so 2Δx noise
    blows up h_g).  The seed is clipped to [alpha_floor, 1 − alpha_floor] (mass
    fraction stays ~1e-6, so flashing physics is unmoved; only the passive-ghost
    VOLUME is inflated).

    Returns (U_h_0, info) with info = {path, n_source, alpha_floor,
    alpha_raw_min, n_lifted}.
    """
    import pandas as pd

    path = resolve_path(run, base_dir)
    if path.is_dir():
        path = path / csv_name
    D = pd.read_csv(path)
    xs = np.asarray(D["distance"])
    U_h_0 = np.column_stack([
        np.interp(x, xs, D[c])
        for c in ("pressure", "velocity_gas", "velocity_liquid", "void_fraction",
                  "enthalpy_gas", "enthalpy_liquid")])
    if not np.isfinite(U_h_0).all():
        raise ValueError(f"restart solution contains NaN/inf: {path}")
    alpha_raw_min = float(U_h_0[:, 3].min())
    n_lifted = int(np.sum(np.asarray(np.interp(x, xs, D["void_fraction"])) < alpha_floor))
    U_h_0[:, 3] = np.clip(U_h_0[:, 3], alpha_floor, 1.0 - alpha_floor)
    return U_h_0, dict(path=path, n_source=xs.size, alpha_floor=alpha_floor,
                       alpha_raw_min=alpha_raw_min, n_lifted=n_lifted)


# ═════════════════════════════════════════════════════════════════════════════
# nozzlex solution pickle  /  live nozzlex solve (formerly nozzlex_ic.py)
# ═════════════════════════════════════════════════════════════════════════════
# Compute a two-fluid time-marching initial condition from a nozzlex space-marching
# solve.
#
# Runs nozzlex's `nozzle_two_fluid_one_component_autonomous` pipe model (identical
# setup to demo/demo_pipe_nozzlex.py), evaluates the dense solution onto the
# time-marching x-grid, and returns
#
#     U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]   shape (Nx, 6)
#
# ready for `W_from_U_h_batch`.  The nozzlex INLET state (p0_in, alpha_g_in, u_in,
# h_g=h_sat_g, h_l=h_sat_l+dh_l) matches the uniform IC used before — this module
# only replaces the *interior* with the relaxed downstream profile, so the existing
# inlet BC stays valid.
#
# Why: seeding the transient with nozzlex's already-relaxed profile starts the
# solve ON the slow manifold instead of slamming a deeply metastable uniform state
# into every cell at once (which lights up the zero-slip checkerboard).


def _nozzlex_api():
    """Lazy import of the nozzlex package (and diffrax / optimistix): only the
    live nozzlex solve needs it, so restart / pickle seeds work without it."""
    import diffrax as dfx
    import optimistix as optx
    from nozzlex.two_fluid.nozzle_model_solver import NozzleParams, replace_param
    from nozzlex.two_fluid.nozzle_model_core import (
        get_pipe,
        nozzle_two_fluid_one_component_autonomous,
    )
    return (dfx, optx, NozzleParams, replace_param, get_pipe,
            nozzle_two_fluid_one_component_autonomous)


def _solve_pipe(
    fluid_g, fluid_l, *,
    p0_in, T0_in, u_g_in, u_l_in, alpha_g_in, dh_l, length,
    Nb, Nd, Ma_in, solver_name, rtol, atol, nsteps,
):
    """Run the nozzlex pipe IVP and return (sol, Y_ref, params_model).

    Faithful port of demo_pipe_nozzlex.transonic_nozzle_one_component_lax, with
    the inlet superheat (dh_l), void fraction and velocities exposed as arguments.
    """
    (dfx, optx, NozzleParams, replace_param, get_pipe,
     nozzle_two_fluid_one_component_autonomous) = _nozzlex_api()
    # ── Inlet state (saturated vapour + metastable superheated liquid) ─────────
    sat_v = fluid_g.get_state_saturation(p0_in)
    sat_l = fluid_l.get_state_saturation(p0_in)
    h_sat_l = sat_l["enthalpy"]
    st_l_in = fluid_l.get_state(jxp.HmassP_INPUTS, h_sat_l + dh_l, p0_in)

    p_in     = st_l_in["pressure"]
    rho_l_in = st_l_in["density"]
    rho_v_in = sat_v["density"]
    h_l_in   = st_l_in["enthalpy"]
    h_v_in   = sat_v["enthalpy"]

    alpha_l_in = 1.0 - alpha_g_in
    y_inlet = jnp.array([
        1e-9, alpha_g_in, alpha_l_in, rho_v_in, rho_l_in,
        u_g_in, u_l_in, p_in, h_v_in, h_l_in,
    ])

    params = NozzleParams(
        p0_in=p0_in, T01_in=T0_in, T02_in=T0_in, length=length,
        roughness=0.0, Ma_low=0.95, Ma_high=1.05,
        heat_transfer=0.0, wall_friction=0.0,
        fluid1=fluid_g, fluid2=fluid_l, geometry=get_pipe,
        bubble_number=Nb, droplet_number=Nd,
        u1_in=u_g_in, u2_in=u_l_in,
    )
    params = replace_param(params, "Ma_in", Ma_in)

    # ── A_mat row/col equilibration from the inlet (as in the demo) ────────────
    params = replace_param(params, "row_scales", jnp.ones(9))
    params = replace_param(params, "col_scales", jnp.ones(9))
    params = replace_param(params, "det_ref", jnp.float64(1.0))
    out_inlet     = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params)
    A_mat_inlet   = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params = replace_param(params, "row_scales", row_scales_in)
    params = replace_param(params, "col_scales", col_scales_in)
    out_scaled  = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params)
    det_ref_val = jnp.maximum(jnp.abs(out_scaled["D"]), 1e-10)
    params = replace_param(params, "det_ref", det_ref_val)

    # ── State scaling (Y_hat = Y / Y_ref is O(1)) ─────────────────────────────
    Y_ref = jnp.array([
        length, 1.0, 1.0, rho_v_in, rho_l_in,
        u_g_in, u_l_in, p_in, h_v_in, h_l_in,
    ])

    def ode_full(t, y_hat, args):
        return nozzle_two_fluid_one_component_autonomous(t, y_hat * Y_ref, args)

    def rhs(t, y_hat, args):
        return ode_full(t, y_hat, args)["rhs_autonomous"] / Y_ref

    y_inlet_hat = y_inlet / Y_ref
    solver  = jxp.make_diffrax_solver(solver_name)
    adjoint = jxp.make_diffrax_adjoint("DirectAdjoint")
    term    = dfx.ODETerm(rhs)
    ctrl    = dfx.PIDController(rtol=rtol, atol=atol)

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_two_fluid_one_component_autonomous(0.0, y_phys, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        M = 15.0 - Ma_mix
        return jnp.min(jnp.stack([x / L, 1.0 - x / L, M]))

    event  = dfx.Event(cond_fn=event_fn,
                       root_finder=optx.Newton(rtol=1e-6, atol=1e-6))
    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term, solver, t0=1e-9, t1=1e9, dt0=1e-9, y0=y_inlet_hat,
        args=params, stepsize_controller=ctrl, adjoint=adjoint,
        saveat=saveat, event=event, max_steps=nsteps,
    )
    return sol, Y_ref, params


def nozzlex_initial_condition(
    x_grid, fluid_g, fluid_l, *,
    p0_in, T0_in, u_g_in, u_l_in, alpha_g_in, dh_l,
    length=None, Nb=1e10, Nd=1e8, Ma_in=0.01,
    solver_name="ImplicitEuler", rtol=1e-6, atol=1e-6,
    nsteps=1000, n_eval=200,
):
    """
    Build a time-marching IC from a nozzlex pipe solve.

    Parameters
    ----------
    x_grid : (Nx,) cell-centre positions of the time-marching grid [m]
    fluid_g, fluid_l : FluidBicubic vapour / liquid tables (reused, must match)
    p0_in, T0_in     : inlet pressure [Pa] / stagnation temperature [K]
    u_g_in, u_l_in   : inlet phase velocities [m/s]
    alpha_g_in       : inlet void fraction [-]
    dh_l             : inlet liquid superheat above saturation [J/kg]
    length           : pipe length [m] (defaults to x_grid[-1])
    Nb, Nd           : bubble / droplet number densities
    Ma_in            : nozzlex continuation Mach parameter (demo uses 0.01)
    n_eval           : # dense-output samples along the pipe before interpolation

    Returns
    -------
    U_h_0 : (Nx, 6) ndarray — [p, u_g, u_l, alpha_g, h_g, h_l] per cell
    prof  : dict of the raw nozzlex profile (x, p, u_g, u_l, alpha_g, h_g, h_l)
            for plotting / diagnostics
    """
    (dfx, optx, NozzleParams, replace_param, get_pipe,
     nozzle_two_fluid_one_component_autonomous) = _nozzlex_api()
    x_grid = np.asarray(x_grid, dtype=float)
    if length is None:
        length = float(x_grid[-1])

    sol, Y_ref, params = _solve_pipe(
        fluid_g, fluid_l,
        p0_in=p0_in, T0_in=T0_in, u_g_in=u_g_in, u_l_in=u_l_in,
        alpha_g_in=alpha_g_in, dh_l=dh_l, length=length,
        Nb=Nb, Nd=Nd, Ma_in=Ma_in, solver_name=solver_name,
        rtol=rtol, atol=atol, nsteps=nsteps,
    )

    # Dense output → physical states → full nozzlex output dicts
    ts   = jnp.linspace(sol.t0, sol.ts[0], n_eval)
    outs = [nozzle_two_fluid_one_component_autonomous(0.0, sol.evaluate(t) * Y_ref, params)
            for t in ts]

    def _col(key):
        return np.asarray([float(o[key]) for o in outs])

    x_nx  = _col("x")
    p_nx  = _col("p")
    ug_nx = _col("v1")
    ul_nx = _col("v2")
    ag_nx = _col("alpha1")
    hg_nx = _col("h1")
    hl_nx = _col("h2")

    # Sort by x (dense samples should already be monotonic) and interpolate.
    # np.interp clamps to end values beyond the solved range (fills last cells
    # if the pipe solve stopped short of x_grid[-1]).
    order = np.argsort(x_nx)
    x_s   = x_nx[order]

    def _i(f):
        return np.interp(x_grid, x_s, f[order])

    U_h_0 = np.column_stack([
        _i(p_nx), _i(ug_nx), _i(ul_nx), _i(ag_nx), _i(hg_nx), _i(hl_nx),
    ])

    prof = dict(
        x=x_s, p=p_nx[order], u_g=ug_nx[order], u_l=ul_nx[order],
        alpha_g=ag_nx[order], h_g=hg_nx[order], h_l=hl_nx[order],
    )
    return U_h_0, prof


def ic_from_nozzlex_pickle(pkl_path, x_grid, *, phase1="auto"):
    """
    Build a two-fluid time-marching IC from a SAVED nozzlex solution pickle,
    e.g. the output of case_study/elliot_1979/diffrax_nozzle_two_fluid_jump.py:

        {'pass1': {...}, 'pass2': {...|None}, 'pass3': {...|None}}

    Each pass dict holds dense PHYSICAL profiles keyed by x, p, v1, v2, alpha1,
    alpha2, h1, h2, rho1, rho2 (all shape (n,)).  The passes are stitched in x
    (subsonic approach → supersonic → subsonic tail), interpolated onto x_grid,
    and mapped to  U_h = [p, u_g, u_l, alpha_g, h_g, h_l].

    Phase ordering gotcha
    ---------------------
    nozzlex index-1 is NOT always the gas.  In the one-component phase-change
    setup (nozzlex_initial_condition, above) index-1 = VAPOUR; in the two-
    component Elliot setup (diffrax_nozzle_two_fluid_jump: fluid_name=
    ['water','nitrogen']) index-1 = LIQUID (water).  `phase1='auto'` picks the
    ordering from the densities (liquid is the heavier phase), so the mapping is
    correct regardless of which script wrote the pickle.

    Parameters
    ----------
    pkl_path : path to the saved solution pickle
    x_grid   : (Nx,) time-marching cell-centre positions [m]
    phase1   : 'auto' | 'liquid' | 'gas' — which physical phase is nozzlex idx-1

    Returns
    -------
    U_h_0 : (Nx, 6) [p, u_g, u_l, alpha_g, h_g, h_l]
    prof  : dict of the stitched raw profile for plotting / diagnostics
    """
    import pickle

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    passes = [data[k] for k in ("pass1", "pass2", "pass3")
              if isinstance(data, dict) and data.get(k) is not None]
    if not passes:
        raise ValueError(f"No non-empty pass segments found in {pkl_path}")

    def _cat(key):
        return np.concatenate(
            [np.asarray(p[key], dtype=float).ravel() for p in passes])

    x    = _cat("x")
    p    = _cat("p")
    v1   = _cat("v1");     v2   = _cat("v2")
    a1   = _cat("alpha1"); a2   = _cat("alpha2")
    h1   = _cat("h1");     h2   = _cat("h2")
    rho1 = _cat("rho1");   rho2 = _cat("rho2")

    if phase1 == "auto":
        # Liquid is the denser phase; whichever index carries it is phase-1.
        phase1 = "liquid" if np.nanmean(rho1) >= np.nanmean(rho2) else "gas"

    if phase1 == "liquid":     # fluid1 = liquid, fluid2 = gas  (Elliot pickle)
        u_g, u_l   = v2, v1
        alpha_g    = a2
        h_g, h_l   = h2, h1
    elif phase1 == "gas":      # fluid1 = gas,    fluid2 = liquid
        u_g, u_l   = v1, v2
        alpha_g    = a1
        h_g, h_l   = h1, h2
    else:
        raise ValueError(f"phase1 must be 'auto'|'liquid'|'gas', got {phase1!r}")

    order = np.argsort(x)
    x_s   = x[order]
    x_grid = np.asarray(x_grid, dtype=float)

    def _i(f):
        return np.interp(x_grid, x_s, f[order])   # clamps beyond the solved span

    U_h_0 = np.column_stack([
        _i(p), _i(u_g), _i(u_l), _i(alpha_g), _i(h_g), _i(h_l),
    ])

    U_h_0[0, 3] = U_h_0[1, 3] 

    prof = dict(
        x=x_s, p=p[order], u_g=u_g[order], u_l=u_l[order],
        alpha_g=alpha_g[order], h_g=h_g[order], h_l=h_l[order],
        phase1=phase1,
    )
    return U_h_0, prof


# ═════════════════════════════════════════════════════════════════════════════
# Dispatcher used by main_two_fluid.py
# ═════════════════════════════════════════════════════════════════════════════

IC_TYPES = ("uniform", "stagnation", "simple", "isentropic", "isentropic_shock",
            "isentropic_subsonic", "subsonic", "bernoulli",
            "restart", "nozzlex_csv", "nozzlex")


def build_initial_condition(ic, *, x, A, p0, T_g0, T_l0, alpha_g, slip, H_g0, H_l0,
                            p_out, fluid_g, fluid_l, dom, inlet_static, base_dir):
    """U_h_0 (Nx, 6) for the YAML `ic:` block, with the driver's log lines.

    ic           : the raw `ic:` dict from the settings file (ic["type"] selects)
    inlet_static : (p_in, u_g_in, u_l_in, h_g_in, h_l_in), the driver's isentropic
                   inlet-static estimate (used by 'uniform')
    base_dir     : folder that relative ic.run / ic.pickle paths are resolved
                   against (dev_scripts/)
    """
    kind = ic["type"]
    Nx = len(x)

    if kind == "uniform":
        p_in, u_g_in, u_l_in, _, _ = inlet_static
        U_h_0 = uniform_ic(Nx, inlet_static, alpha_g)
        print(f"IC (uniform):  p={p_in/1e5:.1f} bar  alpha_g={alpha_g:.6f}"
              f"  u_g={u_g_in:.1f} m/s  u_l={u_l_in:.1f} m/s")

    elif kind == "stagnation":
        U_h_0 = stagnation_ic(Nx, p0, alpha_g, H_g0, H_l0)
        print(f"IC (stagnation):  at rest, p={p0/1e5:.1f} bar  alpha_g={alpha_g:.6f}"
              f"  H_g0={H_g0/1e3:.1f}  H_l0={H_l0/1e3:.1f} kJ/kg")

    elif kind == "simple":
        U_h_0 = simple_ic(Nx, p0, p_out, alpha_g, slip, H_g0, H_l0)
        print(f"IC (simple cold-start):  p ramp {p0/1e5:.1f}→{p_out/1e5:.2f} bar  "
              f"alpha_g={alpha_g:.4f}  u_g={U_h_0[0, 1]:.1f}  u_l={U_h_0[0, 2]:.1f} m/s "
              f"(slip={slip:.2f}, du≈0)")

    elif kind == "isentropic":
        # Analytic choked perfect-expansion IC (gas-dominant mirror of bernoulli):
        # area–Mach per cell, sub branch before the throat, sup branch after.
        # This profile IS the steady solution up to discretization error and
        # doubles as the reference for check_isentropic.py.
        U_h_0, isen = isentropic_ic(
            x, A, p0=p0, T0_g=T_g0, T0_l=T_l0, alpha_g=alpha_g, slip=slip,
            fluid_g=fluid_g, fluid_l=fluid_l,
        )
        print(f"IC (isentropic, choked):  gamma={isen['gamma']:.4f}  "
              f"R={isen['R']:.2f} J/kg/K  A*={isen['A_star']:.4e} m² "
              f"(throat cell {isen['i_throat']})")
        print(f"  M {U_h_0[0,1]/np.sqrt(isen['gamma']*isen['R']*isen['T'][0]):.3f}"
              f" → M_exit={isen['M_exit']:.3f}   p {U_h_0[:,0].max()/1e5:.3f}"
              f"..{U_h_0[:,0].min()/1e5:.4f} bar   T {isen['T'].max():.1f}"
              f"..{isen['T'].min():.1f} K   mdot*={isen['mdot_choked']:.4f} kg/s")

    elif kind == "isentropic_shock":
        # Analytic choked profile with a STANDING NORMAL SHOCK at ic.x_shock.
        # bc.p_out MUST equal the construction's p_exit (printed below) — the
        # subsonic outlet then HOLDS the shock at its design cell.
        x_shock = float(ic["x_shock"])
        U_h_0, isen = isentropic_shock_ic(
            x, A, p0=p0, T0_g=T_g0, T0_l=T_l0, alpha_g=alpha_g, slip=slip,
            fluid_g=fluid_g, fluid_l=fluid_l, x_shock=x_shock,
        )
        print(f"IC (isentropic + normal shock @ x={x_shock} → cell {isen['i_shock']}):"
              f"  M1={isen['M1']:.4f} → M2={isen['M2']:.4f}"
              f"  p2/p1={isen['p_ratio']:.4f}  p02/p0={isen['p02_ratio']:.4f}")
        print(f"  M_exit={isen['M_exit']:.4f}  p_exit={isen['p_exit']:.1f} Pa"
              f"  (bc.p_out={p_out:.1f} — {'CONSISTENT' if abs(p_out - isen['p_exit']) < 500 else 'WARNING: MISMATCH, shock will drift'})"
              f"  mdot*={isen['mdot_choked']:.4f} kg/s")

    elif kind in ("isentropic_subsonic", "subsonic"):
        # Analytic SUBSONIC (unchoked) venturi IC, anchored on bc.p_out.  IS the
        # steady solution up to discretization error → converges immediately.
        U_h_0, isen = isentropic_subsonic_ic(
            x, A, p0=p0, T0_g=T_g0, T0_l=T_l0, alpha_g=alpha_g, slip=slip,
            fluid_g=fluid_g, fluid_l=fluid_l, p_out=p_out,
        )
        print(f"IC (isentropic subsonic):  gamma={isen['gamma']:.4f}  "
              f"R={isen['R']:.2f} J/kg/K  M_throat={isen['M_throat']:.4f} "
              f"(throat cell {isen['i_throat']})")
        print(f"  M_exit={isen['M_exit']:.4f}   p {U_h_0[:,0].min()/1e5:.4f}"
              f"..{U_h_0[:,0].max()/1e5:.4f} bar   T {isen['T'].min():.1f}"
              f"..{isen['T'].max():.1f} K   mdot={isen['mdot']:.4f} kg/s")

    elif kind == "bernoulli":
        # Analytic incompressible (Bernoulli) cold-start for a nearly incompressible
        # dominant phase; ic.u_in = inlet dominant-phase velocity (default 1 m/s).
        u_in = float(ic.get("u_in", 1.0))
        U_h_0, bern = bernoulli_ic(
            x, A, p0=p0, T0_g=T_g0, T0_l=T_l0, u_in=u_in, alpha_g=alpha_g, slip=slip,
            fluid_g=fluid_g, fluid_l=fluid_l, dom=dom,
        )
        print(f"IC (bernoulli, dom={dom}):  p0={p0/1e5:.2f} bar  u_in={u_in:.2f} m/s  "
              f"rho_dom={bern['rho_dom']:.2f} kg/m³  Ma_in={bern['Ma_in']:.4f}")
        print(f"  p {U_h_0[:,0].min()/1e5:.3f}..{U_h_0[:,0].max()/1e5:.3f} bar"
              f"  u_dom {bern['u_dom'].min():.2f}..{bern['u_dom'].max():.2f} m/s"
              f"  alpha_g={alpha_g:.6f}  slip={slip:.2f}")
        if bern["p_floor_hit"]:
            print("  WARNING: Bernoulli pressure hit the floor (p≤p_floor) — the "
                  "throat dip is too deep; lower ic.u_in or widen the throat.")

    elif kind == "restart":
        if ic.get("run") is None:
            raise ValueError("ic.type='restart' requires ic.run: <run folder or solution.csv>")
        U_h_0, info = restart_ic(
            ic["run"], x, base_dir=base_dir,
            smooth_passes=int(ic.get("smooth_passes", 0)),
            slave_ghost=bool(ic.get("slave_ghost", False)),
        )
        print(f"IC (restart):  {info['path'].parent.name}  ({info['n_source']} cells → Nx={Nx})")
        if info["smooth_passes"] or info["slave_ghost"]:
            print(f"  smoothed: {info['smooth_passes']} passes of 1-2-1"
                  + ("  + ghost gas slaved (u_g=u_l, T_g=T_l)" if info["slave_ghost"] else ""))
        print(f"  p {U_h_0[:,0].min()/1e5:.4f}..{U_h_0[:,0].max()/1e5:.4f} bar"
              f"  u_l {U_h_0[:,2].min():.2f}..{U_h_0[:,2].max():.2f} m/s"
              f"  alpha_g {U_h_0[:,3].min():.2e}..{U_h_0[:,3].max():.2e}"
              f"  (bc.p_out={p_out/1e5:.4f} bar)")

    elif kind == "nozzlex_csv":
        if ic.get("run") is None:
            raise ValueError("ic.type='nozzlex_csv' requires ic.run: <folder or nozzlex CSV>")
        U_h_0, info = nozzlex_csv_ic(
            ic["run"], x, base_dir=base_dir,
            alpha_floor=float(ic.get("alpha_floor", 1.0e-4)),
        )
        a_floor = info["alpha_floor"]
        print(f"IC (restart):  {info['path'].parent.name}  ({info['n_source']} cells → Nx={Nx})")
        print(f"  alpha_g floored: raw min {info['alpha_raw_min']:.2e} → clip [{a_floor:.1e}, "
              f"{1.0 - a_floor:.4f}]  ({info['n_lifted']} cells lifted)")
        print(f"  p {U_h_0[:,0].min()/1e5:.4f}..{U_h_0[:,0].max()/1e5:.4f} bar"
              f"  u_l {U_h_0[:,2].min():.2f}..{U_h_0[:,2].max():.2f} m/s"
              f"  alpha_g {U_h_0[:,3].min():.2e}..{U_h_0[:,3].max():.2e}"
              f"  (bc.p_out={p_out/1e5:.4f} bar)")

    elif kind == "nozzlex":
        # Seed from a SAVED nozzlex space-marching solution (pickle); phase
        # ordering auto-detected from the densities (ic.phase1 overrides).
        if ic.get("pickle") is None:
            raise ValueError("ic.type='nozzlex' requires ic.pickle: <path to solution pickle>")
        pkl = resolve_path(ic["pickle"], base_dir)
        U_h_0, prof = ic_from_nozzlex_pickle(
            str(pkl), x, phase1=ic.get("phase1", "auto"))
        print(f"IC (nozzlex pickle):  {pkl.name}")
        print(f"  phase1={prof['phase1']}  (nozzlex idx-1)   profile x:"
              f" {prof['x'].min():.4f}..{prof['x'].max():.4f} m")
        print(f"  p {U_h_0[:,0].min()/1e5:.2f}..{U_h_0[:,0].max()/1e5:.2f} bar"
              f"  u_g {U_h_0[:,1].min():.1f}..{U_h_0[:,1].max():.1f}"
              f"  u_l {U_h_0[:,2].min():.1f}..{U_h_0[:,2].max():.1f} m/s"
              f"  alpha_g {U_h_0[:,3].min():.3f}..{U_h_0[:,3].max():.3f}")

    else:
        raise ValueError(f"unknown ic.type {kind!r}; available: {', '.join(IC_TYPES)}")

    return U_h_0
