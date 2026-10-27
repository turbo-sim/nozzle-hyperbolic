"""Analytical nozzle profiles and initial states for two-fluid verification.

Includes the dominant-phase isentropic and incompressible Bernoulli limits.
The initial-state builders return both the six-variable two-fluid state and a
reference profile; subsonic_profile provides the plotting reference without
constructing an initial state.
"""

import numpy as np
import jax.numpy as jnp
import jaxprop as jxp


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


def _area_ratio(M, g):
    """A/A* from the area-Mach relation (inverse of _area_mach)."""
    return (1.0 / M) * ((2.0 / (g + 1.0))
                        * (1.0 + 0.5 * (g - 1.0) * M * M)
                        ) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def subsonic_profile(x, A, *, p0, T0, p_out, gamma, R):
    """Analytic UNCHOKED subsonic isentropic profile set by the back-pressure.

    M_exit from p0/p_out, virtual sonic area A* = A_exit / (A/A*)(M_exit), then
    the subsonic area-Mach branch at every station.  Returns None (with a note)
    if the pressure ratio would choke the nozzle, i.e. if the required A* is
    larger than the actual throat area.
    """
    g = gamma
    if not (p_out < p0):
        print("    subsonic reference: p_out >= p0 — skipped")
        return None
    M_e = np.sqrt(2.0 / (g - 1.0) * ((p0 / p_out) ** ((g - 1.0) / g) - 1.0))
    if M_e >= 1.0:
        print(f"    subsonic reference: exit would be supersonic "
              f"(M_exit={M_e:.3f}) — skipped")
        return None
    A_star = float(A[-1]) / _area_ratio(M_e, g)
    if A_star > float(A.min()) * (1.0 + 1e-12):
        print(f"    subsonic reference: back-pressure chokes the nozzle "
              f"(A*={A_star:.4e} > A_throat={A.min():.4e}) — skipped")
        return None
    M = np.array([_area_mach(a / A_star, g, "sub") for a in A])
    T = T0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    p = p0 * (T / T0) ** (g / (g - 1.0))
    u = M * np.sqrt(g * R * T)
    print(f"    subsonic reference: M_exit={M_e:.4f}  M_throat={M.min():.4f}"
          f"..{M.max():.4f}  A*/A_throat={A_star/float(A.min()):.4f}")
    return dict(x=x, A=A, M=M, p=p, T=T, u=u, gamma=g, R=R)

