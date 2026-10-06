"""
Analytic isentropic (area-Mach) initial condition for the two-fluid solver.

Purpose
-------
GAS-DOMINANT mirror of bernoulli_ic: a self-contained cold-start IC for the
compressible perfect-expansion regime (alpha_g ~ 1 - 1e-4, choked C-D nozzle,
shock-free supersonic diverging section).  For a perfect-gas dominant phase the
exact steady quasi-1D solution is the area-Mach relation

    A/A* = (1/M) [ (2/(g+1)) (1 + (g-1)/2 M^2) ]^((g+1)/(2(g-1)))

with the SUBSONIC branch upstream of the throat and the SUPERSONIC branch
downstream (perfect/adapted expansion), and

    T = T0 / (1 + (g-1)/2 M^2),   p = p0 (T/T0)^(g/(g-1)),   u = M sqrt(g R T).

This profile doubles as the ANALYTIC REFERENCE the converged two-fluid solution
is validated against (check_isentropic.py), including the choked mass flux

    mdot* = A* p0 sqrt(g/(R T0)) (2/(g+1))^((g+1)/(2(g-1))) .

Ghost (passive) phase
---------------------
Near-massless second phase (alpha_l ~ 1e-4): constant void fraction, co-flowing
(slip S = u_g/u_l), enthalpy from (p, T_gas) per its own EOS — with SRC on the
drag+heat slave it to the gas along the whole expansion.  Use an UNBOUNDED
(perfect-gas) backend for the ghost: the gas T drops to ~0.5 T0 at the exit,
far off any liquid table.

gamma and R are DERIVED from fluid_g at (p0, T0) (R = p0/(rho0 T0),
gamma = a0^2/(R T0)) so the reference is self-consistent with the solver's EOS.

Returns  U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]  shape (Nx, 6).
"""

import numpy as np
import jax.numpy as jnp
import jaxprop as jxp


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
