"""
Analytic incompressible (Bernoulli) initial condition for the two-fluid solver.

Purpose
-------
A self-contained, geometry-agnostic cold-start IC for the LOW-SPEED LIQUID regime
(the flashing-inlet regime) that needs NO nozzlex solve.  For a nearly
incompressible dominant phase the exact steady quasi-1D solution is

    mass    :  u_dom(x) = u_in * A_in / A(x)                    (rho ~ const)
    Bernoulli: p(x)     = p0 - 1/2 * rho_dom * u_dom(x)^2       (reservoir p0)

which is exact to O(Ma^2).  This same profile doubles as the ANALYTIC REFERENCE
the converged two-fluid solution is validated against (the ghost limit must
reproduce single-phase incompressible nozzle flow).

Ghost (passive) phase
---------------------
The near-massless second phase (alpha ~ 1e-4) is given, per the physical picture:
  * constant void fraction  alpha_g = alpha_g_in,
  * NO SLIP by default (slip=1 -> u_ghost = u_dom); general slip S = u_g/u_l,
  * SAME temperature T0 as the dominant phase, ISOTHERMAL along x, so its
    enthalpy h_ghost(x) = h(p(x), T0) via PT_INPUTS is always a valid in-table
    EOS state (no risk of driving the ghost off its table on step 0).

Both phases are built ISOTHERMAL at T0 (the flow is ~isothermal at low Ma), so
h_i(x) = h_i(p(x), T0_i) via PT_INPUTS -> every cell is a valid EOS state.  The
tiny H0 = h + u^2/2 drift this implies (order 1/2 u^2, a few J/kg at low speed)
relaxes away in the first transient.

Returns  U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]  shape (Nx, 6).
"""

import numpy as np
import jax.numpy as jnp
import jaxprop as jxp


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
