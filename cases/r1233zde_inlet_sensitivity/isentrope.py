"""
Metastable single-phase isentrope for the stagnation-inlet pressure closure.
===========================================================================

The two-fluid stagnation inlet closes the shared static pressure on the
DOMINANT phase's isentrope: it needs h_dom(p) along s_dom = s_dom0.

Why not CoolProp's PSmass call
------------------------------
CoolProp returns the EQUILIBRIUM state, so as soon as the isentrope crosses the
saturation line it hands back a two-phase MIXTURE enthalpy - which the solver
then uses as the LIQUID-phase enthalpy.  For a subcooled reservoir that is
harmless (the isentrope stays single-phase liquid over a wide pressure range).
For a reservoir sitting exactly ON the saturation line, as in this case, it is
wrong immediately and badly:

    p [bar]   h_equilibrium   h_l,sat(p)   imposed "liquid" superheat
     29.83        416.01        416.01          +0.00 kJ/kg
     29.00        415.90        412.62          +3.28 kJ/kg
     20.00        413.76        375.77         +37.99 kJ/kg
      2.00        386.24        245.50        +140.74 kJ/kg

i.e. the boundary condition itself declares the liquid to be superheated by an
amount that grows without bound as p falls, driving spurious flashing at the
inlet and pushing the state past the spinodal, where the metastable table is
frozen and the Newton system goes singular.

What this module computes instead
---------------------------------
The physically correct object for a two-fluid model is the isentrope of the
liquid AS A SINGLE PHASE - the metastable continuation, not the equilibrium
mixture.  For an isentropic single-phase process

    dh = v dp = dp / rho        (T ds = dh - v dp,  ds = 0)

so the isentrope is the solution of the ODE  dh/dp = 1/rho(p, h)  anchored at
(p0, h0).  Integrating it with the SOLVER'S OWN liquid table makes the inlet
closure exactly consistent with the EOS the solver uses, and monotone by
construction (rho > 0 => dh/dp > 0).

Integration is second-order (midpoint) and marches outward from p0 in both
directions over the supplied pressure grid.
"""

import numpy as np
import jax.numpy as jnp
import jaxprop as jxp


def metastable_isentrope(fluid, p_grid, p_ref, h_ref):
    """h(p) along s = const for a single phase, from dh/dp = 1/rho(p,h).

    Parameters
    ----------
    fluid  : FluidBicubic - the phase's own (metastable) table
    p_grid : (N,) ascending pressures [Pa]
    p_ref  : anchor pressure [Pa]      (must lie within p_grid)
    h_ref  : anchor enthalpy [J/kg] at p_ref

    Returns
    -------
    h : (N,) enthalpy along the isentrope [J/kg]
    """
    p_grid = np.asarray(p_grid, dtype=float)
    h = np.empty_like(p_grid)

    def inv_rho(p, hh):
        st = fluid.get_state(jxp.HmassP_INPUTS,
                             jnp.array([float(hh)]), jnp.array([float(p)]))
        return 1.0 / float(st["density"][0])

    def midpoint(p_from, h_from, p_to):
        dp = p_to - p_from
        k1 = inv_rho(p_from, h_from)
        k2 = inv_rho(p_from + 0.5 * dp, h_from + 0.5 * dp * k1)
        return h_from + dp * k2

    i0 = int(np.argmin(np.abs(p_grid - p_ref)))
    h[i0] = midpoint(p_ref, h_ref, p_grid[i0])
    for i in range(i0 + 1, p_grid.size):            # upward in p
        h[i] = midpoint(p_grid[i - 1], h[i - 1], p_grid[i])
    for i in range(i0 - 1, -1, -1):                 # downward in p
        h[i] = midpoint(p_grid[i + 1], h[i + 1], p_grid[i])
    return h


def _self_test():
    """Validate against CoolProp where the liquid really is single-phase.

    Above p0 the reservoir entropy is BELOW the saturated-liquid entropy, so the
    state is subcooled liquid and CoolProp's equilibrium PSmass call returns the
    genuine liquid branch - a region where the two must agree.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import case_config as cfg
    import CoolProp.CoolProp as CP

    _, fluid_l = cfg.build_fluids(verbose=False)
    p = np.linspace(cfg.P_TABLE_MIN * 1.01, cfg.P_TABLE_MAX * 0.99, 400)
    h = metastable_isentrope(fluid_l, p, cfg.P0, cfg.H_L0)

    print(f"metastable liquid isentrope: h = {h.min()/1e3:.3f} .. "
          f"{h.max()/1e3:.3f} kJ/kg  monotone={bool(np.all(np.diff(h) > 0))}")

    # (a) anchor is reproduced
    print(f"  h at p0: {np.interp(cfg.P0, p, h)/1e3:.4f} kJ/kg  "
          f"(reservoir h_l0 = {cfg.H_L0/1e3:.4f})")

    # (b) agreement with CoolProp in the SUBCOOLED window p > p0
    print("\n  subcooled window (CoolProp PSmass is the true liquid branch here):")
    print("    p[bar]   ODE h    CoolProp h     diff")
    worst = 0.0
    for pq in [30.5, 31.5, 33.0, 34.5, 35.5]:
        hq = np.interp(pq * 1e5, p, h)
        hc = CP.PropsSI("H", "P", pq * 1e5, "S", cfg.S_L0, cfg.FLUID_NAME)
        worst = max(worst, abs(hq - hc))
        print(f"    {pq:6.2f}  {hq/1e3:8.3f}  {hc/1e3:11.3f}  {(hq-hc)/1e3:+8.4f} kJ/kg")
    print(f"    worst deviation: {worst/1e3:.4f} kJ/kg")

    # (c) what it fixes: liquid superheat implied at the inlet, both closures
    print("\n  implied LIQUID superheat h_l(p) - h_l,sat(p):")
    print("    p[bar]   equilibrium (old)   metastable (new)")
    for pq in [29.0, 25.0, 20.0, 10.0, 2.0]:
        P = pq * 1e5
        h_eq = CP.PropsSI("H", "P", P, "S", cfg.S_L0, cfg.FLUID_NAME)
        h_ms = np.interp(P, p, h)
        h_ls = CP.PropsSI("H", "P", P, "Q", 0, cfg.FLUID_NAME)
        print(f"    {pq:6.2f}   {(h_eq-h_ls)/1e3:+13.2f}      {(h_ms-h_ls)/1e3:+12.2f} kJ/kg")


if __name__ == "__main__":
    _self_test()
