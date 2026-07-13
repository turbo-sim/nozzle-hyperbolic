"""
Compute a two-fluid time-marching initial condition from a nozzlex space-marching
solve.

Runs nozzlex's `nozzle_two_fluid_one_component_autonomous` pipe model (identical
setup to demo/demo_pipe_nozzlex.py), evaluates the dense solution onto the
time-marching x-grid, and returns

    U_h_0 = [p, u_g, u_l, alpha_g, h_g, h_l]   shape (Nx, 6)

ready for `W_from_U_h_batch`.  The nozzlex INLET state (p0_in, alpha_g_in, u_in,
h_g=h_sat_g, h_l=h_sat_l+dh_l) matches the uniform IC used before — this module
only replaces the *interior* with the relaxed downstream profile, so the existing
inlet BC stays valid.

Why: seeding the transient with nozzlex's already-relaxed profile starts the
solve ON the slow manifold instead of slamming a deeply metastable uniform state
into every cell at once (which lights up the zero-slip checkerboard).
"""

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx
import optimistix as optx
import jaxprop as jxp

from nozzlex.two_fluid.nozzle_model_solver import NozzleParams, replace_param
from nozzlex.two_fluid.nozzle_model_core import (
    get_pipe,
    nozzle_two_fluid_one_component_autonomous,
)


def _solve_pipe(
    fluid_g, fluid_l, *,
    p0_in, T0_in, u_g_in, u_l_in, alpha_g_in, dh_l, length,
    Nb, Nd, Ma_in, solver_name, rtol, atol, nsteps,
):
    """Run the nozzlex pipe IVP and return (sol, Y_ref, params_model).

    Faithful port of demo_pipe_nozzlex.transonic_nozzle_one_component_lax, with
    the inlet superheat (dh_l), void fraction and velocities exposed as arguments.
    """
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


# ---------------------------------------------------------------------------
# Consume a SAVED nozzlex solution pickle (no re-solve) → time-marching IC.
# ---------------------------------------------------------------------------
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
