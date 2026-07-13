# %%
"""
Two-fluid FVS solver — pipe phase-change test case.

Initial condition: uniform two-phase water mixture at 10 bar with the liquid
phase in a superheated (metastable) state (h_l = h_sat_liq + 100 kJ/kg).
With mass-transfer source terms active the liquid should be driven toward
saturation by evaporation.  Without sources (current default) this test verifies
that the numerical scheme handles metastable EOS states correctly.

Plotting mirrors demo_two_fluid_pipe.py (nozzlex space-marching reference):
  - pressure
  - velocities (gas and liquid)
  - temperature (gas and liquid) with T_sat reference
  - void fraction
  - enthalpies (gas and liquid) with h_sat reference lines
  + a separate p-h diagram with the thermodynamic path of each phase

Run from dev_scripts/:
    python main_phase_change.py
"""

# TODO: the clip gives nan. important to check why

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import pathlib
_cache_dir = pathlib.Path(__file__).parent.parent / ".jax_cache"
_cache_dir.mkdir(exist_ok=True)
# Persistent XLA compilation cache — saves compiled kernels to disk so
# subsequent runs skip the slow first-time compilation.
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))

import time as _time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step,
    W_from_U_h_batch,
    U_h_from_W_batch,
    phase_props,
)

import settings_phase_change as cfg
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure,
    D_VE_algebric,
)

# ── Output directory (mirrors single-phase main.py) ───────────────────────────
# One folder per run, tagged with the key options + a timestamp so successive
# runs don't clobber each other.  A copy of the settings YAML is saved inside so
# the exact options used are always recoverable.
import shutil
import datetime as _dt

_here     = pathlib.Path(__file__).parent
_yaml_src = _here / "settings_phase_change.yaml"
_stamp    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name  = f"phase_change_N{cfg.Nx}_{cfg.RECON}_CFL{cfg.CFL:g}_{_stamp}"
out_dir   = _here / "results" / run_name
out_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(_yaml_src, out_dir / "settings_phase_change.yaml")
print(f"Output directory: {out_dir.resolve()}")

# ── EOS tables ────────────────────────────────────────────────────────────────
print("Building / loading EOS tables...")
fluid_g = jxp.FluidBicubic(**cfg.LUT_G)
fluid_l = jxp.FluidBicubic(**cfg.LUT_L)
print("  Done.")

# ── Grid ─────────────────────────────────────────────────────────────────────
x  = np.linspace(cfg.x1, cfg.x2, cfg.Nx)
dx = float(x[1] - x[0])

# Geometry: same mechanism as single-phase main.py — dispatch the named factory
# in geometry.py with its kwargs, so nozzle cases reuse the SAME geometry file.
_geo_fn = getattr(_geom, cfg.GEO_FUNCTION)
geo     = _geo_fn(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A     = np.asarray(geo.A,    dtype=float)
dAdx  = np.asarray(geo.dAdx, dtype=float)
A_face = 0.5 * (A[:-1] + A[1:])

print(f"Grid: Nx={cfg.Nx}  dx={dx:.4f} m  geometry={cfg.GEO_FUNCTION}  "
      f"A: {A.min():.4e}..{A.max():.4e} m^2")

# ── Initial condition — uniform superheated-liquid ────────────────────────────
_p0     = jnp.array(cfg.P_INIT)
_sat_g  = fluid_g.get_state_saturation(_p0)
_sat_l  = fluid_l.get_state_saturation(_p0)

h_g_sat = float(_sat_g["enthalpy"])
h_l_sat = float(_sat_l["enthalpy"])
T_sat_0 = float(_sat_g["temperature"])

h_g_0 = h_g_sat + cfg.H_G_OFFSET
h_l_0 = h_l_sat + cfg.H_L_OFFSET

print(
    f"\nSaturation at {cfg.P_INIT/1e5:.1f} bar:  T_sat={T_sat_0:.1f} K"
    f"  h_g_sat={h_g_sat/1e3:.1f} kJ/kg  h_l_sat={h_l_sat/1e3:.1f} kJ/kg"
)
print(
    f"Enthalpy offsets:  dh_g={cfg.H_G_OFFSET/1e3:+.1f} kJ/kg"
    f"  dh_l={cfg.H_L_OFFSET/1e3:+.1f} kJ/kg"
    f"\n  -> h_g={h_g_0/1e3:.1f}  h_l={h_l_0/1e3:.1f} kJ/kg"
)

# Liquid temperature from EOS to confirm superheated state
_state_l0 = fluid_l.get_state(
    jxp.HmassP_INPUTS,
    jnp.array([h_l_0]),
    jnp.array([cfg.P_INIT]),
)
T_l_0 = float(_state_l0["temperature"][0])
print(
    f"  Liquid at IC:  T_l={T_l_0:.1f} K"
    f"  (dT above T_sat = {T_l_0 - T_sat_0:+.1f} K  -> metastable superheated liquid)"
)

U_h_0 = np.tile(
    [cfg.P_INIT, cfg.U_G_INIT, cfg.U_L_INIT, cfg.ALPHA_G, h_g_0, h_l_0],
    (cfg.Nx, 1),
).astype(float)
print(
    f"IC:  p={cfg.P_INIT/1e5:.1f} bar  alpha_g={cfg.ALPHA_G:.2f}"
    f"  u_g={cfg.U_G_INIT:.1f} m/s  u_l={cfg.U_L_INIT:.1f} m/s"
)

# ── nozzlex-seeded IC (override interior with the relaxed space-marching profile)
# The nozzlex INLET matches the uniform IC above, so the inlet BC/prints stay
# valid; only the interior is replaced with the on-slow-manifold profile.
_nozzlex_prof = None
if getattr(cfg, "IC_TYPE", "uniform") == "nozzlex":
    from nozzlex_ic import nozzlex_initial_condition
    print("\nSeeding IC from nozzlex space-marching solve...")
    U_h_0, _nozzlex_prof = nozzlex_initial_condition(
        x, fluid_g, fluid_l,
        p0_in=cfg.P_INIT, T0_in=cfg.NZ_T0_IN,
        u_g_in=cfg.U_G_INIT, u_l_in=cfg.U_L_INIT,
        alpha_g_in=cfg.ALPHA_G, dh_l=cfg.H_L_OFFSET,
        length=float(cfg.x2), Nb=cfg.BUBBLE_NUMBER, Nd=cfg.DROPLET_NUMBER,
        Ma_in=cfg.NZ_MA_IN, solver_name=cfg.NZ_SOLVER,
        rtol=cfg.NZ_RTOL, atol=cfg.NZ_ATOL,
        nsteps=cfg.NZ_NSTEPS, n_eval=cfg.NZ_N_EVAL,
    )
    U_h_0 = np.asarray(U_h_0, dtype=float)
    print(f"  nozzlex IC:  alpha_g {U_h_0[:,3].min():.3f}–{U_h_0[:,3].max():.3f}"
          f"  p {U_h_0[:,0].min()/1e5:.2f}–{U_h_0[:,0].max()/1e5:.2f} bar"
          f"  u_g {U_h_0[:,1].min():.2f}–{U_h_0[:,1].max():.2f} m/s"
          f"  h_l {U_h_0[:,5].min()/1e3:.1f}–{U_h_0[:,5].max()/1e3:.1f} kJ/kg")

W_0 = W_from_U_h_batch(jnp.array(U_h_0), fluid_g, fluid_l)
print(f"W_0 shape: {W_0.shape}   dtype: {W_0.dtype}")

# Round-trip sanity check
U_h_check = np.array(U_h_from_W_batch(W_0))
max_err = np.max(np.abs(U_h_check - U_h_0) / (np.abs(U_h_0) + 1e-300))
print(f"Round-trip error  U_h -> W -> U_h:  max relative = {max_err:.2e}  (expected ~1e-15)")

# ── Source-term closure ───────────────────────────────────────────────────────
# Wraps two_fluid_closure (drag, HTC, mass transfer) + D_VE_algebric.
# fluid_g, fluid_l, and cfg.K_VM are captured from the enclosing scope.
# sat_g / sat_l are canonical-key dicts from fluid.get_state_saturation(p).
#
# SRC_SCALE — Diagnostic knob: multiply the whole D_VE source by this factor.
#   0.0 = source OFF  → isolates flux + BC (localizes whether the outlet crash is
#         source-driven, since flux-only is known stable);
#   1.0 = physical;  0.3 = weak source (does the spike still crash the last cell?).
SRC_SCALE = 1.0
print(f"[Diagnostic] SRC_SCALE = {SRC_SCALE}")

def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
    closure = two_fluid_closure(
        u_h, eos_g, eos_l, fluid_g, fluid_l,
        T_sat   = sat_g["temperature"],
        h_sat_g = sat_g["enthalpy"],
        h_sat_l = sat_l["enthalpy"],
        dlnA_dx = dlnA,
        Nb      = cfg.BUBBLE_NUMBER,
        Nd      = cfg.DROPLET_NUMBER,
    )
    return SRC_SCALE * D_VE_algebric(u_h, cq, closure, eos_g, eos_l, cfg.K_VM)

# ── Inlet BC conditions ───────────────────────────────────────────────────────
# total_state : impose stagnation enthalpies H_k0 = h_k + u_k^2/2 + alpha_g0,
#               extrapolate p, u from the interior.
# static_state: impose the full fixed static inlet state (mirrors the demo):
#               p, u_g, u_l, alpha_g, h_g=h_sat_g(p), h_l=h_sat_l(p)+dh_l.
#               This is exactly the IC row U_h_0[0].
H_g0      = h_g_0 + 0.5 * cfg.U_G_INIT**2
H_l0      = h_l_0 + 0.5 * cfg.U_L_INIT**2
alpha_g0  = cfg.ALPHA_G
U_h_inlet = jnp.array([cfg.P_INIT, cfg.U_G_INIT, cfg.U_L_INIT, cfg.ALPHA_G, h_g_0, h_l_0])
print(f"\nInlet BC: {cfg.BC_INLET}   Outlet BC: {cfg.BC_OUTLET}"
      + (f"  (p_out={cfg.P_OUT/1e5:.2f} bar)" if cfg.BC_OUTLET == "pressure" else ""))
if cfg.BC_INLET == "subsonic_inflow":
    print(f"  impose u_g={cfg.U_G_INIT:.1f} u_l={cfg.U_L_INIT:.1f} m/s  alpha_g={cfg.ALPHA_G:.2f}"
          f"  h_g={h_g_0/1e3:.1f} h_l={h_l_0/1e3:.1f} kJ/kg  (p extrapolated)")
if cfg.BC_INLET == "total_state":
    print(f"  H_g0={H_g0/1e3:.1f} kJ/kg  H_l0={H_l0/1e3:.1f} kJ/kg  alpha_g0={alpha_g0:.2f}")
elif cfg.BC_INLET == "static_state":
    print(f"  p={cfg.P_INIT/1e5:.1f} bar  u_g={cfg.U_G_INIT:.1f}  u_l={cfg.U_L_INIT:.1f} m/s"
          f"  alpha_g={cfg.ALPHA_G:.2f}  h_g={h_g_0/1e3:.1f} (sat)  h_l={h_l_0/1e3:.1f} kJ/kg")

# ── Diagnostic 1 toggles ──────────────────────────────────────────────────────
# PROJECT : post-step EOS-manifold projection in solver.step() (True = on).
#   Run with PROJECT=False FIRST to see the RAW dynamics — where p/h_g/h_l leave
#   the table and rel_eos drifts (the underlying trigger), WITHOUT the projection
#   ↔ adaptive-dt interaction.  Then flip True to see the projection's effect.
# MONITOR : per-step min/max of p, h_g, h_l (flags "!" at table bounds) + rel_eos.
PROJECT = True
MONITOR = False
print(f"\n[Diagnostic 1]  PROJECT={PROJECT}  MONITOR={MONITOR}")

# ── Solver factory ────────────────────────────────────────────────────────────
print(f"Building solver (first JIT compilation may take a minute)... "
      f"jacobian={cfg.JACOBIAN}")
_ret = make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx, dx,
    CFL           = cfg.CFL,
    k             = cfg.K_VM,
    delta_factor  = cfg.DELTA_FACTOR,
    newton_iter   = cfg.NEWTON_ITER,
    recon         = cfg.RECON,
    sources_fn    = _sources_fn,
    debug_newton  = cfg.DEBUG_NEWTON,
    jacobian_mode = cfg.JACOBIAN,
    linear_solve  = cfg.LINEAR_SOLVE,
    bc_inlet      = cfg.BC_INLET,
    bc_outlet     = cfg.BC_OUTLET,
    H_g0          = H_g0,
    H_l0          = H_l0,
    alpha_g0      = alpha_g0,
    U_h_inlet     = U_h_inlet,
    p_out         = cfg.P_OUT,
    p_out_relax   = cfg.P_OUT_RELAX,
    eos_project   = PROJECT,
    return_debug  = (cfg.JACOBIAN_VALIDATE or cfg.PROFILE),
)

if cfg.JACOBIAN_VALIDATE or cfg.PROFILE:
    step, compute_dt, _resid, _jacfn, _jac_diff = _ret
else:
    step, compute_dt = _ret

# ── Colored-Jacobian sparsity guard ───────────────────────────────────────────
# Compare colored vs dense AD Jacobian at a GRADIENT state (a uniform field hides
# the MUSCL ±2 coupling, so a wrong sw=1 coloring would falsely pass). Must be ~0;
# nonzero ⇒ a non-local term broke the banded sparsity assumption → use dense.
if cfg.JACOBIAN_VALIDATE:
    _s   = np.sin(np.linspace(0.0, 3.0, cfg.Nx))
    _Ug  = U_h_0.copy()
    _Ug[:, 0] *= 1.0 + 0.03 * _s              # p ramp
    _Ug[:, 1] += 0.8 * _s                      # u_g ramp
    _Ug[:, 3]  = np.clip(0.1 + 0.2 * (_s + 1.0), 0.02, 0.9)  # alpha_g ramp
    _Ug[:, 5] += 4.0e4 * _s                    # h_l ramp
    _Wg  = W_from_U_h_batch(jnp.array(_Ug), fluid_g, fluid_l)
    _dtg = jnp.float64(float(compute_dt(_Wg)))
    _d   = float(_jac_diff(_Wg, _dtg))
    print(f"  Jacobian validation (gradient state): max|J_colored - J_dense| = {_d:.3e}")
    if _d > 1e-8:
        print("  WARNING: colored Jacobian differs from dense — banded sparsity "
              "assumption violated (non-local term?). Use jacobian: dense.")

print("  Warming up compute_dt...")
dt0 = float(compute_dt(W_0))
print(f"  dt = {dt0:.4e} s")
print("  Warming up step (triggers full Newton + Jacobian compilation)...")
W_test, ssr_test = step(W_0, jnp.float64(dt0))
print(f"  SSR after first step: {float(ssr_test):.4e}")
print("  Compilation done.")

# ── Diagnostic monitor: per-step min/max p, h_g, h_l vs table bounds + rel_eos ─
# A trailing "!" on a field means it has reached/exceeded a table edge — the
# bicubic then extrapolates → NaN densities → the projection clip fires and the
# state drifts off the manifold.  relEOS=...@k gives the worst EOS-inconsistency
# and its cell.  This pinpoints WHICH variable/cell/step leaves the valid domain.
_HG_LO, _HG_HI = cfg.LUT_G["h_min"], cfg.LUT_G["h_max"]
_HL_LO, _HL_HI = cfg.LUT_L["h_min"], cfg.LUT_L["h_max"]
_P_LO,  _P_HI  = cfg.LUT_G["p_min"], cfg.LUT_G["p_max"]

# Choking probe: frozen acoustic eigenstructure at a given cell.
from nozzle_hyperbolic.fvm_solver.two_fluid.matrices import common_quantities, eigenstructure
def _choking(uh):
    eg = phase_props(fluid_g, uh[0], uh[4]); el = phase_props(fluid_l, uh[0], uh[5])
    cq = common_quantities(uh, eg, el, cfg.K_VM)
    ev = np.array(eigenstructure(uh, eg, el, cq)[0])
    umix = 0.5 * (float(uh[1]) + float(uh[2]))
    a_fr = 0.5 * (ev.max() - ev.min())        # frozen mixture sound speed
    return umix / (a_fr + 1e-30), np.min(np.abs(ev))   # Mach, min|eigenvalue|

def _proj_probe(W_cell):
    """Numpy replica of the solver's per-cell EOS projection — trace why it fails.
    Shows whether V is UNREALIZABLE (|r|/m never →0) or the 2x2 Jacobian is
    SINGULAR (det →0), and what (p, alpha_g) it lands on (the 85.8 bar mystery)."""
    p0, ag0, m_g, u_g, m_l, u_l, mg_eg, ml_el = np.array(W_cell, float)
    def _r(pp, aa):
        al = 1.0 - aa
        hg_ = (mg_eg + pp * aa) / m_g;  hl_ = (ml_el + pp * al) / m_l
        rg_ = float(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array([hg_]), jnp.array([pp]))["density"][0])
        rl_ = float(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array([hl_]), jnp.array([pp]))["density"][0])
        return np.array([aa * rg_ - m_g, al * rl_ - m_l])
    pa = np.array([p0, ag0])
    print(f"    [proj] V: m_g={m_g:.3e} m_l={m_l:.3e} mg_eg={mg_eg:.3e} ml_el={ml_el:.3e}")
    for it in range(8):
        r0 = _r(*pa)
        ep = max(abs(pa[0]) * 1e-6, 1.0); ea = 1e-7
        J = np.column_stack([(_r(pa[0] + ep, pa[1]) - r0) / ep,
                             (_r(pa[0], pa[1] + ea) - r0) / ea])
        det = np.linalg.det(J)
        try: dpa = np.linalg.solve(J, -r0)
        except Exception: dpa = np.zeros(2)
        pa = pa + dpa; pa[1] = np.clip(pa[1], 1e-4, 1 - 1e-4)
        print(f"    [proj] it{it}: p={pa[0]:.4e} ag={pa[1]:.4f} "
              f"|r|/m_g={np.linalg.norm(r0)/(abs(m_g)+1e-30):.2e} detJ={det:.2e}")

def _monitor(W, n):
    U  = np.array(U_h_from_W_batch(W))
    p, hg, hl, ag = U[:, 0], U[:, 4], U[:, 5], U[:, 3]
    Wn = np.array(W); mg = Wn[:, 2]
    rg = np.array(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(hg), jnp.array(p))["density"])
    res = np.abs(mg - ag * rg) / (np.abs(mg) + 1e-300)
    k   = int(np.argmax(res))
    # frozen Mach + degeneracy at the worst cell AND at the outlet cell -2
    Ma_k,  ev_k  = _choking(jnp.array(U[k]))
    Ma_o,  ev_o  = _choking(jnp.array(U[-2]))
    def _f(v, lo, hi): return "!" if (v.min() <= lo or v.max() >= hi) else " "
    print(f"  [mon] n={n:5d}  "
          f"p[{p.min():.3e},{p.max():.3e}]{_f(p,_P_LO,_P_HI)} "
          f"hg[{hg.min():.3e},{hg.max():.3e}]{_f(hg,_HG_LO,_HG_HI)} "
          f"hl[{hl.min():.3e},{hl.max():.3e}]{_f(hl,_HL_LO,_HL_HI)} "
          f"ag[{ag.min():.3f},{ag.max():.3f}]  relEOS={res.max():.2e}@{k}"
          f"  Ma@{k}={Ma_k:.3f} minEV@{k}={ev_k:.1f}"
          f"  Ma@out={Ma_o:.3f} minEV@out={ev_o:.1f}", flush=True)
    # Trace the projection at the worst cell the FIRST time it goes bad, to see
    # why it fails there (unrealizable V vs singular Jacobian vs 85.8-bar landing).
    if res.max() > 0.1 and not getattr(_monitor, "_probed", False):
        _monitor._probed = True
        print(f"  [proj-probe @ cell {k}, step {n}] ─────────────────────────────")
        _proj_probe(W[k])

# ── Per-step cost breakdown (what makes a step slow) ──────────────────────────
# Times the three jitted Newton-inner components (residual, Jacobian, lstsq)
# separately.  A step does newton_iter of each, so per-step ≈ newton_iter × sum.
if cfg.PROFILE:
    _wi   = W_0[1:-1].reshape(-1)
    _wn   = W_0.reshape(-1)
    _dtp  = jnp.float64(dt0)
    _res_j = jax.jit(_resid)
    _jac_j = jax.jit(_jacfn)
    if cfg.LINEAR_SOLVE == "normal_eq":
        import jax.scipy.linalg as _jsl
        def _solve(J, R):
            JJt = J @ J.T
            reg = 1e-10 * jnp.mean(jnp.diagonal(JJt))
            return J.T @ _jsl.solve(JJt + reg*jnp.eye(J.shape[0]), R, assume_a='pos')
        _lstsq_j = jax.jit(_solve)
    elif cfg.LINEAR_SOLVE == "qr":
        import jax.scipy.linalg as _jsl
        def _solve(J, R):
            Q, Rm = jnp.linalg.qr(J.T, mode='reduced')
            return Q @ _jsl.solve_triangular(Rm, R, lower=False, trans='T')
        _lstsq_j = jax.jit(_solve)
    else:
        _lstsq_j = jax.jit(lambda J, R: jnp.linalg.lstsq(J, R, rcond=None)[0])
    _Jc = _jac_j(_wi, _wn, _dtp); jax.block_until_ready(_Jc)   # compile
    _Rc = _res_j(_wi, _wn, _dtp); jax.block_until_ready(_Rc)
    _x  = _lstsq_j(_Jc, -_Rc);    jax.block_until_ready(_x)

    def _bench(fn, reps=8):
        _tb = _time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn())
        return (_time.perf_counter() - _tb) / reps * 1e3   # ms

    t_res = _bench(lambda: _res_j(_wi, _wn, _dtp))
    t_jac = _bench(lambda: _jac_j(_wi, _wn, _dtp))
    t_sol = _bench(lambda: _lstsq_j(_Jc, -_Rc))
    _per_step = (t_res + t_jac + t_sol) * cfg.NEWTON_ITER
    print("\n  ── Per-step cost breakdown ──────────────────────────────────────")
    print(f"    residual      : {t_res:7.1f} ms  × {cfg.NEWTON_ITER} = {t_res*cfg.NEWTON_ITER:7.1f} ms")
    print(f"    jacobian ({cfg.JACOBIAN:<8}): {t_jac:7.1f} ms  × {cfg.NEWTON_ITER} = {t_jac*cfg.NEWTON_ITER:7.1f} ms")
    print(f"    lstsq solve   : {t_sol:7.1f} ms  × {cfg.NEWTON_ITER} = {t_sol*cfg.NEWTON_ITER:7.1f} ms")
    print(f"    -> est. per step (×{cfg.NEWTON_ITER} Newton iters): {_per_step:.0f} ms")
    print( "  ─────────────────────────────────────────────────────────────────")

# ── Time-stepping loop ────────────────────────────────────────────────────────
# Two modes (cfg.ADAPTIVE_DT):
#   adaptive : grow dt toward the CFL cap, then halve-and-retry each step while any
#              primitive increment (Δp/p, Δh/h, Δu/(|u|+a), Δα_g) exceeds its
#              threshold — or the state goes NaN.  Städtke's automatic dt control.
#   fixed    : take the full CFL-capped dt every step, accept unconditionally.
W_n       = W_0
t_phys    = 0.0
n         = 0
ssr       = float("inf")
_win      = cfg.PLATEAU_WINDOW
_ssr_hist = [float("inf")] * _win
_stop_reason = f"max steps ({cfg.N_STEPS}) reached"

_adaptive = cfg.ADAPTIVE_DT
_DT_MIN   = dx / 1e12
dt        = float(compute_dt(W_0))   # current dt target (grows/shrinks adaptively)

print(f"\nRunning (max {cfg.N_STEPS} steps  SSR_tol={cfg.SSR_TOL:.1e}"
      + (f"  t_end={cfg.T_END:.3e} s" if cfg.T_END < float("inf") else "") + ")")
print(f"Adaptive dt limiter: {_adaptive}"
      + (f"  (max_halvings={cfg.MAX_HALVINGS}  dp<{cfg.DP_MAX} dh<{cfg.DH_MAX}"
         f" du<{cfg.DU_MAX} dα<{cfg.DA_MAX}  grow×{cfg.DT_GROW})" if _adaptive else ""))
print(f"{'Step':>6}  {'t [s]':>12}  {'dt [s]':>12}  {'SSR':>12}")
print("-" * 50)


def _increments(W_old, W_new, a_char):
    """Max relative primitive increments over interior cells (Städtke crit. 3)."""
    Uo = U_h_from_W_batch(W_old)[1:-1]
    Un = U_h_from_W_batch(W_new)[1:-1]
    bad = bool(jnp.any(jnp.isnan(Un)) | jnp.any(jnp.isinf(Un)))
    dp  = float(jnp.max(jnp.abs(Un[:, 0] - Uo[:, 0]) / (jnp.abs(Uo[:, 0]) + 1e-10)))
    dug = float(jnp.max(jnp.abs(Un[:, 1] - Uo[:, 1]) / (jnp.abs(Uo[:, 1]) + a_char + 1e-10)))
    dul = float(jnp.max(jnp.abs(Un[:, 2] - Uo[:, 2]) / (jnp.abs(Uo[:, 2]) + a_char + 1e-10)))
    dag = float(jnp.max(jnp.abs(Un[:, 3] - Uo[:, 3])))
    dhg = float(jnp.max(jnp.abs(Un[:, 4] - Uo[:, 4]) / (jnp.abs(Uo[:, 4]) + 1e-10)))
    dhl = float(jnp.max(jnp.abs(Un[:, 5] - Uo[:, 5]) / (jnp.abs(Uo[:, 5]) + 1e-10)))
    return bad, dp, max(dug, dul), max(dhg, dhl), dag


_t0 = _time.perf_counter()
_tm = {'dt': 0.0, 'step': 0.0, 'check': 0.0}   # accumulated wall time
try:
    while n < cfg.N_STEPS:
        halvings = 0
        if _adaptive:
            # ── CFL cap for current state; grow dt toward it ──────────────────
            _ta      = _time.perf_counter()
            dt_cap   = float(compute_dt(W_n))
            a_char   = cfg.CFL * dx / max(dt_cap, 1e-300)   # max wave speed
            dt       = min(dt * cfg.DT_GROW, dt_cap)
            _tm['dt'] += _time.perf_counter() - _ta

            if dt < _DT_MIN:
                _stop_reason = f"dt collapsed to {dt:.2e} (< {_DT_MIN:.2e})"
                break

            # ── Halve-and-retry until increments are within thresholds ────────
            for halvings in range(cfg.MAX_HALVINGS + 1):
                _tb          = _time.perf_counter()
                W_try, ssr_j = step(W_n, jnp.float64(dt))
                jax.block_until_ready(W_try)
                _tc          = _time.perf_counter()
                _tm['step'] += _tc - _tb
                bad, dp, du, dh, da = _increments(W_n, W_try, a_char)
                _tm['check'] += _time.perf_counter() - _tc

                ok = (not bad) and dp < cfg.DP_MAX and dh < cfg.DH_MAX \
                    and du < cfg.DU_MAX and da < cfg.DA_MAX
                if ok:
                    break
                if halvings < cfg.MAX_HALVINGS:
                    dt *= 0.5

            W_n = W_try
            # Newton stalled (max halvings hit): inflate SSR so plateau/threshold
            # stops can't fire on an untrustworthy residual.
            ssr = (max(float(ssr_j), cfg.SSR_TOL * 10)
                   if halvings >= cfg.MAX_HALVINGS else float(ssr_j))
        else:
            # ── Fixed CFL dt, accept unconditionally ──────────────────────────
            _ta         = _time.perf_counter()
            dt          = float(compute_dt(W_n))
            _tb         = _time.perf_counter()
            W_n, ssr_j  = step(W_n, jnp.float64(dt))
            ssr         = float(ssr_j)
            jax.block_until_ready(W_n)
            _tc         = _time.perf_counter()
            _tm['dt']  += _tb - _ta
            _tm['step']+= _tc - _tb

        t_phys   += dt
        n        += 1
        _ssr_hist = _ssr_hist[1:] + [ssr]

        if MONITOR and (n <= 300 or halvings > 0):
            _monitor(W_n, n)

        if n <= 10 or n % cfg.PRINT_EVERY == 0 or halvings > 0:
            _el = _time.perf_counter() - _t0
            _extra = (f"  step={_tm['step']/n*1e3:.0f}ms dt={_tm['dt']/n*1e3:.0f}ms"
                      if cfg.PROFILE else "")
            _hv = (f"  [{halvings} halving(s) dp={dp:.2f} dh={dh:.2f}"
                   f" du={du:.2f} dα={da:.3f}]" if (_adaptive and halvings > 0) else "")
            print(f"{n:>6}  {t_phys:>12.4e}  {dt:>12.4e}  {ssr:>12.4e}"
                  f"  ({_el:.1f}s{_extra}){_hv}", flush=True)

        if ssr < cfg.SSR_TOL:
            _stop_reason = f"SSR={ssr:.3e} < tol={cfg.SSR_TOL:.1e}"
            break

        if n >= _win:
            _hmax = max(_ssr_hist);  _hmin = min(_ssr_hist) + 1e-300
            if _hmax / _hmin < cfg.PLATEAU_RATIO:
                _stop_reason = (f"SSR plateau ({_hmin:.3e}-{_hmax:.3e}"
                                f"  ratio={_hmax/_hmin:.2f})")
                break

        if t_phys >= cfg.T_END:
            _stop_reason = f"t_end={cfg.T_END:.3e} s reached"
            break

except KeyboardInterrupt:
    _stop_reason = "interrupted by user"

_elapsed = _time.perf_counter() - _t0
print(f"\nStopped: {_stop_reason}")
print(f"  steps={n}  t={t_phys:.3e} s  SSR={ssr:.3e}  wall={_elapsed:.1f} s")
if cfg.PROFILE and n > 0:
    print(f"  wall breakdown:  step {_tm['step']:.1f}s ({_tm['step']/n*1e3:.0f} ms/step)"
          f"  |  compute_dt {_tm['dt']:.1f}s ({_tm['dt']/n*1e3:.0f} ms/step)"
          f"  |  incr-check {_tm['check']:.1f}s ({_tm['check']/n*1e3:.0f} ms/step)")

# ── Post-processing ───────────────────────────────────────────────────────────
U_h_f   = np.array(U_h_from_W_batch(W_n))
p_f     = U_h_f[:, 0]
u_g_f   = U_h_f[:, 1]
u_l_f   = U_h_f[:, 2]
alpha_g = U_h_f[:, 3]
h_g_f   = U_h_f[:, 4]
h_l_f   = U_h_f[:, 5]

# ── EOS-consistency check ─────────────────────────────────────────────────────
# With the post-step projection in solver.step(), W must stay on the EOS manifold
# m_g = alpha_g*rho_g(p,h_g), m_l = (1-alpha_g)*rho_l(p,h_l).  rel_eos ~ machine
# zero means the plotted (p, alpha_g, h) are physically consistent with the
# conserved masses; a large value means the solution has drifted and the profiles
# below are NOT trustworthy (was ~0.55 before the projection fix — see test_T3b).
_Wn    = np.array(W_n)
_mg    = _Wn[:, 2]; _ml = _Wn[:, 4]
_rho_g = np.array(fluid_g.get_state(jxp.HmassP_INPUTS, jnp.array(h_g_f), jnp.array(p_f))["density"])
_rho_l = np.array(fluid_l.get_state(jxp.HmassP_INPUTS, jnp.array(h_l_f), jnp.array(p_f))["density"])
_res_g = _mg - alpha_g * _rho_g
_res_l = _ml - (1.0 - alpha_g) * _rho_l
_rel_g = np.sqrt(np.mean(_res_g**2)) / (np.sqrt(np.mean(_mg**2)) + 1e-300)
_rel_l = np.sqrt(np.mean(_res_l**2)) / (np.sqrt(np.mean(_ml**2)) + 1e-300)
print(f"\nEOS consistency:  rel_eos_g={_rel_g:.2e}  rel_eos_l={_rel_l:.2e}  "
      f"max|m_g-a*rho_g|={np.max(np.abs(_res_g)):.2e}")
print("  " + ("OK — solution is on the EOS manifold." if max(_rel_g, _rel_l) < 1e-8
              else "WARNING: W drifted off the EOS manifold — profiles NOT trustworthy."))

# Temperature and saturation properties at each cell (via EOS)
print("\nComputing temperatures and saturation props from EOS...")
T_g_f   = np.array(fluid_g.get_state(
    jxp.HmassP_INPUTS, jnp.array(h_g_f), jnp.array(p_f))["temperature"])
T_l_f   = np.array(fluid_l.get_state(
    jxp.HmassP_INPUTS, jnp.array(h_l_f), jnp.array(p_f))["temperature"])

# Saturation lines at each cell's current pressure
h_sat_g_f = np.array([float(fluid_g.get_state_saturation(jnp.array(p))["enthalpy"])
                       for p in p_f])
h_sat_l_f = np.array([float(fluid_l.get_state_saturation(jnp.array(p))["enthalpy"])
                       for p in p_f])
T_sat_f   = np.array([float(fluid_g.get_state_saturation(jnp.array(p))["temperature"])
                       for p in p_f])

# ── Mass source term per cell (sigma_Mg = evaporation rate) ───────────────────
# Recompute the mass-transfer closure at each converged cell state so we can
# plot it exactly like the space-marching demo's "Mass source term" panel.
import jax
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import two_fluid_closure as _tfc

def _mdot_cell(u_h):
    eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
    eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
    sat_g_c = fluid_g.get_state_saturation(u_h[0])
    sat_l_c = fluid_l.get_state_saturation(u_h[0])
    cl = _tfc(
        u_h, eos_g_c, eos_l_c, fluid_g, fluid_l,
        T_sat=sat_g_c["temperature"], h_sat_g=sat_g_c["enthalpy"],
        h_sat_l=sat_l_c["enthalpy"], dlnA_dx=0.0,
        Nb=cfg.BUBBLE_NUMBER, Nd=cfg.DROPLET_NUMBER,
    )
    return cl.sigma_Mg

m_dot_lv = np.array(jax.vmap(_mdot_cell)(jnp.array(U_h_f)))

# ── Save solution to CSV (mirrors single-phase main.py) ───────────────────────
csv_header = ("x,p,u_g,u_l,alpha_g,h_g,h_l,T_g,T_l,T_sat,"
              "h_sat_g,h_sat_l,m_dot_lv")
csv_data   = np.column_stack([
    x, p_f, u_g_f, u_l_f, alpha_g, h_g_f, h_l_f,
    T_g_f, T_l_f, T_sat_f, h_sat_g_f, h_sat_l_f, m_dot_lv,
])
sol_path = out_dir / "solution.csv"
np.savetxt(sol_path, csv_data, delimiter=",", header=csv_header, comments="")
print(f"Solution saved → {sol_path}")

# ── Main 5-panel figure (layout mirrors nozzlex demo_two_fluid_pipe.py) ───────
#   1 pressure | 2 velocity | 3 mass source | 4 temperature | 5 void fraction
# vapour = red, liquid = blue (same colour convention as the demo).
fig, axes = plt.subplots(5, 1, figsize=(5, 10), sharex=True)
fig.suptitle(
    f"Two-fluid pipe (time-marching)  water  {cfg.P_INIT/1e5:.0f} bar"
    f"  dh_l={cfg.H_L_OFFSET/1e3:+.0f} kJ/kg  N={cfg.Nx}  steps={n}",
    fontsize=9,
)
xf = x   # plot the full domain

# Row 1 — pressure (bar)
axes[0].plot(xf, p_f / 1e5, color="purple", lw=1.0, marker="o", markersize=3)
axes[0].set_ylabel("Pressure (bar)")

# Row 2 — velocities (vapour 'o', liquid '^')
axes[1].plot(xf, u_g_f, color="red",  marker="o", markersize=3, lw=1.0, label=r"$u_g$ (vapour)")
axes[1].plot(xf, u_l_f, color="blue", marker="^", markersize=3, lw=1.0, label=r"$u_l$ (liquid)")
axes[1].set_ylabel("Velocity (m/s)")
axes[1].legend(fontsize="x-small")

# Row 3 — mass source term (evaporation rate sigma_Mg)
axes[2].plot(xf, m_dot_lv, color="purple", marker="o", markersize=3, lw=1.0)
axes[2].set_ylabel("Mass source term")

# Row 4 — temperature (vapour red, liquid blue, T_sat dashed)
axes[3].plot(xf, T_g_f,   color="red",    lw=1.0, label=r"$T_g$ (vapour)")
axes[3].plot(xf, T_l_f,   color="blue",   lw=1.0, label=r"$T_l$ (liquid)")
axes[3].plot(xf, T_sat_f, color="purple", lw=0.8, ls="--", label=r"$T_\mathrm{sat}(p)$")
axes[3].set_ylabel("Temperature")
axes[3].legend(fontsize="x-small")

# Row 5 — volume fraction (vapour red, liquid blue)
axes[4].plot(xf, alpha_g,       color="red",  marker="o", markersize=3, lw=1.0, label=r"$\alpha_g$ (vapour)")
axes[4].plot(xf, 1.0 - alpha_g, color="blue", marker="o", markersize=3, lw=1.0, label=r"$\alpha_l$ (liquid)")
axes[4].set_ylabel("Vol. Fraction")
axes[4].legend(fontsize="x-small")

axes[-1].set_xlabel("x [m]")
plt.tight_layout()
out_main = out_dir / "phase_change_result.png"
fig.savefig(out_main, dpi=150, bbox_inches="tight")
print(f"Main plot saved: {out_main}")

# ── Temperature-departure-from-saturation diagnostic ──────────────────────────
# Panel 1: T_k - T_sat(p) per phase — how far each phase is from equilibrium.
#          Gas superheat (T_g - T_sat > 0) is the thermal-non-equilibrium mode
#          under investigation; liquid superheat (T_l - T_sat > 0) is metastable.
# Panel 2: pressure along x — surfaces the pressure drift that shifts T_sat(p)
#          (the concern flagged for later debugging: sat quantities move with p).
dT_g = T_g_f - T_sat_f
dT_l = T_l_f - T_sat_f

fig_dt, ax_dt = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
fig_dt.suptitle(
    f"Departure from saturation  water  N={cfg.Nx}  steps={n}", fontsize=9)

ax_dt[0].axhline(0.0, color="purple", lw=0.8, ls="--", label=r"$T_\mathrm{sat}(p)$")
ax_dt[0].plot(xf, dT_g, color="red",  marker="o", markersize=3, lw=1.0,
              label=r"$T_g - T_\mathrm{sat}$ (vapour superheat)")
ax_dt[0].plot(xf, dT_l, color="blue", marker="^", markersize=3, lw=1.0,
              label=r"$T_l - T_\mathrm{sat}$ (liquid superheat)")
ax_dt[0].set_ylabel(r"$T_k - T_\mathrm{sat}$  [K]")
ax_dt[0].legend(fontsize="x-small")

ax_dt[1].plot(xf, p_f / 1e5, color="purple", marker="o", markersize=3, lw=1.0)
ax_dt[1].axhline(cfg.P_INIT / 1e5, color="gray", lw=0.8, ls=":",
                 label=f"inlet {cfg.P_INIT/1e5:.1f} bar")
ax_dt[1].set_ylabel("Pressure [bar]")
ax_dt[1].set_xlabel("x [m]")
ax_dt[1].legend(fontsize="x-small")

plt.tight_layout()
out_dt = out_dir / "phase_change_dT_sat.png"
fig_dt.savefig(out_dt, dpi=150, bbox_inches="tight")
print(f"Temperature-departure plot saved: {out_dt}")

# ── p-h diagram ───────────────────────────────────────────────────────────────
try:
    fluid_ph = jxp.Fluid(cfg.FLUID_NAME, "HEOS")
    fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
        x_prop="h", y_prop="p",
        x_scale="linear", y_scale="log",
        plot_critical_point=True, plot_quality_isolines=True,
    )

    # Mark IC state
    ax_ph.plot(h_g_0, cfg.P_INIT, "r*", markersize=10, zorder=6, label="IC gas")
    ax_ph.plot(h_l_0, cfg.P_INIT, "b*", markersize=10, zorder=6, label="IC liquid")

    ax_ph.plot(h_g_f,  p_f, "-o", color="red",  markersize=3, label="vapour path")
    ax_ph.plot(h_l_f,  p_f, "-s", color="blue", markersize=3, label="liquid path")

    # Overlay the nozzlex space-marching reference the IC was seeded from
    if _nozzlex_prof is not None:
        ax_ph.plot(_nozzlex_prof["h_g"], _nozzlex_prof["p"], "--",
                   color="darkred",  lw=1.0, alpha=0.7, label="nozzlex vapour (seed)")
        ax_ph.plot(_nozzlex_prof["h_l"], _nozzlex_prof["p"], "--",
                   color="navy",     lw=1.0, alpha=0.7, label="nozzlex liquid (seed)")

    ax_ph.set_xlabel("Specific enthalpy [J/kg]")
    ax_ph.set_ylabel("Pressure [Pa]")
    ax_ph.legend(fontsize=7)
    fig_ph.suptitle("p-h diagram — phase-change pipe")
    fig_ph.tight_layout()
    out_ph = out_dir / "phase_change_ph_diagram.png"
    fig_ph.savefig(out_ph, dpi=150, bbox_inches="tight")
    print(f"p-h diagram saved: {out_ph}")
except Exception as _e:
    print(f"p-h diagram skipped: {_e}")

plt.show()

# ── Mass-flux diagnostics ─────────────────────────────────────────────────────
from nozzle_hyperbolic.fvm_solver.two_fluid import V_from_W
import jax

V_f   = np.array(jax.vmap(V_from_W)(W_n))
m_g   = V_f[:, 0]
m_l   = V_f[:, 1]
mdot_g = m_g * u_g_f * A
mdot_l = m_l * u_l_f * A

print("\n── Conservative-variable diagnostics (interior cells) ──────────────")
for label, mdot in [("gas   ", mdot_g), ("liquid", mdot_l)]:
    mm  = np.mean(mdot[1:-1])
    rms = np.sqrt(np.mean((mdot[1:-1] - mm)**2)) / (abs(mm) + 1e-300)
    print(f"  Mass flux {label}  mean={mm:.4e} kg/s   RMS variation={rms*100:.3f} %")

# Superheating diagnostic: how far is liquid from saturation?
dh_l = h_l_f - h_sat_l_f
dT_l = T_l_f - T_sat_f
print("\n── Liquid superheat diagnostic (interior) ──────────────────────────")
print(f"  dh_l = h_l - h_sat_l:  mean={np.mean(dh_l[1:-1])/1e3:.2f}  "
      f"max={np.max(dh_l[1:-1])/1e3:.2f}  min={np.min(dh_l[1:-1])/1e3:.2f}  kJ/kg")
print(f"  dT_l = T_l - T_sat:    mean={np.mean(dT_l[1:-1]):.2f}  "
      f"max={np.max(dT_l[1:-1]):.2f}  min={np.min(dT_l[1:-1]):.2f}  K")
print("────────────────────────────────────────────────────────────────────")
