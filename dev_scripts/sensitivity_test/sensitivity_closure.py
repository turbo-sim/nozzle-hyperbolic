"""
sensitivity_closure.py

AD sensitivities of integral nozzle quantities to the interphase closure
parameters (N_b, N_d), computed at a CONVERGED steady state by the implicit
function theorem — no re-solve, no differentiation of the time-marching loop.

Why the IFT and not "one AD pass through the solver"
----------------------------------------------------
The solver reaches steady state by implicit time marching (thousands of adaptive
steps).  Differentiating that trajectory would mean unrolling every step:
infeasible.  But at convergence the steady residual satisfies R(W; θ) = 0, so
differentiating the CONSTRAINT rather than the path gives

    ∂R/∂W · dW/dθ + ∂R/∂θ = 0     ⇒     dW/dθ = −(∂R/∂W)⁻¹ ∂R/∂θ
    dJ/dθ = (∂J/∂W)(dW/dθ)

and ∂R/∂W is the Jacobian the Newton solver already assembles every iteration.
The only new object is ∂R/∂θ: two columns of forward-mode AD.  Cost per
sensitivity ≈ one extra linear solve of a system the code already solves several
times per timestep.

Isolating the STEADY Jacobian
-----------------------------
The solver's residual is  R = Vc·dV + dt·fdiv + dt·nc − dt·Vc·src  (solver.py),
i.e. dt MULTIPLIES the steady terms while the transient term Vc·dV carries no
1/dt.  At the fixed point dV = 0 so R = 0 for any dt, but ∂R/∂W keeps the
Vc·dV/dW term.  Dividing by dt and taking dt large suppresses it:
    ∂R_ss/∂W ≈ ∂R/∂W / dt        (dt → large)
DT_SS below sets that dt; --dt-sweep verifies the answer is dt-independent.

Quantities of interest
----------------------
Both are polynomials in W = (p, α_g, m_g, u_g, m_l, u_l, m_g·e_g, m_l·e_l) —
no EOS in the chain, because W carries the partial masses m_i = α_i ρ_i directly:
    mdot    = (m_g·u_g + m_l·u_l)·A        [kg/s]
    x_exit  = m_g u_g / (m_g u_g + m_l u_l)  at the exit cell  (FLOWING quality)

Usage
-----
    poetry run python sensitivity_closure.py <run_dir>
    poetry run python sensitivity_closure.py <run_dir> --dt-sweep

Env:
    DT_SS   steady-Jacobian dt (default 1e6 × the CFL dt)
    STATION QoI station for mdot: 'exit' (default) | 'throat' | 'mean'

This script only READS the run directory and its settings yaml.  Nothing under
nozzle_hyperbolic/ is touched, and no existing config is modified.
"""
import os, sys, glob, pathlib
sys.stdout.reconfigure(encoding="utf-8")
from typing import NamedTuple
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # dev_scripts/ (config_loader)
from config_loader import load_config
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.two_fluid import (
    make_two_fluid_step, W_from_U_h_batch, U_h_from_W_batch)
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure, D_VE_algebric, D_VE_algebric_two_component, D_VE_area)

# ── Knobs ─────────────────────────────────────────────────────────────────────
SRC_SCALE = float(os.environ.get("SRC_SCALE", "1.0"))   # must match the run
AREA_SRC  = True                                        # driver default
# dt_ss = DT_FACTOR × dt_CFL.  1e8 verified: the flashing N100 case converges to 5
# significant figures by 1e6 and is flat from 1e6 to 1e10 (--dt-sweep).
DT_FACTOR = float(os.environ.get("DT_SS", "1e8"))
STATION   = os.environ.get("STATION", "mean")           # 'mean' | 'throat' | 'exit'
# Outlet BC override.  IMPORTANT for reproducing OLD runs: on 2026-07-28 the
# 'characteristic' outlet was reimplemented (eigenvector projection) and the
# previous one renamed 'characteristic_old'.  A run recorded BEFORE that date has
# `outlet: characteristic` in its yaml but was actually integrated with the LEGACY
# operator, so reconstructing it verbatim gives a residual that is NOT zero at the
# stored state.  Set BC_OUTLET=characteristic_old for such runs (check 1 below is
# what detects the mismatch).
BC_OUTLET = os.environ.get("BC_OUTLET", "")             # "" → use the yaml's value
DT_SWEEP  = "--dt-sweep" in sys.argv
_PG = ("perfect_gas", "perfectgas", "ideal_gas", "idealgas")
_here = pathlib.Path(__file__).parent


# ── Locate the run ────────────────────────────────────────────────────────────
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
if _args:
    run_dir = pathlib.Path(_args[0])
    if not run_dir.is_absolute():
        run_dir = _here / run_dir
else:
    run_dir = pathlib.Path(sorted(glob.glob(
        str(_here / "results_two_phase" / "*" / "solution.csv")),
        key=os.path.getmtime)[-1]).parent
cfg = load_config(sorted(run_dir.glob("settings*.yaml"))[0])

print("=" * 78)
print(f"Run: {run_dir.name}")
print(f"  mass_transfer={cfg.MASS_TRANSFER}  SRC_SCALE={SRC_SCALE}  "
      f"Nb={cfg.BUBBLE_NUMBER:.3e}  Nd={cfg.DROPLET_NUMBER:.3e}")
print(f"  bc_inlet={cfg.BC_INLET}  bc_outlet={BC_OUTLET or cfg.BC_OUTLET}"
      + ("  [OVERRIDDEN from yaml's "
         f"'{cfg.BC_OUTLET}']" if BC_OUTLET else ""))
if SRC_SCALE == 0.0:
    sys.exit("SRC_SCALE=0 → the interphase closure is OFF, so dR/dtheta is "
             "identically zero and every sensitivity would be a spurious 0.")


# ── Fluids (mirrors main_two_fluid_single_phase.py) ───────────────────────────
def _build_fluid(lut):
    if str(lut.get("backend", "")).lower() in _PG:
        return jxp.FluidPerfectGas(lut["fluid_name"],
                                   T_ref=lut["T_ref"], p_ref=lut["p_ref"]), True
    return jxp.FluidBicubic(**lut), False


fluid_g, _pg_g = _build_fluid(cfg.LUT_G)
fluid_l, _pg_l = _build_fluid(cfg.LUT_L)
# Reservoir (PT) / reverse (PS) evaluator per phase — perfect gas does it
# analytically, a real fluid needs the FluidJAX inverse call.
fl_g_inv = fluid_g if _pg_g else jxp.FluidJAX(cfg.LUT_G["fluid_name"])
fl_l_inv = fluid_l if _pg_l else jxp.FluidJAX(cfg.LUT_L["fluid_name"])

# ── Grid and geometry — rebuilt from cfg (NOT the csv) so A/dAdx are identical
#    to the run's, including the well-balanced dAdx.
x  = np.linspace(cfg.x1, cfg.x2, cfg.Nx)
dx = float(x[1] - x[0])
geo = getattr(_geom, cfg.GEO_FUNCTION)(x, roughness=cfg.GEO_ROUGHNESS, **cfg.GEO_KWARGS)
A      = np.asarray(geo.A, dtype=float)
dAdx   = np.array(geo.dAdx, dtype=float)
A_face = 0.5 * (A[:-1] + A[1:])
dAdx[1:-1] = (A_face[1:] - A_face[:-1]) / dx          # well-balanced form

# ── Stagnation-inlet setup (mirrors the driver) ───────────────────────────────
P0   = cfg.BC_P0 if cfg.BC_P0 is not None else cfg.P_INIT
T_G0 = cfg.BC_T_G0
T_L0 = cfg.BC_T_L0 if cfg.BC_T_L0 is not None else T_G0
SLIP = float(cfg.BC_SLIP)

_stg0 = fl_g_inv.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_G0]))
_stl0 = fl_l_inv.get_state(jxp.PT_INPUTS, jnp.array([P0]), jnp.array([T_L0]))
H_g0 = float(_stg0["enthalpy"][0]); s_g0 = float(_stg0["s"][0])
H_l0 = float(_stl0["enthalpy"][0]); s_l0 = float(_stl0["s"][0])
rho_g0 = float(_stg0["density"][0]); rho_l0 = float(_stl0["density"][0])

# Ghost-vapor reservoir fix for a SUBCOOLED (flashing) reservoir: the equilibrium
# PT call returns the liquid branch for the gas phase, below the vapor table
# floor → take the ghost vapor at SATURATION at p0 instead.
_hg_min = float(getattr(fluid_g, "h_min", -np.inf))
if H_g0 < _hg_min and not _pg_g:
    _satg0 = fluid_g.get_state_saturation(jnp.array([P0]))
    H_g0   = float(_satg0["enthalpy"][0])
    s_g0   = float(_satg0["entropy"][0])
    rho_g0 = float(_satg0["density"][0])
    print(f"  [ghost-vapor fix] using SATURATED VAPOR at p0: "
          f"H_g0={H_g0/1e3:.1f} kJ/kg  s_g0={s_g0:.2f} J/kg/K")

if cfg.BC_QUALITY is not None:
    QUALITY = float(cfg.BC_QUALITY)
    ALPHA_G = 1.0 / (1.0 + (1.0 - QUALITY) / QUALITY * (rho_g0 / rho_l0) * SLIP)
elif cfg.BC_VOID is not None:
    ALPHA_G = float(cfg.BC_VOID)
    QUALITY = 1.0 / (1.0 + (1.0 / ALPHA_G - 1.0) / ((rho_g0 / rho_l0) * SLIP))
else:
    QUALITY, ALPHA_G = None, None

if cfg.DOM_PHASE in ("gas", "liquid"):
    dom = cfg.DOM_PHASE
else:
    dom = "gas" if (QUALITY is not None and QUALITY >= 0.5) else "liquid"

# Reservoir isentrope for the stagnation-inlet pressure closure (robust FluidJAX
# reverse call, same p grid the solver would use).
_iso_h = _iso_p = None
if cfg.BC_INLET in ("stagnation", "stagnation_ramped"):
    _dom_inv = fl_g_inv if dom == "gas" else fl_l_inv
    _s_dom0  = s_g0 if dom == "gas" else s_l0
    _iso_p = np.linspace(float(getattr(fluid_g, "p_min", 1e4)) * 1.01,
                         float(getattr(fluid_g, "p_max", 300e5)) * 0.99, 400)
    _iso_h = np.asarray(_dom_inv.get_state(
        jxp.PSmass_INPUTS, jnp.asarray(_iso_p),
        jnp.full(_iso_p.size, jnp.float64(_s_dom0)))["h"])
    print(f"  isentrope ({dom}): h_iso {_iso_h.min()/1e3:.1f}..{_iso_h.max()/1e3:.1f} kJ/kg"
          f"  (monotone={bool(np.all(np.diff(_iso_h) > 0))})")

# ── Converged state ───────────────────────────────────────────────────────────
D = np.genfromtxt(run_dir / "solution.csv", delimiter=",", names=True)
U_h_star = np.column_stack([D["p"], D["u_g"], D["u_l"],
                            D["alpha_g"], D["h_g"], D["h_l"]])
if U_h_star.shape[0] != cfg.Nx:
    sys.exit(f"grid mismatch: solution.csv has {U_h_star.shape[0]} cells, "
             f"yaml says Nx={cfg.Nx}")
W_star = W_from_U_h_batch(jnp.asarray(U_h_star), fluid_g, fluid_l)
N  = cfg.Nx
Nc = N - 2
A_j = jnp.asarray(A, dtype=jnp.float64)
print(f"  N={N}  alpha_g range: {U_h_star[:,3].min():.6f}..{U_h_star[:,3].max():.6f}")


# ── theta-parametrized source closure ─────────────────────────────────────────
class _AC(NamedTuple):
    dlnA_dx: jnp.ndarray


_D_VE = D_VE_algebric if cfg.MASS_TRANSFER else D_VE_algebric_two_component


def make_sources_fn(theta):
    """Build the driver's source closure with (N_b, N_d) taken from `theta`.

    theta is a TRACED array here, so jacfwd w.r.t. it flows through
    interfacial_area() → the dispersed length scale d → drag and both
    heat-transfer coefficients.
    """
    def _sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
        Dv = jnp.zeros(6)
        if SRC_SCALE != 0.0:
            clo = two_fluid_closure(
                u_h, eos_g, eos_l, fluid_g, fluid_l,
                T_sat   = sat_g["temperature"],
                h_sat_g = sat_g["enthalpy"],
                h_sat_l = sat_l["enthalpy"],
                dlnA_dx = dlnA,
                Nb      = theta[0],
                Nd      = theta[1],
                mass_transfer = cfg.MASS_TRANSFER,
            )
            Dv = Dv + SRC_SCALE * _D_VE(u_h, cq, clo, eos_g, eos_l, cfg.K_VM)
        if AREA_SRC:
            Dv = Dv + D_VE_area(u_h, cq, _AC(dlnA_dx=dlnA), eos_g, eos_l)
        return Dv
    return _sources_fn


def build(theta):
    """Rebuild the solver with theta-dependent sources; return (step, dt_fn, residual, jac).

    The factory itself never touches theta (its float() conversions and the numpy
    isentrope build above are theta-independent), so theta stays a tracer inside
    _sources_fn only — which is what lets this work without touching solver.py.
    """
    return make_two_fluid_step(
        fluid_g, fluid_l, A, A_face, dAdx, dx,
        CFL           = cfg.CFL,
        k             = cfg.K_VM,
        delta_factor  = cfg.DELTA_FACTOR,
        newton_iter   = cfg.NEWTON_ITER,
        recon         = cfg.RECON,
        sources_fn    = make_sources_fn(theta),
        lowmach_diss  = cfg.LOWMACH_DISS,
        lowmach_diss_bc = cfg.LOWMACH_DISS_BC,
        nc_scale      = cfg.NC_SCALE,
        eps_void      = cfg.EPS_VOID,
        eps_void_ucut   = cfg.EPS_VOID_UCUT,
        eps_void_uwidth = cfg.EPS_VOID_UWIDTH,
        eps_void_scale  = cfg.EPS_VOID_SCALE,
        eps_ghost        = cfg.EPS_GHOST,
        eps_ghost_acut   = cfg.EPS_GHOST_ACUT,
        eps_ghost_awidth = cfg.EPS_GHOST_AWIDTH,
        eps_ghost_k2     = cfg.EPS_GHOST_K2,
        eps_ghost_alo    = cfg.EPS_GHOST_ALO,
        ssr_mode         = cfg.SSR_MODE,
        du_blend_lo   = cfg.DU_BLEND_LO,
        du_blend_hi   = cfg.DU_BLEND_HI,
        jacobian_mode = cfg.JACOBIAN,
        linear_solve  = cfg.LINEAR_SOLVE,
        bc_inlet      = cfg.BC_INLET,
        bc_outlet     = (BC_OUTLET or cfg.BC_OUTLET),
        H_g0 = H_g0, s_g0 = s_g0, H_l0 = H_l0, s_l0 = s_l0,
        dom_phase = dom, iso_h = _iso_h, iso_p = _iso_p,
        alpha_g0 = ALPHA_G, quality = QUALITY, slip = SLIP,
        U_h_inlet = jnp.asarray(U_h_star[0]),
        p_out = cfg.P_OUT, p_out_relax = cfg.P_OUT_RELAX,
        eos_project = cfg.EOS_PROJECT, proj_s_ref = cfg.PROJ_S_REF,
        eos_rows = cfg.EOS_ROWS, eos_rows_banded = cfg.EOS_ROWS_BANDED,
        jac_lag = cfg.JAC_LAG, nc_form = cfg.NC_FORM,
        use_saturation = cfg.MASS_TRANSFER,
        w_char = cfg.W_CHAR, alpha_eps = cfg.ALPHA_EPS,
        pin_ghost = cfg.PIN_GHOST, pin_ghost_momentum = cfg.PIN_GHOST_MOMENTUM,
        return_debug = True,
    )


theta0 = jnp.array([float(cfg.BUBBLE_NUMBER), float(cfg.DROPLET_NUMBER)])
W_int0 = W_star[1:-1].reshape(-1)
W_full = W_star.reshape(-1)

print("\nBuilding solver (first JIT compilation may take a minute)...")
step, compute_dt, residual, jac_fn, _jac_diff, dbg = build(theta0)
dt_cfl = float(compute_dt(W_star))
dt_ss  = DT_FACTOR * dt_cfl
print(f"  dt_CFL = {dt_cfl:.4e} s   dt_steady = {dt_ss:.4e} s  (×{DT_FACTOR:g})")


# ── CHECK 1: does the reconstructed operator reproduce the run's SSR? ─────────
_, ssr_here = step(W_star, jnp.float64(dt_cfl))
print(f"\n[check 1] SSR of the reconstructed solver at the loaded state: "
      f"{float(ssr_here):.4e}")
print("          (compare with the run_log's final SSR — they must agree to "
      "within a step; a mismatch means this reconstruction is NOT the operator "
      "the run converged, and everything below would be meaningless)")


# ── The FULL system.  With eos_rows=True the Newton solves 8 rows per cell:
#   R_full = [ _residual (6/cell, the balance laws) ; _eos_residual (2/cell) ]
#   J_full = [ jac_fn     (6/cell)                 ; eos_jac_dense  (2/cell) ]
# against 8 unknowns per cell, i.e. a SQUARE system (solver.py:1974-1981).  Using
# the balance rows alone would leave 6·Nc equations for 8·Nc unknowns — an
# UNDERDETERMINED system whose min-norm dW/dtheta is arbitrary in a 2·Nc-dimensional
# null space, and the resulting "sensitivities" are pure truncation noise.
eos_residual  = dbg["eos_residual"]
eos_jac_dense = dbg["eos_jac_dense"]
if not cfg.EOS_ROWS:
    print("\n  WARNING: this run has eos_rows=False, so the Newton system is\n"
          "  genuinely rectangular and dW/dtheta is only defined up to the\n"
          "  min-norm choice.  The sensitivities below are NOT well posed.")

# Column scaling + row equilibration, exactly as the Newton body does — these are
# exact transformations of a square system (they change conditioning, not the
# answer), and without them cond(J) ~ 1e16 sits at the edge of float64.
_w_char = np.asarray(dbg["w_char"])
w_ref   = np.maximum(np.abs(np.asarray(W_int0)), _w_char)       # (8*Nc,)


def R_full_of(th, W_int_flat, W_full_flat, dt):
    *_, res, _jf, _jd, d = build(th)
    return jnp.concatenate([res(W_int_flat, W_full_flat, dt),
                            d["eos_residual"](W_int_flat)])


R_star = np.asarray(R_full_of(theta0, W_int0, W_full, jnp.float64(dt_ss)))
_nrm_row = np.maximum(np.abs(R_star).max(), 1e-300)
print(f"[check 1b] ||R_full(W*)||_inf = {np.abs(R_star).max():.4e}"
      f"   (absolute; the SSR above is the solver's normalized version)")


# ── CHECK 2: is dR/dtheta actually non-zero? ─────────────────────────────────
print("\n[check 2] forward-AD dR/dtheta (rebuilds the solver under trace)...")
R_theta = np.asarray(jax.jacfwd(R_full_of, argnums=0)(
    theta0, W_int0, W_full, jnp.float64(dt_ss)))           # (8*Nc, 2)
nrm = np.max(np.abs(R_theta), axis=0)
print(f"          ||dR/dN_b||_inf = {nrm[0]:.4e}    "
      f"||dR/dN_d||_inf = {nrm[1]:.4e}   shape={R_theta.shape}")
# The EOS rows are m_i - alpha_i*rho_i(p,h_i): no theta in them, so their
# theta-derivative must be identically zero.  Cheap structural self-check.
_n6 = 6 * Nc
print(f"          balance rows |dR/dtheta| = {np.abs(R_theta[:_n6]).max():.4e}, "
      f"EOS rows = {np.abs(R_theta[_n6:]).max():.4e} (must be exactly 0)")

# ── Flow-regime census — explains any structurally INERT parameter ────────────
# interfacial_area()/the drag closure select on alpha_g with lax.cond at the
# closure bounds (0.3 / 0.7 by default): bubbly below, droplet above, linear blend
# between.  AD returns the SELECTED branch's derivative, so a parameter belonging
# to a branch the flow never enters has an EXACTLY zero sensitivity — a structural
# fact about the regime, not a numerical artefact.  N_b lives in the bubbly branch,
# N_d in the droplet branch.
_ag = U_h_star[1:-1, 3]
_ab, _ad = 0.3, 0.7                       # interfacial_area defaults
_cens = (int((_ag <= _ab).sum()), int(((_ag > _ab) & (_ag < _ad)).sum()),
         int((_ag >= _ad).sum()))
print(f"          regime census over {Nc} interior cells: "
      f"bubbly(a<={_ab})={_cens[0]}  transition={_cens[1]}  droplet(a>={_ad})={_cens[2]}")

PARAM = ["N_b", "N_d"]
inert = [p for p, n in zip(PARAM, nrm) if n == 0.0]
if inert:
    print(f"          NOTE: {', '.join(inert)} is structurally INERT here — the flow "
          f"never enters its regime,\n"
          f"          so its sensitivity is exactly zero and must be reported as "
          f"'n/a (regime)', not 0.")
if not np.any(nrm > 0):
    sys.exit("BOTH parameters are inert — theta never reaches the physics. Check "
             "SRC_SCALE and that make_sources_fn is the closure actually in use.")


# ── Quantities of interest: polynomials in W, no EOS ─────────────────────────
# W = (p, alpha_g, m_g, u_g, m_l, u_l, m_g*e_g, m_l*e_l)
def _qoi(W_int_flat):
    W = W_int_flat.reshape(Nc, 8)
    mflux = W[:, 2] * W[:, 3] + W[:, 4] * W[:, 5]     # m_g u_g + m_l u_l
    mdot_x = mflux * A_j[1:-1]                        # per-cell mass flow
    if STATION == "throat":
        i = int(np.argmin(A[1:-1]))
        mdot = mdot_x[i]
    elif STATION == "mean":
        mdot = jnp.mean(mdot_x)
    else:                                             # 'exit'
        mdot = mdot_x[-1]
    gas = W[-1, 2] * W[-1, 3]                         # exit gas mass flux
    tot = gas + W[-1, 4] * W[-1, 5]
    x_exit = gas / tot
    return jnp.array([mdot, x_exit])


J_star = np.asarray(_qoi(W_int0))
dJ_dW  = np.asarray(jax.jacfwd(_qoi)(W_int0))          # (2, 8*Nc)
print(f"\nQoI at the converged state (station={STATION}):")
print(f"  mdot   = {J_star[0]:.6e} kg/s")
print(f"  x_exit = {J_star[1]:.6e}  (flowing quality)")

# mass-flow station-independence.  RMS variation is the run_log's own convention,
# so this line is directly comparable with its "total mass flux" diagnostic.
_W = np.asarray(W_int0).reshape(Nc, 8)
_mdot_all = (_W[:, 2] * _W[:, 3] + _W[:, 4] * _W[:, 5]) * A[1:-1]
_m_mean = _mdot_all.mean()
print(f"  mdot: mean={_m_mean:.6e}  throat={_mdot_all[int(np.argmin(A[1:-1]))]:.6e}  "
      f"exit={_mdot_all[-1]:.6e} kg/s")
print(f"  mdot RMS variation = {np.sqrt(np.mean((_mdot_all/_m_mean-1)**2))*100:.3f} %"
      f"   (max-min = {(_mdot_all.max()-_mdot_all.min())/_m_mean*100:.3f} %)"
      f"  — compare the run_log's 'total mass flux' line")


# ── IFT solve ─────────────────────────────────────────────────────────────────
def _linearize(dt_use):
    """Full square Jacobian and dR/dtheta at the converged state, SCALED.

    Column scaling by w_ref and row equilibration mirror the Newton body
    (solver.py:1974-1981).  Both are exact for a square system — they fix
    conditioning without changing the solution — and are what makes this solvable
    in float64 at all.  No explicit /dt is needed: at large dt the balance rows of
    both J and dR/dtheta scale with dt, so it cancels in the solve.
    """
    J_bal = np.asarray(jac_fn(W_int0, W_full, jnp.float64(dt_use)))     # (6Nc, 8Nc)
    J_eos = np.asarray(eos_jac_dense(W_int0))                          # (2Nc, 8Nc)
    J = np.concatenate([J_bal, J_eos], axis=0)                         # (8Nc, 8Nc)
    Rth = np.asarray(jax.jacfwd(R_full_of, argnums=0)(
        theta0, W_int0, W_full, jnp.float64(dt_use)))                  # (8Nc, 2)
    J_sc = J * w_ref[None, :]                                          # columns
    r_sc = 1.0 / (np.max(np.abs(J_sc), axis=1) + 1e-300)               # rows
    return J_sc * r_sc[:, None], Rth * r_sc[:, None]


def sensitivities(dt_use, rcond=1e-12, svd=None):
    """dJ/dtheta via the IFT, with an explicit SVD truncation threshold.

    Solves the SCALED system for y = dW/w_ref, then unscales: dW = w_ref*y.
    The truncation must not matter — if the sensitivity moves with rcond, the
    quantity is not determined by the steady state and must not be reported.
    """
    if svd is None:
        J_sc, Rth = _linearize(dt_use)
        svd = (np.linalg.svd(J_sc, full_matrices=False), Rth)
    (U, S, Vt), Rth = svd
    keep = S > rcond * S[0]
    y = -(Vt[keep].T @ ((U[:, keep].T @ Rth) / S[keep, None]))          # (8Nc, 2)
    dW_dth = y * w_ref[:, None]                                         # unscale
    # dW_dth is the FULL field sensitivity (8*Nc, n_theta).  Contracting it to
    # dJ/dtheta discards where in the nozzle the parameter acts; it is returned
    # so sensitivity_plots.py can put that back.
    return dJ_dW @ dW_dth, svd, int(keep.sum()), dW_dth


print("\nSolving the IFT system...")
J_sc, Rth = _linearize(dt_ss)
_svd = (np.linalg.svd(J_sc, full_matrices=False), Rth)
_S = _svd[0][1]
print(f"  system: {J_sc.shape[0]} equations x {J_sc.shape[1]} unknowns "
      f"({'square' if J_sc.shape[0]==J_sc.shape[1] else 'RECTANGULAR'})")
print(f"  singular values (scaled): max={_S[0]:.3e}  min={_S[-1]:.3e}  "
      f"cond={_S[0]/_S[-1]:.3e}")
for _rc in (1e-8, 1e-10, 1e-12, 1e-14):
    print(f"    rank at rcond={_rc:.0e}: {int((_S > _rc*_S[0]).sum())} / {_S.size}")

dJ_dth, _svd, _rank, dW_dth = sensitivities(dt_ss, rcond=1e-12, svd=_svd)
print(f"  using rcond=1e-12  (rank {_rank}/{_S.size})")

names = ["mdot [kg/s]", "x_exit [-]"]
print("\n" + "=" * 78)
print(f"SENSITIVITY TABLE — {run_dir.name}")
print("=" * 78)
print(f"{'quantity J':<14} {'J':>13} {'dJ/dN_b':>13} {'dJ/dN_d':>13} "
      f"{'dlnJ/dlnN_b':>13} {'dlnJ/dlnN_d':>13}")



def _cell(val, j):
    """Format a sensitivity, flagging structurally inert parameters."""
    return "    n/a (regime)" if nrm[j] == 0.0 else f"{float(val):13.4e}"


for i, nm in enumerate(names):
    lb = dJ_dth[i, 0] * theta0[0] / J_star[i]
    ld = dJ_dth[i, 1] * theta0[1] / J_star[i]
    print(f"{nm:<14} {J_star[i]:13.6e} {_cell(dJ_dth[i,0],0)} {_cell(dJ_dth[i,1],1)} "
          f"{_cell(lb,0)} {_cell(ld,1)}")
print("\nThe log-sensitivities dlnJ/dlnN are the dimensionless, "
      "reportable form:\n  a value of 0.01 means a 10x change in N moves J by ~2.3%.")


# ── VERIFICATION: insensitivity to the pseudo-inverse truncation ─────────────
print("\n" + "=" * 78)
print("rcond SWEEP — the answer must not depend on where the null space is cut")
print("=" * 78)
print(f"{'rcond':>10} {'rank':>7} {'dlnMdot/dlnNb':>16} {'dlnMdot/dlnNd':>16} "
      f"{'dlnX/dlnNb':>14} {'dlnX/dlnNd':>14}")
RCOND_ROWS = []           # (rcond, rank, [elasticities]) — replotted in fig 4
for _rc in (1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
    _d, _, _rk, _ = sensitivities(dt_ss, rcond=_rc, svd=_svd)
    RCOND_ROWS.append((_rc, _rk, [float(_d[i, j] * theta0[j] / J_star[i])
                                  for i in range(2) for j in range(2)]))
    print(f"{_rc:10.0e} {_rk:7d} "
          f"{float(_d[0,0]*theta0[0]/J_star[0]):16.6e} "
          f"{float(_d[0,1]*theta0[1]/J_star[0]):16.6e} "
          f"{float(_d[1,0]*theta0[0]/J_star[1]):14.6e} "
          f"{float(_d[1,1]*theta0[1]/J_star[1]):14.6e}")
print("Stable columns ⇒ dJ/dW has no weight in the discarded null space and the\n"
      "sensitivity is well posed despite the rank deficiency.  Drifting columns ⇒\n"
      "the quantity is not determined by the steady state alone; do not report it.")


# ── FINITE-DIFFERENCE VALIDATION ──────────────────────────────────────────────
# FD_LO / FD_HI point at two re-converged runs with the parameter perturbed.  The
# QoI is evaluated by the SAME _qoi() used for the AD contraction, so the only
# difference between the two columns is the derivative method.
FD_LO = os.environ.get("FD_LO", "")
FD_HI = os.environ.get("FD_HI", "")
FD_PAR = os.environ.get("FD_PAR", "N_b")            # which parameter was perturbed
if FD_LO and FD_HI:
    def _load_case(p):
        d = pathlib.Path(p)
        if not d.is_absolute():
            d = _here / d
        Dd = np.genfromtxt(d / "solution.csv", delimiter=",", names=True)
        Uh = np.column_stack([Dd["p"], Dd["u_g"], Dd["u_l"],
                              Dd["alpha_g"], Dd["h_g"], Dd["h_l"]])
        Wf = W_from_U_h_batch(jnp.asarray(Uh), fluid_g, fluid_l)
        c2 = load_config(sorted(d.glob("settings*.yaml"))[0])
        if Wf.shape[0] != N:
            sys.exit(f"FD run {d.name} has {Wf.shape[0]} cells, reference has "
                     f"{N} — the QoI contraction assumes a common grid.")
        return (np.asarray(_qoi(Wf[1:-1].reshape(-1))),
                float(c2.BUBBLE_NUMBER), float(c2.DROPLET_NUMBER), d.name,
                np.asarray(Wf[1:-1].reshape(-1)))

    J_lo, nb_lo, nd_lo, n_lo, W_lo = _load_case(FD_LO)
    J_hi, nb_hi, nd_hi, n_hi, W_hi = _load_case(FD_HI)
    p_lo_, p_hi_ = ((nb_lo, nb_hi) if FD_PAR == "N_b" else (nd_lo, nd_hi))
    jcol = 0 if FD_PAR == "N_b" else 1
    dP = p_hi_ - p_lo_

    print("\n" + "=" * 78)
    print(f"FINITE-DIFFERENCE VALIDATION — parameter {FD_PAR}")
    print("=" * 78)
    print(f"  lo: {n_lo}   {FD_PAR}={p_lo_:.6e}")
    print(f"  hi: {n_hi}   {FD_PAR}={p_hi_:.6e}")
    print(f"  central step: d{FD_PAR}/{FD_PAR} = "
          f"{dP/theta0[jcol]*100:.3f} %   (dln{FD_PAR} = {np.log(p_hi_/p_lo_):.5f})")
    if dP == 0.0:
        print("  ** the two runs have the SAME parameter value — nothing to compare **")
    else:
        print(f"\n{'quantity':<14} {'AD dJ/dP':>14} {'FD dJ/dP':>14} "
              f"{'rel.diff':>11} {'AD dlnJ/dlnP':>14} {'FD dlnJ/dlnP':>14}")
        for i, nm in enumerate(names):
            fd  = (J_hi[i] - J_lo[i]) / dP
            ad  = dJ_dth[i, jcol]
            rel = (fd - ad) / (abs(ad) if ad != 0 else np.inf)
            print(f"{nm:<14} {ad:14.5e} {fd:14.5e} {rel*100:10.2f}% "
                  f"{ad*theta0[jcol]/J_star[i]:14.5e} "
                  f"{fd*theta0[jcol]/J_star[i]:14.5e}")
        print("\nAgreement to a few percent validates the whole chain: the IFT "
              "linearization,\nthe scaled solve, and the theta-traced closure. "
              "Large disagreement points at\nthe FD step being outside the "
              "[truncation | convergence-noise] plateau, at the\nreference state "
              "not being converged, or at a regime switch between the runs.")


# ── Optional: dt-insensitivity sweep (verification 3) ────────────────────────
DT_ROWS = []              # (dt/dt_cfl, cond, [elasticities]) — replotted in fig 4
if DT_SWEEP:
    print("\n" + "=" * 78)
    print("dt-INSENSITIVITY SWEEP — the steady Jacobian must stop depending on dt")
    print("=" * 78)
    print(f"{'dt/dt_CFL':>12} {'cond':>11} {'dlnMdot/dlnNb':>16} "
          f"{'dlnMdot/dlnNd':>16} {'dlnX/dlnNb':>14} {'dlnX/dlnNd':>14}")
    for f in (1e0, 1e2, 1e4, 1e6, 1e8, 1e10):
        try:
            d_, sv_, _, _ = sensitivities(f * dt_cfl)
            S_ = sv_[0][1]
            DT_ROWS.append((f, S_[0] / S_[-1],
                            [float(d_[i, j] * theta0[j] / J_star[i])
                             for i in range(2) for j in range(2)]))
            print(f"{f:12.0e} {S_[0]/S_[-1]:11.3e} "
                  f"{float(d_[0,0]*theta0[0]/J_star[0]):16.6e} "
                  f"{float(d_[0,1]*theta0[1]/J_star[0]):16.6e} "
                  f"{float(d_[1,0]*theta0[0]/J_star[1]):14.6e} "
                  f"{float(d_[1,1]*theta0[1]/J_star[1]):14.6e}")
        except Exception as e:                       # noqa: BLE001
            print(f"{f:12.0e}   failed: {type(e).__name__}: {e}")
    print("Converged columns ⇒ the transient term Vc·dV/dW is fully suppressed "
          "and these are true steady sensitivities.")


# ── FIGURES ───────────────────────────────────────────────────────────────────
# The table above is a contraction of dW/dtheta against dJ/dW, i.e. the spatial
# structure summed away.  These figures plot dW/dtheta itself, so the question
# "10x more bubbles costs 13% of the mass flow — earned WHERE?" has an answer.
if "--no-plot" not in sys.argv:
    import matplotlib
    if "--show" not in sys.argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nozzle_hyperbolic as hyp
    sys.path.insert(0, str(_here))          # sensitivity_plots sits beside this file
    import sensitivity_plots as sp
    from nozzle_hyperbolic.fvm_solver.two_fluid.sources import interfacial_area

    hyp.set_plot_options(fontsize=11, grid=False)
    matplotlib.rcParams["savefig.dpi"] = 200

    out = run_dir / "sensitivity"
    out.mkdir(exist_ok=True)
    x_int = x[1:-1]
    x_throat = float(x[int(np.argmin(A))])
    ag_int = np.asarray(U_h_star[1:-1, 3])

    # Derived fields as one pure function of W, so a forward-mode JVP along the
    # sensitivity direction gives every field's sensitivity at once — including
    # the derived ones (slip, mdot) — with no hand-differentiation.
    def _fields(W_flat):
        Wm = W_flat.reshape(Nc, 8)
        p_, ag_, mg_, ug_, ml_, ul_ = (Wm[:, 0], Wm[:, 1], Wm[:, 2],
                                       Wm[:, 3], Wm[:, 4], Wm[:, 5])
        return {"p": p_, "alpha_g": ag_, "u_g": ug_, "u_l": ul_,
                "du": ug_ - ul_, "mdot": (mg_ * ug_ + ml_ * ul_) * A_j[1:-1]}

    state = {k: np.asarray(v) for k, v in _fields(W_int0).items()}
    # tangent = theta_j * dW/dtheta_j  ⇒  the JVP output is d(field)/d(ln theta_j)
    sens = {k: np.zeros((len(PARAM), Nc)) for k in state}
    for j in range(len(PARAM)):
        _, tang = jax.jvp(_fields, (W_int0,),
                          (jnp.asarray(theta0[j] * dW_dth[:, j]),))
        for k, v in tang.items():
            sens[k][j] = np.asarray(v)

    f1 = sp.plot_sensitivity_fields(
        x_int, state, sens, PARAM, inert=inert, alpha_g=ag_int,
        x_throat=x_throat,
        title=f"{run_dir.name} — state (top) and its sensitivity per e-fold "
              f"of the closure parameter (bottom)")
    f1.savefig(out / "sens_fields.png", bbox_inches="tight")

    # The interfacial area is the ONLY channel through which N_b/N_d reach the
    # physics, so its profile and its own derivative explain every zero and every
    # kink in the figure above.
    _d_min_b, _d_min_d = 5e-6, 1e-5        # sources.py call site / library default
    _ag_j = jnp.asarray(ag_int)

    # lax.map, NOT vmap.  interfacial_area() branches on alpha through a nested
    # lax.cond cascade; vmap batches the predicate and lowers it to a select,
    # which evaluates the DEAD branches too — and those contain alpha**(2/3) and
    # (1-alpha)**(2/3), whose tangents are infinite at alpha = 0 and 1.  That is
    # the dead-branch NaN trap, and hitting it here would be an artefact of the
    # plotting code, not of the solver.  lax.map keeps each cond a real branch.
    def _ia_fields(th):
        def one(a):
            ai_, d_ = interfacial_area(a, alpha_b=0.3, alpha_d=0.7,
                                       Nb=th[0], Nd=th[1], d_min_b=_d_min_b)
            return jnp.stack([ai_, d_])
        return jax.lax.map(one, _ag_j)                       # (Nc, 2)

    _iad = np.asarray(jax.jit(_ia_fields)(theta0))
    ai_p, d_p = _iad[:, 0], _iad[:, 1]
    _jia = np.asarray(jax.jit(jax.jacfwd(_ia_fields))(theta0))        # (Nc,2,2)
    dai = (_jia[:, 0, :] * np.asarray(theta0)[None, :]).T             # (2, Nc)
    f2 = sp.plot_regime_map(
        x_int, ag_int, ai_p, d_p, dai, PARAM,
        alpha_b_star=float(np.pi * theta0[0] * _d_min_b**3 / 6.0),
        alpha_d_star=float(1.0 - np.pi * theta0[1] * _d_min_d**3 / 6.0),
        x_throat=x_throat,
        title=f"{run_dir.name} — flow regime and the interfacial-area channel")
    f2.savefig(out / "sens_regime.png", bbox_inches="tight")

    f4 = sp.plot_diagnostics(
        _S, RCOND_ROWS, DT_ROWS, ["mdot", "x_exit"], PARAM,
        title=f"{run_dir.name} — is the IFT solve trustworthy?")
    f4.savefig(out / "sens_diagnostics.png", bbox_inches="tight")

    saved = ["sens_fields.png", "sens_regime.png", "sens_diagnostics.png"]

    # AD vs FD field by field — the strong form of the validation gate.  Two
    # scalars that disagree tell you nothing about WHY; two fields do.
    if FD_LO and FD_HI and dP != 0.0:
        dln = float(np.log(p_hi_ / p_lo_))
        fl = {k: np.asarray(v) for k, v in _fields(jnp.asarray(W_lo)).items()}
        fh = {k: np.asarray(v) for k, v in _fields(jnp.asarray(W_hi)).items()}
        sens_fd = {k: (fh[k] - fl[k]) / dln for k in fl}
        sens_ad = {k: sens[k][jcol] for k in fl}
        f3 = sp.plot_ad_vs_fd(
            x_int, sens_ad, sens_fd, list(fl.keys()), FD_PAR,
            alpha_g=ag_int, x_throat=x_throat,
            title=f"{run_dir.name} — AD vs central FD in log space, "
                  f"dln{FD_PAR}={dln:.4f}")
        f3.savefig(out / "sens_ad_vs_fd.png", bbox_inches="tight")
        saved.append("sens_ad_vs_fd.png")

    print("\n" + "=" * 78)
    print(f"FIGURES → {out}")
    for s in saved:
        print(f"  {s}")
    if "--show" in sys.argv:
        plt.show()
