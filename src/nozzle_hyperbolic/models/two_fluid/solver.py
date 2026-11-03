"""Deprecated historical two-fluid implementation.

.. deprecated:: 0.0.1
    Use ``TwoFluidModel`` with ``nozzle_hyperbolic.solvers.fvm`` drivers.
    Retained only for historical comparisons and remaining legacy callers.
    This module is a deletion candidate after those callers/tests migrate.
    The current model is not a drop-in replacement for every legacy API.

Implicit Newton time-stepping for Städtke's two-fluid FVS solver.

Newton variable:  W = (p, α_g, m_g, u_g, m_l, u_l, m_g·e_g, m_l·e_l)   shape (N, 8)

  m_i = α_i · ρ_i   (partial densities)
  e_i = h_i - p/ρ_i (specific internal energies)

Two key polynomial maps (no EOS, no inner Newton):

  V(W) = (m_g, m_l, m_g·u_g, m_l·u_l,
          m_g·e_g + m_g·u_g²/2,  m_l·e_l + m_l·u_l²/2)    ← conservative

  U_h(W) = (p, u_g, u_l, α_g,
             (m_g·e_g + p·α_g)/m_g,                         ← h_g = e_g + p/ρ_g
             (m_l·e_l + p·(1−α_g))/m_l)                    ← h_l

EOS is needed only for fluxes, nc correction, and sources (same as before).

Residual for interior cell i (6-component, same physics as before):

  R_i = Vc_i·(V(W_new)_i − V(W_n)_i)
      + dt·(A_{i+1/2}·F̂_{i+1/2} − A_{i-1/2}·F̂_{i-1/2})
      + dt·A_i · X_i @ solve(K_i, F̂_{i+1/2} − F̂_{i-1/2})
      − dt·Vc_i · D_VE_i

  R ∈ R^{6·(N-2)},  W_int ∈ R^{8·(N-2)}
  Jacobian ∂R/∂W is (6·(N-2)) × (8·(N-2)) — rectangular.
  Newton update: dW = lstsq(J, −R)  [minimum-norm solution].

API
---
  W_n   = W_from_U_h_batch(U_h_0, fluid_g, fluid_l)   ← initial condition
  step, compute_dt = make_two_fluid_step(...)
  W_new, ssr = step(W_n, dt)
  U_h   = U_h_from_W_batch(W_new)                      ← back to primitives
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .boundaries import _apply_ramped_inlet

import jax.scipy.linalg as jsl
import jaxprop as jxp

from .matrices import (
    phase_props, common_quantities,
    K_matrix, X_matrix, eigenstructure, flux_vector,
)
from .flux import fvs_interface_flux, nonconservative_correction
from .sources import (TwoFluidClosure, D_VE_conservative,
                      D_VE_algebric_two_component, evaporation_weight)

# Void-fraction floor: α_g is hard-clipped to (ε, 1-ε) in the projection and after
# each Newton update to guard the EOS lookups and keep both phases realizable.
# Set to 1e-6 so a "ghost" phase (single-phase-limit tests) can sit at α≈1e-6
# without the clamp fighting it — matching the space-marching solver's floor.
_ALPHA_EPS = 1e-6


# ---------------------------------------------------------------------------
# W ↔ V  and  W ↔ U_h  (polynomial / algebraic — no EOS)
# ---------------------------------------------------------------------------

def V_from_W(W):
    """W (8,) → V (6,) — polynomial, no EOS."""
    p, ag, m_g, u_g, m_l, u_l, mg_eg, ml_el = W
    return jnp.array([
        m_g,
        m_l,
        m_g * u_g,
        m_l * u_l,
        mg_eg + 0.5 * m_g * u_g**2,
        ml_el + 0.5 * m_l * u_l**2,
    ])


def U_h_from_W(W):
    """W (8,) → U_h (6,) — algebraic, no EOS.

    h_g = e_g + p/ρ_g = (m_g·e_g + p·α_g) / m_g   (since p/ρ_g = p·α_g/m_g)
    h_l = e_l + p/ρ_l = (m_l·e_l + p·α_l) / m_l
    """
    p, ag, m_g, u_g, m_l, u_l, mg_eg, ml_el = W
    al  = 1.0 - ag
    h_g = (mg_eg + p * ag) / m_g
    h_l = (ml_el + p * al) / m_l
    return jnp.array([p, u_g, u_l, ag, h_g, h_l])


def W_from_U_h_batch(U_h, fluid_g, fluid_l):
    """U_h (N,6) → W (N,8).  Needs one EOS batch call to get ρ_g, ρ_l."""
    p   = U_h[:, 0];  u_g = U_h[:, 1];  u_l = U_h[:, 2]
    ag  = U_h[:, 3];  h_g = U_h[:, 4];  h_l = U_h[:, 5]
    al  = 1.0 - ag
    stg = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
    stl = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p)
    rg  = stg["rho"];  rl = stl["rho"]
    m_g = ag * rg;  m_l = al * rl
    e_g = h_g - p / rg;  e_l = h_l - p / rl
    return jnp.stack([p, ag, m_g, u_g, m_l, u_l, m_g * e_g, m_l * e_l], axis=1)


def U_h_from_W_batch(W):
    """W (N,8) → U_h (N,6) — purely algebraic, no EOS."""
    return jax.vmap(U_h_from_W)(W)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_two_fluid_step(
    fluid_g, fluid_l,
    A, A_face, dAdx,
    dx,
    CFL=0.5,
    k=0.5,
    delta_factor=1e-6,
    newton_iter=5,
    recon='muscl_van_leer',
    va_eps_mode='magnitude',
    va_rel=None,
    va_r0=0.5,
    va_rm=4,
    va_mean_stop_gradient=True,
    sources_fn=None,
    lowmach_diss=0.0,
    lowmach_diss_bc=None,
    nc_scale=1.0,
    eps_void=0.0,
    eps_void_ucut=None,
    eps_void_uwidth=None,
    eps_void_scale='wood',
    eps_void_mass_acut=None,
    ghost_mass_upwind_acut=None,
    eps_ghost=0.0,
    eps_ghost_acut=3e-3,
    eps_ghost_awidth=0.25,
    eps_ghost_k2=0.0,
    eps_ghost_alo=1e-5,
    du_blend_lo=1e-2,
    du_blend_hi=1e-1,
    debug_newton=False,
    jacobian_mode='dense',
    linear_solve='lstsq',
    linear_reg=0.0,
    bc_inlet='transmissive',
    bc_outlet='transmissive',
    H_g0=None,
    H_l0=None,
    alpha_g0=None,
    s_g0=None,
    s_l0=None,
    dom_phase='gas',
    iso_h=None,
    iso_p=None,
    quality=None,
    slip=None,
    U_h_inlet=None,
    p_out=None,
    p_out_relax=1.0,
    eos_project=True,
    proj_s_ref=2e-2,
    eos_rows=False,
    eos_rows_banded=False,
    jac_lag=1,
    nc_form='cell',
    use_saturation=True,
    w_char=None,
    ssr_mode='all',
    return_debug=False,
    alpha_eps=_ALPHA_EPS,
    pin_ghost=False,
    p_mt_max=None,
    pin_ghost_momentum=False,
    pin_mom_alpha=None,
    pin_mom_tau=None,
    fast_wb_flux=False,
    flux_single_branch=False,
    flux_local_jac=False,
    cell_local_jac=False,
    newton_tol=0.0,
    thomas='orig',
):
    """
    Build @jax.jit step and compute_dt functions for Städtke's two-fluid system.

    Parameters
    ----------
    fluid_g, fluid_l : FluidBicubic tables (metastable vapor / liquid)
    A       : (N,)   cell cross-section areas [m²]
    A_face  : (N-1,) face cross-section areas [m²]
    dAdx    : (N,)   dA/dx at cell centres [m]
    dx      : scalar uniform cell width [m]
    CFL     : Courant number
    k       : virtual-mass coefficient
    delta_factor : Harten entropy-fix fraction
    newton_iter  : Newton iterations (unrolled inside @jax.jit)
    recon   : 'muscl_van_leer' | 'muscl_van_albada' (smooth C¹, no slope flicker
              on ghost-scale noise — use for ghost-limit cases) | 'muscl_minmod'
              | 'mc' | 'constant'
    sources_fn : optional callable with signature
                   (u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l) → D_VE (6,)
                 where sat_g = fluid_g.get_state_saturation(p) — sat. vapor props dict,
                       sat_l = fluid_l.get_state_saturation(p) — sat. liquid props dict.
                 Use sat_g["enthalpy"], sat_g["temperature"], etc. (canonical names).
                 If None, only the geometric area-variation source is included.
    lowmach_diss, lowmach_diss_bc : small blend factor(s) for a Rusanov/LF
                 stabilization term, used to damp the low-Mach checkerboard mode
                 at the interface flux. lowmach_diss_bc optionally overrides the
                 blend at the two boundary faces (None → same as lowmach_diss).
    nc_scale   : float, default 1.0. Scales the non-conservative geometric
                 coupling term (H^nc·ΔF, see nc_form below). nc_scale=0 turns
                 NC coupling OFF — a DIAGNOSTIC setting, not for production runs.
    eps_void, eps_void_ucut, eps_void_uwidth, eps_void_scale :
                 4th-order JST-style filter damping the low-Mach void/slip
                 checkerboard mode. eps_void=0 (default) disables it.
                 eps_void_scale: 'wood' (default, true dipping mixture sound
                 speed) or 'frozen' (non-dipping surrogate) — both are
                 deliberately-supported production choices, not legacy vs new.
                 eps_void_ucut/_uwidth optionally gate the filter to low-velocity
                 regions only (None → ungated).
    eps_ghost, eps_ghost_acut, eps_ghost_awidth, eps_ghost_k2, eps_ghost_alo :
                 shock-aware (JST-sensor cross-faded) filter damping the
                 ghost-phase 2Δx limit cycle near the void floor. eps_ghost=0
                 (default) disables it. NOTE: eps_ghost_k2 defaults to 0.0, but
                 an inline comment at its use site recommends ~4 (from a
                 certified run) to keep the damping rate constant through the
                 shock-sensor cross-fade — doc/default drift, see
                 dev_scripts/OPEN_QUESTIONS.md item 5.
    du_blend_lo, du_blend_hi : velocity-difference thresholds blending between
                 the exact two-velocity FVS flux and a shared-velocity
                 (homogeneous) low-Mach-safe flux at the interface.
    debug_newton : bool, default False. When True, runs Newton as a plain
                 Python for-loop with print() diagnostics each iteration (not
                 JIT-able) — interactive debugging only, never production.
    jacobian_mode : 'dense' (default) | 'colored'. 'colored' is a graph-colored,
                 cheaper Jacobian assembly, verified numerically equivalent to
                 'dense' via the jac_diff sparsity guard — pure perf choice.
    linear_solve : 'lstsq' (default, SVD min-norm, most robust) |
                 'normal_eq' / 'normal_eq_banded' (κ²-squared, faster) | 'qr'.
                 Deliberate accuracy/speed tradeoffs, not legacy vs new.
    linear_reg : float, default 0.0. Tikhonov regularization added to the
                 linear solve; 0 = off.
    bc_inlet   : 'transmissive' (default), 'total_state', 'stagnation',
                 'stagnation_ramped', 'static_state', or 'subsonic_inflow'.
                 'total_state' mirrors _bc_inlet_real from the single-phase solver:
                   extrapolate p, u_g, u_l from interior; impose H_g0, H_l0, alpha_g0.
    bc_outlet  : 'transmissive' (default), 'pressure', 'characteristic', or
                 'characteristic_old'.
                 'characteristic' auto-switches subsonic ↔ supersonic on the
                   Städtke mixture Mach and builds the subsonic ghost by
                   projecting the imposed back-pressure onto the INCOMING acoustic
                   right eigenvector r₄ (λ₄ = û − a), so no reflection is emitted.
                 'characteristic_old' is the legacy form of the same BC: identical
                   switch, but a shared Δu on both phases through the homogeneous
                   (Wood) impedance, with α_g and h_k frozen.  Kept as an
                   INTENTIONAL A/B reference (still selected by some driver
                   scripts to reproduce old runs) — the two agree to ~0.06% at
                   steady state. Not a cleanup target.
    H_g0       : total gas enthalpy at inlet [J/kg]  (required when bc_inlet='total_state')
    H_l0       : total liquid enthalpy at inlet [J/kg]
    alpha_g0   : void fraction at inlet [-]
    s_g0, s_l0 : stagnation entropy at inlet [J/(kg·K)], gas/liquid — closes the
                 reservoir isentrope for the 'stagnation'/'stagnation_ramped'
                 inlets; which one is used is selected by dom_phase.
    dom_phase  : 'gas' (default) | 'liquid'. Selects which phase's isentrope
                 (s_g0 vs s_l0) drives the stagnation-inlet pressure fixed
                 point: 'gas' for gas-dominant tiers, 'liquid' for
                 liquid-dominant/flashing cases.
    iso_h, iso_p : optional caller-precomputed (h, p) isentrope tables,
                 bypassing the in-function bicubic isentrope build — needed when
                 the dominant phase's bicubic EOS has no root at low p on its
                 own isentrope (e.g. a liquid isentrope landing two-phase).
    quality, slip : required by the 'stagnation'/'stagnation_ramped' inlets —
                 gas mass fraction x0 and velocity slip ratio S0 = u_g/u_l,
                 jointly fixing the inlet void fraction and velocity split.
    U_h_inlet  : full 6-vector Dirichlet inlet state (p, u_g, u_l, α_g, h_g, h_l),
                 used by bc_inlet='static_state'/'subsonic_inflow'.
    p_out      : back-pressure [Pa] for bc_outlet='pressure'/'characteristic'/
                 'characteristic_old'.
    p_out_relax : float, default 1.0. Under-relaxation factor blending p_out
                 into the outlet ghost each step (1.0 = full imposition).
    eos_project : True (default, classic 2×2 (p, α) pair projection) | 'damped'
                 (conditioning-gated pair, ghost-safe — use for flashing) |
                 False (off). 'triangular' FAILED certification (see
                 dev_scripts/debug_projection_tri.py, which documents the
                 failure) and now raises a ValueError rather than being
                 silently selectable.
    proj_s_ref : float, default 2e-2. Reference entropy scale conditioning the
                 'damped' projection's gate.
    eos_rows, eos_rows_banded : bool, default False. eos_rows=True switches to
                 an EXPERIMENTAL 8-row E3-augmented residual system that folds
                 the EOS-manifold projection directly into the Newton residual
                 instead of post-hoc projecting; self-labeled "prototype"
                 (2026-07-15), fixes a measured secular liquid-mass drain but
                 is not yet the production default — see
                 dev_scripts/OPEN_QUESTIONS.md item 4. eos_rows_banded=True
                 switches the dense eos_rows path to a verified-equivalent
                 O(N) block-banded twin (only meaningful when eos_rows=True).
    jac_lag    : int, default 1. Newton Jacobian refresh cadence on the
                 eos_rows_banded path; 1 = refresh every iteration (certified
                 default); >1 lags the refresh but additionally force-refreshes
                 whenever the residual failed to decrease (added after a
                 naively-scheduled lag diverged on strong transients).
    nc_form    : 'cell' (default, "ours") | 'staedtke' (book eq. 8.38) | 'wb'
                 (well-balanced, 2026-07-15). Correctness is an UNRESOLVED
                 pairing property between this term and the area-source
                 content, discriminated only by dev_scripts/debug_nc_contact.py
                 — see dev_scripts/OPEN_QUESTIONS.md item 1. Production stays
                 on 'cell' regardless of this docstring's existence.
    use_saturation : bool, default True. Whether sources_fn's sat_g/sat_l
                 saturation-property lookups are computed; set False only for
                 fluids/sources that never reference them (e.g. non-saturating
                 gases, or mass-transfer-off configs). Must be True if
                 pin_ghost=True.
    w_char     : None (default, fixed two-component-calibrated Newton min-norm
                 floors) | 'auto' (local rolling-max floor scaled to the field,
                 certified by dev_scripts/debug_wref_sensitivity.py for a
                 ghost-limit thrash bug) | explicit length-8 list override. See
                 dev_scripts/OPEN_QUESTIONS.md item 3 for why 'auto' isn't yet
                 the default.
    ssr_mode   : 'all' (default, equal-weighted steady-state residual across
                 all rows — self-labeled legacy in-code) | 'mass' (mass-row-
                 weighted; fixes a measured pathology where ghost rows owned up
                 to 99.98% of the plain SSR² on a converged run). See
                 dev_scripts/OPEN_QUESTIONS.md item 2.
    return_debug : bool, default False. When True, step() additionally returns
                 an internal-state debug dict (Jacobian, residual breakdown,
                 etc.) for probe scripts; adds overhead, not for production.
    alpha_eps  : float, default _ALPHA_EPS. Per-run void-fraction floor; the
                 default is tuned for the gas-dominant tier — flashing runs
                 with a thin liquid-side ghost should raise it (~1e-4..1e-3).
    pin_ghost, pin_ghost_momentum : bool, default False. Algebraic pins slaving
                 the ghost-phase energy/momentum toward the saturation curve /
                 dominant-phase value, guarding against the ghost-phase 2Δx
                 amplifier. Both carry an explicit "NO-DEADLOCK INVARIANT — do
                 not break" contract at their implementation site (the pin may
                 only ever WIDEN toward 1, never narrow) that any future edit
                 must preserve.
    p_mt_max   : float [Pa] or None. No phase change at p >= p_mt_max: the
                 evaporation weight is 0 there, so the ghost pin stays on.  Pass
                 the SAME value to two_fluid_closure in sources_fn.  For a
                 supercritical inlet (saturation table ends at 0.98 p_crit).
                 None (default) → no cut-off.

    THIN-GHOST SWITCHES (2026-10-08).  All default to None = the previous code
    path.  Together they let a flashing inlet ghost sit at alpha ~1e-7 (tested:
    SMD supersonic case, N300/N600, SSR < 1e-8); at alpha_in = 1e-3 they change
    the solution by <= 2e-7 of each field's range.
    ghost_mass_upwind_acut : float.  At faces where either side has alpha below
                 ~acut (log-alpha gate), the ghost phase's MASS flux is the pure
                 upwind alpha*rho*u instead of the FVS value, whose acoustic
                 correction (built at the average face state) moves ghost mass
                 ~alpha_R*rho*u across a front and empties the thin side.
    eps_void_mass_acut : float.  Fades the eps_void 4th-order filter out of the
                 ghost phase's mass row below ~acut (the filter is not
                 positivity preserving and digs alpha holes across decades).
    pin_mom_alpha : float.  alpha at which the ghost momentum pin is half
                 released (default 10*alpha_eps).
    pin_mom_tau : float [s].  Makes the momentum pin a relaxation source scaled
                 by dt/pin_mom_tau, so the steady state is dt-independent where
                 the pin is partly active (the algebraic blend leaves an SSR
                 floor there).  ~1e-6 s (drag time of a 5 um bubble).
    va_mean_stop_gradient : bool, default True.  va_eps_mode='magnitude' only:
                 stop_gradient on the domain mean in the van Albada eps, so the
                 Jacobian drops its global coupling instead of folding it into
                 wrong in-band colored entries (error ∝ 1/alpha_ghost: 21-31% of
                 the row max at alpha_in=1e-7).  Same residual, same converged
                 state; only the Newton path changes.  False = the earlier code
                 (bit-for-bit; set by defaults_legacy.yaml for legacy files).

    SPEED-UP SWITCHES (2026-10-05).  All default to the previous code path, so
    omitting them reproduces earlier runs exactly.  The first five leave the
    residual unchanged (values to ~1e-15, Jacobian to ~1e-13; an N600 restart
    reproduced a 3,410-step reference run to 1e-13 in every field):
    fast_wb_flux : bool.  nc_form='wb' only: evaluate the cell-centre flux F_c
                 as the physical flux F(U_i) instead of fvs_interface_flux(U_i,
                 U_i) — identical by consistency of the FVS flux (U_L = U_R).
    flux_single_branch : bool.  Pass single_branch=True to fvs_interface_flux
                 (each slip branch evaluated once instead of the zero-slip
                 branch twice under vmap).
    flux_local_jac : bool.  Give the face-flux map a custom JVP that linearizes
                 each face ONCE (exact 6×12 block from 12 forward tangents) and
                 serves every colored Jacobian seed as a mat-vec.
    cell_local_jac : bool.  Same for the cell-local maps (NC correction,
                 sources, filter spectral radius, ghost pin, wb F_c).
    thomas     : 'orig' | 'chol'.  E3-banded block-Thomas: 'chol' factors each
                 super-block once (Cholesky) and back-substitutes with stored
                 Dp⁻¹U blocks; 'orig' re-factors three times.  Same solution to
                 roundoff (~1e-11 relative).
    newton_tol : float, default 0.0 (= always newton_iter iterations).  > 0
                 stops the Newton loop once max|dW_scaled| < newton_tol (the
                 remaining error is ~newton_tol² by quadratic convergence);
                 newton_iter stays the cap.  Per-step results then differ from
                 the fixed-iteration ones only at that level (1e-8 recommended).
                 Not available together with jac_lag > 1.

    See also: dAdx must satisfy dAdx·dx ≡ ΔA_face (the well-balanced area-
    gradient invariant) or the smooth-flow flux divergence will not exactly
    cancel the geometric source, reintroducing a 2Δx checkerboard. This
    function does not currently assert the invariant itself — see
    dev_scripts/main_two_fluid.py's "WELL-BALANCED area gradient" section for
    the caller-side fix, and dev_scripts/OPEN_QUESTIONS.md for the note that
    this should move into an internal check.

    Returns
    -------
    step       : jit-compiled  step(W_n, dt) → (W_new, ssr)
    compute_dt : jit-compiled  compute_dt(W_n) → dt

    Initial condition: W_n = W_from_U_h_batch(U_h_0, fluid_g, fluid_l)
    Post-processing:   U_h = U_h_from_W_batch(W_new)
    """
    # Per-run void-fraction floor.  Default 1e-6 (gas tier); flashing runs with
    # a thin liquid-side ghost should set ~1e-4..1e-3 so the solver cannot
    # re-thin α below the IC floor mid-march (h_g = E/(αρ) amplifier guard).
    alpha_eps = float(alpha_eps)
    # Ghost momentum pin half-release alpha (historically tied to the floor).
    _a_pin_mom = 10.0 * alpha_eps if pin_mom_alpha is None else float(pin_mom_alpha)
    # van Albada eps normalisation (see _muscl below).  'magnitude' is the
    # original behaviour and the default, so omitting these keys reproduces every
    # pre-2026-09-09 run exactly.  va_rel=None picks the value calibrated for the
    # chosen mode (1e-3 magnitude / 1e-4 range) so switching mode alone cannot
    # land on a mis-scaled floor.
    if va_eps_mode not in ('magnitude', 'range', 'smoothness'):
        raise ValueError(f"va_eps_mode must be 'magnitude', 'range' or "
                         f"'smoothness', got {va_eps_mode!r}")
    _va_rel_eff = float(va_rel) if va_rel is not None else (
        {'range': 1e-4, 'smoothness': 1e-12}.get(va_eps_mode, 1e-3))
    _va_r0 = float(va_r0)
    _va_rm = int(va_rm)
    if _va_r0 <= 0.0 or _va_rm < 4 or _va_rm % 2:
        raise ValueError("va_r0 must be > 0 and va_rm an EVEN integer >= 4: "
                         "even so the gate can be evaluated on r^2 (no abs() "
                         "in the AD'd residual), >=4 so the smooth-region "
                         "perturbation stays O(h^m) below the O(h^2) "
                         "truncation error")
    if pin_ghost and not use_saturation:
        raise ValueError("pin_ghost=True requires use_saturation=True "
                         "(the pin reads h_sat_g(p) from the saturation table)")
    if thomas not in ('orig', 'chol'):
        raise ValueError(f"thomas must be 'orig' or 'chol', got {thomas!r}")
    newton_tol = float(newton_tol)
    if newton_tol > 0.0 and eos_rows and eos_rows_banded and jac_lag > 1:
        raise ValueError("newton_tol > 0 is not implemented together with "
                         "jac_lag > 1 (lagged-Jacobian Newton); use one or the other")
    N       = len(A)
    A_j     = jnp.asarray(A,      dtype=jnp.float64)
    Af_j    = jnp.asarray(A_face, dtype=jnp.float64)
    V_cell  = A_j * dx
    dlnA_dx = jnp.asarray(dAdx, dtype=jnp.float64) / A_j

    # EOS table bounds for MUSCL fallback check (on U_h values)
    p_lo  = float(getattr(fluid_g, 'p_min', 1e4))   * 1.01
    p_hi  = float(getattr(fluid_g, 'p_max', 300e5)) * 0.99
    hg_lo = float(getattr(fluid_g, 'h_min', 50e3))  + 1.0
    hg_hi = float(getattr(fluid_g, 'h_max', 850e3)) - 1.0
    hl_lo = float(getattr(fluid_l, 'h_min', 50e3))  + 1.0
    hl_hi = float(getattr(fluid_l, 'h_max', 850e3)) - 1.0

    # ── Precomputed reservoir GAS isentrope (s = s_g0) for the stagnation inlets ──
    # The inlet needs p ON the reservoir isentrope at the current static h_g.  A
    # per-step REVERSE (h,s)→p solve (HmassSmass) is fragile — it hard-crashes via
    # lineax when h_g falls below the h where the isentrope leaves the table
    # (s=s_g0 wants p<p_min) — and slow.  Instead sample the isentrope ONCE here
    # (setup): fixed p grid → h via the reliable forward-ish PSmass call, giving a
    # monotone h_iso→p_iso lookup.  The inlet then does p_s = interp(h_g, h_iso,
    # p_iso): exact on-isentrope pressure, always finite, no runtime reverse solve.
    # jnp.interp CLAMPS below the range (h_dom < h_iso[0] → p_min) so the runaway
    # regime is bounded instead of crashing.  Only built when the closing entropy
    # is given (the stagnation inlets); other BCs never reference _h_iso/_p_iso.
    #
    # The reservoir→inlet acceleration pins ONE shared p on ONE phase's isentrope
    # (imposing both h_g,h_l AND both s_g0,s_l0 over-determines a single p).  We
    # close on the DOMINANT phase — the mass-weighted mixture entropy s_mix =
    # x·s_g + (1−x)·s_l is, to O(x), the dominant phase's entropy, and the trace
    # (ghost) phase's isentrope is compressible/ill-conditioned in its metastable
    # corner.  dom_phase='gas' → close on s_g0/h_g (gas-dominant tier, unchanged);
    # 'liquid' → close on s_l0/h_l (liquid-dominant flashing).
    _dom_gas = (dom_phase == 'gas')
    _s_dom0  = s_g0 if _dom_gas else s_l0
    _fl_dom  = fluid_g if _dom_gas else fluid_l
    if iso_h is not None and iso_p is not None:
        # Isentrope precomputed by the caller (robust FluidJAX/CoolProp reverse
        # solve): the DOMINANT-phase bicubic PSmass root-find has no root at low p
        # for the LIQUID isentrope (s_l0 lands two-phase there) and diverges — the
        # caller's equilibrium reverse call returns a finite, monotone h(p) instead.
        _p_iso = jnp.asarray(iso_p, dtype=jnp.float64)
        _h_iso = jnp.asarray(iso_h, dtype=jnp.float64)
    elif _s_dom0 is not None:
        _N_ISO = 400
        _p_iso = jnp.linspace(p_lo, p_hi, _N_ISO)
        _h_iso = _fl_dom.get_state(jxp.PSmass_INPUTS, _p_iso,
                                   jnp.full(_N_ISO, jnp.float64(_s_dom0)))["h"]

    # ── BC in U_h space ───────────────────────────────────────────────────────
    # BCs are applied to U_h = (p, u_g, u_l, α_g, h_g, h_l) — purely algebraic,
    # no EOS calls.  W is only used for the Newton iterate (dV term); ghost cells
    # in W are placeholders and do not carry physical BC information.
    #
    # Outlet: always transmissive (zero-gradient).
    # Inlet: transmissive (default) or total-state imposition.
    #
    # Total-state inlet:
    #   1. Extrapolate p, u_g, u_l from interior cells [1] and [2].
    #   2. Static enthalpies: h_k = H_k0 - u_k²/2  (no EOS needed).
    #   3. Impose α_g = alpha_g0.
    #
    # Key difference vs single-phase: p is extrapolated (not derived from an
    # isentrope) because two-phase flow has no single per-mixture isentrope.
    # U_h indices: 0=p, 1=u_g, 2=u_l, 3=α_g, 4=h_g, 5=h_l

    # ── Inlet BC (writes ghost cell U_h[0]) ───────────────────────────────────
    if bc_inlet == 'total_state':
        _H_g0     = jnp.float64(H_g0)
        _H_l0     = jnp.float64(H_l0)
        _alpha_g0 = jnp.float64(alpha_g0)

        def _bc_inlet(U_h):
            p_in   = 2.0 * U_h[1, 0] - U_h[2, 0]
            u_g_in = 2.0 * U_h[1, 1] - U_h[2, 1]
            u_l_in = 2.0 * U_h[1, 2] - U_h[2, 2]
            p_in   = jnp.clip(p_in,   p_lo,  p_hi)
            h_g_in = jnp.clip(_H_g0 - 0.5 * u_g_in**2, hg_lo, hg_hi)
            h_l_in = jnp.clip(_H_l0 - 0.5 * u_l_in**2, hl_lo, hl_hi)
            U_h0   = jnp.array([p_in, u_g_in, u_l_in, _alpha_g0, h_g_in, h_l_in])
            return U_h.at[0].set(U_h0)

    elif bc_inlet == 'stagnation':
        # Stagnation (total-state) inlet — characteristic-consistent reservoir BC.
        #
        # At a subsonic inlet only ONE characteristic is outgoing (u_mix − a < 0),
        # so we may extrapolate exactly ONE quantity: the mass-weighted MIXTURE
        # velocity u_mix = (α_g ρ_g u_g + α_l ρ_l u_l)/ρ_mix (linear extrapolation
        # from interior cells 1,2).  The two phase velocities are NOT independent —
        # they are tied by the imposed SLIP ratio S = u_g/u_l and, together with the
        # imposed QUALITY x (gas mass fraction), fix the void fraction via
        #     α_g = 1 / (1 + ((1−x)/x)·(ρ_g/ρ_l)·S).
        # Imposed (5): quality x0, slip S0, stagnation enthalpies H_g0/H_l0, and gas
        # stagnation entropy s_g0.  Derived: α_g (from x0,S0,ρ), the velocity SPLIT
        # (from u_mix and S0), h_k = H_k0 − u_k²/2, and the shared static p from the
        # GAS isentrope s_g(h_g,p)=s_g0.  Because ρ_g,ρ_l depend on (p,h) which depend
        # on the velocities, this is a small fixed point → a few unrolled passes.
        # The velocity MAGNITUDE (mass flux) still emerges from the interior u_mix,
        # so the nozzle can choke; slip/quality only set the composition + split.
        _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        _s_g0 = jnp.float64(s_g0)
        _x0   = jnp.float64(quality)          # gas quality (mass fraction)
        _S0   = jnp.float64(slip)             # slip ratio u_g/u_l
        _rx   = (1.0 - _x0) / _x0             # (1−x)/x

        def _umix_of(uh_cell):
            p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
            al_ = 1.0 - ag_
            rg_ = phase_props(fluid_g, p_, hg_).rho
            rl_ = phase_props(fluid_l, p_, hl_).rho
            return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)

        def _split(u_mix, ag, rg, rl):
            al = 1.0 - ag
            rmix = ag * rg + al * rl
            u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
            return _S0 * u_l, u_l          # (u_g, u_l)

        def _bc_inlet(U_h):
            # ZERO-ORDER extrapolation of the single outgoing characteristic (u_mix
            # = first interior cell), NOT the linear 2·u1−u2.  Linear extrapolation
            # is a HIGH-PASS filter: it amplifies inlet 2Δx noise and, if a cell
            # momentarily reverses, injects that reverse flow into the ghost → the
            # velocity split flips sign → garbage h_g → the isentrope Newton below
            # explodes p to the table edge.  Zero-order is 1st-order accurate, stable,
            # and matches Städtke's use of the actual first-cell mixture velocity.
            u_mix = _umix_of(U_h[1])
            p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)
            h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
            h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
            for _ in range(4):                                  # fixed point (ρ ⇄ u ⇄ p,h)
                rg = phase_props(fluid_g, p_s, h_g).rho
                rl = phase_props(fluid_l, p_s, h_l).rho
                ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)        # void from quality + slip
                u_g, u_l = _split(u_mix, ag, rg, rl)
                h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
                h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
                # p ON the reservoir isentrope at the DOMINANT phase's static h —
                # PRECOMPUTED isentrope lookup (interp), NOT a per-step reverse
                # (h,s)→p solve.  Exact on-isentrope pressure, always finite, clamps
                # below-range instead of crashing.  h_dom = h_g (gas-dominant) or
                # h_l (liquid-dominant flashing) — matches the _h_iso/_p_iso table.
                h_dom = h_g if _dom_gas else h_l
                p_s = jnp.clip(jnp.interp(h_dom, _h_iso, _p_iso), p_lo, p_hi)
            rg = phase_props(fluid_g, p_s, h_g).rho
            rl = phase_props(fluid_l, p_s, h_l).rho
            ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
            u_g, u_l = _split(u_mix, ag, rg, rl)
            U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])
            return U_h.at[0].set(U_h0)

    elif bc_inlet == 'stagnation_ramped':
        # Same characteristic-consistent reservoir BC as 'stagnation', but blend
        # the imposed state into the first three cells to soften the inlet forcing.
        _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        _s_g0 = jnp.float64(s_g0)
        _x0   = jnp.float64(quality)          # gas quality (mass fraction)
        _S0   = jnp.float64(slip)             # slip ratio u_g/u_l
        _rx   = (1.0 - _x0) / _x0             # (1−x)/x

        def _umix_of(uh_cell):
            p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
            al_ = 1.0 - ag_
            rg_ = phase_props(fluid_g, p_, hg_).rho
            rl_ = phase_props(fluid_l, p_, hl_).rho
            return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)

        def _split(u_mix, ag, rg, rl):
            al = 1.0 - ag
            rmix = ag * rg + al * rl
            u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
            return _S0 * u_l, u_l          # (u_g, u_l)

        def _bc_inlet(U_h):
            u_mix = 2.0 * _umix_of(U_h[1]) - _umix_of(U_h[2])   # extrapolate the 1 outgoing char
            p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)
            h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
            h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
            for _ in range(4):                                  # fixed point (ρ ⇄ u ⇄ p,h)
                rg = phase_props(fluid_g, p_s, h_g).rho
                rl = phase_props(fluid_l, p_s, h_l).rho
                ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)        # void from quality + slip
                u_g, u_l = _split(u_mix, ag, rg, rl)
                h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
                h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
                st  = fluid_g.get_state(jxp.HmassP_INPUTS,
                                        jnp.expand_dims(h_g, 0), jnp.expand_dims(p_s, 0))
                p_s = jnp.clip(p_s + st["rho"][0] * st["T"][0] * (st["s"][0] - _s_g0),
                               p_lo, p_hi)                       # gas isentrope Newton step
            rg = phase_props(fluid_g, p_s, h_g).rho
            rl = phase_props(fluid_l, p_s, h_l).rho
            ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
            u_g, u_l = _split(u_mix, ag, rg, rl)
            U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])
            return _apply_ramped_inlet(U_h, U_h0, n_cells=3)

    elif bc_inlet == "stagnation_ramped_soft":
        # RETIRED (2026-08-04): never selected anywhere in the repo (undocumented
        # in this docstring's bc_inlet enum, zero callers found by a full grep of
        # dev_scripts/), and its inline alternate gas-isentrope Newton step is
        # explicitly marked broken below ("BROKE convergence"). The under-
        # relaxed/damped-extrapolation implementation is left commented out
        # (not deleted) as the historical record of what was tried; see 'stagnation'
        # and 'stagnation_ramped' above for the maintained reservoir-BC variants.
        raise ValueError(
            "bc_inlet='stagnation_ramped_soft' is retired (never used, contains "
            "a known-broken inline Newton step). Use 'stagnation' or "
            "'stagnation_ramped' instead."
        )
        # # Characteristic-consistent reservoir BC, blended into the first three
        # # cells via UNDER-RELAXATION (not a hard overwrite) to avoid injecting
        # # grid-scale noise at the interior/BC interface. The extrapolation of
        # # the single outgoing characteristic is also damped (limited) instead
        # # of a raw linear extrapolation, since raw linear extrapolation is a
        # # high-pass filter on 2dx noise and will amplify checkerboard modes.
        # _H_g0 = jnp.float64(H_g0);  _H_l0 = jnp.float64(H_l0)
        # _s_g0 = jnp.float64(s_g0)
        # _x0   = jnp.float64(quality)
        # _S0   = jnp.float64(slip)
        # _rx   = (1.0 - _x0) / _x0
        #
        # _OMEGA   = jnp.float64(0.3)     # relaxation gain, e.g. 0.2-0.4
        # _EXTRAP_LIMIT = jnp.float64(0.5)        # 0 = no extrap (1st order),
        #                                         # 1 = full linear extrap
        #
        # def _umix_of(uh_cell):
        #     p_, ug_, ul_, ag_, hg_, hl_ = uh_cell
        #     al_ = 1.0 - ag_
        #     rg_ = phase_props(fluid_g, p_, hg_).rho
        #     rl_ = phase_props(fluid_l, p_, hl_).rho
        #     return (ag_ * rg_ * ug_ + al_ * rl_ * ul_) / (ag_ * rg_ + al_ * rl_)
        #
        # def _split(u_mix, ag, rg, rl):
        #     al = 1.0 - ag
        #     rmix = ag * rg + al * rl
        #     u_l  = u_mix * rmix / (ag * rg * _S0 + al * rl)
        #     return _S0 * u_l, u_l
        #
        # def _damped_extrap(u1, u2, u3):
        #     # Minmod-limited extrapolation of the outgoing characteristic instead
        #     # of raw 2*u1 - u2. Falls back toward u1 (1st order / zero-gradient)
        #     # whenever the interior gradient changes sign or is noisy, which is
        #     # exactly the checkerboard signature -> kills it at the source.
        #     d1 = u1 - u2
        #     d2 = u2 - u3
        #     same_sign = jnp.sign(d1) == jnp.sign(d2)
        #     slope = jnp.where(same_sign,
        #                     jnp.sign(d1) * jnp.minimum(jnp.abs(d1), jnp.abs(d2)),
        #                     0.0)
        #     return u1 + _EXTRAP_LIMIT * slope
        #
        # def _bc_inlet(U_h):
        #     u_mix = _damped_extrap(_umix_of(U_h[1]), _umix_of(U_h[2]), _umix_of(U_h[3]))
        #
        #     # p_s = jnp.clip(U_h[1, 0], p_lo, p_hi)   # (was) seed from the SPIKING cell 1
        #     p_s = jnp.clip(U_h[0, 0], p_lo, p_hi)     # warm-start from previous inlet (ghost)
        #     h_g = jnp.clip(U_h[1, 4], hg_lo, hg_hi)
        #     h_l = jnp.clip(U_h[1, 5], hl_lo, hl_hi)
        #     for _ in range(4):
        #         rg = phase_props(fluid_g, p_s, h_g).rho
        #         rl = phase_props(fluid_l, p_s, h_l).rho
        #         ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
        #         u_g, u_l = _split(u_mix, ag, rg, rl)
        #         h_g = jnp.clip(_H_g0 - 0.5 * u_g**2, hg_lo, hg_hi)
        #         h_l = jnp.clip(_H_l0 - 0.5 * u_l**2, hl_lo, hl_hi)
        #         # OLD gas-isentrope NEWTON step (commented — ρT~6000 → overshoots p to
        #         # the table edge → the inlet spike; under-relax/cap BROKE convergence):
        #         # st  = fluid_g.get_state(jxp.HmassP_INPUTS,
        #         #                         jnp.expand_dims(h_g, 0), jnp.expand_dims(p_s, 0))
        #         # p_s = jnp.clip(p_s + st["rho"][0] * st["T"][0] * (st["s"][0] - _s_g0),
        #         #             p_lo, p_hi)
        #         # p ON the reservoir isentrope at h_g — PRECOMPUTED isentrope lookup
        #         # (interp), NOT a per-step reverse (h,s)→p solve.  Exact on-isentrope
        #         # pressure, always finite, clamps below-range instead of crashing.
        #         p_s = jnp.clip(jnp.interp(h_g, _h_iso, _p_iso), p_lo, p_hi)
        #     rg = phase_props(fluid_g, p_s, h_g).rho
        #     rl = phase_props(fluid_l, p_s, h_l).rho
        #     ag = 1.0 / (1.0 + _rx * (rg / rl) * _S0)
        #     u_g, u_l = _split(u_mix, ag, rg, rl)
        #     U_h0 = jnp.array([p_s, u_g, u_l, ag, h_g, h_l])
        #
        #     return _apply_ramped_inlet_relaxed(U_h, U_h0, n_cells=3, omega=_OMEGA)
        #
        # def _apply_ramped_inlet_relaxed(U_h, U_h0, n_cells, omega):
        #     # Instead of overwriting cells 1..n_cells with a spatial ramp toward
        #     # U_h0 (hard reset every step -> impedance mismatch at cell n_cells+1
        #     # -> reflected 2dx noise), nudge each cell a fraction `omega` of the
        #     # way toward a *per-cell* target that itself ramps in weight from the
        #     # boundary inward. This is a soft spring, not a Dirichlet reset, so
        #     # residual-based (implicit) convergence is unaffected at steady state
        #     # -- omega only controls transient path, not the converged solution,
        #     # since at convergence U_h == U_h0 makes the correction vanish anyway.
        #     # GHOST cell 0 MUST carry the imposed reservoir state — it is what the
        #     # inlet flux (face between ghost 0 and cell 1) reads.  The soft ramp only
        #     # touches cells 1..n_cells, so WITHOUT this the ghost floats to garbage
        #     # (observed: p→0, u→−55, T<0 spike at x≈0).  Hard-set it like every other
        #     # inlet variant; the under-relaxation below is only for the interior cells.
        #     U_h = U_h.at[0].set(U_h0)
        #     idx = jnp.arange(1, n_cells + 1)
        #     ramp_w = jnp.linspace(1.0, 1.0 / n_cells, n_cells)   # 1 at inlet -> weaker inward
        #     target = U_h0[None, :]                                # same target state, weighted blend
        #     blended_target = ramp_w[:, None] * target + (1.0 - ramp_w[:, None]) * U_h[idx]
        #     U_h = U_h.at[idx].set(U_h[idx] + omega * (blended_target - U_h[idx]))
        #     return U_h

    elif bc_inlet == 'static_state':
        # Fully-fixed (Dirichlet) STATIC inlet — mirrors the space-marching demo:
        # impose the whole inlet state U_h_inlet = (p, u_g, u_l, α_g, h_g, h_l),
        # with p at the static inlet pressure, h_g = h_sat_g(p) (saturated vapour →
        # T_g = T_sat), h_l = h_sat_l(p) + Δh_l (superheated liquid).  Pinning p also
        # anchors the pressure level (transmissive outlet alone leaves it free).
        # Over-specifies the subsonic inflow by one characteristic (as the demo's
        # IVP does), but is the faithful reproduction of the demo inlet.
        _U_h_in = jnp.asarray(U_h_inlet, dtype=jnp.float64)

        def _bc_inlet(U_h):
            return U_h.at[0].set(_U_h_in)

    elif bc_inlet == 'subsonic_inflow':
        # Subsonic inflow: impose the 5 INCOMING characteristics — velocities,
        # void fraction, and phase enthalpies (u_g, u_l, α_g, h_g, h_l) from
        # U_h_inlet — and EXTRAPOLATE p (the one outgoing/backward-acoustic
        # characteristic, u_mix − a_mix < 0) from the interior.  Pressure is NOT
        # pinned here — it must be anchored downstream by a back-pressure outlet
        # (bc_outlet='pressure'); otherwise the pressure level is a free mode.
        _U_h_in = jnp.asarray(U_h_inlet, dtype=jnp.float64)

        def _bc_inlet(U_h):
            p_in = jnp.clip(U_h[1, 0], p_lo, p_hi)   # zero-order extrapolation from cell 1
            U_h0 = jnp.array([p_in, _U_h_in[1], _U_h_in[2],
                              _U_h_in[3], _U_h_in[4], _U_h_in[5]])
            return U_h.at[0].set(U_h0)

    else:  # transmissive: zero-gradient ghost
        def _bc_inlet(U_h):
            return U_h.at[0].set(U_h[1])

    # ── Outlet BC (writes ghost cell U_h[-1]) ─────────────────────────────────
    # p_out_t: optional RUNTIME back-pressure (traced), used by a ramped start-up
    # (driver bc.p_out_ramp_steps).  None -> the factory's p_out constant
    if bc_outlet == 'transmissive':
        def _bc_outlet(U_h, p_out_t=None):
            return U_h.at[-1].set(U_h[-2])
    elif bc_outlet == 'pressure':
        # Non-reflecting subsonic-outflow back-pressure (LODI / characteristic).
        # A naive "impose p_out, copy u" ghost changes ONLY pressure at the outlet
        # face → the FVS flux reads that pure pressure jump as an incoming acoustic
        # → REFLECTS it → the last cell's V is driven unrealizable and its EOS
        # projection fails (relEOS blows up) → dt collapses.  Instead we move p AND
        # u TOGETHER along the OUTGOING acoustic characteristic, preserving the
        # Riemann invariant J⁺ = u + ∫dp/(ρa), so imposing the back-pressure emits
        # NO reflected wave:
        #     u_ghost = u_interior + (p_interior − p_ghost) / (ρ_mix a_mix)
        # ρ_mix a_mix is the mixture acoustic impedance (Wood's frozen speed).  The
        # same velocity correction is applied to both phase velocities.  p_out_relax
        # blends the imposed pressure (1 = full back-pressure; <1 = softer anchor).
        _p_out = jnp.float64(p_out)
        _relax = jnp.float64(p_out_relax)

        def _bc_outlet(U_h, p_out_t=None):
            _po  = _p_out if p_out_t is None else p_out_t
            uh   = U_h[-2]
            p98  = uh[0];  ag = uh[3];  al = 1.0 - ag
            eg   = phase_props(fluid_g, p98, uh[4])
            el   = phase_props(fluid_l, p98, uh[5])
            rmix = ag * eg.rho + al * el.rho
            comp = ag / (eg.rho * eg.a**2) + al / (el.rho * el.a**2)
            imped = jnp.sqrt(rmix / comp)             # ρ_mix · a_Wood
            p_gh = p98 + _relax * (_po - p98)          # relax=1 → full back-pressure
            du   = (p98 - p_gh) / imped                # co-move u along outgoing char
            U_hN = jnp.array([p_gh, uh[1] + du, uh[2] + du, ag, uh[4], uh[5]])
            return U_h.at[-1].set(U_hN)
    elif bc_outlet == 'characteristic_old':
        # LEGACY (pre-2026-07-28) version of 'characteristic', kept verbatim as a
        # fallback / A-B reference.  Identical regime switch, but the velocity
        # co-move uses the HOMOGENEOUS (Wood) impedance and applies the SAME Δu to
        # both phases, leaving α_g and the enthalpies untouched.  That is a
        # homogeneous-mixture approximation of the true acoustic invariant; the
        # new 'characteristic' below projects onto the model's own acoustic
        # eigenvector instead.  The two agree in the single-phase limits (measured
        # a_Wood/a_Städtke = 1.000 at the gasdom outlet) and differ by up to ~2.3×
        # in the impedance at intermediate α_g.  Since the residual Δu at steady
        # state is O(Δx) (~0.2% of the exit velocity in the shipped runs), results
        # from the two paths are interchangeable to ~0.06% there.
        #
        # Regime-aware outlet that AUTO-SWITCHES subsonic ↔ supersonic, mirroring
        # the single-phase Städtke outlet (fvm.py make_jax_stadtke_step._bc_outlet).
        # The two-fluid eigenstructure is analogous: exactly ONE acoustic wave
        # (u_mix − a_mix) can enter at a subsonic outlet; all material waves
        # (u_g, u_l) are outgoing.  So:
        #   subsonic  (M_mix<1): impose p_out and co-move u along the outgoing
        #                        acoustic (the LODI 'pressure' ghost above),
        #   supersonic(M_mix>1): extrapolate all 6 (the 'transmissive' ghost).
        # Blend the two by a SMOOTH tanh weight on the MIXTURE Mach so the implicit
        # Newton keeps a non-zero BC Jacobian near M=1 (lets p_out propagate
        # upstream).  Because both ghosts share the interior (α, h_g, h_l), the
        # blend collapses to scaling the LODI pressure correction by w:
        #     p_ghost = p98 + w·relax·(p_out − p98),   u += (p98 − p_ghost)/impedance
        # w→1 (subsonic) = full back-pressure;  w→0 (supersonic) = pure extrapolation.
        #
        # CRITICAL: the sub/supersonic Mach MUST be measured with the SCHEME's OWN
        # characteristic speeds — the Städtke eigenstructure `u_mix ± a` (which the
        # flux and CFL use, incl. virtual mass k via rho_hat_s and slip via
        # delta_a_sq) — NOT the Wood speed a_W = √(1/(ρ_mix·Σαᵢ/(ρᵢaᵢ²))).  Using the
        # Wood speed (which differs from a_Städtke) mis-classifies the outlet: it
        # reads M_W ≥ 1 → w ≈ 0 → the back-pressure is silently dropped and the
        # outlet behaves TRANSMISSIVE even when the true outgoing acoustic u_mix−a is
        # still < 0 (subsonic).  We therefore take M_out from eigenstructure().  The
        # Wood impedance is kept ONLY for the LODI velocity co-move (shared with the
        # working 'pressure' outlet; it sets the reflection quality, not the regime).
        _p_out = jnp.float64(p_out)
        _relax = jnp.float64(p_out_relax)
        _msw   = jnp.float64(0.05)          # tanh half-width in Mach (switch sharpness)

        def _bc_outlet(U_h, p_out_t=None):
            _po  = _p_out if p_out_t is None else p_out_t
            uh   = U_h[-2]
            p98  = uh[0];  ag = uh[3];  al = 1.0 - ag
            eg   = phase_props(fluid_g, p98, uh[4])
            el   = phase_props(fluid_l, p98, uh[5])
            rmix = ag * eg.rho + al * el.rho
            comp = ag / (eg.rho * eg.a**2) + al / (el.rho * el.a**2)
            imped = jnp.sqrt(rmix / comp)                  # ρ_mix · a_Wood (LODI co-move only)
            # Regime switch from the scheme's OWN eigenstructure (Städtke wave speeds):
            # eigvals = [u_g, u_l, u_mix+a, u_mix−a, u_g, u_l]  ⇒  a=½(ev₂−ev₃),
            # u_mix=½(ev₂+ev₃).  Subsonic outlet ⇔ outgoing acoustic ev₃=u_mix−a < 0.
            cq   = common_quantities(uh, eg, el, k)
            ev   = eigenstructure(uh, eg, el, cq)[0]
            a_s  = 0.5 * (ev[2] - ev[3])                   # Städtke mixture sound speed (>0)
            u_s  = 0.5 * (ev[2] + ev[3])                   # hat-weighted mixture velocity
            M_out = jnp.abs(u_s) / (a_s + 1e-300)
            w     = 0.5 * (1.0 + jnp.tanh((1.0 - M_out) / _msw))  # 1 sub, 0 super
            p_gh  = p98 + w * _relax * (_po - p98)
            du    = (p98 - p_gh) / imped
            U_hN  = jnp.array([p_gh, uh[1] + du, uh[2] + du, ag, uh[4], uh[5]])
            return U_h.at[-1].set(U_hN)
    elif bc_outlet == 'characteristic':
        # Regime-aware outlet, EIGENVECTOR-PROJECTION form (2026-07-28).  Same
        # regime switch as 'characteristic_old' (tanh blend on the Städtke mixture
        # Mach), but the ghost state is built the way the characteristic theory
        # actually prescribes.
        #
        # At a subsonic outlet exactly ONE wave enters: the acoustic λ₄ = û − a < 0.
        # A non-reflecting ghost must therefore perturb the interior state ALONG
        # the corresponding RIGHT eigenvector r₄ and along nothing else — that
        # leaves the five outgoing wave amplitudes untouched, so imposing p_out
        # emits no reflection:
        #     U_ghost = U[N-2] + s·r₄,     s = (p_ghost − p[N-2]) / (r₄)_p
        # The scaling makes the pressure component land exactly on the (blended)
        # target back-pressure, and every other component follows from r₄.
        #
        # What this buys over the legacy shared-Δu form:
        #   * (r₄)_{u_g} = R_24 ≠ (r₄)_{u_l} = R_34 — the two phases get DIFFERENT
        #     velocity shifts (they coincide only at zero slip and equal ρ̂),
        #   * (r₄)_α = χ ≠ 0 — the acoustic wave carries a void-fraction change,
        #   * (r₄)_{h_k} = ρ̂²ã²/ρ_k, so s·(r₄)_{h_k} = (p_ghost − p)/ρ_k, i.e.
        #     exactly the isentropic dh = dp/ρ an acoustic wave transports (the
        #     legacy ghost froze h_k at the interior value).
        # No Wood impedance appears anywhere: the eigenvector supplies its own,
        # consistent with the flux, the CFL and the regime switch.
        #
        # (r₄)_p = ρ̂²ã² > 0 always (ρ̂² = ρ_gρ_l + kρ_mix², ã² > 0), so the scaling
        # never divides by zero; the ε below is belt-and-braces for the ghost limit.
        # As w→0 (supersonic) p_ghost→p[N-2] ⇒ s→0 ⇒ pure extrapolation, so the
        # operator stays smooth across the sonic point exactly as before.
        _p_out = jnp.float64(p_out)
        _relax = jnp.float64(p_out_relax)
        _msw   = jnp.float64(0.05)          # tanh half-width in Mach (switch sharpness)
        _a_lo  = jnp.float64(alpha_eps)     # ghost α_g floor (same one the interior uses)

        def _bc_outlet(U_h, p_out_t=None):
            _po  = _p_out if p_out_t is None else p_out_t
            uh   = U_h[-2]
            p98  = uh[0]
            eg   = phase_props(fluid_g, p98, uh[4])
            el   = phase_props(fluid_l, p98, uh[5])
            cq   = common_quantities(uh, eg, el, k)
            ev, R_h = eigenstructure(uh, eg, el, cq)[:2]
            # Regime from the scheme's OWN wave speeds (see 'characteristic_old':
            # using the Wood speed here mis-classifies the outlet and silently
            # drops the back-pressure).
            a_s   = 0.5 * (ev[2] - ev[3])                 # mixture sound speed (>0)
            u_s   = 0.5 * (ev[2] + ev[3])                 # hat-weighted mixture velocity
            M_out = jnp.abs(u_s) / (a_s + 1e-300)
            w     = 0.5 * (1.0 + jnp.tanh((1.0 - M_out) / _msw))   # 1 sub, 0 super
            p_gh  = p98 + w * _relax * (_po - p98)
            # Project the pressure change onto the INCOMING acoustic eigenvector.
            r4    = R_h[:, 3]                             # col 4 ↔ λ₄ = û − a
            s     = (p_gh - p98) / (r4[0] + 1e-300)
            U_hN  = uh + s * r4
            # Realizability guards.  The perturbation is tiny in practice (Δp of a
            # few hundred Pa ⇒ Δh = Δp/ρ of order 10 J/kg), so these clips are
            # inactive — identity, and thus invisible to the Newton Jacobian —
            # everywhere except a pathological transient.  They exist so a large
            # startup Δp cannot push the ghost off the EOS tables or through the
            # void bounds, which is how the legacy path used to die.
            U_hN  = U_hN.at[0].set(jnp.clip(U_hN[0], p_lo, p_hi))
            U_hN  = U_hN.at[3].set(jnp.clip(U_hN[3], _a_lo, 1.0 - _a_lo))
            U_hN  = U_hN.at[4].set(jnp.clip(U_hN[4], hg_lo, hg_hi))
            U_hN  = U_hN.at[5].set(jnp.clip(U_hN[5], hl_lo, hl_hi))
            return U_h.at[-1].set(U_hN)
    else:
        raise ValueError(f"unknown bc_outlet={bc_outlet!r} (only 'transmissive', "
                         f"'pressure', 'characteristic', 'characteristic_old')")

    def _bc_U_h(U_h, p_out_t=None):
        return _bc_outlet(_bc_inlet(U_h), p_out_t)

    # ── MUSCL van Leer in U_h space ───────────────────────────────────────────

    def _muscl(U_h):
        dU = U_h[1:] - U_h[:-1]
        if recon == 'muscl_van_albada':
            # SMOOTH limiter (2026-07-14): ab(a+b)/(a²+b²+ε²), gated at ab<0.
            # van Leer/minmod SWITCH slopes cell-by-cell; on the ghost-scale 2Δx
            # noise of the ghost-limit case that flicker slowly pumps the state
            # (same pathology class as the du_blend branch alternation).  The ε²
            # floor makes the slope response to sub-ε noise O(noise³/ε²) → no
            # flicker; smooth-gradient slopes (≫ε) are untouched.
            # eps normalisation is selectable (va_eps_mode):
            #
            # 'magnitude' (DEFAULT, original 2026-07-14 behaviour, va_rel=1e-3):
            #     eps ~ |U| per cell, with a domain-mean fallback so u through 0
            #     keeps a finite floor.  Reproduces every run made before
            #     2026-09-09 bit-for-bit.
            #
            # 'range' (EXPERIMENTAL 2026-09-09, va_rel=1e-4):
            #     eps ~ (max U - min U) per variable.  The |U|-based floor tracks
            #     field MAGNITUDE, so any variable that is a large offset with a
            #     small variation gets a floor above its own cell-to-cell
            #     increments and is limited to 1st order at every resolution --
            #     measured on the N=500 two-component run, exit region:
            #         alpha_g (0.978 +/- 0.019) kept  0.03% of its true slope
            #         h_l     (9.7e4 +/- 4.3e3) kept  9.5%
            #         h_g                       kept 12.7%
            #     while u_g/u_l (variation ~ magnitude) were untouched, which is
            #     why only the velocities showed 2nd order and the observed order
            #     sat at ~1.2 instead of 2.
            #     va_rel must stay well below 1/N (the smooth-region cell
            #     increment is ~range/N): 1e-4 leaves ~17x headroom at N=600.
            #     KNOWN TRADE-OFF: removing the over-limiting also removes noise
            #     suppression it was providing -- on N=256 the 2dx checkerboard in
            #     u_l went 0.27% -> 1.70% of range and alpha 0.49% -> 1.14%, and
            #     the mdot conservation spread worsened 2.35% -> 2.84%.  Not yet
            #     recommended for production; see also an h-scaled floor
            #     (eps^2 ~ dx^3) which would fix the asymptotics without this cost.
            #
            # The +1e-12*mean term in 'range' is a degenerate-field guard (rng==0).
            a, b = dU[:-1], dU[1:]
            if va_eps_mode == 'range':
                _va_scale = (jnp.max(U_h, axis=0) - jnp.min(U_h, axis=0)
                             + 1e-12 * jnp.mean(jnp.abs(U_h), axis=0))[None, :]
            else:                       # 'magnitude' and 'smoothness'
                _va_mean = jnp.mean(jnp.abs(U_h), axis=0)
                if va_eps_mode == 'magnitude' and va_mean_stop_gradient:
                    # The domain mean only sets an eps SCALE: keep its value,
                    # drop its derivative.  That derivative couples every cell
                    # to every other one (not representable in the banded/
                    # colored Jacobian, which folds it into wrong in-band
                    # entries), and through a ghost cell's h_g = E/(αρ) it grows
                    # like 1/α_ghost: colored-vs-dense Jacobian error 3e-5 of the
                    # row max at α_in=1e-3 but 21-31% at 1e-7, which held SER in
                    # a limit cycle and blew up α_in=1e-8.  The residual is
                    # unchanged, so converged states are too (to SSR tolerance);
                    # only the Newton path differs.  ('smoothness' scales this
                    # term by 1e-12 and needs no fix.)
                    _va_mean = jax.lax.stop_gradient(_va_mean)
                _va_scale = (jnp.abs(U_h[1:-1]) + 0.1 * _va_mean[None, :])
            eps  = _va_rel_eff * _va_scale
            s    = jnp.where(a * b > 0.0,
                             a * b * (a + b) / (a*a + b*b + eps*eps), 0.0)
            if va_eps_mode == 'smoothness':
                # Detect the 2dx mode by the SHAPE of the stencil rather than by
                # thresholding its amplitude -- which is what both amplitude
                # modes get wrong, in opposite directions.
                #
                #   r = |a-b| / |a+b|
                #     numerator   = |2nd difference|  ~ O(h^2) where smooth
                #     denominator = |centred diff|    ~ O(h)   where smooth
                #   => r ~ O(h) in smooth cells (gate -> 1, FULL 2nd order as
                #      h -> 0), but r ~ O(1) on a 2dx mode, independent of h.
                # Measured on two_component N=256/358/500 (magnitude runs):
                #   r median  alpha_g 0.01377 / 0.00993 / 0.00712  (ratios 1.39,
                #   1.39 vs the h ratio 1.40 -- r tracks h almost exactly), while
                #   r_max stayed ~8-14 on every grid.
                # Because a*b>0 in the active branch, |a+b| = |a|+|b| >= |a-b|,
                # so r <= 1 there; r0=0.5 puts the half-suppression point inside
                # that range.  m >= 3 keeps the smooth-region perturbation
                # O(h^m), i.e. below the O(h^2) truncation error.
                # Retention measured at N=500 (active cells, bulk r<p50 / rough
                # r>p99):  magnitude 4-98% / 5-80%   (mis-targeted: it passes
                # 79.6% of p's roughest slopes while killing 96% of alpha's
                # smoothest);  range 94-99.7% / 64-97% (no tail suppression);
                # smoothness 100% / 7-31%.
                # r is clipped only to keep r**m finite when a+b -> 0 exactly;
                # the gate is already ~1e-6 there.
                # SMOOTH FORM (required): the residual is differentiated by
                # jax.jacfwd, so abs()/minimum() would put kinks straight into
                # the Jacobian -- and |a-b| kinks exactly at a==b, which IS the
                # smooth case (93.8% of cells have r<0.1).  Newton then loses
                # quadratic convergence.  Since va_rm is even,
                #     (r/r0)**m == (r^2/r0^2)**(m/2)
                # and r^2 needs no abs at all, so this is the SAME gate written
                # without a single non-differentiable operation.  The clip is
                # unnecessary too: as (a+b)->0, q->inf and the gate -> 0 on its
                # own (written as a product so inf never reaches a subtraction).
                _q    = (a - b) ** 2 / ((a + b) ** 2 + 1e-300)
                _gate = 1.0 / (1.0 + (_q / (_va_r0 * _va_r0)) ** (_va_rm // 2))
                s     = s * _gate
            sigma = jnp.zeros_like(U_h).at[1:-1].set(s)
            U_L_m  = U_h[:-1] + 0.5 * sigma[:-1]
            U_R_m  = U_h[1:]  - 0.5 * sigma[1:]
            bad_L  = ((U_L_m[:, 0] < p_lo)  | (U_L_m[:, 0] > p_hi)  |
                      (U_L_m[:, 4] < hg_lo) | (U_L_m[:, 4] > hg_hi) |
                      (U_L_m[:, 5] < hl_lo) | (U_L_m[:, 5] > hl_hi))
            bad_R  = ((U_R_m[:, 0] < p_lo)  | (U_R_m[:, 0] > p_hi)  |
                      (U_R_m[:, 4] < hg_lo) | (U_R_m[:, 4] > hg_hi) |
                      (U_R_m[:, 5] < hl_lo) | (U_R_m[:, 5] > hl_hi))
            bad    = (bad_L | bad_R)[:, None]
            return jnp.where(bad, U_h[:-1], U_L_m), jnp.where(bad, U_h[1:], U_R_m)
        if recon == 'muscl_van_leer':
            lim = lambda a, b: jnp.where(
                a * b > 0, 2.0 * a * b / (a + b + 1e-300), 0.0)
        elif recon == 'muscl_minmod':
            lim = lambda a, b: jnp.where(
                a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
        else:   # mc
            lim = lambda a, b: jnp.where(
                a * b > 0,
                jnp.sign(a) * jnp.minimum(
                    jnp.minimum(2*jnp.abs(a), 2*jnp.abs(b)),
                    0.5*jnp.abs(a + b)), 0.0)
        sigma  = jnp.zeros_like(U_h).at[1:-1].set(lim(dU[:-1], dU[1:]))
        U_L_m  = U_h[:-1] + 0.5 * sigma[:-1]
        U_R_m  = U_h[1:]  - 0.5 * sigma[1:]
        bad_L  = ((U_L_m[:, 0] < p_lo)  | (U_L_m[:, 0] > p_hi)  |
                  (U_L_m[:, 4] < hg_lo) | (U_L_m[:, 4] > hg_hi) |
                  (U_L_m[:, 5] < hl_lo) | (U_L_m[:, 5] > hl_hi))
        bad_R  = ((U_R_m[:, 0] < p_lo)  | (U_R_m[:, 0] > p_hi)  |
                  (U_R_m[:, 4] < hg_lo) | (U_R_m[:, 4] > hg_hi) |
                  (U_R_m[:, 5] < hl_lo) | (U_R_m[:, 5] > hl_hi))
        bad    = (bad_L | bad_R)[:, None]
        return jnp.where(bad, U_h[:-1], U_L_m), jnp.where(bad, U_h[1:], U_R_m)

    # ── FVS fluxes at all N-1 faces ───────────────────────────────────────────

    # BOUNDARY-FACE low-Mach exemption (2026-07-17): the back-pressure (and the
    # reservoir state) reach the interior ONLY through the acoustic jump terms
    # of the first/last face flux — the very terms scaled by z_ac=clip(Ma,
    # lowmach_diss,1).  With lowmach_diss=0 at Ma~0.007 the BC coupling is
    # ~150× weakened: the imposed p_out parks as a STANDING JUMP at the outlet
    # face and the mass flow only creeps toward the BC-consistent value
    # (diagnosed on the 2026-07-16 overnight runs: grid-dependent settled
    # inlet u, slow outlet-tail drain).  `lowmach_diss_bc` (default None =
    # inherit lowmach_diss, no behavior change) sets the z_ac FLOOR on the two
    # boundary faces only; 1.0 = full upwind dissipation there = stiff BC
    # enforcement.  Regime-safe: at supersonic outlets z_ac=1 already (no-op)
    # and the characteristic BC extrapolates (no jump); at high subsonic it
    # moves ≤×2 toward the standard full-upwind scheme.
    _lm_face = jnp.full(N - 1, jnp.float64(lowmach_diss))
    if lowmach_diss_bc is not None:
        _lm_face = (_lm_face.at[0].set(jnp.float64(lowmach_diss_bc))
                            .at[-1].set(jnp.float64(lowmach_diss_bc)))

    def _local_jac_map(f):
        """Batched map X (B, n) -> f(x) per row whose JVP linearizes f ONCE per
        row — the exact AD Jacobian from n forward tangents — and serves every
        tangent seed as a small mat-vec (cell_local_jac).  The colored Jacobian
        pushes ~40 seeds through the residual, so per-cell work drops from 40
        tangent passes to n.  Values and derivatives are unchanged."""
        @jax.custom_jvp
        def F(X):
            return jax.vmap(f)(X)

        @F.defjvp
        def F_jvp(primals, tangents):
            (X,), (dX,) = primals, tangents

            def one(x):
                y, lin = jax.linearize(f, x)
                return y, jax.vmap(lin, out_axes=-1)(jnp.eye(x.shape[0], dtype=x.dtype))

            Y, Jc = jax.vmap(one)(X)
            return Y, jnp.einsum('b...i,bi->b...', Jc, dX)
        return F

    def _flux_one(ul, ur, lm):
        F = fvs_interface_flux(
            ul, ur, fluid_g, fluid_l, k, delta_factor, lowmach_diss=lm,
            du_blend_lo=du_blend_lo, du_blend_hi=du_blend_hi,
            single_branch=flux_single_branch,
        )
        if ghost_mass_upwind_acut is not None:
            F = _ghost_mass_upwind(F, ul, ur)
        return F

    # ── Positivity-preserving ghost MASS flux (ghost_mass_upwind_acut) ────────
    # The FVS acoustic correction (u−a wave) is built from the AVERAGE face
    # state, so its ghost-mass component scales with (α_L+α_R)/2: across a front
    # where α jumps by decades (flashing onset, change of inlet α) it moves ghost
    # mass ~α_R·ρ·u through the face and empties the thin side (measured 6.5× the
    # upwind flux at the onset face of the α_in=1e-7 SMD case → α hole → NaN).
    # Where EITHER side is a thin ghost, the ghost phase's mass row becomes the
    # pure upwind flux α·ρ·u (conservative, positivity preserving); the other
    # five rows and all faces with α ≫ acut on both sides keep the FVS value.
    # Reads only U_L, U_R → the colored Jacobian stays exact (jac_diff 1e-14).
    if ghost_mass_upwind_acut is not None:
        _c_gu = float(jnp.log10(ghost_mass_upwind_acut))

        def _wghost(a):                       # 1 for a ≪ acut, 0 for a ≫ acut
            return 0.5 * (1.0 - jnp.tanh(
                (jnp.log10(jnp.clip(a, 1e-300, 1.0)) - _c_gu) / 0.25))

        def _ghost_mass_upwind(F, ul, ur):
            rgL = phase_props(fluid_g, ul[0], ul[4]).rho
            rgR = phase_props(fluid_g, ur[0], ur[4]).rho
            rlL = phase_props(fluid_l, ul[0], ul[5]).rho
            rlR = phase_props(fluid_l, ur[0], ur[5]).rho
            ug_f = 0.5 * (ul[1] + ur[1]);  ul_f = 0.5 * (ul[2] + ur[2])
            sg = jnp.tanh(ug_f / 1e-3);    sl = jnp.tanh(ul_f / 1e-3)   # smooth sign
            Fg_up = (0.5 * (1.0 + sg) * ul[3] * rgL * ul[1]
                     + 0.5 * (1.0 - sg) * ur[3] * rgR * ur[1])
            Fl_up = (0.5 * (1.0 + sl) * (1.0 - ul[3]) * rlL * ul[2]
                     + 0.5 * (1.0 - sl) * (1.0 - ur[3]) * rlR * ur[2])
            wg = 1.0 - (1.0 - _wghost(ul[3])) * (1.0 - _wghost(ur[3]))
            wl = 1.0 - (1.0 - _wghost(1.0 - ul[3])) * (1.0 - _wghost(1.0 - ur[3]))
            return (F.at[0].set(wg * Fg_up + (1.0 - wg) * F[0])
                     .at[1].set(wl * Fl_up + (1.0 - wl) * F[1]))

    # flux_local_jac: each face flux depends on only 12 inputs (U_L, U_R); the
    # custom JVP linearizes it once per face (exact 6×12 block) and serves every
    # colored seed as a 6×12 mat-vec.  The low-Mach floor is a per-face constant.
    def _flux_and_jac_one(ul, ur, lm):
        x = jnp.concatenate([ul, ur])
        F, lin = jax.linearize(lambda x_: _flux_one(x_[:6], x_[6:], lm), x)
        return F, jax.vmap(lin, out_axes=1)(jnp.eye(12, dtype=x.dtype))   # (6,12)

    @jax.custom_jvp
    def _flux_faces_local(U_L, U_R):
        return jax.vmap(_flux_one)(U_L, U_R, _lm_face)

    @_flux_faces_local.defjvp
    def _flux_faces_local_jvp(primals, tangents):
        (U_L, U_R), (dL, dR) = primals, tangents
        F, Jx = jax.vmap(_flux_and_jac_one)(U_L, U_R, _lm_face)
        return F, jnp.einsum('fij,fj->fi', Jx, jnp.concatenate([dL, dR], axis=1))

    def _fvs_all_faces(U_h):
        if recon != 'constant':
            U_L, U_R = _muscl(U_h)
        else:
            U_L, U_R = U_h[:-1], U_h[1:]
        if flux_local_jac:
            return _flux_faces_local(U_L, U_R)
        return jax.vmap(_flux_one)(U_L, U_R, _lm_face)

    # ── Non-conservative correction at interior cells ─────────────────────────

    def _nc_all_interior(U_h, F_hat):
        # nc_form — WHERE the NC-geometric term H·F̄·ΔA lives (2026-07-15):
        #   'cell'     (ours): H^nc·(F̂₊−F̂₋), all geometry in D_VE_area.
        #   'staedtke' (book eq. 8.38): H^nc·(A₊F̂₊−A₋F̂₋)/A_i — geometry
        #              partially inside the NC face sum.
        # The two differ by H^nc·F̄·(dA/dx)·Δx/A; correctness is a PAIRING
        # property with the area-source content — discriminated by the
        # α-contact step-map probe (debug_nc_contact.py).
        if nc_form == 'staedtke':
            dF = (Af_j[1:, None] * F_hat[1:]
                  - Af_j[:-1, None] * F_hat[:-1]) / A_j[1:-1, None]
        elif nc_form == 'wb':
            # WELL-BALANCED form (2026-07-15): face-area-weighted one-sided
            # differences around the CELL flux.  The conservative+source pact
            # leaves the remainder p/2·[A₊Δα₊ + A₋Δα₋]; the plain central
            # 'cell' form supplies p·A_i·(Δα₊+Δα₋)/2 instead — mismatch
            # ~(p/4)·ΔA·Δα, convicted by the α-contact-on-throat probe
            # (kick ×1.5/step at u≈0).  'wb' reduces to 'cell' on uniform A,
            # vanishes identically on uniform states (F̂ = F(U_i)), and
            # supplies exactly the area-weighted difference the pact needs.
            if fast_wb_flux:
                # SPEED-UP: with U_L = U_R the FVS flux reduces EXACTLY to the
                # physical flux (zero-slip: F_12b = F and y_dF = K⁻¹·0 = 0;
                # full: rho·rho⁻¹·F = F), so skip the full FVS machinery.
                def _fc_one(u):
                    return flux_vector(u, phase_props(fluid_g, u[0], u[4]),
                                       phase_props(fluid_l, u[0], u[5]))
                F_c = (_local_jac_map(_fc_one) if cell_local_jac
                       else jax.vmap(_fc_one))(U_h[1:-1])
            else:
                F_c = jax.vmap(
                    lambda u: fvs_interface_flux(
                        u, u, fluid_g, fluid_l, k, delta_factor,
                        lowmach_diss=lowmach_diss,
                        du_blend_lo=du_blend_lo, du_blend_hi=du_blend_hi,
                        single_branch=flux_single_branch)
                )(U_h[1:-1])
            dF = (Af_j[1:, None] * (F_hat[1:] - F_c)
                  + Af_j[:-1, None] * (F_c - F_hat[:-1])) / A_j[1:-1, None]
        else:
            dF = F_hat[1:] - F_hat[:-1]

        def _nc_one(u_h, df):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            return nonconservative_correction(u_h, df, eos_g_c, eos_l_c, cq_c)

        if cell_local_jac:
            return _local_jac_map(lambda x: _nc_one(x[:6], x[6:]))(
                jnp.concatenate([U_h[1:-1], dF], axis=1))
        return jax.vmap(_nc_one)(U_h[1:-1], dF)

    # ── Source terms at interior cells ────────────────────────────────────────

    def _src_all_interior(U_h):
        def _src_one(u_h, dlnA):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            if sources_fn is not None:
                # Saturation properties at current cell pressure (scalar p = u_h[0]).
                # fluid_g is the vapor table → sat. vapor props.
                # fluid_l is the liquid table → sat. liquid props.
                # sources_fn signature:
                #   (u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l) → D_VE (6,)
                # use_saturation=False: tables built WITHOUT the metastable/saturation
                # extension (e.g. Air, or a purely subcooled liquid) have no saturation
                # props, so skip the lookup and pass zero placeholders.  Only safe when
                # the sources_fn does not read sat_g/sat_l (e.g. area source only, or
                # mass transfer disabled).
                if use_saturation:
                    sat_g_c = fluid_g.get_state_saturation(u_h[0])
                    sat_l_c = fluid_l.get_state_saturation(u_h[0])
                else:
                    _z = jnp.zeros(())
                    sat_g_c = {"temperature": _z, "enthalpy": _z}
                    sat_l_c = {"temperature": _z, "enthalpy": _z}
                return sources_fn(u_h, eos_g_c, eos_l_c, cq_c, dlnA, sat_g_c, sat_l_c)
            # FALLBACK ONLY (sources_fn is None): primitive-frame area source with a
            # zero physical closure.  NOT used when a sources_fn is supplied (the
            # driver always supplies one → the ALGEBRAIC D_VE_algebric/D_VE_area path
            # above runs instead).  Kept for the no-sources_fn convenience case.
            z  = jnp.zeros(())
            cl = TwoFluidClosure(
                sigma_Mg=z, sigma_Qg=z, sigma_Ql=z, Fv_g=z,
                F_g=z, F_l=z, Q_g=z, Q_l=z,
                u_ex=z, u_int=z, h_ex_g=z, h_ex_l=z, dlnA_dx=dlnA,
            )
            return D_VE_conservative(u_h, eos_g_c, eos_l_c, cq_c, cl, k)

        if cell_local_jac:
            return _local_jac_map(lambda x: _src_one(x[:6], x[6]))(
                jnp.concatenate([U_h[1:-1], dlnA_dx[1:-1, None]], axis=1))
        return jax.vmap(_src_one)(U_h[1:-1], dlnA_dx[1:-1])

    # ── Residual:  R ∈ R^{6·(N-2)},  W_int ∈ R^{8·(N-2)} ───────────────────

    def _residual(W_int_flat, W_n_flat, dt, p_out_t=None):
        W_int = W_int_flat.reshape(N - 2, 8)
        W_n   = W_n_flat.reshape(N, 8)

        W = jnp.concatenate([W_n[:1], W_int, W_n[-1:]], axis=0)

        # Conservative variables — POLYNOMIAL in W, no EOS chain in dV/dW
        # Only interior cells [1:-1] enter dV, so ghost-cell W values don't matter.
        V_new = jax.vmap(V_from_W)(W)
        V_old = jax.vmap(V_from_W)(W_n)

        # Primitives U_h — ALGEBRAIC from W, then BCs applied in U_h space
        U_h = jax.vmap(U_h_from_W)(W)
        U_h = _bc_U_h(U_h, p_out_t)

        # FVS fluxes (EOS called internally by fvs_interface_flux)
        F_hat = _fvs_all_faces(U_h)

        # Non-conservative correction and sources (from the UNfiltered flux)
        nc  = _nc_all_interior(U_h, F_hat)
        src = _src_all_interior(U_h)

        # ── 4th-order (JST-type) background filter ────────────────────────────
        # Damps the 2Δx void/alpha checkerboard that the FVS leaves undamped at
        # low phase velocity (λ=u_g,u_l→0 ⇒ ~zero material-wave dissipation).
        # Added as a conservative FLUX (telescopes → mass/momentum/energy exactly
        # conserved), scaled by the local spectral radius so it does NOT vanish as
        # u_g,u_l→0.  4th-order ⇒ perturbs the SMOOTH solution only at O(eps_void·Δx³)
        # (negligible / ε-insensitive) while strongly damping the checkerboard.
        # eps_void=0 → identically off.  Needs the ±2 stencil ⇒ _sw widened below.
        #
        # LOW-VELOCITY GATING (eps_void_ucut): the α/void checkerboard is a MATERIAL-
        # wave mode — it rides the phase-velocity eigenvalues λ=u_g,u_l, whose FVS
        # dissipation ~|λ|·Δx vanishes as u_g,u_l→0.  So the physically-correct
        # "smallness" sensor is the PHASE VELOCITY itself, NOT a Mach number: in a
        # two-phase mixture a_mix is the Wood speed, a strongly non-monotone function
        # of α, so dividing by it would move the gate with α rather than with the
        # velocity.  Sensor: u_mat = min(|u_g|,|u_l|) [m/s] — the slowest material
        # wave (min ⇒ the filter stays ON if EITHER phase slows, robust under slip).
        # Gate: multiply the filter by a smooth low-pass weight
        #     w = ½(1 + tanh((u0 − u_mat)/Δu))   → 1 for u_mat≪u0, → 0 for u_mat≫u0,
        # so the filter is active ONLY in the low-velocity (inlet) region and is OFF
        # at the throat/supersonic region (full high-order accuracy preserved there).
        # u0=eps_void_ucut [m/s], Δu=eps_void_uwidth [m/s] (defaults to u0/3 if None).
        # eps_void_ucut=None ⇒ ungated (uniform, the original behaviour).  Reads the
        # velocities from the SAME ±2-stencil cells ⇒ does NOT widen the band.
        if eps_void > 0.0 or eps_ghost > 0.0:
            def _lam_umat_cell(u_h):
                eg_ = phase_props(fluid_g, u_h[0], u_h[4])
                el_ = phase_props(fluid_l, u_h[0], u_h[5])
                ev, _, _, _ = eigenstructure(
                    u_h, eg_, el_, common_quantities(u_h, eg_, el_, k))
                # eigvals = [u_g, u_l, u_mix+a, u_mix−a, u_g, u_l]  (matrices.py)
                u_mat = jnp.minimum(jnp.abs(ev[0]), jnp.abs(ev[1]))  # min(|u_g|,|u_l|)
                # FILTER-STRENGTH scale (eps_void_scale):
                #   'wood'   → the model's true mixture eigenvalue |u_mix|+a_s.  a_s is
                #             the Städtke/Wood mixture speed, which DIPS to a deep
                #             minimum at α≈0.5 → the filter self-weakens exactly where
                #             the slip/void checkerboard peaks (see nc/eps_void notes).
                #   'frozen' → a NON-dipping surrogate |u_mix| + max(a_g,a_l) built from
                #             the single-phase (frozen) sound speeds, which have no
                #             α-dependence.  Keeps the damping magnitude ~constant in α.
                #             Used ONLY here for the 4th-order filter scale — the fluxes
                #             and CFL still use the true (dipping) a_s.  Retune eps_void
                #             down to compensate the larger magnitude.
                if eps_void_scale == 'frozen':
                    u_mix = 0.5 * (ev[2] + ev[3])                    # u_mix = ½((u+a)+(u−a))
                    lam   = jnp.abs(u_mix) + jnp.maximum(eg_.a, el_.a)
                else:                                                # 'wood' (default)
                    lam = jnp.max(jnp.abs(ev))
                return lam, u_mat
            if cell_local_jac:
                _lu_c = _local_jac_map(lambda u: jnp.stack(_lam_umat_cell(u)))(U_h)
                lam_c, umat_c = _lu_c[:, 0], _lu_c[:, 1]
            else:
                lam_c, umat_c = jax.vmap(_lam_umat_cell)(U_h)    # (N,) each
            r_face_raw = 0.5 * (lam_c[:-1] + lam_c[1:])          # (N-1,) face scale (ungated)
            third  = (V_new[3:] - 3.0 * V_new[2:-1]
                      + 3.0 * V_new[1:-2] - V_new[:-3])          # (N-3,6): faces 1..N-3
        if eps_void > 0.0:
            r_face = r_face_raw
            if eps_void_ucut is not None:
                u0 = jnp.float64(eps_void_ucut)
                du = jnp.float64(eps_void_uwidth if eps_void_uwidth is not None
                                 else eps_void_ucut / 3.0)
                w_c    = 0.5 * (1.0 + jnp.tanh((u0 - umat_c) / du))  # 1 slow → 0 fast
                w_face = 0.5 * (w_c[:-1] + w_c[1:])              # (N-1,)
                r_face = r_face * w_face                         # gate the filter scale
            third_v = third
            if eps_void_mass_acut is not None:
                # GHOST MASS ROW GATE (eps_void_mass_acut): a 4th-order filter is
                # not positivity preserving — on the MASS row of a ghost whose α
                # spans decades its undershoot exceeds α itself and digs α holes
                # (same lesson as eps_ghost's "mass rows excluded" below).  Fade
                # the ghost phase's mass row out below ~acut; the weight is the
                # product over the face's 4-cell stencil (the cells `third` reads).
                _cm  = jnp.log10(jnp.float64(eps_void_mass_acut))
                _wgc = 0.5 * (1.0 + jnp.tanh(
                    (jnp.log10(jnp.clip(U_h[:, 3], 1e-300, 1.0)) - _cm) / 0.25))
                _wlc = 0.5 * (1.0 + jnp.tanh(
                    (jnp.log10(jnp.clip(1.0 - U_h[:, 3], 1e-300, 1.0)) - _cm) / 0.25))
                _wg4 = _wgc[:-3] * _wgc[1:-2] * _wgc[2:-1] * _wgc[3:]
                _wl4 = _wlc[:-3] * _wlc[1:-2] * _wlc[2:-1] * _wlc[3:]
                third_v = third * jnp.ones_like(third).at[:, 0].set(_wg4).at[:, 1].set(_wl4)
            d_face = jnp.zeros_like(F_hat).at[1:-1].set(
                eps_void * r_face[1:-1, None] * third_v)         # +sign DAMPS (verified); 0 at bdry
            F_hat  = F_hat + d_face

        # ── GHOST-phase 4th-order filter (eps_ghost) ──────────────────────────
        # The eps_void channel above is VELOCITY-gated (checkerboards ride the
        # material waves as u→0) — so it is fully OFF in fast regions (e.g. the
        # shock layer at u≥100 m/s), which is exactly where a near-massless
        # GHOST phase limit-cycles (its rows carry ~α of the mass ⇒ the FVS
        # dissipation that stabilises the dominant phase is ~α-weak for it, and
        # nothing else damps its 2Δx content: h/u_ghost pumping, shock-cell
        # flicker, stagnation h_g checkerboard).  This channel applies the SAME
        # conservative 4th-order flux, but gated per ROW by the phase's own
        # smallness in log-α:
        #     w_k = ½(1 − tanh((log10 α_k − log10 acut)/awidth))   (per face)
        # → 1 while phase k is a ghost (α_k ≪ acut), → 0 once it carries real
        # mass (never contaminates a physical phase; flashing-safe: the filter
        # hands off smoothly as a phase grows through acut).  Row order of V:
        # [mass_g, mass_l, mom_g, mom_l, E_g, E_l] → gas rows (0,2,4) weighted
        # by w_gas, liquid rows (1,3,5) by w_liq.  NOT velocity-gated: a ghost
        # needs damping at any speed.  eps_ghost=0 → identically off.
        if eps_ghost > 0.0:
            ag_c  = U_h[:, 3]
            c0    = jnp.log10(jnp.float64(eps_ghost_acut))
            aw    = jnp.float64(eps_ghost_awidth)
            lg    = jnp.log10(jnp.clip(ag_c,       1e-14, 1.0))
            ll    = jnp.log10(jnp.clip(1.0 - ag_c, 1e-14, 1.0))
            wg_c  = 0.5 * (1.0 - jnp.tanh((lg - c0) / aw))   # 1 ⇔ gas   is ghost
            wl_c  = 0.5 * (1.0 - jnp.tanh((ll - c0) / aw))   # 1 ⇔ liquid is ghost
            # BAND GATE (2026-07-18): disengage BELOW eps_ghost_alo.  The
            # shock-settling transient transiently squeezes the ghost to the α
            # clip floor (measured α_l 1e-4→1e-6 at shock+3, WITH OR WITHOUT
            # the filter — baseline control heals by drag within ~100 steps).
            # Inside that hole the ghost fields are garbage (u_l ±2000 m/s,
            # h_l ±1e7) and the filter's Δ³V of garbage is a huge random
            # forcing that LOCKS the hole open (all v1-v5 deaths).  So: help a
            # thin-but-HEALTHY ghost (α ∈ [alo, acut]), hands off in a hole
            # (α < alo — drag heals it untouched).
            c1    = jnp.log10(jnp.float64(eps_ghost_alo))
            wg_c  = wg_c * 0.5 * (1.0 + jnp.tanh((lg - c1) / aw))
            wl_c  = wl_c * 0.5 * (1.0 + jnp.tanh((ll - c1) / aw))
            wg_f  = 0.5 * (wg_c[:-1] + wg_c[1:])             # (N-1,)
            wl_f  = 0.5 * (wl_c[:-1] + wl_c[1:])
            # MASS ROWS EXCLUDED (2026-07-18): filtering the ghost MASS row
            # digs an α-hole at a shock — the 2nd-order term transports mass
            # out of the jump's corner cells, and the α-gate then AMPLIFIES
            # (thinner α → w→1 → stronger filter → thinner): measured α_l
            # 1e-4 → 1e-8 over 6 cells, ghost u_l ±2263 m/s, drag feeding
            # garbage back into the GAS (H0 excursion ×3, RH gate FAIL).
            # The limit cycle we target lives in ghost MOMENTUM/ENERGY
            # (99.98% mom_l by row-SSR); mass never needed filtering — α
            # checkerboards are eps_void's (all-row, velocity-gated) job.
            zero  = jnp.zeros_like(wg_f)
            row_w = jnp.stack([zero, zero, wg_f, wl_f, wg_f, wl_f], axis=1)  # (N-1,6)
            # JST SHOCK SENSOR: a pure 4th-order filter is ANTI-diffusive in the
            # two cells flanking a discontinuity (Gibbs kicks — certified fatal
            # on the analytic shock IC: dhl≈123 K at step 4, global checkerboard,
            # blow-up).  Classic JST remedy: sense the shock with the normalized
            # pressure second difference and CROSS-FADE the ghost filter from
            # 4th-order (smooth regions: damps the 2Δx checkerboard) to 2nd-order
            # (shock zone: monotone Laplacian damping — kills the ghost shock-cell
            # limit cycle WITHOUT ringing).  σ≈0 smooth (d2p/4p ~ O(Δx²)),
            # σ→1 at a captured shock (d2p/4p ~ 0.2-0.4 with the ×10 sharpening).
            p_c   = U_h[:, 0]
            d2p   = jnp.abs(p_c[2:] - 2.0 * p_c[1:-1] + p_c[:-2]) \
                    / (p_c[2:] + 2.0 * p_c[1:-1] + p_c[:-2])
            sig_c = jnp.zeros(N).at[1:-1].set(jnp.minimum(1.0, 10.0 * d2p))
            sig_f = jnp.maximum(sig_c[:-1], sig_c[1:])       # (N-1,) face sensor
            diff1 = V_new[1:] - V_new[:-1]                   # (N-1,6): ΔV at faces
            d_gh  = jnp.zeros_like(F_hat).at[1:-1].set(
                eps_ghost * r_face_raw[1:-1, None] * row_w[1:-1]
                * ((1.0 - sig_f[1:-1, None]) * third          # 4th-order, smooth
                   # 2nd-order at the shock.  k2 (default 4): for a 2Δx mode the
                   # 4th difference is 4× the 1st difference, so eps2 = k2·eps4
                   # keeps the checkerboard damping rate CONSTANT through the
                   # sensor cross-fade (k2=1 was ×4 weaker in the shock zone —
                   # held the N600 cycle at ~14% instead of draining it).
                   - jnp.float64(eps_ghost_k2) * sig_f[1:-1, None] * diff1[1:-1]))
            F_hat = F_hat + d_gh

        Vc      = V_cell[1:-1, None]
        dV      = V_new[1:-1] - V_old[1:-1]
        fdiv    = Af_j[1:, None] * F_hat[1:] - Af_j[:-1, None] * F_hat[:-1]
        nc_term = nc_scale * A_j[1:-1, None] * nc   # nc_scale=0 → NC coupling OFF (diagnostic)

        R = Vc * dV + dt * fdiv + dt * nc_term - dt * Vc * src

        # ── GHOST-VAPOR SATURATION PIN (2026-07-21, nozzlex parity) ───────────
        # Pre-flashing the vapor is a numerical ghost: its dynamic h_g carries
        # no physical information but rings the E/(αρ_g) 2Δx amplifier — every
        # flashing blow-up localized to floor-pinned front cells (probe ladder
        # 65→74→116→220 healthy steps; σ-sharpness probe NEGATIVE).  nozzlex
        # slaves the ghost to the saturation curve (dh_v/dx = dh_sat_v/dp·dp/dx,
        # nozzle_model_core.py:1777) gated by the SAME evaporation flag.  Here:
        # BLEND the gas-ENERGY residual row with the algebraic pin
        #     R_Eg ← w·R_Eg + (1−w)·Vc·m_g·(h_g − h_sat_g(p)),
        # w = evaporation_weight (LIQUID superheat only — shared with the σ
        # gate, so pin and evaporation are complementary by construction).
        # ρ_g consistency follows FREE via the existing gas EOS row
        # (m_g = α_g·ρ_g(p, h_g) with h_g = h_sat_g(p) ⇒ ρ_g = ρ_sat_g(p)).
        #
        # NO-DEADLOCK INVARIANT (do not break): w may only ever be WIDENED toward
        # 1 — nothing in the α-growth chain may be gated OFF by it — and the pin
        # must touch ONLY row 4 (gas energy).  (Originally stated as "w must
        # depend only on liquid quantities, never on α"; that was stricter than
        # necessary.  evaporation_weight now also forces w=1 for α>5% — see the
        # α-availability clause in sources.py — which only ADDS activation and so
        # preserves the invariant while un-pinning genuinely real vapor.)  The liquid then
        # superheats freely as p drops, w opens, σ>0 (liquid-side heat; pinned
        # T_g = T_sat makes Qvi = 0), gas CONTINUITY (row 0, untouched) grows
        # α, and the pin releases.  Nothing in the α-growth chain is gated.
        #
        # Limitation: needs p < p_crit (saturation curve exists) — tables end
        # at 130 bar << 220.6 bar for water, fine for this tier.  Above the
        # saturation table (0.98 p_crit) the lookup is clamped, and p_mt_max
        # forces w = 0 there.  That narrows w, but only on p, which the pin
        # does not control, so it cannot deadlock: below p_mt_max the chain
        # above is unchanged.  p_mt_max must sit above the pressure where
        # flashing can physically start (h_sat_l(p) = reservoir h_l).
        #
        # α-availability gate (shock-wave cases, see TODO_two_phase.md): pure
        # superheat gating re-pins REAL vapor after a recompression shock (liquid
        # re-subcooled at α~0.5) = forced equilibrium condensation.  IMPLEMENTED
        # 2026-07-22 as a binary `| (ag > 5e-2)` clause inside evaporation_weight
        # (sources.py), so w_ev below already carries it — a real phase is never
        # pinned.  Smooth variant, if the binary α threshold ever flickers:
        #   w_ghost = 0.5*(1+tanh((jnp.log(ag_c/(10.0*alpha_eps)))/0.5))
        #   w_pin   = jnp.maximum(w_ev, w_ghost)   # use in place of w_ev below
        if pin_ghost:
            def _pin_one(u_h, m_g_c):
                eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
                sat_g_c = fluid_g.get_state_saturation(u_h[0])
                sat_l_c = fluid_l.get_state_saturation(u_h[0])
                w_ev = evaporation_weight(u_h[5], eos_l_c.T,
                                          sat_l_c["enthalpy"],
                                          sat_l_c["temperature"], u_h[3],
                                          p=u_h[0], p_max=p_mt_max)
                pin  = m_g_c * (u_h[4] - sat_g_c["enthalpy"])
                return w_ev, pin
            if cell_local_jac:
                _wp = _local_jac_map(lambda x: jnp.stack(_pin_one(x[:6], x[6])))(
                    jnp.concatenate([U_h[1:-1], W_int[:, 2:3]], axis=1))
                w_ev, pin = _wp[:, 0], _wp[:, 1]
            else:
                w_ev, pin = jax.vmap(_pin_one)(U_h[1:-1], W_int[:, 2])
            R = R.at[:, 4].set(w_ev * R[:, 4] + (1.0 - w_ev) * Vc[:, 0] * pin)

        # ── Ghost-GAS MOMENTUM pin (nozzlex constraint 2, u_v → u_l) ──────────
        # The thin ghost gas carries a velocity dof with ~α·ρ_g of inertia — on a
        # grid with fixed dt this is a 2Δx resonator (every 1e-6-floor crash has
        # u_g at 97–98% odd-even).  A space-march (nozzlex) has no such mode, so it
        # left u_v free (its constraint 2 stayed commented); on our grid it must be
        # slaved.  BLEND the gas-MOMENTUM residual row with u_g → u_l:
        #     R_mom_g ← (1−w)·R_mom_g + w·Vc·m_g·(u_g − u_l).
        #
        # WEIGHT is α-GATED (not the evaporation weight): the momentum dof owns its
        # velocity only once α has physically grown, so the pin must release on α,
        # not on liquid superheat.  This also makes it SATURATION-FREE (reads only
        # α_g) → it serves the Bernoulli ghost-gas case (no mass transfer) with the
        # same code path, and is independent of / composable with the energy pin.
        #   w = 1 for α_g ≲ a_pin/10 (ghost), → 0 by ~10·a_pin (real gas), with
        #   a_pin = pin_mom_alpha (default 10·alpha_eps — which switches the pin
        #   off exactly when alpha_eps is lowered for a thin inlet ghost).
        # NO-DEADLOCK: touches ONLY row 2 (gas momentum); continuity rows 0/1 and
        # the α-growth chain are untouched, so α grows through the tier and the pin
        # releases BECAUSE α grew.
        #
        # pin_mom_tau (dt-CONSISTENT pin): the algebraic blend mixes a dt-scaled
        # conservation residual with a NON-dt-scaled pin, so where 0 < w < 1 the
        # converged state depends on dt and the dt=1 steady residual cannot reach
        # zero (measured SSR stuck at 1.4e-8, all in mom_g of the onset cells).
        # With pin_mom_tau the pin is a relaxation source ×dt/τ instead.
        if pin_ghost_momentum:
            ag_c  = U_h[1:-1, 3]                                   # α_g interior
            _z    = jnp.log(ag_c / _a_pin_mom) / 0.5
            pin_u = W_int[:, 2] * (U_h[1:-1, 1] - U_h[1:-1, 2])    # m_g·(u_g − u_l)
            if pin_mom_tau is None:
                # Original algebraic blend (bit-for-bit with earlier runs).
                w_mom = 0.5 * (1.0 - jnp.tanh(_z))
                R = R.at[:, 2].set((1.0 - w_mom) * R[:, 2]
                                   + w_mom * Vc[:, 0] * pin_u)
            else:
                # Weight through its complement w_free = sigmoid(2z) = 1 − w_mom:
                # deep in the ghost the tanh form's derivative 1 − tanh² keeps only
                # ~4 digits (colored vs dense Jacobian differed by 1e-6 of the row
                # max there; with this form they agree to 1e-14).
                w_free = jax.nn.sigmoid(2.0 * _z)
                w_mom  = 1.0 - w_free
                R = R.at[:, 2].set(w_free * R[:, 2]
                                   + w_mom * (dt / pin_mom_tau) * Vc[:, 0] * pin_u)

        return R.flatten()   # (6*(N-2),)

    # ── Jacobian: dense (exact AD) or colored (banded compression) ────────────
    # R ∈ R^{6(N-2)}, W ∈ R^{8(N-2)}.  The residual is block-banded: residual
    # row-block for cell i touches only column-blocks i-sw..i+sw
    #   sw = 1 (constant recon)  |  sw = 2 (MUSCL, ±2 reconstruction stencil).
    # Both BCs fold inward (total_state inlet reads cells 1,2; transmissive outlet
    # reads cell -2) so they never widen the band — verified via jac_diff.
    #
    # Column coloring (forward mode, matching jacfwd — avoids transpose solves
    # through the internal K^{-1}/lstsq in flux & nc): columns ≥ col_stride apart
    # never share a residual row, so they are probed together in ONE JVP and
    # de-interleaved exactly.  n_colors is fixed by the stencil (24 constant,
    # 40 MUSCL), INDEPENDENT of N → O(N) instead of O(N²) Jacobian cost.
    # eps_void>0 adds a ±2-stencil 4th-order filter → widen the band to sw=2 even
    # for constant recon so the colored Jacobian stays exact (verify via jac_diff).
    _sw          = 2 if (recon != 'constant' or eps_void > 0.0
                         or eps_ghost > 0.0) else 1
    _col_stride  = 8 * (2 * _sw + 1)             # DOF stride (8 vars/cell): 24 | 40
    _m_cols      = 8 * (N - 2)                     # total W DOFs
    _n_rows      = 6 * (N - 2)                     # total residual rows
    _n_colors    = min(_col_stride, _m_cols)       # graceful if N tiny

    _jac_dense = jax.jacfwd(_residual, argnums=0)

    def _jac_colored(W_int_flat, W_n_flat, dt, p_out_t=None):
        # Linearize once (expensive nonlinear residual + EOS chain runs a single
        # time); each colored seed is then a cheap linear JVP.
        _, jvp_fn = jax.linearize(lambda w: _residual(w, W_n_flat, dt, p_out_t), W_int_flat)
        cols_idx  = jnp.arange(_m_cols)
        color_ids = jnp.arange(_n_colors)
        S  = (cols_idx[None, :] % _col_stride == color_ids[:, None]).astype(W_int_flat.dtype)
        Jc = jax.vmap(jvp_fn)(S)                   # (n_colors, 6(N-2)): summed columns
        # De-interleave: residual row r → cell lr = r//6.  In-band columns are
        # cells lr-sw..lr+sw × 8 vars; each has a unique colour within the band,
        # so J[r, c] = Jc[c % col_stride, r].
        rows = jnp.arange(_n_rows)
        lr   = rows // 6
        J    = jnp.zeros((_n_rows, _m_cols), dtype=W_int_flat.dtype)
        for dc in range(-_sw, _sw + 1):
            for vc in range(8):
                lc     = lr + dc
                c      = lc * 8 + vc
                valid  = (lc >= 0) & (lc <= (N - 3))
                c_safe = jnp.clip(c, 0, _m_cols - 1)
                val    = jnp.where(valid, Jc[c_safe % _col_stride, rows], 0.0)
                J      = J.at[rows, c_safe].add(val)   # invalid → adds 0 at clipped col
        return J

    _jac_fn = _jac_colored if jacobian_mode == 'colored' else _jac_dense

    # ── E3-AUGMENTED residual rows (prototype, 2026-07-15) ────────────────────
    # Append the two EOS closures per cell as RESIDUAL ROWS:
    #     R7 = α ρ_g(p,h_g) − m_g ,   R8 = (1−α) ρ_l(p,h_l) − m_l
    # giving an 8-row × 8-col per-cell system solved by the SAME least-squares
    # min-norm machinery (rank deficiency in degenerate limits stays harmless —
    # the E3 objection to a hard square solve is respected; we ADD information,
    # never demand invertibility).  This closes the per-cell null space that
    # lets the min-norm park flux-residual adjustments in (m_g, m_l): measured
    # 2026-07-15 as a SECULAR liquid-mass drain (−1.5e-4/step at the throat,
    # p/h_l/α standing still) that kills ghost-limit runs at ~4700 steps.
    # Unlike the post-hoc projection (which holds m frozen and pumps p), the
    # joint solve may correct m_l itself — ∂R8/∂m_l = −1 — negotiated in least
    # squares against the m_l flux row: the missing feedback loop.
    # Dense path is the certified default; eos_rows_banded=True switches to the
    # O(N) block-banded twin (_e3_banded_solve below) — same numbers.
    def _eos_res_cell(Wc):
        p, ag, m_g, u_g, m_l, u_l, mg_eg, ml_el = Wc
        h_g  = (mg_eg + p * ag) / m_g
        h_l  = (ml_el + p * (1.0 - ag)) / m_l
        p_c  = jnp.clip(p,   p_lo,  p_hi)
        h_gc = jnp.clip(h_g, hg_lo, hg_hi)
        h_lc = jnp.clip(h_l, hl_lo, hl_hi)
        rg = fluid_g.get_state(jxp.HmassP_INPUTS,
                               jnp.atleast_1d(h_gc), jnp.atleast_1d(p_c))["density"][0]
        rl = fluid_l.get_state(jxp.HmassP_INPUTS,
                               jnp.atleast_1d(h_lc), jnp.atleast_1d(p_c))["density"][0]
        return jnp.array([ag * rg - m_g, (1.0 - ag) * rl - m_l])

    _eos_jac_cells = jax.vmap(jax.jacfwd(_eos_res_cell))     # (Nc, 2, 8)

    def _eos_residual(W_int_flat):
        return jax.vmap(_eos_res_cell)(W_int_flat.reshape(N - 2, 8)).reshape(-1)

    def _eos_jac_dense(W_int_flat):
        Nc = N - 2
        Jc = _eos_jac_cells(W_int_flat.reshape(Nc, 8))       # (Nc, 2, 8)
        rows = 2 * jnp.arange(Nc)[:, None, None] + jnp.arange(2)[None, :, None]
        cols = 8 * jnp.arange(Nc)[:, None, None] + jnp.arange(8)[None, None, :]
        return jnp.zeros((2 * Nc, 8 * Nc)).at[rows, cols].set(Jc)

    # ── E1b: block-banded Jacobian storage (for linear_solve='normal_eq_banded')
    # The de-interleave above already computes exactly the in-band block entries
    # J[cell i → cell i+d-sw]; this variant returns them as (N-2, 2sw+1, 6, 8)
    # blocks instead of scattering into a dense (6(N-2))×(8(N-2)) matrix.  Same
    # coloring, same JVPs, IDENTICAL numbers — only the storage differs.
    _Nc   = N - 2
    _Dtot = 2 * _sw + 1

    def _jac_colored_blocks(W_int_flat, W_n_flat, dt, p_out_t=None):
        _, jvp_fn = jax.linearize(lambda w: _residual(w, W_n_flat, dt, p_out_t), W_int_flat)
        cols_idx  = jnp.arange(_m_cols)
        color_ids = jnp.arange(_n_colors)
        S  = (cols_idx[None, :] % _col_stride == color_ids[:, None]).astype(W_int_flat.dtype)
        Jc = jax.vmap(jvp_fn)(S)                                 # (n_colors, 6(N-2))
        i_c   = jnp.arange(_Nc)
        dc    = jnp.arange(-_sw, _sw + 1)
        lc    = i_c[:, None] + dc[None, :]                       # (Nc, Dtot) column cell
        # colour of column c = lc*8+vc is c % col_stride = 8*(lc mod Dtot) + vc
        color = 8 * jnp.mod(lc, _Dtot)[:, :, None] + jnp.arange(8)[None, None, :]
        rows  = i_c[:, None] * 6 + jnp.arange(6)[None, :]        # (Nc, 6)
        J_blk = Jc[color[:, :, None, :], rows[:, None, :, None]]  # (Nc, Dtot, 6, 8)
        valid = (lc >= 0) & (lc <= _Nc - 1)
        return jnp.where(valid[:, :, None, None], J_blk, 0.0)

    # ── Newton inner linear solve ─────────────────────────────────────────────
    # The Newton system is underdetermined: J is 6(N-2) × 8(N-2) (fewer equations
    # than unknowns), so we take the minimum-norm solution dW = Jᵀ(J Jᵀ)⁻¹(−R).
    #
    # 'lstsq'     : dense SVD min-norm least-squares (robust, rank-tolerant).
    #               Gold-standard accuracy; slowest.
    # 'normal_eq' : min-norm via the normal equations, Cholesky on J Jᵀ (SPD).
    #               ~15× faster than SVD, but FORMS J Jᵀ → squares the condition
    #               number (κ(JJᵀ)=κ(J)²), losing ~half the digits on the stiff
    #               source-coupled Jacobian.  Tiny Tikhonov for rank-deficiency.
    # 'normal_eq_banded' (E1b): SAME numbers as 'normal_eq', but J/JJᵀ stay in
    #               block-banded storage and the SPD solve is a block-tridiagonal
    #               Thomas sweep → O(N) memory and flops instead of O(N²)/O(N³).
    #               Speed/memory only — κ² conditioning unchanged.  See below.
    # 'qr'        : min-norm via the LQ factorization (QR of Jᵀ).  With Jᵀ = Q R
    #               (economic), J Jᵀ = RᵀR, so the min-norm solution is
    #                   dW = Jᵀ(JJᵀ)⁻¹(−R) = Q R⁻ᵀ(−R)
    #               i.e. solve Rᵀ y = (−R) (triangular), dW = Q y.  QR does NOT
    #               square κ (κ(R)=κ(J)) → ~SVD accuracy at ~3-5× the SVD speed.
    #               Assumes J full row rank; near exact zero slip add a small
    #               Tikhonov or fall back to 'lstsq'.
    _I_rows = jnp.eye(6 * (N - 2), dtype=jnp.float64)

    if eos_rows:
        # E3-augmented prototype: rhs is (8Nc,).  Least-squares NORMAL
        # equations (JᵀJ, Cholesky) — ~10× faster than SVD lstsq at N=300
        # (measured 15-20 s/step vs ~2 s/step).  The tiny Tikhonov floor
        # regularizes the genuinely degenerate ghost columns (u_g, h_g rows
        # all scale with α) — there it acts like the min-norm's small-step
        # selection; everywhere else it is ~roundoff.
        _I_cols = jnp.eye(8 * (N - 2), dtype=jnp.float64)

        def _lin_solve(J_sc, rhs):
            JtJ = J_sc.T @ J_sc
            reg = 1e-10 * jnp.mean(jnp.diagonal(JtJ))
            return jsl.solve(JtJ + reg * _I_cols, J_sc.T @ rhs, assume_a='pos')
    elif linear_solve in ('normal_eq', 'normal_eq_banded'):
        # Dense normal equations.  For 'normal_eq_banded' this dense version is
        # only the debug_newton fallback — the production Newton body uses
        # _min_norm_banded below (same numbers, O(N) storage and flops).
        def _lin_solve(J_sc, rhs):                     # rhs = -R
            JJt = J_sc @ J_sc.T
            # With row equilibration cond(J)~1e3 → JJt is well-conditioned, so
            # linear_reg defaults to 0 (no Tikhonov).  A small value only floors
            # exact singularity; large values BIAS the min-norm step (they swamp
            # the small singular value that dominates the true solution).
            reg = linear_reg * jnp.mean(jnp.diagonal(JJt))
            y   = jsl.solve(JJt + reg * _I_rows, rhs, assume_a='pos')
            return J_sc.T @ y
    elif linear_solve == 'qr':
        def _lin_solve(J_sc, rhs):                     # rhs = -R
            # Jᵀ (n×m, tall) = Q (n×m) · R (m×m upper-tri);  dW = Q · (Rᵀ \ rhs)
            Q, R = jnp.linalg.qr(J_sc.T, mode='reduced')
            y    = jsl.solve_triangular(R, rhs, lower=False, trans='T')
            return Q @ y
    else:
        def _lin_solve(J_sc, rhs):
            return jnp.linalg.lstsq(J_sc, rhs, rcond=None)[0]

    # ── E1b: banded normal-equations min-norm solve ───────────────────────────
    # Same min-norm solution as 'normal_eq' — J is block-banded (residual cell i
    # touches only cells i±sw), so J, JJᵀ (block half-bandwidth 2sw cells) and
    # the SPD solve never need dense storage:
    #   dense:  form JJᵀ O(N²·band) + Cholesky O(N³)      memory O(N²)
    #   banded: form JJᵀ O(N·band²) + block-Thomas O(N)   memory O(N)
    # The solve re-blocks JJᵀ into a block-TRIDIAGONAL system with super-blocks
    # of s=2sw cells (S=12sw scalars: 24×24 for MUSCL/filter) and runs a block
    # Thomas forward/backward sweep via lax.scan.  In exact arithmetic the
    # result is IDENTICAL to the dense path; only the rounding order differs
    # (block elimination vs dense Cholesky) → dW agrees to ~κ·eps ≈ 1e-10 rel.
    # Conditioning is unchanged (still the normal equations, κ(JJᵀ)=κ(J)²) —
    # this is a speed/memory upgrade only, certified by debug_banded_solve.py.
    _use_banded = (linear_solve == 'normal_eq_banded') and not eos_rows

    _q_bw   = 2 * _sw                       # JJᵀ block half-bandwidth [cells]
    _s_blk  = 2 * _sw                       # cells per super-block
    _M_blk  = -(-_Nc // _s_blk)             # ceil(Nc / s)
    _Nc_pad = _M_blk * _s_blk               # padded cells: identity rows, rhs 0
    _S_blk  = 6 * _s_blk                    # scalar super-block size
    if _use_banded and _m_cols < _col_stride:
        raise ValueError("normal_eq_banded needs N large enough for full coloring "
                         f"(8*(N-2)={_m_cols} < col_stride={_col_stride})")

    # static index maps for super-block assembly (baked into the jit)
    _a_loc  = jnp.arange(_s_blk)
    _off_dd = _a_loc[None, :] - _a_loc[:, None]            # col cell − row cell (s,s)
    _msk_dd = jnp.abs(_off_dd) <= _q_bw
    _q_dd   = jnp.clip(_off_dd + _q_bw, 0, 2 * _q_bw)
    _off_uu = _s_blk + _a_loc[None, :] - _a_loc[:, None]   # cols in next super-block
    _msk_uu = jnp.abs(_off_uu) <= _q_bw
    _q_uu   = jnp.clip(_off_uu + _q_bw, 0, 2 * _q_bw)

    def _min_norm_banded(J_blk, R_k, w_ref):
        """Min-norm dW_sc = J_scᵀ(J_sc J_scᵀ)⁻¹(−R_sc) in block-banded storage.
        Mirrors the dense path exactly: column scaling (w_ref) → row
        equilibration → JJᵀ + Tikhonov → SPD solve → Jᵀy."""
        # column scaling: block (i,d) holds columns of cell i+d-sw
        w     = w_ref.reshape(_Nc, 8)
        w_pad = jnp.pad(w, ((_sw, _sw), (0, 0)))
        w_sh  = jnp.stack([w_pad[d:d + _Nc] for d in range(_Dtot)], axis=1)  # (Nc,Dtot,8)
        J_sc  = J_blk * w_sh[:, :, None, :]
        # row equilibration (max over all in-band entries of each residual row)
        r_sc  = 1.0 / (jnp.max(jnp.abs(J_sc), axis=(1, 3)) + 1e-300)          # (Nc,6)
        J_sc  = J_sc * r_sc[:, None, :, None]
        rhs   = (-R_k).reshape(_Nc, 6) * r_sc

        # JJᵀ upper block diagonals: JJt_up[e][i] = Σ_d J_i,d · J_{i+e},{d-e}ᵀ
        Jp = jnp.pad(J_sc, ((0, _q_bw), (0, 0), (0, 0), (0, 0)))
        JJt_up = []
        for e in range(_q_bw + 1):
            acc = jnp.zeros((_Nc, 6, 6), dtype=J_sc.dtype)
            for d in range(e, _Dtot):
                acc = acc + jnp.einsum('iab,icb->iac', J_sc[:, d], Jp[e:e + _Nc, d - e])
            JJt_up.append(acc)

        # Tikhonov identical to dense: reg = linear_reg · mean(diag JJᵀ)
        reg = linear_reg * jnp.mean(jnp.diagonal(JJt_up[0], axis1=1, axis2=2))
        JJt_up[0] = JJt_up[0] + reg * jnp.eye(6, dtype=J_sc.dtype)

        # full banded storage (Nc_pad, 2q+1, 6, 6); lower diags by symmetry,
        # padded cells get identity diagonal (decoupled, rhs 0 → solution 0)
        JJt_all = jnp.zeros((_Nc_pad, 2 * _q_bw + 1, 6, 6), dtype=J_sc.dtype)
        for e in range(_q_bw + 1):
            up = jnp.pad(JJt_up[e], ((0, _Nc_pad - _Nc), (0, 0), (0, 0)))
            JJt_all = JJt_all.at[:, _q_bw + e].set(up)
            if e > 0:   # block (i, i-e) = block (i-e, i)ᵀ
                lo = jnp.pad(jnp.swapaxes(JJt_up[e], 1, 2),
                             ((e, _q_bw), (0, 0), (0, 0)))[:_Nc_pad]
                JJt_all = JJt_all.at[:, _q_bw - e].set(lo)
        if _Nc_pad > _Nc:
            JJt_all = JJt_all.at[_Nc:, _q_bw].set(jnp.eye(6, dtype=J_sc.dtype))

        # assemble block-tridiagonal super-blocks D (M,S,S), U (M-1,S,S); L = Uᵀ
        i_sb    = jnp.arange(_M_blk)
        cell_dd = i_sb[:, None, None] * _s_blk + _a_loc[None, :, None]   # (M,s,s)
        Dm = JJt_all[cell_dd, _q_dd[None]]                                # (M,s,s,6,6)
        Dm = jnp.where(_msk_dd[None, :, :, None, None], Dm, 0.0)
        Dm = Dm.transpose(0, 1, 3, 2, 4).reshape(_M_blk, _S_blk, _S_blk)
        if _M_blk > 1:
            i_su    = jnp.arange(_M_blk - 1)
            cell_uu = i_su[:, None, None] * _s_blk + _a_loc[None, :, None]
            Um = JJt_all[cell_uu, _q_uu[None]]
            Um = jnp.where(_msk_uu[None, :, :, None, None], Um, 0.0)
            Um = Um.transpose(0, 1, 3, 2, 4).reshape(_M_blk - 1, _S_blk, _S_blk)

        b_sb = jnp.pad(rhs, ((0, _Nc_pad - _Nc), (0, 0))).reshape(_M_blk, _S_blk)

        # block-tridiagonal Thomas (symmetric: lower block = upper blockᵀ)
        def _fwd(carry, inp):
            Dp_prev, yp_prev = carry
            D_i, U_prev, b_i = inp
            G  = jnp.linalg.solve(Dp_prev, U_prev)
            g  = jnp.linalg.solve(Dp_prev, yp_prev)
            Lt = U_prev.T
            Dp = D_i - Lt @ G
            yp = b_i - Lt @ g
            return (Dp, yp), (Dp, yp)

        if _M_blk > 1:
            _, (Dps, yps) = jax.lax.scan(_fwd, (Dm[0], b_sb[0]),
                                         (Dm[1:], Um, b_sb[1:]))
            Dps = jnp.concatenate([Dm[:1], Dps], axis=0)
            yps = jnp.concatenate([b_sb[:1], yps], axis=0)
        else:
            Dps, yps = Dm, b_sb

        x_last = jnp.linalg.solve(Dps[-1], yps[-1])

        def _bwd(x_next, inp):
            Dp_i, yp_i, U_i = inp
            x_i = jnp.linalg.solve(Dp_i, yp_i - U_i @ x_next)
            return x_i, x_i

        if _M_blk > 1:
            _, xs = jax.lax.scan(_bwd, x_last, (Dps[:-1], yps[:-1], Um),
                                 reverse=True)
            y_all = jnp.concatenate([xs, x_last[None]], axis=0)
        else:
            y_all = x_last[None]

        y_c = y_all.reshape(_Nc_pad, 6)[:_Nc]

        # dW_sc = J_scᵀ y — block-banded transpose multiply
        dW = jnp.zeros((_Nc, 8), dtype=J_sc.dtype)
        for d in range(_Dtot):
            sh  = d - _sw                                   # row cell i → col cell i+sh
            src = jnp.einsum('iav,ia->iv', J_sc[:, d], y_c)
            if sh >= 0:
                dW = dW.at[sh:].add(src[:_Nc - sh] if sh > 0 else src)
            else:
                dW = dW.at[:_Nc + sh].add(src[-sh:])
        return dW.reshape(-1)

    # ── E3-BANDED: block-banded solve for the eos_rows-augmented system ───────
    # NEW OPTION (eos_rows_banded, 2026-07-16) — the certified dense eos_rows
    # path above stays the default; this reproduces ITS numbers in O(N).
    # Rows per cell = 6 flux residuals + 2 EOS closures.  The EOS rows are
    # CELL-LOCAL, so they live in the diagonal block (d = sw) only — stencil,
    # coloring and n_colors are unchanged.  The augmented system is square
    # (8×8 per cell) but may be rank-deficient in the ghost limit (u_g/h_g
    # columns scale with α), so we keep EXACTLY the dense path's least-squares
    # normal equations, JᵀJ + 1e-10·mean(diag) Tikhonov.  JᵀJ is SPD and
    # block-banded (half-bandwidth 2sw cells) → solved by the same super-block
    # Thomas sweep as E1b, with 8×8 blocks.  In exact arithmetic the result
    # equals the dense eos_rows path; only rounding order differs (~1e-10 rel).
    # Certify with debug_banded_e3.py before switching production YAMLs.
    _use_banded_e3 = bool(eos_rows and eos_rows_banded)
    if _use_banded_e3 and _m_cols < _col_stride:
        raise ValueError("eos_rows_banded needs N large enough for full coloring "
                         f"(8*(N-2)={_m_cols} < col_stride={_col_stride})")
    _S8 = 8 * _s_blk                     # scalar super-block size (32 for sw=2)

    def _jac_blocks_e3(W_int_flat, W_n_flat, dt, p_out_t=None):
        """(Nc, Dtot, 8, 8) blocks: flux rows 0..5 from the colored JVPs, EOS
        rows 6..7 (cell-local, vmapped exact jacfwd) in the diagonal block."""
        Jf = _jac_colored_blocks(W_int_flat, W_n_flat, dt, p_out_t)  # (Nc,Dtot,6,8)
        Je = _eos_jac_cells(W_int_flat.reshape(_Nc, 8))         # (Nc,2,8)
        J8 = jnp.zeros((_Nc, _Dtot, 8, 8), dtype=Jf.dtype)
        J8 = J8.at[:, :, :6, :].set(Jf)
        return J8.at[:, _sw, 6:, :].set(Je)

    def _e3_banded_solve(J_blk, R8, w_ref):
        """dW_sc = (J_scᵀJ_sc + reg)⁻¹ J_scᵀ(−R_sc) in block-banded storage.
        Mirrors the dense eos_rows path exactly: column scaling (w_ref) → row
        equilibration → JᵀJ + Tikhonov → SPD block-Thomas.  R8 is (Nc, 8) with
        per-cell row order [6 flux | 2 EOS]; row scaling is per-row and JᵀJ /
        Jᵀrhs sum over rows in any order, so the interleaved layout yields the
        same numbers as the dense path's [all-flux; all-EOS] stacking.
        Unlike the min-norm path there is NO trailing Jᵀy multiply — the
        normal-equations solution IS the (scaled) Newton step."""
        # column scaling: block (i,d) holds columns of cell i+d-sw
        w     = w_ref.reshape(_Nc, 8)
        w_pad = jnp.pad(w, ((_sw, _sw), (0, 0)))
        w_sh  = jnp.stack([w_pad[d:d + _Nc] for d in range(_Dtot)], axis=1)
        J_sc  = J_blk * w_sh[:, :, None, :]
        # row equilibration (max over all in-band entries of each residual row)
        r_sc  = 1.0 / (jnp.max(jnp.abs(J_sc), axis=(1, 3)) + 1e-300)   # (Nc,8)
        J_sc  = J_sc * r_sc[:, None, :, None]
        rhs   = -R8 * r_sc                                              # (Nc,8)

        # JᵀJ upper block diagonals + Jᵀ·rhs.  Column cell i is touched by row
        # cells k = i+sw-d (d = its block offset); padding J by sw cells makes
        # J_blk[i+sw-d] = Jp[i+2sw-d] → contiguous slice starting at 2sw-d.
        Jp  = jnp.pad(J_sc, ((_sw, _sw), (0, 0), (0, 0), (0, 0)))
        rp  = jnp.pad(rhs,  ((_sw, _sw), (0, 0)))
        Jtr = jnp.zeros((_Nc, 8), dtype=J_sc.dtype)
        for d in range(_Dtot):
            s0  = 2 * _sw - d
            Jtr = Jtr + jnp.einsum('iba,ib->ia',
                                   Jp[s0:s0 + _Nc, d], rp[s0:s0 + _Nc])
        JtJ_up = []
        for e in range(_q_bw + 1):
            acc = jnp.zeros((_Nc, 8, 8), dtype=J_sc.dtype)
            for d in range(_Dtot - e):
                s0  = 2 * _sw - d
                blk = Jp[s0:s0 + _Nc]
                acc = acc + jnp.einsum('iba,ibc->iac', blk[:, d], blk[:, d + e])
            JtJ_up.append(acc)

        # Tikhonov identical to the dense eos_rows path (ghost-limit rank guard)
        reg = 1e-10 * jnp.mean(jnp.diagonal(JtJ_up[0], axis1=1, axis2=2))
        JtJ_up[0] = JtJ_up[0] + reg * jnp.eye(8, dtype=J_sc.dtype)

        # full banded storage; lower diags by symmetry (JᵀJ symmetric),
        # padded cells get identity diagonal (decoupled, rhs 0 → solution 0)
        JtJ_all = jnp.zeros((_Nc_pad, 2 * _q_bw + 1, 8, 8), dtype=J_sc.dtype)
        for e in range(_q_bw + 1):
            up = jnp.pad(JtJ_up[e], ((0, _Nc_pad - _Nc), (0, 0), (0, 0)))
            JtJ_all = JtJ_all.at[:, _q_bw + e].set(up)
            if e > 0:   # block (i, i-e) = block (i-e, i)ᵀ
                lo = jnp.pad(jnp.swapaxes(JtJ_up[e], 1, 2),
                             ((e, _q_bw), (0, 0), (0, 0)))[:_Nc_pad]
                JtJ_all = JtJ_all.at[:, _q_bw - e].set(lo)
        if _Nc_pad > _Nc:
            JtJ_all = JtJ_all.at[_Nc:, _q_bw].set(jnp.eye(8, dtype=J_sc.dtype))

        # super-block tridiagonal assembly (same cell-index maps as E1b)
        i_sb    = jnp.arange(_M_blk)
        cell_dd = i_sb[:, None, None] * _s_blk + _a_loc[None, :, None]
        Dm = JtJ_all[cell_dd, _q_dd[None]]
        Dm = jnp.where(_msk_dd[None, :, :, None, None], Dm, 0.0)
        Dm = Dm.transpose(0, 1, 3, 2, 4).reshape(_M_blk, _S8, _S8)
        if _M_blk > 1:
            i_su    = jnp.arange(_M_blk - 1)
            cell_uu = i_su[:, None, None] * _s_blk + _a_loc[None, :, None]
            Um = JtJ_all[cell_uu, _q_uu[None]]
            Um = jnp.where(_msk_uu[None, :, :, None, None], Um, 0.0)
            Um = Um.transpose(0, 1, 3, 2, 4).reshape(_M_blk - 1, _S8, _S8)

        b_sb = jnp.pad(Jtr, ((0, _Nc_pad - _Nc), (0, 0))).reshape(_M_blk, _S8)

        if thomas == 'chol' and _M_blk > 1:
            # SPEED-UP (thomas='chol'): ONE Cholesky factorization per super-block.
            # With [G_i | g_i] = Dp_i⁻¹ [U_i | yp_i] (one factor, S+1 right-hand
            # sides) the back substitution is x_i = g_i − G_i·x_{i+1}: mat-vecs
            # only.  The 'orig' sweep below factors Dp twice forward, once back.
            Um_pad = jnp.concatenate([Um, jnp.zeros((1, _S8, _S8), Um.dtype)], 0)

            def _fwd_chol(carry, inp):
                U_prev, G_prev, g_prev = carry
                D_i, U_i, b_i = inp
                Dp  = D_i - U_prev.T @ G_prev
                yp  = b_i - U_prev.T @ g_prev
                Z   = jsl.cho_solve((jnp.linalg.cholesky(Dp), True),
                                    jnp.concatenate([U_i, yp[:, None]], axis=1))
                return (U_i, Z[:, :-1], Z[:, -1]), (Z[:, :-1], Z[:, -1])

            z0 = jnp.zeros((_S8, _S8), Dm.dtype)
            _, (Gs, gs) = jax.lax.scan(_fwd_chol, (z0, z0, jnp.zeros(_S8, Dm.dtype)),
                                       (Dm, Um_pad, b_sb))

            def _bwd_chol(x_next, inp):
                G_i, g_i = inp
                x_i = g_i - G_i @ x_next
                return x_i, x_i

            _, x_all = jax.lax.scan(_bwd_chol, jnp.zeros(_S8, Dm.dtype), (Gs, gs),
                                    reverse=True)
            return x_all.reshape(_Nc_pad, 8)[:_Nc].reshape(-1)

        # block-tridiagonal Thomas (symmetric: lower block = upper blockᵀ)
        def _fwd(carry, inp):
            Dp_prev, yp_prev = carry
            D_i, U_prev, b_i = inp
            G  = jnp.linalg.solve(Dp_prev, U_prev)
            g  = jnp.linalg.solve(Dp_prev, yp_prev)
            Lt = U_prev.T
            Dp = D_i - Lt @ G
            yp = b_i - Lt @ g
            return (Dp, yp), (Dp, yp)

        if _M_blk > 1:
            _, (Dps, yps) = jax.lax.scan(_fwd, (Dm[0], b_sb[0]),
                                         (Dm[1:], Um, b_sb[1:]))
            Dps = jnp.concatenate([Dm[:1], Dps], axis=0)
            yps = jnp.concatenate([b_sb[:1], yps], axis=0)
        else:
            Dps, yps = Dm, b_sb

        x_last = jnp.linalg.solve(Dps[-1], yps[-1])

        def _bwd(x_next, inp):
            Dp_i, yp_i, U_i = inp
            x_i = jnp.linalg.solve(Dp_i, yp_i - U_i @ x_next)
            return x_i, x_i

        if _M_blk > 1:
            _, xs = jax.lax.scan(_bwd, x_last, (Dps[:-1], yps[:-1], Um),
                                 reverse=True)
            x_all = jnp.concatenate([xs, x_last[None]], axis=0)
        else:
            x_all = x_last[None]

        # normal-equations solution IS dW_sc (square system, no Jᵀy step)
        return x_all.reshape(_Nc_pad, 8)[:_Nc].reshape(-1)

    def jac_diff(W_n, dt):
        """max|J_colored - J_dense| at state W_n — sparsity validation guard.
        Must be ~0 (same AD graph, reordered sums).  VALIDATE AT A GRADIENT STATE:
        at a uniform field the MUSCL ±2 coupling vanishes and a wrong sw=1 coloring
        would falsely pass."""
        W_int = W_n[1:-1].reshape(-1)
        W_nf  = W_n.reshape(-1)
        return jnp.max(jnp.abs(_jac_colored(W_int, W_nf, dt)
                               - _jac_dense(W_int, W_nf, dt)))

    # ── CFL time step ─────────────────────────────────────────────────────────

    # Per-cell spectral radius (max |eigenvalue|) of the coupled two-fluid system,
    # i.e. the Städtke mixture eigenvalues (u_mix ± a_mix, u_g, u_l), NOT the
    # per-phase EOS sound speeds a_g, a_l.  For bubbly flow a_mix ≪ a_l, so using
    # a_l would over-estimate the wave speed ~10×.  Same eigenstructure the flux
    # uses, so the CFL bound and the SSR time-scale are consistent with the scheme.
    def _max_wave_speed_cells(U_h_all):
        def _lam_one(u_h):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            eigvals, _, _, _ = eigenstructure(u_h, eos_g_c, eos_l_c, cq_c)
            return jnp.max(jnp.abs(eigvals))
        return jax.vmap(_lam_one)(U_h_all)

    @jax.jit
    def compute_dt(W_n):
        U_h     = jax.vmap(U_h_from_W)(W_n)   # W → U_h algebraic, no EOS
        lam_max = _max_wave_speed_cells(U_h)
        return CFL * dx / jnp.max(lam_max[1:-1])

    # ── Implicit Newton step ──────────────────────────────────────────────────

    # Characteristic scales for W = (p, α_g, m_g, u_g, m_l, u_l, mg_eg, ml_el).
    # lstsq minimises ||dW||² in absolute space, which causes huge steps in
    # dimensionless variables like α_g (whose Jacobian column is O(dt) ≈ 1e-5×
    # smaller than the m_g column from the Vc*dV term).  Scaling the columns of
    # J by w_ref before solving — and unscaling dW afterward — makes lstsq
    # minimise ||dW / w_ref||² instead, i.e. relative changes, which keeps α_g
    # safely inside (0, 1).
    # DEFAULT floors are calibrated for TWO-COMPONENT states (alpha ~ 0.1-0.9).
    # In the GHOST LIMIT (alpha ~ 1e-4) they exceed the ghost's actual W values
    # by x20-500 (alpha 0.05 vs 1e-4, m_g 0.1 vs 2e-3, mg_eg 1e4 vs ~500), so
    # the min-norm sees ghost DOFs as nearly FREE directions and dumps the
    # residual into them → h_g/alpha thrash while the liquid stays clean
    # (diagnosed 2026-07-14, Bernoulli ghost case).  Pass `w_char` (8 floors,
    # W order: p, α_g, m_g, u_g, m_l, u_l, mg_eg, ml_el) to override, or
    # w_char='auto' for state-derived floors (below).
    #
    # 'auto' (2026-07-17, certified by debug_wref_sensitivity.py on the ghost
    # state): the ideal column scale is |W| itself — a floor only acts where
    # it EXCEEDS the local |W|, and any floor at-or-below the field scale
    # reproduces the hand-tuned floors BIT-IDENTICALLY (both leave w_ref=|W|
    # a.e.).  Floors far ABOVE the field scale (the two-comp defaults on a
    # ghost state) inflate the reachable SSR floor ×180 via the Tikhonov
    # metric on the near-singular ghost columns; all-ones floors degrade it
    # to 1.7e-2 (no explosion under E3 — the old min-norm catastrophe is
    # gone).  'auto' therefore sets, per variable, 0.05 × max_cells|W_v|,
    # recomputed from the CURRENT Newton iterate — guards only genuinely
    # zero DOFs (u at rest, α at a flashing inlet) and adapts as the field
    # evolves (α traversing 1e-4 → 0.9).  No per-case tuning.
    _w_auto = isinstance(w_char, str) and w_char == 'auto'
    _w_char = jnp.tile(
        jnp.array([1e5, 0.05, 0.1, 0.5, 10.0, 0.5, 1e4, 1e6]
                  if (w_char is None or _w_auto)
                  else [float(v) for v in w_char]),
        N - 2,
    )   # (8*(N-2),) — floor values; actual w_ref = max(|W_k|, _w_char)

    def _w_ref_fn(W_k):
        """Column scales w_ref for the Newton solve (see block comment).

        'auto' uses a LOCAL field scale: rolling max of |W| over ±_W_WIN
        cells (edge-clamped), not the global max.  In a FLASHING case the α
        field spans decades (1e-4 inlet → 0.9 outlet); a global-max floor
        would over-floor the inlet ghost cells ×~500 — the measured
        SSR-inflation regime.  The local window tracks the ghost scale at the
        inlet and the dominant scale downstream, transitioning over a few
        cells at the front.  Where the field is uniform to within the window
        it reduces to the (certified) global behavior: the floor shrinks or
        stays, and any floor at/below local |W| is inert."""
        if _w_auto:
            loc = jnp.abs(W_k.reshape(N - 2, 8))
            for _ in range(4):                       # ±4-cell window
                up  = jnp.concatenate([loc[1:], loc[-1:]], axis=0)
                dn  = jnp.concatenate([loc[:1], loc[:-1]], axis=0)
                loc = jnp.maximum(loc, jnp.maximum(up, dn))
            return jnp.maximum(jnp.abs(W_k), (0.05 * loc).reshape(-1))
        return jnp.maximum(jnp.abs(W_k), _w_char)

    # ── EOS-manifold projection ────────────────────────────────────────────────
    # The Newton system is underdetermined (6 conservation eqs, 8 unknowns/cell),
    # so the min-norm step leaves (p, α_g) UNPINNED and W drifts off the EOS
    # manifold  m_g = α_g ρ_g(p,h_g),  m_l = (1-α_g) ρ_l(p,h_l)  (diagnosed in
    # test_T3b: rel_eos → 0.55 over 400 steps → the α_g checkerboard/void mode).
    # V_from_W does NOT depend on p or α_g, so we re-pin them to the conserved V
    # by Newton-solving the two closures while holding the other 6 W-components
    # (hence V) fixed.  Provably a no-op on already-consistent states
    # (test_eos_projection), so it never disturbs a converged / steady state.
    _PROJ_ITER = 8

    def _project_cell(W):
        p0, ag0, m_g, u_g, m_l, u_l, mg_eg, ml_el = W

        def _res(pa):
            p, ag = pa
            al  = 1.0 - ag
            h_g = (mg_eg + p * ag) / m_g
            h_l = (ml_el + p * al) / m_l
            # Clip EOS inputs to the table domain so the projection Newton NEVER
            # extrapolates into the NaN-producing region (same guard the closure
            # and flux paths use).  α_g enters the residual directly (α_g·ρ_g),
            # so the 2×2 Jacobian stays non-singular even where h is clamped.
            p_c  = jnp.clip(p,   p_lo,  p_hi)
            h_gc = jnp.clip(h_g, hg_lo, hg_hi)
            h_lc = jnp.clip(h_l, hl_lo, hl_hi)
            rg  = fluid_g.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_gc), jnp.atleast_1d(p_c))["density"][0]
            rl  = fluid_l.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_lc), jnp.atleast_1d(p_c))["density"][0]
            return jnp.array([ag * rg - m_g, al * rl - m_l])

        _jac = jax.jacfwd(_res)

        def _body(_, pa):
            dpa = jnp.linalg.solve(_jac(pa), -_res(pa))
            dpa = jnp.where(jnp.all(jnp.isfinite(dpa)), dpa, 0.0)   # singular → no move
            # Damp: cap the per-iteration change so an overshoot can't leave the
            # valid region (p by 50%, α_g by 0.2), then clip α_g to (ε, 1-ε).
            dpa = dpa.at[0].set(jnp.clip(dpa[0], -0.5 * jnp.abs(pa[0]), 0.5 * jnp.abs(pa[0])))
            dpa = dpa.at[1].set(jnp.clip(dpa[1], -0.2, 0.2))
            pa  = pa + dpa
            return pa.at[1].set(jnp.clip(pa[1], alpha_eps, 1.0 - alpha_eps))

        pa = jax.lax.fori_loop(0, _PROJ_ITER, _body, jnp.array([p0, ag0]))
        # Safety net: a failed projection must never be worse than not projecting —
        # if anything went non-finite, keep the pre-projection (p, α_g).
        pa = jnp.where(jnp.all(jnp.isfinite(pa)), pa, jnp.array([p0, ag0]))
        return W.at[0].set(pa[0]).at[1].set(pa[1])

    # ── GHOST-SAFE variant: TRIANGULAR (Gauss-Seidel) projection ──────────────
    # The exact 2×2 (p,α) root above has intrinsic gain dp/dm_g = ρ_l/|det| ≈
    # 1e8 Pa per unit gas mass in the ghost limit (both constraint rows nearly
    # parallel to the α axis), so ghost-mass noise becomes bar-scale p noise —
    # the pump convicted 2026-07-14.  Triangular split with the SAME manifold
    # and fixed points, but m_g noise never routes into p:
    #   1. p from the TOTAL-mass closure  αρ_g(p)+(1−α)ρ_l(p) = m_g+m_l
    #      (scalar Newton; d/dp = αρg_p+(1−α)ρl_p = mixture compressibility —
    #       NEVER degenerate at either α extreme),
    #   2. α = m_g / ρ_g(p, h_g)  directly (well-behaved at both ends),
    #   iterated _PROJ_ITER times (Gauss-Seidel).  On consistent states both
    #   formulations return the identical (p, α): same no-op property.
    def _project_cell_tri(W):
        p0, ag0, m_g, u_g, m_l, u_l, mg_eg, ml_el = W
        m_tot = m_g + m_l

        def _rho_gl(p, ag):
            al  = 1.0 - ag
            h_g = (mg_eg + p * ag) / m_g
            h_l = (ml_el + p * al) / m_l
            p_c  = jnp.clip(p,   p_lo,  p_hi)
            h_gc = jnp.clip(h_g, hg_lo, hg_hi)
            h_lc = jnp.clip(h_l, hl_lo, hl_hi)
            rg = fluid_g.get_state(jxp.HmassP_INPUTS,
                                   jnp.atleast_1d(h_gc), jnp.atleast_1d(p_c))["density"][0]
            rl = fluid_l.get_state(jxp.HmassP_INPUTS,
                                   jnp.atleast_1d(h_lc), jnp.atleast_1d(p_c))["density"][0]
            return rg, rl

        def _F_tot(p, ag):
            rg, rl = _rho_gl(p, ag)
            return ag * rg + (1.0 - ag) * rl - m_tot

        _dF = jax.grad(_F_tot, argnums=0)

        def _body(_, pa):
            p, ag = pa[0], pa[1]
            # 1) scalar Newton on p (α frozen): total-mass closure
            F  = _F_tot(p, ag)
            dF = _dF(p, ag)
            dp = jnp.where(jnp.abs(dF) > 1e-300, -F / dF, 0.0)
            dp = jnp.where(jnp.isfinite(dp), dp, 0.0)
            dp = jnp.clip(dp, -0.5 * jnp.abs(p), 0.5 * jnp.abs(p))
            p  = p + dp
            # 2) α from the trace-phase mass at the new p (direct, no inversion)
            rg, _ = _rho_gl(p, ag)
            ag = jnp.clip(m_g / (rg + 1e-300), alpha_eps, 1.0 - alpha_eps)
            return jnp.array([p, ag])

        pa = jax.lax.fori_loop(0, _PROJ_ITER, _body, jnp.array([p0, ag0]))
        pa = jnp.where(jnp.all(jnp.isfinite(pa)), pa, jnp.array([p0, ag0]))
        return W.at[0].set(pa[0]).at[1].set(pa[1])

    # ── GHOST-SAFE variant 2: DAMPED pair projection ──────────────────────────
    # Same manifold and 2×2 Newton as _project_cell, but every update is scaled
    # by a per-cell, per-pass damping factor λ computed from the projection
    # Jacobian's OWN conditioning:
    #     s = |det J_sc| / (‖row₀‖·‖row₁‖),   J_sc = J·diag(|p|, 1)
    # s is the sine of the angle between the two constraint gradients in the
    # column-scaled (p, α) plane: O(1) where the gas phase is real/compressible,
    # → 0 in the ghost / stiff-liquid limit where the EXACT root has intrinsic
    # gain dp/dm ≈ ρ_l/|det| ≈ 1e8 (the p-noise pump convicted 2026-07-14;
    # note the triangular GS variant above FAILED certification — any method
    # converging to the exact root inherits that gain).  λ = s²/(s²+s_ref²)
    # blends continuously between the two CERTIFIED endpoint behaviours:
    #     λ → 1 : identical to the classic pair projection   (two-component)
    #     λ → 0 : identical to eos_project=False              (ghost limit)
    # so a flashing run (α traversing 1e-4 → 0.9 in one domain) gets full
    # manifold pinning where it is well-posed and none where it would pump.
    # The δp trust cap (λ·1% of p per pass) bounds the worst-case per-step
    # pressure impact QUADRATICALLY in λ.  s_ref via `proj_s_ref`, calibrated
    # by debug_projection_damped.py check 0 (two-comp s≈0.36, ghost s≈9e-4 →
    # s_ref=2e-2 mid-gap gives λ=0.997 / 2e-3).
    def _project_cell_damped(W):
        p0, ag0, m_g, u_g, m_l, u_l, mg_eg, ml_el = W

        def _res(pa):
            p, ag = pa
            al  = 1.0 - ag
            h_g = (mg_eg + p * ag) / m_g
            h_l = (ml_el + p * al) / m_l
            p_c  = jnp.clip(p,   p_lo,  p_hi)
            h_gc = jnp.clip(h_g, hg_lo, hg_hi)
            h_lc = jnp.clip(h_l, hl_lo, hl_hi)
            rg  = fluid_g.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_gc), jnp.atleast_1d(p_c))["density"][0]
            rl  = fluid_l.get_state(jxp.HmassP_INPUTS,
                                    jnp.atleast_1d(h_lc), jnp.atleast_1d(p_c))["density"][0]
            return jnp.array([ag * rg - m_g, al * rl - m_l])

        _jac = jax.jacfwd(_res)

        def _body(_, pa):
            J    = _jac(pa)
            J_sc = J * jnp.array([jnp.abs(pa[0]) + 1e-300, 1.0])[None, :]
            det  = J_sc[0, 0] * J_sc[1, 1] - J_sc[0, 1] * J_sc[1, 0]
            n0   = jnp.sqrt(J_sc[0, 0] ** 2 + J_sc[0, 1] ** 2)
            n1   = jnp.sqrt(J_sc[1, 0] ** 2 + J_sc[1, 1] ** 2)
            s    = jnp.abs(det) / (n0 * n1 + 1e-300)
            # QUADRATIC gate (probe-calibrated 2026-07-15: two-comp s≈0.36,
            # ghost s≈9e-4, ×400 apart — the linear gate left the ghost at
            # λ≈0.5, a half-engaged pump).  δp cap is ALSO λ-scaled: with a
            # fixed cap a small λ still walks 1%·p per pass toward the
            # far-away ill-conditioned root (measured 7.7% p drift at α=1e-4).
            s2   = s * s
            lam  = s2 / (s2 + proj_s_ref * proj_s_ref)
            dpa  = jnp.linalg.solve(J, -_res(pa))
            dpa  = jnp.where(jnp.all(jnp.isfinite(dpa)), dpa, 0.0)   # singular → no move
            dpa  = lam * dpa
            # DIRECTIONAL trust cap: shrink the WHOLE step by one factor β so
            # |δp| ≤ λ²·1%·p and |δα| ≤ 0.2.  Capping the components
            # independently breaks the Newton direction (α moves as if p had
            # followed) and WORSENS consistency — probe round 2 measured
            # inc ×12 at α=1e-4 with p nearly frozen.  The cap must be λ²:
            # in the ill-conditioned limit the Newton direction is pure-p and
            # noise-following, so the cap IS the per-pass push — with λ¹ the
            # ghost march accumulated it systematically (round 3: p-osc ×10
            # control over 400 steps at 40 Pa/pass).  λ² makes it 0.08 Pa.
            beta = jnp.minimum(1.0, jnp.minimum(
                lam * lam * 0.01 * jnp.abs(pa[0]) / (jnp.abs(dpa[0]) + 1e-300),
                0.2 / (jnp.abs(dpa[1]) + 1e-300)))
            pa   = pa + beta * dpa
            return pa.at[1].set(jnp.clip(pa[1], alpha_eps, 1.0 - alpha_eps))

        pa = jax.lax.fori_loop(0, _PROJ_ITER, _body, jnp.array([p0, ag0]))
        pa = jnp.where(jnp.all(jnp.isfinite(pa)), pa, jnp.array([p0, ag0]))
        return W.at[0].set(pa[0]).at[1].set(pa[1])
        # NOTE 2026-07-15: a "ghost-slave" third stage (α from the liquid
        # constraint + m_g re-slaved, weight 1−λ) was tried here to absorb the
        # secular e_l drift and FAILED at step 1: the α correction the liquid
        # constraint demands is ABSOLUTE (~e_l·(1−α)), i.e. of order α itself
        # at the ghost limit, and infeasible outright when m_l > ρ_l.  The
        # drift has NO legitimate absorber in W (p = pump, α = saturates,
        # m_l/ml_el = real conserved) — it must be fixed at its production
        # site in the scheme, not projected away.

    # eos_project: True (classic 2×2 pair) | 'damped' (conditioning-gated
    # pair, ghost-safe and α-uniform — use for flashing) | False.
    # 'triangular' FAILED certification (see dev_scripts/debug_projection_tri.py,
    # which exists to document the failure) and is retired to a hard error below
    # rather than silently selectable — _project_cell_tri is kept, unreachable,
    # as the historical record of what was tried (2026-08-04 downgrade).
    if eos_project == 'triangular':
        raise ValueError(
            "eos_project='triangular' FAILED certification and is retired — "
            "see dev_scripts/debug_projection_tri.py for the certification run "
            "that documents the failure (secular e_l drift, fails at step 1). "
            "Use eos_project=True (classic) or 'damped' (ghost-safe) instead."
        )
    _project_one = (_project_cell_damped if eos_project == 'damped'
                    else _project_cell)

    def _project_interior(W_int_flat):
        Wp = jax.vmap(_project_one)(W_int_flat.reshape(N - 2, 8))
        return Wp.reshape(-1)

    def step(W_n, dt, p_out=None):
        """
        Advance one implicit time step.

        Parameters
        ----------
        W_n : (N, 8) — W state at time n
        dt  : scalar
        p_out : optional scalar — RUNTIME outlet back-pressure [Pa] for the
                'pressure'/'characteristic' outlets (ramped start-up).  None
                (default) uses the factory's p_out: the original code path.

        Returns
        -------
        W_new : (N, 8)
        ssr   : scalar — normalised steady-state residual
        """
        W_n_flat = W_n.reshape(-1)          # (8*N,)
        W_k      = W_n[1:-1].reshape(-1)    # (8*(N-2),) initial Newton guess

        if debug_newton:
            import numpy as _np
            _eq  = ["m_g", "m_l", "mom_g", "mom_l", "E_g", "E_l"]
            _var = ["p", "ag", "m_g", "u_g", "m_l", "u_l", "mg_eg", "ml_el"]
            print(f"\n  [Newton]  dt={float(dt):.3e}  N_int={N-2}")
            for _it in range(newton_iter):
                R_k = _residual(W_k, W_n_flat, dt, p_out)
                _nan_R = bool(jnp.any(jnp.isnan(R_k)))
                _inf_R = bool(jnp.any(jnp.isinf(R_k)))
                print(f"    it {_it:2d}  max|R|={float(jnp.nanmax(jnp.abs(R_k))):.3e}"
                      f"  NaN={_nan_R}  Inf={_inf_R}")
                # Per-equation-block max|R| + its cell (H1 localization, 2026-07-21):
                # names WHICH equation family and WHICH cell stops contracting.
                # Cell index is GLOBAL (interior i → cell i+1).
                _R_abs = _np.abs(_np.array(R_k)).reshape(N - 2, 6)
                print("           per-eq max|R|: " + "  ".join(
                    f"{_eq[_q]}={_R_abs[:, _q].max():.2e}@c{int(_R_abs[:, _q].argmax()) + 1}"
                    for _q in range(6)))
                if _nan_R or _inf_R:
                    _R_mat = _np.array(R_k).reshape(N - 2, 6)
                    _W_mat = _np.array(W_k).reshape(N - 2, 8)
                    _bad = _np.where(_np.isnan(_R_mat).any(1) | _np.isinf(_R_mat).any(1))[0]
                    print(f"           bad cells: {_bad[:8]}")
                    for _c in _bad[:3]:
                        print(f"             cell {_c}  R = {dict(zip(_eq, _R_mat[_c]))}")
                        print(f"                      W = {dict(zip(_var, _W_mat[_c]))}")
                    break

                J_k = _jac_fn(W_k, W_n_flat, dt, p_out)
                _nan_J = bool(jnp.any(jnp.isnan(J_k)))
                _inf_J = bool(jnp.any(jnp.isinf(J_k)))
                print(f"           max|J|={float(jnp.nanmax(jnp.abs(J_k))):.3e}"
                      f"  NaN={_nan_J}  Inf={_inf_J}")
                if _nan_J or _inf_J:
                    _J_np  = _np.array(J_k)
                    _bad_r = _np.where(_np.isnan(_J_np).any(1) | _np.isinf(_J_np).any(1))[0]
                    _bad_c = _np.where(_np.isnan(_J_np).any(0) | _np.isinf(_J_np).any(0))[0]
                    print(f"           bad J rows: {_bad_r[:8]}  → cells {_bad_r[:8]//6}  eqs {[_eq[r%6] for r in _bad_r[:8]]}")
                    print(f"           bad J cols: {_bad_c[:8]}  → cells {_bad_c[:8]//8}  vars {[_var[c%8] for c in _bad_c[:8]]}")
                    break

                _w_ref = _w_ref_fn(W_k)
                _J_sc  = J_k * _w_ref[None, :]
                _r_sc  = 1.0 / (jnp.max(jnp.abs(_J_sc), axis=1) + 1e-300)  # row equilibration
                _dW_sc = _lin_solve(_J_sc * _r_sc[:, None], -R_k * _r_sc)
                dW     = _dW_sc * _w_ref
                _nan_dW = bool(jnp.any(jnp.isnan(dW)))
                print(f"           max|dW_rel|={float(jnp.nanmax(jnp.abs(_dW_sc))):.3e}"
                      f"  max|dW|={float(jnp.nanmax(jnp.abs(dW))):.3e}"
                      f"  NaN={_nan_dW}")
                # Per-variable max scaled step + its cell: names WHICH W column
                # (soft subspace) takes the overshoot.  Cell index GLOBAL (i+1).
                _dW_abs = _np.abs(_np.array(_dW_sc)).reshape(N - 2, 8)
                print("           per-var max|dW_rel|: " + "  ".join(
                    f"{_var[_q]}={_dW_abs[:, _q].max():.2e}@c{int(_dW_abs[:, _q].argmax()) + 1}"
                    for _q in range(8)))
                W_k = W_k + dW
                # Hard clip α_g to (ε, 1-ε): physically required, guards EOS
                W_k = W_k.at[1::8].set(jnp.clip(W_k[1::8], alpha_eps, 1.0 - alpha_eps))
                _Wm = _np.array(W_k).reshape(N - 2, 8)
                print(f"           p=[{_Wm[:,0].min():.3e},{_Wm[:,0].max():.3e}]"
                      f"  ag=[{_Wm[:,1].min():.3f},{_Wm[:,1].max():.3f}]"
                      f"  u_g=[{_Wm[:,3].min():.2f},{_Wm[:,3].max():.2f}]"
                      f"  u_l=[{_Wm[:,5].min():.2f},{_Wm[:,5].max():.2f}]")
                if _nan_dW:
                    break
        else:
            def _newton_update(W_k):
                """One Newton iteration → (W_new, max|dW_scaled|)."""
                R_k   = _residual(W_k, W_n_flat, dt, p_out)        # (6*(N-2),)
                w_ref = _w_ref_fn(W_k)
                if _use_banded:
                    # E1b: block-banded Jacobian + banded normal equations —
                    # same scaling/solve as below, O(N) instead of O(N³).
                    J_blk = _jac_colored_blocks(W_k, W_n_flat, dt, p_out)
                    dW_sc = _min_norm_banded(J_blk, R_k, w_ref)
                elif _use_banded_e3:
                    # E3-BANDED (new option 2026-07-16): same JᵀJ + Tikhonov
                    # numbers as the dense eos_rows branch below, O(N) storage
                    # and flops.  Per-cell row order [6 flux | 2 EOS].
                    R8    = jnp.concatenate(
                        [R_k.reshape(_Nc, 6),
                         jax.vmap(_eos_res_cell)(W_k.reshape(_Nc, 8))], axis=1)
                    dW_sc = _e3_banded_solve(
                        _jac_blocks_e3(W_k, W_n_flat, dt, p_out), R8, w_ref)
                elif eos_rows:
                    # E3-augmented prototype: 8 rows/cell, same scaling + lstsq
                    R_k   = jnp.concatenate([R_k, _eos_residual(W_k)])
                    J_k   = jnp.concatenate(
                        [_jac_fn(W_k, W_n_flat, dt, p_out), _eos_jac_dense(W_k)], axis=0)
                    J_sc  = J_k * w_ref[None, :]
                    r_sc  = 1.0 / (jnp.max(jnp.abs(J_sc), axis=1) + 1e-300)
                    dW_sc = _lin_solve(J_sc * r_sc[:, None], -R_k * r_sc)
                else:
                    J_k   = _jac_fn(W_k, W_n_flat, dt, p_out)      # (6*(N-2), 8*(N-2))
                    J_sc  = J_k * w_ref[None, :]            # column scaling (relative dW)
                    # Row equilibration: scale each residual row to O(1).  For the
                    # underdetermined min-norm this is EXACTLY invariant (same solution
                    # set) — it only fixes conditioning (cond ~1e8 → ~1e3 from the
                    # mass/energy row-magnitude spread), not the answer.
                    r_sc  = 1.0 / (jnp.max(jnp.abs(J_sc), axis=1) + 1e-300)
                    dW_sc = _lin_solve(J_sc * r_sc[:, None], -R_k * r_sc)
                W_new = W_k + dW_sc * w_ref
                # Hard clip α_g to (ε, 1-ε): physically required, guards EOS
                W_new = W_new.at[1::8].set(jnp.clip(W_new[1::8], alpha_eps, 1.0 - alpha_eps))
                # Project each iterate onto the EOS manifold, INSIDE the loop, so
                # the Newton evaluates its next residual on an EOS-consistent state
                # and co-converges to a jointly conservation+EOS-consistent W.
                # (Projecting only ONCE after the loop changes the fluxes post-hoc
                # and fights the min-norm → dp≈1 at the inlet → adaptive-dt collapse.)
                if eos_project:
                    W_new = _project_interior(W_new)
                return W_new, jnp.max(jnp.abs(dW_sc))

            def _newton_body(_, W_k):
                return _newton_update(W_k)[0]

            if _use_banded_e3 and jac_lag > 1:
                # ── JACOBIAN LAGGING (modified Newton, new option 2026-07-16) ──
                # Recompute J only every jac_lag-th iteration; the residual is
                # STILL evaluated every iteration, so a converged answer is
                # unchanged — only the Newton DIRECTION uses a stale J (local
                # convergence drops from quadratic to linear, harmless for the
                # small per-step corrections of a CFL-limited implicit march).
                # jac_lag >= newton_iter → one Jacobian per step: the ~40
                # colored JVPs (the dominant cost) are paid once instead of
                # newton_iter times.  E3-banded path only; jac_lag=1 keeps the
                # certified per-iteration body below, bit for bit.
                # RESIDUAL-GROWTH GUARD (2026-07-16 evening): a purely
                # scheduled lag DIVERGES on strong transients (restart-pickle
                # startup: the state moves far within one step, the
                # start-of-step J stops contracting → three identical
                # production explosions).  Refresh J additionally whenever
                # ‖R‖ failed to decrease since the previous iteration: on
                # transients this self-reverts to per-iteration Newton, near
                # steady state (‖R‖ contracting) the schedule rules and the
                # full speedup is realized.  Cost of the guard: one norm.
                def _newton_body_lag(it, carry):
                    W_k, J_blk, r_prev = carry
                    R_k   = _residual(W_k, W_n_flat, dt, p_out)
                    w_ref = _w_ref_fn(W_k)
                    R8    = jnp.concatenate(
                        [R_k.reshape(_Nc, 6),
                         jax.vmap(_eos_res_cell)(W_k.reshape(_Nc, 8))], axis=1)
                    r_now = jnp.linalg.norm(R8)
                    J_blk = jax.lax.cond(
                        (it % jac_lag == 0) | (r_now > r_prev),
                        lambda W: _jac_blocks_e3(W, W_n_flat, dt, p_out),
                        lambda W: J_blk, W_k)
                    dW_sc = _e3_banded_solve(J_blk, R8, w_ref)
                    W_new = W_k + dW_sc * w_ref
                    W_new = W_new.at[1::8].set(
                        jnp.clip(W_new[1::8], alpha_eps, 1.0 - alpha_eps))
                    if eos_project:
                        W_new = _project_interior(W_new)
                    return (W_new, J_blk, r_now)

                _J0 = jnp.zeros((_Nc, _Dtot, 8, 8), dtype=jnp.float64)
                W_k, _, _ = jax.lax.fori_loop(0, newton_iter, _newton_body_lag,
                                              (W_k, _J0, jnp.float64(jnp.inf)))
            elif newton_tol > 0.0:
                # SPEED-UP (newton_tol): stop once the scaled update drops below
                # newton_tol.  At CFL ~2 Newton reaches roundoff in 4–5 iterations;
                # the remaining fixed iterations change W by ~1e-12.  newton_iter
                # stays the hard cap.
                def _nt_cond(c):
                    return (c[0] < newton_iter) & (c[2] > newton_tol)

                def _nt_body(c):
                    W_next, err = _newton_update(c[1])
                    return (c[0] + 1, W_next, err)

                _, W_k, _ = jax.lax.while_loop(
                    _nt_cond, _nt_body, (0, W_k, jnp.float64(jnp.inf)))
            else:
                W_k = jax.lax.fori_loop(0, newton_iter, _newton_body, W_k)

        # Re-pin (p, α_g) to the conserved V — removes the min-norm EOS drift
        # (applies to both the debug and production Newton paths).
        if eos_project:
            W_k = _project_interior(W_k)

        W_new = jnp.concatenate([W_n[:1], W_k.reshape(N - 2, 8), W_n[-1:]], axis=0)

        # Steady-state residual (dV = 0 since same W passed for new and old)
        R_ss = _residual(
            W_new[1:-1].reshape(-1), W_new.reshape(-1), jnp.float64(1.0), p_out
        ).reshape(N - 2, 6)

        U_h_new = _bc_U_h(jax.vmap(U_h_from_W)(W_new), p_out)
        lam_max = _max_wave_speed_cells(U_h_new)   # Städtke mixture wave speed
        t_char  = dx / (jnp.max(lam_max) + 1e-300)
        V_new   = jax.vmap(V_from_W)(W_new)
        S_ss    = (t_char * R_ss
                   / (V_cell[1:-1, None] * jnp.abs(V_new[1:-1]) + 1e-300))
        if ssr_mode == 'mass':
            # MASS-WEIGHTED SSR: each phase-equation row votes with its phase's
            # LOCAL mass fraction m_k/(m_g+m_l), per cell.  A near-massless
            # ghost (m~1e-4 of the mixture) then cannot hold the stop criterion
            # hostage with its shock-layer limit cycle (measured: ghost rows
            # owned 99.98% of the plain SSR² on a converged N600 shock run,
            # ghost/dominant ratio 7553×), while in a flashing nozzle both
            # phases vote fully wherever both carry mass — no dominant-phase
            # switch, smooth in α.  Row order: [mass_g,mass_l,mom_g,mom_l,E_g,E_l].
            m_g  = W_new[1:-1, 2]
            m_l  = W_new[1:-1, 4]
            w_g  = m_g / (m_g + m_l + 1e-300)
            w_l  = 1.0 - w_g
            w_r  = jnp.stack([w_g, w_l, w_g, w_l, w_g, w_l], axis=1)
            ssr  = jnp.sqrt(jnp.sum(w_r * S_ss ** 2) / jnp.sum(w_r))
        else:                                       # 'all' (legacy, equal votes)
            ssr = jnp.sqrt(jnp.mean(S_ss ** 2))

        # Write the BC-consistent ghost cells back into W so post-processing sees
        # the actual boundary state (transmissive outlet copy / fixed inlet), not
        # the stale input ghosts.  Re-applied every residual eval, so this does
        # not affect the iteration — only the returned/plotted state.  (2 EOS calls.)
        W_gh  = W_from_U_h_batch(jnp.stack([U_h_new[0], U_h_new[-1]]), fluid_g, fluid_l)
        W_new = W_new.at[0].set(W_gh[0]).at[-1].set(W_gh[-1])

        return W_new, ssr

    # JIT the production path (fori_loop is XLA-compatible).
    # debug_newton uses Python for-loops and print() — not JIT-able.
    if not debug_newton:
        step = jax.jit(step)

    # Diagnostics hook (Jacobian coloring / benchmarking): expose the raw
    # residual and its dense AD Jacobian.  Backward compatible — default off.
    if return_debug:
        return step, compute_dt, _residual, _jac_fn, jac_diff, {
            'jac_colored':        _jac_colored,
            'jac_colored_blocks': _jac_colored_blocks,
            'min_norm_banded':    _min_norm_banded,
            'jac_blocks_e3':      _jac_blocks_e3,
            'e3_banded_solve':    _e3_banded_solve,
            'eos_jac_dense':      _eos_jac_dense,
            'eos_residual':       _eos_residual,
            'lin_solve':          _lin_solve,
            'sw': _sw, 'Nc': _Nc, 'w_char': _w_char,
        }

    return step, compute_dt

# TODO: check if the time stepping works correctly. Are we actually computing the correct pipeeline with W. Also we need Hnc and D at the timestep n+1!
