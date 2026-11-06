"""
FVS flux kernel and non-conservative correction for Städtke's two-fluid model.

Implements §13.6 of stadtke_enthalpy_frame_derivation_complete.md:
  - Per-face FVS interface flux via rank-one projectors in flux space
  - Non-conservative H^nc correction via X @ solve(K, dF)

Entropy fix: STRICT sonic-crossing activation (matches the single-phase Städtke
  solver _jax_F_fvs_stadtke).  The Harten C^1 smoothing is applied to a wave
  ONLY at a face where its eigenvalue changes sign L->R (lam_L<0<lam_R, an
  expansion sonic transition); everywhere else s = exact sign(lam) (full upwind,
  no added dissipation).  delta_factor*|a_mix| is only a tiny robustness floor.
  This replaces the earlier GLOBAL magnitude floor (eps=delta_factor*|a_mix| on
  all 6 waves), which over-fired on the slow material/void waves at low Mach.

Zero-slip regularization (pipeline §4.2):
  When |du| = |u_g - u_l| < eps (Harten threshold), the void-wave right
  eigenvectors c1, c2 become parallel → rho_mat = K @ R_h is rank-deficient
  → jnp.linalg.solve has undefined JVP under jacfwd → NaN in Newton Jacobian.

  Fix: branch on |du| < eps.  In the zero-slip branch, skip forming rho_mat
  and instead compute only the four non-singular projectors (acoustic P3/P4,
  entropy P5/P6), then recover the combined void-wave contribution as
    P12 @ F_blend = F_blend - P3@F_blend - P4@F_blend - P5@F_blend - P6@F_blend
  using the completeness identity P1+P2+P3+P4+P5+P6 = I.

  Result for the zero-slip branch:
    F_hat = F_12_blend
          + Σ_{k=3,4,5,6}  0.5*(s_k - s_12) · rho_k · (l_k · y_dF) / D_k
  where F_12_blend = upwind blend of F_L, F_R at the void-wave speed u_mix,
        y_dF       = K⁻¹ @ (F_L - F_R)  (one K-solve, K non-singular at u≠0),
        rho_k      = K @ r_k  (mat-vec, always stable).

  Under jax.lax.cond both branches are traced for JVP, but lax.select discards
  the NaN tangents from the unselected (full) branch when du ≈ 0.
"""

import jax
import jax.numpy as jnp
from .matrices import (
    PhaseProps, phase_props, CommonQty, common_quantities,
    K_matrix, X_matrix, eigenstructure, flux_vector, J_matrix,
)


# ---------------------------------------------------------------------------
# Equilibrated linear solves
# ---------------------------------------------------------------------------
# The two-fluid matrices mix units: mass rows ~O(10), momentum ~O(1e6), energy
# ~O(1e8) → kappa(rho_mat) ~ 1e13 from SCALING alone (after one max-abs row+col
# equilibration it drops to ~1e2; measured at the throat checkerboard cell).
# A raw solve at kappa~1e13 injects ~kappa*eps ≈ 1e-3 RELATIVE roundoff noise
# into the wave decomposition every step — a stochastic forcing that lands
# hardest on the small-magnitude VOID row (the observed throat checkerboard,
# and the SSR plateau at ~1e-3 = the noise floor).  Equilibration is an exact
# identity in real arithmetic — it changes nothing but the roundoff.

def _equil_solve(A, B):
    """solve(A, B) with max-abs row+col equilibration (B may be (6,) or (6,m))."""
    dr = 1.0 / jnp.maximum(jnp.max(jnp.abs(A), axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / jnp.maximum(jnp.max(jnp.abs(As), axis=0), 1e-300)
    As = As * dc[None, :]
    Bs = B * (dr[:, None] if B.ndim == 2 else dr)
    Y  = jnp.linalg.solve(As, Bs)
    return Y * (dc[:, None] if B.ndim == 2 else dc)


def _equil_lstsq(A, B, rcond=1e-14):
    """lstsq(A, B) with the same equilibration (rcond acts on the SCALED matrix,
    so it no longer risks truncating genuine directions hidden by bad units)."""
    dr = 1.0 / jnp.maximum(jnp.max(jnp.abs(A), axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / jnp.maximum(jnp.max(jnp.abs(As), axis=0), 1e-300)
    As = As * dc[None, :]
    Bs = B * (dr[:, None] if B.ndim == 2 else dr)
    Y  = jnp.linalg.lstsq(As, Bs, rcond=rcond)[0]
    return Y * (dc[:, None] if B.ndim == 2 else dc)


# ---------------------------------------------------------------------------
# Harten entropy fix
# ---------------------------------------------------------------------------

def smoothed_sign(lam, eps):
    """
    C^1-smooth sign function with Harten entropy fix.

    Returns s ∈ [-1, 1] such that:
      s = sign(lam)  when |lam| >= eps
      s → lam/eps    for |lam| << eps  (linear near zero)

    The smooth absolute value used internally:
      |lam|_smooth = |lam|             when |lam| >= eps
                   = (lam^2+eps^2)/(2*eps)  otherwise

    Transition is C^1 at |lam| = eps.  Used to blend upwind (|s|=1) to
    centered (s=0) smoothly near sonic points.
    """
    abs_smooth = jnp.where(
        jnp.abs(lam) >= eps,
        jnp.abs(lam),
        (lam**2 + eps**2) / (2.0 * eps),
    )
    return lam / abs_smooth


# ---------------------------------------------------------------------------
# FVS interface flux
# ---------------------------------------------------------------------------

def fvs_interface_flux(U_h_L, U_h_R, fluid_g, fluid_l, k,
                       delta_factor=0.1, lowmach_diss=0.0,
                       du_blend_lo=1e-2, du_blend_hi=1e-1,
                       single_branch=False):
    """
    FVS interface flux F_hat at one face of the two-fluid system.

    Implements flux-vector-splitting in enthalpy-primitive variables
    (derivation §13.6).  Two internal branches selected by slip criterion:

    Full branch (|du| >= eps):
      Standard decomposition via rho_mat = K @ R_h.  rho_mat is rank-6 when
      du ≠ 0 so the direct solve is well-conditioned.

    Zero-slip branch (|du| < eps, pipeline §4.2):
      c1, c2 become parallel at du=0 → rho_mat rank-deficient → solve NaN.
      Uses  P12 = I - P3 - P4 - P5 - P6  for the combined void-wave block;
      only one K-solve needed (for F_L - F_R), not a full 6×6 system inversion.
      Under jax.lax.cond both branches are JVP-traced; lax.select discards the
      NaN tangents from the full branch when du ≈ 0.

    Parameters
    ----------
    U_h_L, U_h_R : (6,) — left/right primitive states [p,u_g,u_l,ag,h_g,h_l]
    fluid_g, fluid_l : FluidBicubic tables (metastable vapor / liquid)
    k            : virtual-mass coefficient
    delta_factor : Harten floor fraction — eps = delta_factor * |a_mix|
                   also used as the slip threshold for the zero-slip branch
    single_branch : bool, default False.  True evaluates the zero-slip and full
                   branches once each and selects with jnp.where (same values,
                   ~half the flux cost under vmap); False keeps the original
                   lax.cond, which under vmap runs the zero-slip branch twice.

    Returns
    -------
    F_hat : (6,) — interface flux (conservative ordering)
    """
    # TODO: Why we are using bar and not pasccals??? Pr bar is the face value?
    # ── Average state at the face ────────────────────────────────────────────
    U_h_bar = 0.5 * (U_h_L + U_h_R)
    p_b, u_g_b, u_l_b, ag_b, h_g_b, h_l_b = U_h_bar

    eos_g_b = phase_props(fluid_g, p_b, h_g_b)
    eos_l_b = phase_props(fluid_l, p_b, h_l_b)
    cq_b    = common_quantities(U_h_bar, eos_g_b, eos_l_b, k)

    # ── Eigenstructure and K (computed once, shared by both branches) ─────────
    eigvals, R_h, L_h, D = eigenstructure(U_h_bar, eos_g_b, eos_l_b, cq_b)
    K_mat = K_matrix(U_h_bar, eos_g_b, eos_l_b)

    # ── Physical fluxes at actual L/R states ─────────────────────────────────
    eos_g_L = phase_props(fluid_g, U_h_L[0], U_h_L[4])
    eos_l_L = phase_props(fluid_l, U_h_L[0], U_h_L[5])
    F_L = flux_vector(U_h_L, eos_g_L, eos_l_L)

    eos_g_R = phase_props(fluid_g, U_h_R[0], U_h_R[4])
    eos_l_R = phase_props(fluid_l, U_h_R[0], U_h_R[5])
    F_R = flux_vector(U_h_R, eos_g_R, eos_l_R)

    # ── Strict sonic-crossing entropy fix (port of the single-phase Städtke
    #    _jax_F_fvs_stadtke fix, fvm.py §5) ──────────────────────────────────
    # Activate the Harten smoothing on a wave ONLY where its eigenvalue changes
    # sign across the face (λ_L < 0 < λ_R — an EXPANSION sonic transition); never
    # at shocks, never in smooth flow.  Off-sonic, s = exact sign(λ_avg) (full
    # upwind, ZERO added dissipation).  In forward flow only the acoustic− wave
    # (idx 3, u_mix−a) ever crosses zero, so the material/void/entropy waves
    # (idx 0,1,4,5) stay untouched — reproducing the single-phase behaviour and
    # avoiding the GLOBAL smearing of the old magnitude-floor fix (eps=δ·|a_mix|
    # applied to all 6 waves), which fired on the slow waves everywhere at low Ma.
    eigvals_L = eigenstructure(
        U_h_L, eos_g_L, eos_l_L, common_quantities(U_h_L, eos_g_L, eos_l_L, k))[0]
    eigvals_R = eigenstructure(
        U_h_R, eos_g_R, eos_l_R, common_quantities(U_h_R, eos_g_R, eos_l_R, k))[0]

    a_mix        = 0.5 * (eigvals[2] - eigvals[3])       # mixture sound speed (>0)
    delta_global = delta_factor * jnp.abs(a_mix)         # tiny robustness floor ONLY

    sonic   = (eigvals_L < 0.0) & (eigvals_R > 0.0)      # (6,) expansion sonic crossing
    delta_k = jnp.where(sonic, eigvals_R - eigvals_L, delta_global)
    # Harten C¹ smoothed |λ|: (λ²+δ²)/(2δ) at a sonic-crossing face, |λ| elsewhere.
    lam_safe = jnp.where(
        sonic,
        (eigvals**2 + delta_k**2) / (2.0 * jnp.maximum(delta_k, 1e-300)),
        jnp.maximum(jnp.abs(eigvals), delta_global),
    )
    # Signed upwind weight s ∈ [−1,1] = sign(λ)·(|λ|/|λ|_safe): ±1 (full upwind)
    # off-sonic, blends toward 0 (central, added dissipation) only at a crossing.
    s = eigvals / lam_safe

    du = cq_b.du   # u_g - u_l

    # ── Low-Mach dissipation scaling (Rieper/Thornber-type fix) ──────────────
    # At low Ma, the standard FVS dissipation on the acoustic pair (idx 2,3)
    # over-dissipates the pressure-velocity coupling relative to the physical
    # scale of the pressure fluctuation (O(Ma) too strong), which is what
    # produces the undamped checkerboard mode in alpha_g/u_l at low Ma.
    # z_ac -> 1 recovers the full standard scheme as Ma -> O(1) (shocks,
    # sonic throat unaffected). z_ac floors at `lowmach_diss` rather than 0
    # to avoid *removing* dissipation entirely, which would reopen the
    # checkerboard from the other direction (undamped central scheme).
    # _, u_g, u_l, ag, _, _ = U_h
    # al = 1.0 - ag
    # TODO: this fixing is not avoiding thee checkerboard at the inlet
    rg = eos_g_b.rho;  ag_ = eos_g_b.a;  rl = eos_l_b.rho;  al_ = eos_l_b.a
    al_b = 1.0 - ag_b
    cq = cq_b
    rg_hat  = cq.rg_hat;  rl_hat  = cq.rl_hat;  rh2    = cq.rh2
    rho_mix = cq.rho_mix; rho_s   = cq.rho_s;   a0sq   = cq.a0sq
    du      = cq.du

    # ── Extra abbreviations for eigenstructure (§13.1) ──────────────────────
    # k*rho_mix is not stored in cq; derive from rg_hat = rg + k*rho_mix.
    k_rho_mix  = rg_hat - rg
    rho_hat_s  = rho_s + k_rho_mix          # hat_rho_s = rho_s + k*rho_mix
    rh         = jnp.sqrt(rh2)              # hat_rho = sqrt(hat_rho^2)

    a_tilde_sq = (rho_hat_s / rho_s) * (rg * rl / rh2) * a0sq
    delta_a_sq = ag_b * al_b * rg * rl * rg_hat * rl_hat / rh2**2 * du**2
    a_sq       = a_tilde_sq - delta_a_sq
    a          = jnp.sqrt(jnp.maximum(a_sq, 0.0))  # real mixture sound speed

    hat_ag = ag_b * rg * rl_hat / rh2         # volume-fraction weight for u_g
    hat_al = al_b * rl * rg_hat / rh2         # volume-fraction weight for u_l
    u_mix  = hat_ag * u_g_b + hat_al * u_l_b    # mixture velocity

    
    Ma_face = jnp.abs(u_mix) / jnp.maximum(jnp.abs(a), 1e-8)
    z_ac    = jnp.clip(Ma_face, lowmach_diss, 1.0)

    # ── Zero-slip branch: |du| < eps ─────────────────────────────────────────
    def _zero_slip(_):
        """
        Void-wave block handled as P12 = I - P3 - P4 - P5 - P6.

        P3,P4 (acoustic) and P5,P6 (entropy) are individually non-singular
        at du=0.  Only one K-solve needed: y_dF = K^{-1} @ (F_L - F_R).
        K is non-singular whenever u_g, u_l ≠ 0 (§4.3 separate issue).

        Formula (derived from completeness P1+...+P6 = I):
          F_hat = F_12_blend
                + Σ_{k=3,4,5,6} 0.5*(s_k - s_12) * (K·r_k) * (l_k·y_dF) / D_k

        where F_12_blend uses the volume-weighted mixture velocity u1 as the
        common speed for both void waves (exact at du=0; O(du) error otherwise).
        """
        # Volume-weighted mixture velocity as speed for the void-wave block.
        # u1 is a MATERIAL speed (never crosses sonic in forward flow), so no
        # entropy fix is warranted — the tiny delta_global floor only keeps the
        # sign C¹-smooth near u1=0 for a well-defined Newton Jacobian.
        s_12  = smoothed_sign(cq_b.u1, delta_global)
        F_12b = 0.5 * (1.0 + s_12) * F_L + 0.5 * (1.0 - s_12) * F_R

        # One K-solve for the flux difference (lstsq guards the §4.3 u_i→0 case);
        # equilibrated: kappa(K) is also ~1e13 from unit scaling alone.
        y_dF = _equil_lstsq(K_mat, F_L - F_R, rcond=1e-14)

        # Accumulate corrections from acoustic (k=2,3) and entropy (k=4,5) waves
        # (0-indexed: indices 2,3 = acoustic+/−, indices 4,5 = entropy g/l)
        F_hat = F_12b
        for ki in [2, 3, 4, 5]:
            rho_k = K_mat @ R_h[:, ki]
            ell_k = jnp.dot(L_h[ki], y_dF) / D[ki]
            # F_hat = F_hat + 0.5 * (s[ki] - s_12) * rho_k * ell_k
            # z_ki  = z_ac if ki in (2, 3) else 1.0     # only scale acoustic pair
            z_ki = jnp.where((ki == 2) | (ki == 3), z_ac, 1.0)
            F_hat = F_hat + z_ki * 0.5 * (s[ki] - s_12) * rho_k * ell_k
        return F_hat

    # ── Full branch: |du| >= eps ──────────────────────────────────────────────
    def _full_slip(_):
        """
        Standard rho_mat = K @ R_h decomposition.
        rho_mat is rank-6 when |du| >= eps so the direct solve is stable.
        """
        rho_mat = K_mat @ R_h
        # Equilibrated solve: raw kappa(rho_mat)~1e13 is pure unit scaling (mass
        # vs energy rows) → ~1e-3 relative roundoff on the wave strengths, which
        # acts as stochastic anti-diffusion on the small VOID row (the throat
        # checkerboard pump).  After scaling kappa~1e2 → noise ~1e-14.
        beta_LR = _equil_solve(
            rho_mat, jnp.stack([F_L, F_R], axis=1)   # (6,2) RHS
        )
        # w_plus  = 0.5 * (1.0 + s)
        # w_minus = 0.5 * (1.0 - s)
        # beta    = w_plus * beta_LR[:, 0] + w_minus * beta_LR[:, 1]
        dbeta   = beta_LR[:, 0] - beta_LR[:, 1]
        z_vec   = jnp.ones(6).at[2].set(z_ac).at[3].set(z_ac)  # only idx 2,3 scaled
        dbeta_lm = z_vec * dbeta

        beta_c  = 0.5 * (beta_LR[:, 0] + beta_LR[:, 1])
        beta    = beta_c + 0.5 * s * dbeta_lm
        return rho_mat @ beta

    # ── Smooth blend between the zero-slip and full branches ──────────────────
    # The full branch inverts rho_mat = K@R_h, whose two void-wave columns c1,c2
    # become PARALLEL as du→0 (κ(rho_mat) ~ 1/du): rank-deficient (NaN) exactly at
    # du=0, and NOISY in a whole RING |du|≲du_blend_hi around every slip sign-change.
    # The old hard switch |du|<1e-4 protected only the exact singularity, leaving that
    # noisy ring on the full branch → a 2Δx checkerboard localized at du zero-crossings
    # (clean between crossings).  No amount of background dissipation fixes it because
    # the pump is the flux itself there.
    #
    # Fix: well-conditioned zero-slip branch for |du|<du_blend_lo, accurate full branch
    # for |du|>du_blend_hi, C¹ cosine blend between.  The OUTER cond guarantees
    # _full_slip only executes for |du|>=du_blend_lo (finite) so no NaN leaks into the
    # blend.  The zero-slip branch is only O(du) less accurate (both void waves share
    # the mixture speed u1) — negligible across the narrow blend window, and it is the
    # correct limit at du=0.  Widen [du_blend_lo,du_blend_hi] on coarser meshes (where
    # du changes more per cell) so the transition still spans a few cells.
    def _blend(_):
        Fz = _zero_slip(None)
        Ff = _full_slip(None)                    # |du| >= du_blend_lo here ⇒ finite
        t  = jnp.clip((jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo),
                      0.0, 1.0)
        w  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))   # 1 at du_blend_lo → 0 at du_blend_hi (C¹)
        return w * Fz + (1.0 - w) * Ff

    if single_branch:
        # SPEED-UP (2026-10-05, numerics.flux_single_branch): under vmap the
        # batched-predicate lax.cond below lowers to select and runs BOTH branches,
        # and _blend re-runs _zero_slip — the zero-slip branch (incl. its SVD lstsq)
        # is evaluated twice per face.  Evaluate each branch exactly once and
        # select: identical values (where() discards the full branch's NaN at
        # du≈0 in value and tangent alike, as select already did).
        Fz = _zero_slip(None)
        Ff = _full_slip(None)
        t  = jnp.clip((jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo),
                      0.0, 1.0)
        w  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        return jnp.where(jnp.abs(du) < du_blend_lo, Fz, w * Fz + (1.0 - w) * Ff)

    F_hat = jax.lax.cond(jnp.abs(du) < du_blend_lo, _zero_slip, _blend, None)
    # F_hat = _full_slip(None)

    return F_hat


# ---------------------------------------------------------------------------
# Upwinded primitive face state (K-free non-conservative term)
# ---------------------------------------------------------------------------

def fvs_interface_state(U_h_L, U_h_R, fluid_g, fluid_l, k,
                        delta_factor=0.1, lowmach_diss=0.0,
                        du_blend_lo=1e-2, du_blend_hi=1e-1):
    """
    Upwinded primitive face state U_hat = ½(U_L+U_R) + ½ S (U_L-U_R).

    S is the primitive sign matrix of fvs_interface_flux: the same wave signs
    (incl. the sonic entropy fix), the same low-Mach weights z_ac and the same
    zero-slip / full blend.  The face flux is F_hat ≈ ½(F_L+F_R) + ½ K S ΔU, so
    U_hat is its primitive counterpart without K⁻¹.  The K-free cell term is
    X_i (U_hat_{i+1/2} - U_hat_{i-1/2}), bounded where det K = 0.
    """
    U_h_bar = 0.5 * (U_h_L + U_h_R)
    dU = U_h_L - U_h_R

    eos_g_b = phase_props(fluid_g, U_h_bar[0], U_h_bar[4])
    eos_l_b = phase_props(fluid_l, U_h_bar[0], U_h_bar[5])
    cq_b = common_quantities(U_h_bar, eos_g_b, eos_l_b, k)
    eigvals, R_h, L_h, D = eigenstructure(U_h_bar, eos_g_b, eos_l_b, cq_b)

    eos_g_L = phase_props(fluid_g, U_h_L[0], U_h_L[4])
    eos_l_L = phase_props(fluid_l, U_h_L[0], U_h_L[5])
    eos_g_R = phase_props(fluid_g, U_h_R[0], U_h_R[4])
    eos_l_R = phase_props(fluid_l, U_h_R[0], U_h_R[5])
    eigvals_L = eigenstructure(
        U_h_L, eos_g_L, eos_l_L, common_quantities(U_h_L, eos_g_L, eos_l_L, k))[0]
    eigvals_R = eigenstructure(
        U_h_R, eos_g_R, eos_l_R, common_quantities(U_h_R, eos_g_R, eos_l_R, k))[0]

    a_mix = 0.5 * (eigvals[2] - eigvals[3])
    delta_global = delta_factor * jnp.abs(a_mix)
    sonic = (eigvals_L < 0.0) & (eigvals_R > 0.0)
    delta_k = jnp.where(sonic, eigvals_R - eigvals_L, delta_global)
    lam_safe = jnp.where(
        sonic,
        (eigvals**2 + delta_k**2) / (2.0 * jnp.maximum(delta_k, 1e-300)),
        jnp.maximum(jnp.abs(eigvals), delta_global),
    )
    s = eigvals / lam_safe

    # Low-Mach weight z_ac, as in fvs_interface_flux
    ag_b = U_h_bar[3]
    al_b = 1.0 - ag_b
    rg = eos_g_b.rho
    rl = eos_l_b.rho
    cq = cq_b
    du = cq.du
    k_rho_mix = cq.rg_hat - rg
    rho_hat_s = cq.rho_s + k_rho_mix
    a_tilde_sq = (rho_hat_s / cq.rho_s) * (rg * rl / cq.rh2) * cq.a0sq
    delta_a_sq = ag_b * al_b * rg * rl * cq.rg_hat * cq.rl_hat / cq.rh2**2 * du**2
    a = jnp.sqrt(jnp.maximum(a_tilde_sq - delta_a_sq, 0.0))
    hat_ag = ag_b * rg * cq.rl_hat / cq.rh2
    hat_al = al_b * rl * cq.rg_hat / cq.rh2
    u_mix = hat_ag * U_h_bar[1] + hat_al * U_h_bar[2]
    z_ac = jnp.clip(jnp.abs(u_mix) / jnp.maximum(jnp.abs(a), 1e-8), lowmach_diss, 1.0)
    z_vec = jnp.ones(6).at[2].set(z_ac).at[3].set(z_ac)

    # Zero-slip: S = s12 I + sum_{k=3..6} z_k (s_k - s12) r_k l_k / D_k
    s_12 = smoothed_sign(cq_b.u1, delta_global)
    SdU_z = s_12 * dU
    for ki in [2, 3, 4, 5]:
        SdU_z = SdU_z + z_vec[ki] * (s[ki] - s_12) * R_h[:, ki] * jnp.dot(L_h[ki], dU) / D[ki]

    # Full: S = sum_k z_k s_k r_k l_k / D_k
    SdU_f = R_h @ (z_vec * s * (L_h @ dU) / D)

    t = jnp.clip((jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo), 0.0, 1.0)
    w = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    SdU = jnp.where(jnp.abs(du) < du_blend_lo, SdU_z, w * SdU_z + (1.0 - w) * SdU_f)
    return U_h_bar + 0.5 * SdU


# ---------------------------------------------------------------------------
# Inverse-free face flux and path-integrated non-conservative term
# (port of the 'inverse_old_sonic' package of the main branch,
#  development/two_fluid_test_v2/scripts/numerics.py, 2026-11-05)
# ---------------------------------------------------------------------------

def inverse_free_interface_flux(U_h_L, U_h_R, fluid_g, fluid_l, k,
                                delta_factor=0.1, lowmach_diss=0.0,
                                du_blend_lo=1e-2, du_blend_hi=1e-1):
    """
    Face flux without K⁻¹: primitive-jump viscosity

        F_hat = ½(F_L + F_R) − ½ J(Ū) Σ_k z_k m_k r_k l_k (U_R − U_L) / D_k

    with the magnitudes m_k = λ_k s_k of the native signed weights s_k (same strict
    sonic-crossing entropy fix and delta_factor floor as fvs_interface_flux), the same
    low-Mach weight z_ac on the acoustic pair and the same zero-slip / full blend
    (du_blend_lo, du_blend_hi).  Zero-slip ('grouped'): both void waves move with the
    volume-weighted mixture speed u1, material = u1 s_12.  The grouped viscosity is a
    bounded approximation (not the exact |A| at the material Jordan block).
    """
    U_h_bar = 0.5 * (U_h_L + U_h_R)
    jump = U_h_R - U_h_L

    eos_g_b = phase_props(fluid_g, U_h_bar[0], U_h_bar[4])
    eos_l_b = phase_props(fluid_l, U_h_bar[0], U_h_bar[5])
    cq_b = common_quantities(U_h_bar, eos_g_b, eos_l_b, k)
    eigvals, R_h, L_h, D = eigenstructure(U_h_bar, eos_g_b, eos_l_b, cq_b)

    eos_g_L = phase_props(fluid_g, U_h_L[0], U_h_L[4])
    eos_l_L = phase_props(fluid_l, U_h_L[0], U_h_L[5])
    eos_g_R = phase_props(fluid_g, U_h_R[0], U_h_R[4])
    eos_l_R = phase_props(fluid_l, U_h_R[0], U_h_R[5])
    F_L = flux_vector(U_h_L, eos_g_L, eos_l_L)
    F_R = flux_vector(U_h_R, eos_g_R, eos_l_R)
    eigvals_L = eigenstructure(
        U_h_L, eos_g_L, eos_l_L, common_quantities(U_h_L, eos_g_L, eos_l_L, k))[0]
    eigvals_R = eigenstructure(
        U_h_R, eos_g_R, eos_l_R, common_quantities(U_h_R, eos_g_R, eos_l_R, k))[0]

    # native signed weights (strict sonic-crossing Harten fix, tiny global floor)
    a_mix = 0.5 * (eigvals[2] - eigvals[3])
    delta_global = delta_factor * jnp.abs(a_mix)
    sonic = (eigvals_L < 0.0) & (eigvals_R > 0.0)
    delta_k = jnp.where(sonic, eigvals_R - eigvals_L, delta_global)
    lam_safe = jnp.where(
        sonic,
        (eigvals**2 + delta_k**2) / (2.0 * jnp.maximum(delta_k, 1e-300)),
        jnp.maximum(jnp.abs(eigvals), delta_global),
    )
    s = eigvals / lam_safe

    # native low-Mach weight on the acoustic pair
    ag_b = U_h_bar[3]
    al_b = 1.0 - ag_b
    rg = eos_g_b.rho
    rl = eos_l_b.rho
    cq = cq_b
    du = cq.du
    rho_hat_s = cq.rho_s + (cq.rg_hat - rg)
    a_tilde_sq = (rho_hat_s / cq.rho_s) * (rg * rl / cq.rh2) * cq.a0sq
    delta_a_sq = ag_b * al_b * rg * rl * cq.rg_hat * cq.rl_hat / cq.rh2**2 * du**2
    a = jnp.sqrt(jnp.maximum(a_tilde_sq - delta_a_sq, 0.0))
    u_mix = (ag_b * rg * cq.rl_hat / cq.rh2) * U_h_bar[1] + (al_b * rl * cq.rg_hat / cq.rh2) * U_h_bar[2]
    z_ac = jnp.clip(jnp.abs(u_mix) / jnp.maximum(jnp.abs(a), 1e-8), lowmach_diss, 1.0)
    z_vec = jnp.ones(6).at[2].set(z_ac).at[3].set(z_ac)

    magnitude = eigvals * s * z_vec
    material = cq.u1 * smoothed_sign(cq.u1, delta_global)

    def _grouped(_):
        out = material * jump
        for ki in (2, 3, 4, 5):
            out = out + (magnitude[ki] - material) * R_h[:, ki] * jnp.dot(L_h[ki], jump) / D[ki]
        return out

    def _full(_):
        return R_h @ (magnitude * (L_h @ jump) / D)

    def _transition(_):
        t = jnp.clip((jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo), 0.0, 1.0)
        w = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        return w * _grouped(None) + (1.0 - w) * _full(None)

    action = jax.lax.cond(
        jnp.abs(du) < du_blend_lo, _grouped,
        lambda _: jax.lax.cond(jnp.abs(du) >= du_blend_hi, _full, _transition, None), None)
    return 0.5 * (F_L + F_R) - 0.5 * (J_matrix(U_h_bar, eos_g_b, eos_l_b) @ action)


def path_nonconservative(U_h_a, U_h_b, fluid_g, fluid_l, k):
    """∫ X(U) dU along the straight path U_a → U_b in enthalpy primitives (two-point
    Gauss).  No K⁻¹: bounded where det K = 0."""
    jump = U_h_b - U_h_a

    def one(t):
        u = U_h_a + t * jump
        eos_g = phase_props(fluid_g, u[0], u[4])
        eos_l = phase_props(fluid_l, u[0], u[5])
        return X_matrix(u, eos_g, eos_l, common_quantities(u, eos_g, eos_l, k)) @ jump

    off = 0.5 / jnp.sqrt(3.0)
    return 0.5 * (one(0.5 - off) + one(0.5 + off))


# ---------------------------------------------------------------------------
# Non-conservative H^nc correction
# ---------------------------------------------------------------------------

def nonconservative_correction(U_h_cell, dF,
                                eos_g: PhaseProps,
                                eos_l: PhaseProps,
                                cq: CommonQty) -> jnp.ndarray:
    """
    Non-conservative correction H^nc * dF = X * K^{-1} * dF.

    Implements the operational form from derivation §12.5:
        y   = solve(K, dF)
        nc  = X @ y

    The dense H^nc matrix (24 non-zero entries, algebraically complex) is
    never formed.  K is singular at u_i=0 (§12.4); at those states the
    relevant components of dF vanish by the flux structure, so the linear
    system is consistent even without a dense inverse.

    Parameters
    ----------
    U_h_cell : (6,) primitive state at the cell centre
    dF       : (6,) flux difference  F_hat_{i+1/2} - F_hat_{i-1/2}
    eos_g, eos_l : PhaseProps pre-computed at U_h_cell (avoids duplicate EOS)
    cq       : CommonQty pre-computed at U_h_cell

    Returns
    -------
    nc : (6,) non-conservative contribution to the cell residual
    """
    K_mat = K_matrix(U_h_cell, eos_g, eos_l)
    X_mat = X_matrix(U_h_cell, eos_g, eos_l, cq)
    # Equilibrated (kappa(K)~1e13 is unit scaling; see _equil_solve note above).
    y = _equil_solve(K_mat, dF)
    return X_mat @ y

    # TODO: There should be the calculation of the invers matrix also in another  point. Recallit and check if the impelemnation is correct
