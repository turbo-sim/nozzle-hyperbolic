# Port reference: origin/alberto-branch faafff704e54fd866abff95fcb88fc0a5217405c.
# Intentional safety changes are documented in docs/notes/two_fluid_migration.md.
"""Alberto's compatibility FVS and nonconservative correction kernels.

The face flux uses flux-space projectors; the cell correction is
``X @ compatible_flux_solve(K, dF, U_h)``. At equal slip, the shared void-wave
projector uses completeness instead of a singular full decomposition. The model
keeps scalar ``lax.cond`` execution inside ``lax.map`` for safe JIT/AD.
Sonic-crossing smoothing and the small robustness floor retain the active port's
arithmetic. See docs/notes/two_fluid_migration.md for provenance and policies.
"""

import jax
import jax.numpy as jnp
from .matrices import (
    PhaseProps,
    phase_props,
    CommonQty,
    common_quantities,
    K_matrix,
    X_matrix,
    eigenstructure,
    flux_vector,
)


# ---------------------------------------------------------------------------
# Equilibrated linear solves
# ---------------------------------------------------------------------------
# Alberto's motivating observation: the matrices mix units, with mass rows
# ~O(10), momentum ~O(1e6), energy
# ~O(1e8) → kappa(rho_mat) ~ 1e13 from SCALING alone (after one max-abs row+col
# equilibration it drops to ~1e2; measured at the throat checkerboard cell).
# A raw solve at kappa~1e13 injects ~kappa*eps ≈ 1e-3 RELATIVE roundoff noise
# into the wave decomposition every step — a stochastic forcing that lands
# hardest on the small-magnitude VOID row (the observed throat checkerboard,
# and the observed SSR plateau at ~1e-3). These are historical case-specific
# observations, not bounds on every current solve. Equilibration is an exact
# identity in real arithmetic: it changes scaling and roundoff, not equations.


def _equil_solve(A, B):
    """Solve a six-row system after max-absolute row and column equilibration.

    Parameters
    ----------
    A : jax.Array, shape (6, 6)
        Coefficient matrix.
    B : jax.Array, shape (6,) or (6, M)
        Right-hand side vector or matrix.

    Returns
    -------
    jax.Array
        Solution with the shape of ``B``; equilibration changes scaling only.
    """
    dr = 1.0 / jnp.maximum(jnp.max(jnp.abs(A), axis=1), 1e-300)
    As = A * dr[:, None]
    dc = 1.0 / jnp.maximum(jnp.max(jnp.abs(As), axis=0), 1e-300)
    As = As * dc[None, :]
    Bs = B * (dr[:, None] if B.ndim == 2 else dr)
    Y = jnp.linalg.solve(As, Bs)
    return Y * (dc[:, None] if B.ndim == 2 else dc)


def compatible_flux_solve(A, B, U):
    """Solve in the physical-flux range at phase rest as well as finite velocity.

    Parameters
    ----------
    A : jax.Array, shape (6, 6)
        Physical-flux Jacobian K.
    B : jax.Array, shape (6,)
        Flux-difference right-hand side.
    U : jax.Array, shape (6,)
        Enthalpy primitives used to identify exactly resting phases.

    Returns
    -------
    jax.Array, shape (6,)
        Compatible solution representative, or NaNs when the range check fails.

    Notes
    -----
    At u_g=0, gas energy row is H_g times its mass row and column h_g
    vanishes. Replace only that redundant row by y_hg=0 (likewise liquid).
    X and K annihilate these null columns, so their actions are independent
    of that representative. Reject RHS outside range(K) and other singular
    matrices. No velocity floor, rank truncation or pseudoinverse is used.
    """
    gas_zero, liquid_zero = U[1] == 0.0, U[2] == 0.0
    replaced = A.at[4].set(jnp.where(gas_zero, jnp.eye(6)[4], A[4]))
    replaced = replaced.at[5].set(jnp.where(liquid_zero, jnp.eye(6)[5], A[5]))
    rhs = B.at[4].set(jnp.where(gas_zero, 0.0, B[4]))
    rhs = rhs.at[5].set(jnp.where(liquid_zero, 0.0, B[5]))
    y = _equil_solve(replaced, rhs)
    residual = A @ y - B
    scale = jnp.maximum(jnp.abs(A) @ jnp.abs(y) + jnp.abs(B), 1.0)
    valid = jnp.all(jnp.isfinite(y)) & jnp.all(jnp.abs(residual) <= 1e-10 * scale)
    return jnp.where(valid, y, jnp.nan)


# ---------------------------------------------------------------------------
# Harten entropy fix
# ---------------------------------------------------------------------------


def smoothed_sign(lam, eps):
    """Evaluate the Harten-smoothed upwind sign.

    Parameters
    ----------
    lam : array_like
        Wave speed.
    eps : array_like
        Strictly positive sonic smoothing width in the same units.

    Returns
    -------
    array_like
        ``lam/abs_smooth``, equal to sign(lam) outside the smoothing band.

    Notes
    -----
    Inside the band, ``abs_smooth=(lam**2+eps**2)/(2*eps)``.
    The transition is C1 at both band edges.
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


def fvs_interface_flux(
    U_h_L,
    U_h_R,
    fluid_g,
    fluid_l,
    k,
    delta_factor=0.1,
    lowmach_diss=0.0,
    du_blend_lo=1e-2,
    du_blend_hi=1e-1,
    single_branch=False,
):
    """Evaluate one compatibility FVS flux in conservative ordering.

    Parameters
    ----------
    U_h_L, U_h_R : jax.Array, shape (6,)
        Left/right enthalpy primitives in SI units.
    fluid_g, fluid_l : object
        JAXProp-compatible gas and liquid fluids.
    k : float
        Nonnegative dimensionless virtual-mass coefficient.
    delta_factor : float, optional
        Acoustic-speed fraction setting the entropy-fix robustness floor.
    lowmach_diss : float, optional
        Lower bound of the acoustic dissipation factor, in [0, 1].
    du_blend_lo, du_blend_hi : float, optional
        Absolute-slip thresholds in m/s blending shared-velocity and full formulas.
    single_branch : bool, optional
        If true, evaluate both formulas eagerly and select their values. The
        model uses false and scalar ``lax.cond`` inside ``lax.map`` for safe AD.

    Returns
    -------
    jax.Array, shape (6,)
        Phase mass, momentum and total-energy flux density.

    Notes
    -----
    Alberto's full branch decomposes the flux with ``rho_mat=K@R_h``.
    At zero slip the two void eigenvectors coalesce; the shared-velocity
    branch instead uses the completeness identity ``P12=I-P3-P4-P5-P6``.
    Scalar conditionals avoid differentiating the singular inactive branch;
    vmapping the conditional can evaluate both branches and defeat this.

    An expansion sonic crossing activates the face-dependent Harten width.
    Away from crossings, a small robustness band remains; outside that band
    the sign is exact. Low-level defaults are retained for compatibility;
    ``TwoFluidModel`` explicitly passes ``CompatibilityConfig`` values.
    """
    # The overbar denotes the face-average state; pressure remains in Pa.
    # ── Average state at the face ────────────────────────────────────────────
    U_h_bar = 0.5 * (U_h_L + U_h_R)
    p_b, u_g_b, u_l_b, ag_b, h_g_b, h_l_b = U_h_bar

    eos_g_b = phase_props(fluid_g, p_b, h_g_b)
    eos_l_b = phase_props(fluid_l, p_b, h_l_b)
    cq_b = common_quantities(U_h_bar, eos_g_b, eos_l_b, k)

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
        U_h_L, eos_g_L, eos_l_L, common_quantities(U_h_L, eos_g_L, eos_l_L, k)
    )[0]
    eigvals_R = eigenstructure(
        U_h_R, eos_g_R, eos_l_R, common_quantities(U_h_R, eos_g_R, eos_l_R, k)
    )[0]

    a_mix = 0.5 * (eigvals[2] - eigvals[3])  # mixture sound speed (>0)
    delta_global = delta_factor * jnp.abs(a_mix)  # tiny robustness floor ONLY

    sonic = (eigvals_L < 0.0) & (eigvals_R > 0.0)  # (6,) expansion sonic crossing
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

    du = cq_b.du  # u_g - u_l

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
    # Acoustic rescaling alone does not guarantee suppression of inlet checkerboards.
    rg = eos_g_b.rho
    ag_ = eos_g_b.a
    rl = eos_l_b.rho
    al_ = eos_l_b.a
    al_b = 1.0 - ag_b
    cq = cq_b
    rg_hat = cq.rg_hat
    rl_hat = cq.rl_hat
    rh2 = cq.rh2
    rho_mix = cq.rho_mix
    rho_s = cq.rho_s
    a0sq = cq.a0sq
    du = cq.du

    # ── Extra abbreviations for eigenstructure (§13.1) ──────────────────────
    # k*rho_mix is not stored in cq; derive from rg_hat = rg + k*rho_mix.
    k_rho_mix = rg_hat - rg
    rho_hat_s = rho_s + k_rho_mix  # hat_rho_s = rho_s + k*rho_mix
    rh = jnp.sqrt(rh2)  # hat_rho = sqrt(hat_rho^2)

    a_tilde_sq = (rho_hat_s / rho_s) * (rg * rl / rh2) * a0sq
    delta_a_sq = ag_b * al_b * rg * rl * rg_hat * rl_hat / rh2**2 * du**2
    a_sq = a_tilde_sq - delta_a_sq
    a = jnp.sqrt(a_sq)  # real mixture sound speed

    hat_ag = ag_b * rg * rl_hat / rh2  # volume-fraction weight for u_g
    hat_al = al_b * rl * rg_hat / rh2  # volume-fraction weight for u_l
    u_mix = hat_ag * u_g_b + hat_al * u_l_b  # mixture velocity

    Ma_face = jnp.abs(u_mix) / jnp.abs(a)
    z_ac = jnp.clip(Ma_face, lowmach_diss, 1.0)

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
        s_12 = smoothed_sign(cq_b.u1, delta_global)
        F_12b = 0.5 * (1.0 + s_12) * F_L + 0.5 * (1.0 - s_12) * F_R

        # One K-solve for the flux difference (known null modes handle the §4.3 u_i→0 case);
        # equilibrated: kappa(K) is also ~1e13 from unit scaling alone.
        y_dF = compatible_flux_solve(K_mat, F_L - F_R, U_h_bar)

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
        # Decompose the primitive representative of the flux difference.
        # For nonsingular K this is algebraically identical to solving K R_h
        # for both endpoint fluxes. It also permits the known phase-rest
        # entropy null modes: K r_entropy=0, so the resulting flux is unique.
        y = compatible_flux_solve(K_mat, F_L - F_R, U_h_bar)
        amplitudes = (L_h @ y) / D
        z_vec = jnp.ones(6).at[2].set(z_ac).at[3].set(z_ac)
        return 0.5 * (F_L + F_R) + 0.5 * (K_mat @ R_h) @ (s * z_vec * amplitudes)

    # ── Smooth blend between the zero-slip and full branches ──────────────────
    # The full branch inverts rho_mat = K@R_h, whose two void-wave columns c1,c2
    # become PARALLEL as du→0 (κ(rho_mat) ~ 1/du): rank-deficient (NaN) exactly at
    # du=0, and NOISY in a whole RING |du|≲du_blend_hi around every slip sign-change.
    # The old hard switch |du|<1e-4 protected only the exact singularity, leaving that
    # noisy ring on the full branch → a 2Δx checkerboard localized at du zero-crossings
    # (clean between crossings).  No amount of background dissipation fixes it because
    # the pump is the flux itself there.
    #
    # Retain the shared-velocity projector below du_blend_lo and the full
    # decomposition above du_blend_hi, with a C1 cosine blend between them.
    # Under scalar execution, the outer cond excludes the coalescing full
    # decomposition at du=0. Finite slip alone does not guarantee conditioning
    # of every matrix. The shared branch has the correct zero-slip limit;
    # at finite slip it approximates both void waves by the mixture speed u1.
    # Changing the blend thresholds is a separate numerical-policy choice.
    def _blend(_):
        Fz = _zero_slip(None)
        Ff = _full_slip(None)  # scalar execution reaches this only at finite slip
        t = jnp.clip(
            (jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo), 0.0, 1.0
        )
        w = 0.5 * (
            1.0 + jnp.cos(jnp.pi * t)
        )  # 1 at du_blend_lo → 0 at du_blend_hi (C¹)
        return w * Fz + (1.0 - w) * Ff

    if single_branch:
        # Retained eager alternative: evaluate each branch once and select.
        # A vmapped conditional also evaluates both branches. Neither form
        # guarantees safe derivatives through a singular inactive branch.
        # TwoFluidModel therefore uses single_branch=False with scalar lax.map.
        Fz = _zero_slip(None)
        Ff = _full_slip(None)
        t = jnp.clip(
            (jnp.abs(du) - du_blend_lo) / (du_blend_hi - du_blend_lo), 0.0, 1.0
        )
        w = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        return jnp.where(jnp.abs(du) < du_blend_lo, Fz, w * Fz + (1.0 - w) * Ff)

    F_hat = jax.lax.cond(jnp.abs(du) < du_blend_lo, _zero_slip, _blend, None)

    return F_hat


# ---------------------------------------------------------------------------
# Non-conservative H^nc correction
# ---------------------------------------------------------------------------


def nonconservative_correction(
    U_h_cell, dF, eos_g: PhaseProps, eos_l: PhaseProps, cq: CommonQty
) -> jnp.ndarray:
    """Apply Alberto's cell correction ``X @ solve(K, dF)``.

    Parameters
    ----------
    U_h_cell : array_like, shape (6,)
        Enthalpy primitives ``(p, u_g, u_l, alpha_g, h_g, h_l)`` in SI units.
    dF : jax.Array, shape (6,)
        Selected flux difference, with geometry measures applied by the model.
    eos_g, eos_l : PhaseProps
        Gas and liquid properties evaluated at ``U_h_cell``.
    cq : CommonQty
        Shared mixture abbreviations from :func:`common_quantities`.

    Returns
    -------
    jax.Array, shape (6,)
        Nonconservative action. The generic residual adds this correction.

    Notes
    -----
    The solve uses the range-compatible phase-rest treatment of
    :func:`compatible_flux_solve`.
    """
    K_mat = K_matrix(U_h_cell, eos_g, eos_l)
    X_mat = X_matrix(U_h_cell, eos_g, eos_l, cq)
    # Equilibrated (kappa(K)~1e13 is unit scaling; see _equil_solve note above).
    y = compatible_flux_solve(K_mat, dF, U_h_cell)
    return X_mat @ y
