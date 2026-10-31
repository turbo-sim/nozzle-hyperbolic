"""JAX Euler fluxes, including the preserved legacy Städtke implementation."""
import jax.numpy as jnp
import jaxprop as jxp
from ...numerics.characteristic import (
    harten_absolute as _jax_harten_absolute,
    characteristic_flux_with_entropy_viscosity as _jax_characteristic_flux_with_entropy_viscosity,
)


def _jax_euler_flux(Q, p):
    u = Q[1] / Q[0]
    return jnp.stack([Q[1], Q[1] * u + p, (Q[2] + p) * u])


def _jax_F_rusanov(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R):
    u_L = Q_L[1] / Q_L[0];  u_R = Q_R[1] / Q_R[0]
    F_L = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    a      = jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R)
    A_face = 0.5 * (A_L + A_R)
    return 0.5 * (F_L + F_R - a * (Q_R - Q_L)) * A_face


def _jax_F_hll(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R):
    u_L = Q_L[1] / Q_L[0];  u_R = Q_R[1] / Q_R[0]
    S_L = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R = jnp.maximum(u_L + c_L, u_R + c_R)
    F_L   = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (Q_R - Q_L)) / jnp.maximum(S_R - S_L, 1e-300)
    A_face = 0.5 * (A_L + A_R)
    return jnp.where(S_L >= 0.0, F_L, jnp.where(S_R <= 0.0, F_R, F_hll)) * A_face


def _jax_F_lf(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R, dx, dt):
    F_L = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    A_face = 0.5 * (A_L + A_R)
    return 0.5 * (F_L + F_R - (dx / dt) * (Q_R - Q_L)) * A_face


def _jax_F_hllc(Q_L, Q_R, p_L, c_L, p_R, c_R, A_L, A_R, fluid):
    rho_L = Q_L[0];  rho_R = Q_R[0]
    u_L   = Q_L[1] / rho_L;  u_R = Q_R[1] / rho_R
    E_L   = Q_L[2] / rho_L;  E_R = Q_R[2] / rho_R   # specific total energy
    S_L   = jnp.minimum(u_L - c_L, u_R - c_R)
    S_R   = jnp.maximum(u_L + c_L, u_R + c_R)
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    S_M   = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / jnp.where(jnp.abs(denom) > 1e-300, denom, 1e-300)
    dL    = jnp.where(jnp.abs(S_L - S_M) > 1e-300, S_L - S_M, 1e-300)
    dR    = jnp.where(jnp.abs(S_R - S_M) > 1e-300, S_R - S_M, 1e-300)
    rho_Ls = rho_L * (S_L - u_L) / dL
    rho_Rs = rho_R * (S_R - u_R) / dR
    # Star total energy via Toro (2009) eq. 10.73 — no EOS call required.
    # rhoE* = rho* * (E + (S* - u) * (S* + p / (rho*(S - u))))
    rhoE_Ls = rho_Ls * (E_L + (S_M - u_L) * (S_M + p_L / jnp.where(jnp.abs(rho_L * (S_L - u_L)) > 1e-300, rho_L * (S_L - u_L), 1e-300)))
    rhoE_Rs = rho_Rs * (E_R + (S_M - u_R) * (S_M + p_R / jnp.where(jnp.abs(rho_R * (S_R - u_R)) > 1e-300, rho_R * (S_R - u_R), 1e-300)))
    Q_Ls   = jnp.stack([rho_Ls, rho_Ls * S_M, rhoE_Ls])
    Q_Rs   = jnp.stack([rho_Rs, rho_Rs * S_M, rhoE_Rs])
    F_L    = _jax_euler_flux(Q_L, p_L);  F_R = _jax_euler_flux(Q_R, p_R)
    A_face = 0.5 * (A_L + A_R)
    FA_L   = F_L * A_face;  FA_R = F_R * A_face
    FA_Ls  = FA_L + S_L * (Q_Ls - Q_L) * A_face
    FA_Rs  = FA_R + S_R * (Q_Rs - Q_R) * A_face
    return jnp.where(S_L > 0.0, FA_L, jnp.where(S_M >= 0.0, FA_Ls, jnp.where(S_R >= 0.0, FA_Rs, FA_R)))


# ---------------------------------------------------------------------------
# Städtke FVS flux  (JAX, operates in (p, u, h) primitive space)
# ---------------------------------------------------------------------------

def _jax_F_fvs_stadtke_legacy(U_L, U_R, fluid, delta_factor=0.1):
    """
    Städtke flux-vector-splitting interface flux for all faces simultaneously.

    Primitive states U_L, U_R : (Nfaces, 3)  — columns are (p, u, h).
    Returns conservative flux  F̂ : (Nfaces, 3)  — (ρu, ρu²+p, uρH).

    Algorithm (theory doc §5):
      1. Average state → EOS → ρ, a, G → rho_p, rho_h.
      2. Build split matrices G_k = λ_k v_k ⊗ l_k  (each (Nfaces,3,3)).
      3. Build flux Jacobian K = ∂F/∂U  (Nfaces,3,3).
      4. Projection: R̃_k = K G_k K⁻¹ / λ_k^safe  via batched solve.
      5. Upwind sum:  R̃⁺ = Σ_{λ_k≥0} R̃_k,  R̃⁻ = Σ_{λ_k<0} R̃_k.
      6. F̂ = R̃⁺ F_L + R̃⁻ F_R  (physical fluxes from actual L/R states).
    """
    # ── Averaged state ────────────────────────────────────────────────────
    U_avg  = 0.5 * (U_L + U_R)
    p_a    = U_avg[:, 0];  u_a = U_avg[:, 1];  h_a = U_avg[:, 2]

    st_a   = fluid.get_state(jxp.HmassP_INPUTS, h_a, p_a)
    rho_a  = st_a["rho"]
    a_a    = st_a["a"]
    G_a    = st_a["G"]
    rho_p  = (1.0 + G_a) / a_a**2        # (∂ρ/∂p)_h  — theory §1.3
    rho_h  = -(rho_a * G_a) / a_a**2     # (∂ρ/∂h)_p

    # ── Eigenvalues ───────────────────────────────────────────────────────
    lam1 = u_a + a_a;  lam2 = u_a - a_a;  lam3 = u_a

    # ── Eigenvectors  (Nfaces, 3) ────────────────────────────────────────
    # Right:  v1=(ρa, 1, a),      v2=(-ρa, 1, -a),     v3=(0, 0, 1)
    # Left:   l1=(1/2ρa, 1/2, 0), l2=(-1/2ρa, 1/2, 0), l3=(-1/ρ, 0, 1)
    z = jnp.zeros_like(u_a);  o = jnp.ones_like(u_a)

    v1 = jnp.stack([ rho_a * a_a,  o,  a_a], axis=-1)
    v2 = jnp.stack([-rho_a * a_a,  o, -a_a], axis=-1)
    v3 = jnp.stack([ z,             z,  o  ], axis=-1)
    l1 = jnp.stack([ 0.5 / (rho_a * a_a),  0.5 * o,  z], axis=-1)
    l2 = jnp.stack([-0.5 / (rho_a * a_a),  0.5 * o,  z], axis=-1)
    l3 = jnp.stack([-1.0 / rho_a,           z,        o], axis=-1)

    # ── Flux Jacobian  K = ∂F/∂U  (Nfaces,3,3) ───────────────────────────
    H_a  = h_a + 0.5 * u_a**2
    r0   = jnp.stack([u_a * rho_p,            rho_a,               u_a * rho_h           ], axis=-1)
    r1   = jnp.stack([u_a**2 * rho_p + 1.0,   2.0 * rho_a * u_a,  u_a**2 * rho_h        ], axis=-1)
    r2   = jnp.stack([u_a * rho_p * H_a,       rho_a * (H_a + u_a**2), u_a * (rho_h * H_a + rho_a)], axis=-1)
    K    = jnp.stack([r0, r1, r2], axis=1)   # (Nfaces, 3, 3)
    KT   = K.swapaxes(-1, -2)                # (Nfaces, 3, 3)

    # ── L/R eigenvalues for strict sign-change entropy fix ───────────────
    st_L  = fluid.get_state(jxp.HmassP_INPUTS, U_L[:, 2], U_L[:, 0])
    st_R  = fluid.get_state(jxp.HmassP_INPUTS, U_R[:, 2], U_R[:, 0])
    a_L   = st_L["a"];  a_R = st_R["a"]
    lam1_L = U_L[:, 1] + a_L;  lam1_R = U_R[:, 1] + a_R
    lam2_L = U_L[:, 1] - a_L;  lam2_R = U_R[:, 1] - a_R
    lam3_L = U_L[:, 1];         lam3_R = U_R[:, 1]

    # ── Projection  R̃_k = K G_k K⁻¹ / λ_k^safe  (rank-1 form) ──────────
    # Strict Roe-Harten entropy fix (Roe 1981, Harten & Hyman 1983):
    #   Activate only when λ_k changes sign L→R (expansion sonic transition).
    #   At active faces: δ_k = λ_k^R - λ_k^L  (>0 by construction),
    #                    λ_k^safe = (λ_k^avg² + δ_k²) / (2δ_k)  [C¹ Harten form]
    #   Elsewhere: λ_k^safe = |λ_k^avg|  (coeff = 1, no added dissipation).
    # This means dissipation is added ONLY at the single face where a wave
    # crosses zero from L to R — never at shocks or in smooth flow regions.
    # delta_factor provides an optional global C¹ floor for robustness.
    delta_global = delta_factor * (jnp.abs(u_a) + a_a)   # global floor (Nfaces,)

    def _Rt_vec(v, l, lam_avg, lam_L, lam_R):
        Kv    = jnp.einsum('...ij,...j->...i', K, v)
        KTl   = jnp.linalg.solve(KT, l[..., None]).squeeze(-1)
        # Strict activation: expansion sonic crossing only
        sonic   = (lam_L < 0.0) & (lam_R > 0.0)
        delta_k = jnp.where(sonic, lam_R - lam_L, delta_global)
        lam_safe = jnp.where(sonic,
                             (lam_avg**2 + delta_k**2) / (2.0 * jnp.maximum(delta_k, 1e-300)),
                             jnp.maximum(jnp.abs(lam_avg), delta_global))
        coeff = jnp.abs(lam_avg) / jnp.maximum(lam_safe, 1e-300)
        return coeff[:, None, None] * jnp.einsum('...i,...j->...ij', Kv, KTl)

    Rt1 = _Rt_vec(v1, l1, lam1, lam1_L, lam1_R)
    Rt2 = _Rt_vec(v2, l2, lam2, lam2_L, lam2_R)
    Rt3 = _Rt_vec(v3, l3, lam3, lam3_L, lam3_R)

    # ── Sum by TRUE eigenvalue sign ───────────────────────────────────────
    s1p = jnp.where(lam1[:, None, None] >= 0.0, 1.0, 0.0)
    s2p = jnp.where(lam2[:, None, None] >= 0.0, 1.0, 0.0)
    s3p = jnp.where(lam3[:, None, None] >= 0.0, 1.0, 0.0)

    R_plus  = s1p * Rt1 + s2p * Rt2 + s3p * Rt3
    R_minus = (1.0 - s1p) * Rt1 + (1.0 - s2p) * Rt2 + (1.0 - s3p) * Rt3

    # ── Physical fluxes from ACTUAL L/R states ────────────────────────────
    rho_L = st_L["rho"];  rho_R = st_R["rho"]
    H_L   = U_L[:, 2] + 0.5 * U_L[:, 1]**2
    H_R   = U_R[:, 2] + 0.5 * U_R[:, 1]**2
    F_L   = jnp.stack([rho_L * U_L[:, 1],
                        rho_L * U_L[:, 1]**2 + U_L[:, 0],
                        U_L[:, 1] * rho_L * H_L], axis=-1)
    F_R   = jnp.stack([rho_R * U_R[:, 1],
                        rho_R * U_R[:, 1]**2 + U_R[:, 0],
                        U_R[:, 1] * rho_R * H_R], axis=-1)

    # F̂ = R̃⁺ F_L + R̃⁻ F_R  (batched matvec)
    return (jnp.einsum('...ij,...j->...i', R_plus,  F_L)
          + jnp.einsum('...ij,...j->...i', R_minus, F_R))


def _jax_stadtke_conservative_projectors(U, rho, a, gruneisen):
    """Return conservative Euler projectors without inverting the flux Jacobian.

    For primitive W=(p,u,h), C=dQ/dW has determinant rho**2/a**2.
    The thermodynamic identities rho_p=(1+G)/a**2 and
    rho_h=-rho*G/a**2 give the explicit inverse below. With A the
    primitive Euler matrix and P_i=v_i*l_i, K=dF/dW=C*A, so
    K*P_i*K^-1=C*P_i*C^-1 wherever K is invertible. The C representation
    also remains regular at u=0 and u=+/-a, where K is singular.

    Returns shape (..., 3 waves, 3 balances, 3 conservative variables).
    Admissible EOS states require finite rho>0 and a>0.
    """
    u, h = U[..., 1], U[..., 2]
    rho_p = (1.0 + gruneisen) / a**2
    rho_h = -rho * gruneisen / a**2
    total_h = h + 0.5 * u**2
    zero = jnp.zeros_like(u)
    one = jnp.ones_like(u)
    storage = jnp.stack(
        [
            jnp.stack([rho_p, zero, rho_h], axis=-1),
            jnp.stack([u * rho_p, rho, u * rho_h], axis=-1),
            jnp.stack(
                [total_h * rho_p - one, rho * u, total_h * rho_h + rho],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    chi = 0.5 * u**2 - h
    inverse_storage = jnp.stack(
        [
            jnp.stack(
                [a**2 + gruneisen * chi, -gruneisen * u, gruneisen],
                axis=-1,
            ),
            jnp.stack([-u / rho, one / rho, zero], axis=-1),
            jnp.stack(
                [
                    (a**2 + (gruneisen + one) * chi) / rho,
                    -(gruneisen + one) * u / rho,
                    (gruneisen + one) / rho,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    right = jnp.stack(
        [
            jnp.stack([rho * a, one, a], axis=-1),
            jnp.stack([-rho * a, one, -a], axis=-1),
            jnp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )
    left = jnp.stack(
        [
            jnp.stack([0.5 / (rho * a), 0.5 * one, zero], axis=-1),
            jnp.stack([-0.5 / (rho * a), 0.5 * one, zero], axis=-1),
            jnp.stack([-one / rho, zero, one], axis=-1),
        ],
        axis=-2,
    )
    conservative_right = jnp.einsum("...ij,...kj->...ki", storage, right)
    conservative_left = jnp.einsum("...ki,...ij->...kj", left, inverse_storage)
    return conservative_right[..., :, None] * conservative_left[..., None, :]


def _jax_F_fvs_stadtke(U_L, U_R, fluid, delta_factor=0.1):
    """Regularized characteristic FVS flux with conservative entropy viscosity.

    Primitive face states have columns (p,u,h); output balances are mass,
    momentum, and total energy. The original implementation is retained
    verbatim as _jax_F_fvs_stadtke_legacy for baseline reproduction.

    Conservative projectors C*P_i*C^-1 remove the original sonic/rest flux-
    Jacobian singularity. Smooth physical-flux weights sum to one, preserving
    Fhat(U,U)=F(U) for a general EOS, including states in the sonic band.
    A conservative-state jump adds the viscosity missing from smoothing
    those weights alone. This deliberately changes the discretization inside
    the regularization band; outside it the flux is the original intended
    characteristic FVS, with exact supersonic upwinding when every averaged
    eigenvalue is beyond its positive or negative band.

    The entropy width is delta0 + 0.5*positive_smooth(lambda_R-lambda_L),
    where delta0=delta_factor*a_average and positive_smooth(d) is
    0.5*(d+sqrt(d**2+delta0**2)). There is no strict sonic activation gate.
    For delta_factor>0 and a smooth admissible EOS, flux regularity is C1 at
    rest, sonic states, and the Harten band edges. The caller must supply a
    strictly positive delta_factor; no tracer-dependent Python check is used.

    Arithmetic averaging is not a general-EOS Roe linearization. Although
    frozen linear characteristic viscosity equals the Harten absolute value,
    this construction does not establish a full nonlinear entropy proof.
    """
    average = 0.5 * (U_L + U_R)
    state_average = fluid.get_state(jxp.HmassP_INPUTS, average[:, 2], average[:, 0])
    state_left = fluid.get_state(jxp.HmassP_INPUTS, U_L[:, 2], U_L[:, 0])
    state_right = fluid.get_state(jxp.HmassP_INPUTS, U_R[:, 2], U_R[:, 0])
    rho = state_average["rho"]
    a = state_average["a"]
    projectors = _jax_stadtke_conservative_projectors(
        average, rho, a, state_average["G"]
    )

    def conservative_and_flux(primitive, density):
        p, u, h = primitive[:, 0], primitive[:, 1], primitive[:, 2]
        total_h = h + 0.5 * u**2
        conservative = jnp.stack([density, density * u, density * total_h - p], axis=-1)
        flux = jnp.stack(
            [density * u, density * u**2 + p, density * u * total_h], axis=-1
        )
        return conservative, flux

    conservative_left, flux_left = conservative_and_flux(U_L, state_left["rho"])
    conservative_right, flux_right = conservative_and_flux(U_R, state_right["rho"])

    def characteristics(primitive, sound_speed):
        u = primitive[:, 1]
        return jnp.stack([u + sound_speed, u - sound_speed, u], axis=-1)

    eigenvalues = characteristics(average, a)
    spread = characteristics(U_R, state_right["a"]) - characteristics(
        U_L, state_left["a"]
    )
    delta0 = delta_factor * a[:, None]
    positive_spread = 0.5 * (spread + jnp.sqrt(spread**2 + delta0**2))
    width = delta0 + 0.5 * positive_spread
    return _jax_characteristic_flux_with_entropy_viscosity(
        flux_left,
        flux_right,
        conservative_right - conservative_left,
        projectors,
        eigenvalues,
        width,
    )
