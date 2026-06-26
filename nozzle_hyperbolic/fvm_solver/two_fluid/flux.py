"""
FVS flux kernel and non-conservative correction for Städtke's two-fluid model.

Implements §13.6 of stadtke_enthalpy_frame_derivation_complete.md:
  - Per-face FVS interface flux via rank-one projectors in flux space
  - Non-conservative H^nc correction via X @ solve(K, dF)

Entropy fix: C^1-smooth sign function with global Harten floor
  eps = delta_factor * |a_mix|  applied uniformly to all 6 eigenvalues.
  (Strict per-wave L/R activation, as in the single-phase solver, can be
  added later once solver robustness is confirmed.)
"""

import jax.numpy as jnp
from .matrices import (
    PhaseProps, phase_props, CommonQty, common_quantities,
    K_matrix, X_matrix, eigenstructure, flux_vector,
)


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

# TODO: Is the flux calculation correct? 

def fvs_interface_flux(U_h_L, U_h_R, fluid_g, fluid_l, k,
                       delta_factor=0.1):
    """
    FVS interface flux F_hat at one face of the two-fluid system.

    Implements Städtke's flux-vector-splitting in enthalpy-primitive variables
    (derivation §13.6).  The per-face algorithm:

      1. Average state U_h_bar  →  EOS  →  eigenstructure (R_h, L_h, D, λ)
                                 →  K = ∂F/∂U_h
      2. Flux-space right vectors:  rho_mat = K @ R_h
      3. ONE solve  beta = solve(rho_mat, F) for both L and R simultaneously.
         The rows of rho_mat^{-1} are the normalised flux-space left vectors.
      4. Harten-smooth eigenvalue sign  s_k  (eps = delta_factor * |a_mix|).
      5. Interface flux:
           beta = 0.5*(1+s)*beta_L + 0.5*(1-s)*beta_R
           F_hat = rho_mat @ beta

    Physical fluxes F_L, F_R are evaluated at the actual L/R states.

    Parameters
    ----------
    U_h_L, U_h_R : (6,) — left/right primitive states [p,u_g,u_l,ag,h_g,h_l]
    fluid_g, fluid_l : FluidBicubic tables (metastable vapor / liquid)
    k            : virtual-mass coefficient
    delta_factor : Harten floor fraction — eps = delta_factor * |a_mix|

    Returns
    -------
    F_hat : (6,) — interface flux (conservative ordering)
    """
    # ── Average state at the face ────────────────────────────────────────────
    U_h_bar = 0.5 * (U_h_L + U_h_R)
    p_b, _, _, _, h_g_b, h_l_b = U_h_bar

    eos_g_b = phase_props(fluid_g, p_b, h_g_b)
    eos_l_b = phase_props(fluid_l, p_b, h_l_b)
    cq_b    = common_quantities(U_h_bar, eos_g_b, eos_l_b, k)

    # ── Eigenstructure and K at average state ────────────────────────────────
    eigvals, R_h, L_h, D = eigenstructure(U_h_bar, eos_g_b, eos_l_b, cq_b)
    K_mat = K_matrix(U_h_bar, eos_g_b, eos_l_b)

    # ── Physical fluxes at actual L/R states ─────────────────────────────────
    eos_g_L = phase_props(fluid_g, U_h_L[0], U_h_L[4])
    eos_l_L = phase_props(fluid_l, U_h_L[0], U_h_L[5])
    F_L = flux_vector(U_h_L, eos_g_L, eos_l_L)

    eos_g_R = phase_props(fluid_g, U_h_R[0], U_h_R[4])
    eos_l_R = phase_props(fluid_l, U_h_R[0], U_h_R[5])
    F_R = flux_vector(U_h_R, eos_g_R, eos_l_R)

    # ── Flux-space projectors via LU on rho_mat = K @ R_h ────────────────────
    rho_mat = K_mat @ R_h   # (6,6) — columns are flux-space right eigenvectors

    # Solve for normalised wave amplitudes at L and R simultaneously (one LU)
    #   beta_L = rho_mat^{-1} F_L,  beta_R = rho_mat^{-1} F_R
    # Note: rho_mat is singular when u_g=0 or u_l=0 (K singularity, §12.4);
    # at those states F lies in range(K@R_h) by flux structure, so the system
    # is consistent in exact arithmetic — floating-point regularisation via
    # the Harten floor is sufficient in practice.
    beta_LR = jnp.linalg.solve(
        rho_mat, jnp.stack([F_L, F_R], axis=1)   # (6,2) RHS
    )                                              # returns (6,2)
    beta_L = beta_LR[:, 0]   # normalised wave amplitudes from left
    beta_R = beta_LR[:, 1]   # normalised wave amplitudes from right

    # ── Harten entropy fix ───────────────────────────────────────────────────
    # Mixture sound speed from acoustic eigenvalue pair (eigvals[2,3] = u_mix±a)
    a_mix = 0.5 * (eigvals[2] - eigvals[3])          # >0 for physical states
    eps   = delta_factor * jnp.abs(a_mix)             # global floor

    s       = smoothed_sign(eigvals, eps)             # (6,) ∈ [-1,1]
    w_plus  = 0.5 * (1.0 + s)                        # weight toward F_L
    w_minus = 0.5 * (1.0 - s)                        # weight toward F_R

    # ── Interface flux ───────────────────────────────────────────────────────
    beta  = w_plus * beta_L + w_minus * beta_R
    return rho_mat @ beta


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
    y = jnp.linalg.solve(K_mat, dF)
    return X_mat @ y

    # TODO: There should be the calculation of the invers matrix also in another  point. Recallit and check if the impelemnation is correct
