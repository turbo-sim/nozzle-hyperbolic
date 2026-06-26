"""
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
import jaxprop as jxp

from .matrices import (
    phase_props, common_quantities,
    K_matrix, X_matrix, eigenstructure,
)
from .flux import fvs_interface_flux, nonconservative_correction
from .sources import TwoFluidClosure, D_VE_conservative


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
    delta_factor=0.1,
    newton_iter=10,
    recon='muscl_van_leer',
    sources_fn=None,
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
    recon   : 'muscl_van_leer' or 'constant'
    sources_fn : optional callable (U_h, eos_g, eos_l, cq, dlnA) → D_VE (6,)

    Returns
    -------
    step       : jit-compiled  step(W_n, dt) → (W_new, ssr)
    compute_dt : jit-compiled  compute_dt(W_n) → dt

    Initial condition: W_n = W_from_U_h_batch(U_h_0, fluid_g, fluid_l)
    Post-processing:   U_h = U_h_from_W_batch(W_new)
    """
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

    # ── BC (transmissive / zero-gradient in W space) ──────────────────────────

    def _bc_W(W):
        return W.at[0].set(W[1]).at[-1].set(W[-2])

    # ── MUSCL van Leer in U_h space ───────────────────────────────────────────

    def _muscl(U_h):
        dU = U_h[1:] - U_h[:-1]
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

    def _fvs_all_faces(U_h):
        if recon != 'constant':
            U_L, U_R = _muscl(U_h)
        else:
            U_L, U_R = U_h[:-1], U_h[1:]
        return jax.vmap(
            lambda ul, ur: fvs_interface_flux(ul, ur, fluid_g, fluid_l, k, delta_factor)
        )(U_L, U_R)

    # ── Non-conservative correction at interior cells ─────────────────────────

    def _nc_all_interior(U_h, F_hat):
        dF = F_hat[1:] - F_hat[:-1]

        def _nc_one(u_h, df):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            return nonconservative_correction(u_h, df, eos_g_c, eos_l_c, cq_c)

        return jax.vmap(_nc_one)(U_h[1:-1], dF)

    # ── Source terms at interior cells ────────────────────────────────────────

    def _src_all_interior(U_h):
        def _src_one(u_h, dlnA):
            eos_g_c = phase_props(fluid_g, u_h[0], u_h[4])
            eos_l_c = phase_props(fluid_l, u_h[0], u_h[5])
            cq_c    = common_quantities(u_h, eos_g_c, eos_l_c, k)
            if sources_fn is not None:
                return sources_fn(u_h, eos_g_c, eos_l_c, cq_c, dlnA)
            z  = jnp.zeros(())
            cl = TwoFluidClosure(
                sigma_Mg=z, sigma_Qg=z, Fv_g=z,
                F_g=z, F_l=z, Q_g=z, Q_l=z,
                u_ex=z, u_int=z, h_ex=z, dlnA_dx=dlnA,
            )
            return D_VE_conservative(u_h, eos_g_c, eos_l_c, cq_c, cl, k)

        return jax.vmap(_src_one)(U_h[1:-1], dlnA_dx[1:-1])

    # ── Residual:  R ∈ R^{6·(N-2)},  W_int ∈ R^{8·(N-2)} ───────────────────

    def _residual(W_int_flat, W_n_flat, dt):
        W_int = W_int_flat.reshape(N - 2, 8)
        W_n   = W_n_flat.reshape(N, 8)

        W = jnp.concatenate([W_n[:1], W_int, W_n[-1:]], axis=0)
        W = _bc_W(W)

        # Conservative variables — POLYNOMIAL in W, no EOS chain in dV/dW
        V_new = jax.vmap(V_from_W)(W)
        V_old = jax.vmap(V_from_W)(W_n)

        # Primitives U_h — ALGEBRAIC from W, no EOS
        U_h = jax.vmap(U_h_from_W)(W)

        # FVS fluxes (EOS called internally by fvs_interface_flux)
        F_hat = _fvs_all_faces(U_h)

        # Non-conservative correction and sources
        nc  = _nc_all_interior(U_h, F_hat)
        src = _src_all_interior(U_h)

        Vc      = V_cell[1:-1, None]
        dV      = V_new[1:-1] - V_old[1:-1]
        fdiv    = Af_j[1:, None] * F_hat[1:] - Af_j[:-1, None] * F_hat[:-1]
        nc_term = A_j[1:-1, None] * nc

        R = Vc * dV + dt * fdiv + dt * nc_term - dt * Vc * src
        return R.flatten()   # (6*(N-2),)

    # Jacobian: (6*(N-2)) × (8*(N-2)) — computed via forward-mode AD
    _jac_fn = jax.jacobian(_residual, argnums=0)

    # ── CFL time step ─────────────────────────────────────────────────────────

    @jax.jit
    def compute_dt(W_n):
        U_h = jax.vmap(U_h_from_W)(W_n)   # algebraic, no EOS
        p   = U_h[:, 0];  h_g = U_h[:, 4];  h_l = U_h[:, 5]
        stg = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
        stl = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p)
        a_g = stg["a"];  a_l = stl["a"]
        lam_max = jnp.maximum(jnp.abs(U_h[:, 1]) + a_g, jnp.abs(U_h[:, 2]) + a_l)
        return CFL * dx / jnp.max(lam_max[1:-1])

    # ── Implicit Newton step ──────────────────────────────────────────────────

    @jax.jit
    def step(W_n, dt):
        """
        Advance one implicit time step.

        Parameters
        ----------
        W_n : (N, 8) — W state at time n
        dt  : scalar

        Returns
        -------
        W_new : (N, 8)
        ssr   : scalar — normalised steady-state residual
        """
        W_n_flat = W_n.reshape(-1)          # (8*N,)
        W_k      = W_n[1:-1].reshape(-1)    # (8*(N-2),) initial Newton guess

        def _newton_body(_, W_k):
            R_k = _residual(W_k, W_n_flat, dt)        # (6*(N-2),)
            J_k = _jac_fn(W_k, W_n_flat, dt)          # (6*(N-2), 8*(N-2))
            # Minimum-norm solution of the rectangular J_k dW = -R_k
            dW  = jnp.linalg.lstsq(J_k, -R_k, rcond=None)[0]
            return W_k + dW

        W_k = jax.lax.fori_loop(0, newton_iter, _newton_body, W_k)

        W_new = jnp.concatenate([W_n[:1], W_k.reshape(N - 2, 8), W_n[-1:]], axis=0)
        W_new = _bc_W(W_new)

        # Steady-state residual (dV = 0 since same W passed for new and old)
        R_ss = _residual(
            W_new[1:-1].reshape(-1), W_new.reshape(-1), jnp.float64(1.0)
        ).reshape(N - 2, 6)

        U_h_new = jax.vmap(U_h_from_W)(W_new)
        p   = U_h_new[:, 0];  h_g = U_h_new[:, 4];  h_l = U_h_new[:, 5]
        stg = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
        stl = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p)
        a_g = stg["a"];  a_l = stl["a"]
        lam_max = jnp.maximum(jnp.abs(U_h_new[:, 1]) + a_g,
                               jnp.abs(U_h_new[:, 2]) + a_l)
        t_char  = dx / (jnp.max(lam_max) + 1e-300)
        V_new   = jax.vmap(V_from_W)(W_new)
        ssr     = jnp.sqrt(jnp.mean(
            (t_char * R_ss
             / (V_cell[1:-1, None] * jnp.abs(V_new[1:-1]) + 1e-300)) ** 2
        ))

        return W_new, ssr

    return step, compute_dt
