"""Deprecated historical two-fluid implementation.

.. deprecated:: 0.0.1
    Use ``models.two_fluid.constrained_state`` for the current eight-state model.
    Retained only for historical comparisons and remaining legacy callers.
    This module is a deletion candidate after those callers/tests migrate.
    The current model is not a drop-in replacement for every legacy API.

State-vector converters for the two-fluid solver.

Primitive state (enthalpy frame):
    U_h = (p, u_g, u_l, alpha_g, h_g, h_l)   — per cell, shape (6,)

Conservative state (total-energy frame):
    V = (alpha_g*rho_g,       alpha_l*rho_l,
         alpha_g*rho_g*u_g,   alpha_l*rho_l*u_l,
         alpha_g*rho_g*E_g,   alpha_l*rho_l*E_l)  — shape (6,)

where  E_i = h_i - p/rho_i + u_i^2/2  (total specific energy, Städtke eq. 5.90)

Grid-level arrays use shape (N, 6); single-cell functions operate on (6,) and
are vmapped over the grid in the solver.
"""

import jax
import jax.numpy as jnp
import jaxprop as jxp


# ---------------------------------------------------------------------------
# Forward: primitives → conservative
# ---------------------------------------------------------------------------

def U_h_to_V(U_h, fluid_g, fluid_l):
    """
    Primitive (6,) → conservative (6,).  Direct algebraic, no iteration.

    Parameters
    ----------
    U_h      : shape (6,) — [p, u_g, u_l, alpha_g, h_g, h_l]
    fluid_g  : FluidBicubic for the gas/vapour phase
    fluid_l  : FluidBicubic for the liquid phase

    Returns
    -------
    V : shape (6,) — [alpha_g*rho_g, alpha_l*rho_l, ..., alpha_g*rho_g*E_g, alpha_l*rho_l*E_l]
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    p_arr  = jnp.expand_dims(p, 0)
    st_g   = fluid_g.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_g, 0), p_arr)
    st_l   = fluid_l.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_l, 0), p_arr)
    rho_g  = st_g["rho"][0]
    rho_l  = st_l["rho"][0]

    E_g = h_g - p / rho_g + 0.5 * u_g**2
    E_l = h_l - p / rho_l + 0.5 * u_l**2

    return jnp.array([
        ag * rho_g,
        al * rho_l,
        ag * rho_g * u_g,
        al * rho_l * u_l,
        ag * rho_g * E_g,
        al * rho_l * E_l,
    ])


def U_h_to_V_batch(U_h, fluid_g, fluid_l):
    """
    Primitive (N, 6) → conservative (N, 6) for the full grid.

    Uses batched EOS calls (same as the single-phase solver).
    """
    p   = U_h[:, 0];  u_g = U_h[:, 1];  u_l = U_h[:, 2]
    ag  = U_h[:, 3];  h_g = U_h[:, 4];  h_l = U_h[:, 5]
    al  = 1.0 - ag

    st_g  = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p)
    st_l  = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p)
    rho_g = st_g["rho"]
    rho_l = st_l["rho"]

    E_g = h_g - p / rho_g + 0.5 * u_g**2
    E_l = h_l - p / rho_l + 0.5 * u_l**2

    return jnp.stack([
        ag * rho_g,
        al * rho_l,
        ag * rho_g * u_g,
        al * rho_l * u_l,
        ag * rho_g * E_g,
        al * rho_l * E_l,
    ], axis=1)


# ---------------------------------------------------------------------------
# Inverse: conservative → primitives  (inner Newton on (p, h_g, h_l))
# ---------------------------------------------------------------------------

def V_to_U_h(V, fluid_g, fluid_l, p_lo, p_hi, hg_lo, hg_hi, hl_lo, hl_hi,
             newton_iter=8):
    """
    Conservative (6,) → primitive (6,) via inner Newton on (p, h_g, h_l).

    Given V = (m_g, m_l, mom_g, mom_l, E_g_tot, E_l_tot), the primitives are:
        u_g = mom_g / m_g,   u_l = mom_l / m_l
        alpha_g follows from  m_g / rho_g(p, h_g) + m_l / rho_l(p, h_l) = 1

    The three scalar unknowns (p, h_g, h_l) are found by Newton on:
        f1 = m_g / rho_g(p,h_g) + m_l / rho_l(p,h_l) - 1      (volume sum = 1)
        f2 = E_g_tot/m_g - h_g + p/rho_g(p,h_g) - u_g^2/2     (energy for gas)
        f3 = E_l_tot/m_l - h_l + p/rho_l(p,h_l) - u_l^2/2     (energy for liquid)

    The Jacobian is assembled analytically from the EOS leaves (rho_p, rho_h)
    so the Newton iterations are exact and cheap (no nested autodiff).

    Parameters
    ----------
    V           : shape (6,)
    fluid_g/l   : FluidBicubic tables
    p_lo/hi     : pressure bounds for clipping
    hg_lo/hi    : gas enthalpy bounds
    hl_lo/hi    : liquid enthalpy bounds
    newton_iter : number of Newton steps (unrolled under jit)
    """
    m_g, m_l, mom_g, mom_l, Etot_g, Etot_l = V

    u_g = mom_g / m_g
    u_l = mom_l / m_l
    ke_g = 0.5 * u_g**2
    ke_l = 0.5 * u_l**2

    # Internal energies per unit partial-volume mass
    e_g_spec = Etot_g / m_g - ke_g     # = h_g - p/rho_g  (= u_int,g)
    e_l_spec = Etot_l / m_l - ke_l

    # Initial guess: p from midpoint, h from e + p/rho using midpoint rho
    p   = jnp.array(0.5 * (p_lo + p_hi),   dtype=jnp.float64)
    h_g = jnp.array(0.5 * (hg_lo + hg_hi), dtype=jnp.float64)
    h_l = jnp.array(0.5 * (hl_lo + hl_hi), dtype=jnp.float64)

    def _step(_, carry):
        p_, h_g_, h_l_ = carry
        p_arr = jnp.expand_dims(p_,   0)
        hg_arr = jnp.expand_dims(h_g_, 0)
        hl_arr = jnp.expand_dims(h_l_, 0)

        stg = fluid_g.get_state(jxp.HmassP_INPUTS, hg_arr, p_arr)
        stl = fluid_l.get_state(jxp.HmassP_INPUTS, hl_arr, p_arr)
        rg   = stg["rho"][0];   ag   = stg["a"][0];   Gg   = stg["G"][0]
        rl   = stl["rho"][0];   al   = stl["a"][0];   Gl   = stl["G"][0]

        rg_p = (1.0 + Gg) / ag**2
        rg_h = -(rg * Gg) / ag**2
        rl_p = (1.0 + Gl) / al**2
        rl_h = -(rl * Gl) / al**2

        # Residuals
        f1 = m_g / rg + m_l / rl - 1.0
        f2 = e_g_spec - h_g_ + p_ / rg
        f3 = e_l_spec - h_l_ + p_ / rl

        # Analytic Jacobian  df/d(p, h_g, h_l)
        # df1/dp  = m_g*(-rg_p/rg^2) + m_l*(-rl_p/rl^2)
        # df1/dhg = m_g*(-rg_h/rg^2)
        # df1/dhl = m_l*(-rl_h/rl^2)
        J11 = -m_g * rg_p / rg**2 - m_l * rl_p / rl**2
        J12 = -m_g * rg_h / rg**2
        J13 = -m_l * rl_h / rl**2
        # df2/dp  = (rg - p*rg_p) / rg^2 = 1/rg - p*rg_p/rg^2
        # df2/dhg = -1 - p*rg_h/rg^2
        # df2/dhl = 0
        J21 = 1.0 / rg - p_ * rg_p / rg**2
        J22 = -1.0 - p_ * rg_h / rg**2
        J23 = jnp.zeros_like(p_)
        # df3 symmetric for liquid
        J31 = 1.0 / rl - p_ * rl_p / rl**2
        J32 = jnp.zeros_like(p_)
        J33 = -1.0 - p_ * rl_h / rl**2

        J  = jnp.array([[J11, J12, J13],
                         [J21, J22, J23],
                         [J31, J32, J33]])
        f  = jnp.array([f1, f2, f3])
        dp, dh_g, dh_l = jnp.linalg.solve(J, -f)

        p_new   = jnp.clip(p_   + dp,    p_lo,   p_hi)
        h_g_new = jnp.clip(h_g_ + dh_g,  hg_lo,  hg_hi)
        h_l_new = jnp.clip(h_l_ + dh_l,  hl_lo,  hl_hi)
        return p_new, h_g_new, h_l_new

    p, h_g, h_l = jax.lax.fori_loop(0, newton_iter, _step, (p, h_g, h_l))

    # Recover alpha_g from the converged (p, h_g, h_l)
    p_arr  = jnp.expand_dims(p,   0)
    stg    = fluid_g.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_g, 0), p_arr)
    stl    = fluid_l.get_state(jxp.HmassP_INPUTS, jnp.expand_dims(h_l, 0), p_arr)
    rho_g  = stg["rho"][0]
    rho_l  = stl["rho"][0]
    alpha_g = m_g / rho_g
    alpha_g = jnp.clip(alpha_g, 1e-8, 1.0 - 1e-8)

    return jnp.array([p, u_g, u_l, alpha_g, h_g, h_l])
