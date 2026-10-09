"""
Source terms for two-fluid model
"""

import jax
import jax.numpy as jnp

def calculate_mass_transfer_lv(T_l, T_v, T_sat, htc_l, htc_v, A_i, h_sat_l, h_sat_v, h_l, h_v, Qvi, Qli):
    """
    Interphase mass transfer from Rane & He (2021), Eq. (4).
    Positive m_dot = evaporation (l -> v), Negative = condensation (v -> l).
    """
    # Eq. (4): m_dot = -[htc_l * ai * (Tsat - Tl) + htc_v * ai * (Tsat - Tv)] / (h_qs - h_ps)
    # Distribute the minus sign:
    # numerator = -htc_l * A_i * (T_sat - T_l) - htc_v * A_i * (T_sat - T_v)

    numerator = - Qvi - Qli


    # Denominator depends on direction: Eq. (5) for evaporation, Eq. (6) for condensation
    denominator = h_sat_v - h_sat_l

    m_dot = numerator / denominator

    return m_dot


def calculate_drag_force_one_component(
    alpha1, alpha2, rho1, rho2, mu1, mu2, u1, u2, d,
    alpha_b=0.3, alpha_d=0.7,
):
    """
    Regime-dependent drag force per unit volume for the one-component case.
    The function is based on the Schiller and Naumann model

    Convention: phase 1 = vapor (gas), phase 2 = liquid.
      F > 0 means force on vapor in the direction of u2 - u1.

    Parameters
    ----------
    alpha1, alpha2 : volume fractions (vapor, liquid)
    rho1, rho2     : densities (vapor, liquid)
    mu1, mu2       : dynamic viscosities (vapor, liquid)
    u1, u2         : velocities (vapor, liquid)
    d              : characteristic diameter of dispersed structures
    alpha_b        : upper bound of bubbly regime (vapor fraction)
    alpha_d        : lower bound of droplet regime (vapor fraction)
    """
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    def _Cd_SN(rho_c, mu_c):
        Re = rho_c * u_rel_abs * d / jnp.maximum(mu_c, 1e-30)
        Re_safe = jnp.maximum(Re, 1e-12)   # avoid 0**0.687 → ∞ gradient under jacfwd
        return (24.0 / Re_safe) * (1.0 + 0.15 * Re_safe**0.687)

    prefactor = (3.0 / 4.0) / d * u_rel_abs * u_rel

    def bubbly(_):
        # Vapor dispersed in liquid: alpha_disp = alpha1 (vapor), rho_cont = rho2 (liquid)
        C_d = _Cd_SN(rho2, mu2)
        return prefactor * C_d * alpha1 * rho2

    def droplet(_):
        # Liquid dispersed in vapor: alpha_disp = alpha2 (liquid), rho_cont = rho1 (vapor)
        C_d = _Cd_SN(rho1, mu1)
        return prefactor * C_d * alpha2 * rho1

    def transition(_):
        F_b = prefactor * _Cd_SN(rho2, mu2) * alpha_b * rho2           # bubbly at alpha1 = alpha_b
        F_d = prefactor * _Cd_SN(rho1, mu1) * (1.0 - alpha_d) * rho1  # droplet at alpha1 = alpha_d
        w = (alpha1 - alpha_b) / (alpha_d - alpha_b)
        return F_b + (F_d - F_b) * w

    return jax.lax.cond(
        alpha1 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha1 >= alpha_d, droplet, transition, None),
        None,
    )



def interfacial_area(
    alpha,
    alpha_b=0.3, 
    alpha_d=0.7,
    Nb=1e10,
    Nd=1e10,
    d_min_b=1e-5,
    d_min_d=1e-5,
):
    """
    TA-BD interfacial area density with nucleation and breakup floors.

    Five regions as a function of vapor volume fraction alpha:
      1. alpha <= alpha_b_star         : nucleation floor (constant ai, d = d_min_b)
      2. alpha_b_star < alpha <= 0.3   : bubbly regime
      3. 0.3 < alpha < 0.7             : linear transition
      4. 0.7 <= alpha < alpha_d_star   : droplet regime
      5. alpha >= alpha_d_star         : breakup floor (constant ai, d = d_min_d)

    Parameters
    ----------
    alpha : scalar
        Vapor volume fraction in [0, 1].
    alpha_b, alpha_d : float
        TA-BD transition bounds (0.3 and 0.7 in the paper).
    Nb, Nd : float
        Bubble and droplet number densities [1/m^3].
    d_min_b, d_min_d : float
        Minimum physical diameters for bubbles and droplets [m].

    Returns
    -------
    ai : scalar
        Interfacial area density [1/m].
    d : scalar
        Characteristic length scale [m] (bubble or droplet diameter).
    """

    # Derived thresholds where the natural cube-root diameter hits the floor
    alpha_b_star = jnp.pi * Nb * d_min_b**3 / 6.0
    alpha_d_star = 1.0 - jnp.pi * Nd * d_min_d**3 / 6.0

    # Endpoint values at the TA-BD transition boundaries (alpha_b = 0.3, alpha_d = 0.7)
    Ai_b_at_03 = (6.0 * alpha_b) ** (2.0 / 3.0) * (jnp.pi * Nb) ** (1.0 / 3.0)
    Ai_d_at_07 = (6.0 * (1.0 - alpha_d)) ** (2.0 / 3.0) * (jnp.pi * Nd) ** (1.0 / 3.0)
    d_b_at_03 = (6.0 * alpha_b / (jnp.pi * Nb)) ** (1.0 / 3.0)
    d_d_at_07 = (6.0 * (1.0 - alpha_d) / (jnp.pi * Nd)) ** (1.0 / 3.0)

    def nucleation_floor(_):
        ai = Nb * jnp.pi * d_min_b**2
        d = d_min_b
        return ai, d

    def bubbly(_):
        ai = (6.0 * alpha) ** (2.0 / 3.0) * (jnp.pi * Nb) ** (1.0 / 3.0)
        d = (6.0 * alpha / (jnp.pi * Nb)) ** (1.0 / 3.0)
        return ai, d

    def transition(_):
        w = (alpha - alpha_b) / (alpha_d - alpha_b)
        ai = Ai_b_at_03 + (Ai_d_at_07 - Ai_b_at_03) * w
        d = d_b_at_03 + (d_d_at_07 - d_b_at_03) * w
        return ai, d

    def droplet(_):
        ai = (6.0 * (1.0 - alpha)) ** (2.0 / 3.0) * (jnp.pi * Nd) ** (1.0 / 3.0)
        d = (6.0 * (1.0 - alpha) / (jnp.pi * Nd)) ** (1.0 / 3.0)
        return ai, d

    def breakup_floor(_):
        ai = Nd * jnp.pi * d_min_d**2
        d = d_min_d
        return ai, d

    # Nested lax.cond cascade — walks through the five branches in order.
    # Structure: if alpha <= alpha_b_star -> nucleation_floor
    #            elif alpha <= alpha_b    -> bubbly
    #            elif alpha <  alpha_d    -> transition
    #            elif alpha <  alpha_d_star -> droplet
    #            else                     -> breakup_floor
    ai, d = jax.lax.cond(
        alpha <= alpha_b_star,
        nucleation_floor,
        lambda _: jax.lax.cond(
            alpha <= alpha_b,
            bubbly,
            lambda _: jax.lax.cond(
                alpha < alpha_d,
                transition,
                lambda _: jax.lax.cond(
                    alpha < alpha_d_star,
                    droplet,
                    breakup_floor,
                    None,
                ),
                None,
            ),
            None,
        ),
        None,
    )

    return ai, d


def ranz_marshall_htc_two_fluid(
        state_continuous,
        relative_velocity,
        d_dispersed
):
    """
    Continuous-phase HTC for dispersed particles (Ranz-Marshall 1952).

    Same as ranz_marshall_htc_two_components from HNEM, but uses
    the actual relative velocity instead of the Weber-estimated slip.
    All thermophysical properties are taken from the CONTINUOUS phase.

    Parameters
    ----------
    alpha_dispersed : float
        Volume fraction of the dispersed phase.
    state_continuous : FluidState
        Thermodynamic state of the continuous phase.
    fluid_continuous : Fluid
        Fluid object for the continuous phase (needs fluid_name for surface tension).
    relative_velocity : float
        Absolute relative velocity between phases |u1 - u2| [m/s].
    We : float
        Critical Weber number (default 8.0). Reserved, not used when
        relative_velocity is provided directly.
    Nb : float
        Number density for interfacial area model (default 1e10).
    """
    # --- Continuous-phase properties ---
    rho_c = state_continuous["rho"]
    mu_c  = state_continuous["mu"]
    k_c   = state_continuous["k"]
    cp_c  = state_continuous["cp"]

    # --- Re and Pr use continuous-phase properties ---
    Re = rho_c * jnp.abs(relative_velocity) * d_dispersed / mu_c
    Re_safe = jnp.maximum(Re, 1e-12)   # avoid 0**0.5 → ∞ gradient under jacfwd
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re_safe**0.5 * Pr**(1.0 / 3.0)

    htc = k_c * Nu / d_dispersed

    return htc
