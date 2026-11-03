"""Phase-side interfacial heat transfer and liquid Nusselt alternatives.

Port reference: Alberto revision faafff704e54fd866abff95fcb88fc0a5217405c.
See docs/notes/two_fluid_migration.md for notation and retained policies.
"""

import jax
import jax.numpy as jnp

from .regimes import regime_weight


LIQUID_HTC_BUBBLY = ("ranz_marshall", "jakob", "max", "liao")


LIQUID_HTC_DROPLET = ("internal", "jakob", "max")


def jakob_nusselt(ja):
    """Evaluate the thermal-diffusion bubble-growth Nusselt number.

    Parameters
    ----------
    ja : scalar
        Liquid Jakob number ``rho_l*cp_l*(T_l-T_sat)/(rho_g*h_fg)``.

    Returns
    -------
    scalar
        ``max(2, 12*Ja/pi)``, with the sphere conduction limit as lower bound.

    Notes
    -----
    This closure is independent of slip (Plesset-Zwick bubble-growth scaling).
    """
    return jnp.maximum(2.0, 12.0 / jnp.pi * ja)


def liao_nusselt(ja, pe):
    """Combine liquid superheat and convection in the Liao Nusselt law.

    Parameters
    ----------
    ja : scalar
        Dimensionless liquid Jakob number.
    pe : scalar
        Nonnegative Peclet number based on relative velocity and diameter.

    Returns
    -------
    scalar
        ``sqrt(max(12*Ja**2/pi**2 + Pe/(3*pi), 1e-12))``.

    Notes
    -----
    The positive radicand floor keeps AD finite at zero superheat and slip.
    The upstream policy defines Ja with temperature difference to the first power.
    """
    return jnp.sqrt(jnp.maximum(12.0 / jnp.pi**2 * ja**2 + pe / (3.0 * jnp.pi), 1e-12))


def select_liquid_htc(option, htc_default, htc_jakob, htc_liao=None):
    """Select the liquid-side HTC for one regime.

    Parameters
    ----------
    option : str
        Validated regime law: default, 'jakob', 'max', or 'liao'.
    htc_default, htc_jakob : scalar
        Regime-default and thermal-diffusion HTCs in W/(m**2 K).
    htc_liao : scalar or None, optional
        Convection/superheat HTC required when selecting 'liao'.

    Returns
    -------
    scalar
        Selected HTC in W/(m**2 K); 'max' compares default and Jakob values.

    Notes
    -----
    Selectors are validated by ``ClosureConfig`` or the calling closure.
    """
    if option == "jakob":
        return htc_jakob
    if option == "max":
        return jnp.maximum(htc_default, htc_jakob)
    if option == "liao":
        return htc_liao
    return htc_default


def ranz_marshall_htc_two_fluid(
    state_continuous,
    relative_velocity,
    d_dispersed,
    slip_u0=0.0,
):
    """Evaluate the continuous-phase Ranz-Marshall heat-transfer coefficient.

    Parameters
    ----------
    state_continuous : mapping
        Continuous-phase ``rho``, ``mu``, ``k`` and ``cp`` in SI units.
    relative_velocity : scalar
        Phase-relative velocity in m/s; its magnitude is used.
    d_dispersed : scalar
        Positive dispersed-structure diameter in m.
    slip_u0 : float, optional
        Positive values replace absolute slip by ``sqrt(du**2+slip_u0**2)``.

    Returns
    -------
    scalar
        HTC in W/(m**2 K), with ``Nu=2+0.6*sqrt(Re)*Pr**(1/3)``.

    Notes
    -----
    Reynolds and Prandtl numbers use the continuous phase. The Reynolds
    number has a 1e-12 floor to keep the square-root derivative finite at rest.
    ``slip_u0=0`` retains the original absolute-slip law.
    """
    # --- Continuous-phase properties ---
    rho_c = state_continuous["rho"]
    mu_c = state_continuous["mu"]
    k_c = state_continuous["k"]
    cp_c = state_continuous["cp"]

    # --- Re and Pr use continuous-phase properties ---
    if slip_u0 > 0.0:
        u_rel = jnp.sqrt(relative_velocity**2 + slip_u0**2)
    else:
        u_rel = jnp.abs(relative_velocity)
    Re = rho_c * u_rel * d_dispersed / mu_c
    Re_safe = jnp.maximum(Re, 1e-12)  # avoid 0**0.5 → ∞ gradient under jacfwd
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re_safe**0.5 * Pr ** (1.0 / 3.0)

    htc = k_c * Nu / d_dispersed

    return htc


def calculate_heat_transfer_one_component(
    alpha1,
    state1,
    state2,
    u1,
    u2,
    d,
    Nu1_int=120.0,
    Nu2_int=2.0 * jnp.pi**2 / 3.0,
    alpha_b=0.3,
    alpha_d=0.7,
    blend="linear",
    slip_u0=0.0,
    ja=0.0,
    liquid_bubbly="ranz_marshall",
    liquid_droplet="internal",
):
    """Return phase-side HTCs across the bubbly-to-droplet transition.

    Parameters
    ----------
    alpha1 : scalar
        Gas volume fraction.
    state1, state2 : mapping
        Gas and liquid ``rho``, ``mu``, ``k`` and ``cp`` in SI units.
    u1, u2 : scalar
        Gas and liquid velocities in m/s.
    d : scalar
        Positive dispersed-structure diameter in m.
    Nu1_int, Nu2_int : float, optional
        Internal gas-bubble and liquid-droplet Nusselt numbers.
    alpha_b, alpha_d : float, optional
        Upper bubbly and lower droplet thresholds.
    blend : {'linear', 'smoothstep'}, optional
        Transition weight evaluated at the actual state.
    slip_u0 : float, optional
        Ranz-Marshall slip regularization in m/s; zero retains absolute slip.
    ja : scalar, optional
        Liquid Jakob number used by thermal-diffusion alternatives.
    liquid_bubbly : {'ranz_marshall', 'jakob', 'max', 'liao'}, optional
        Liquid-side law in the bubbly regime.
    liquid_droplet : {'internal', 'jakob', 'max'}, optional
        Liquid-side law in the droplet regime.

    Returns
    -------
    htc1, htc2 : scalar
        Gas and liquid interfacial HTCs in W/(m**2 K).

    Notes
    -----
    Bubbly: internal gas HTC and continuous-liquid Ranz-Marshall HTC.
    Droplet: continuous-gas Ranz-Marshall HTC and internal liquid HTC.
    The transition blends these laws at the current diameter and slip.
    Jakob/Liao alternatives change the liquid side only.
    """
    if (
        liquid_bubbly not in LIQUID_HTC_BUBBLY
        or liquid_droplet not in LIQUID_HTC_DROPLET
    ):
        raise ValueError(
            f"liquid HTC options: bubbly {LIQUID_HTC_BUBBLY}, droplet {LIQUID_HTC_DROPLET}; "
            f"got {liquid_bubbly!r}, {liquid_droplet!r}"
        )
    u_rel_abs = jnp.abs(u1 - u2)
    htc2_jakob = state2["k"] * jakob_nusselt(ja) / d
    u_pe = jnp.sqrt(u_rel_abs**2 + slip_u0**2) if slip_u0 > 0.0 else u_rel_abs
    pe = u_pe * d * state2["rho"] * state2["cp"] / state2["k"]
    htc2_liao = state2["k"] * liao_nusselt(ja, pe) / d

    def bubbly(_):
        # Vapor dispersed in liquid: internal Nu on the vapor side, liquid continuous
        htc1 = state1["k"] * Nu1_int / d
        htc2 = select_liquid_htc(
            liquid_bubbly,
            ranz_marshall_htc_two_fluid(state2, u_rel_abs, d, slip_u0),
            htc2_jakob,
            htc2_liao,
        )
        return htc1, htc2

    def droplet(_):
        # Liquid dispersed in vapor: vapor continuous, internal Nu on the liquid side
        htc1 = ranz_marshall_htc_two_fluid(state1, u_rel_abs, d, slip_u0)
        htc2 = select_liquid_htc(liquid_droplet, state2["k"] * Nu2_int / d, htc2_jakob)
        return htc1, htc2

    def transition(_):
        htc1_b, htc2_b = bubbly(None)
        htc1_d, htc2_d = droplet(None)
        w = (alpha1 - alpha_b) / (alpha_d - alpha_b)
        return htc1_b + (htc1_d - htc1_b) * w, htc2_b + (htc2_d - htc2_b) * w

    if blend != "linear":
        s = regime_weight(alpha1, alpha_b, alpha_d, blend)
        htc1_b, htc2_b = bubbly(None)
        htc1_d, htc2_d = droplet(None)
        return htc1_b + (htc1_d - htc1_b) * s, htc2_b + (htc2_d - htc2_b) * s

    return jax.lax.cond(
        alpha1 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha1 >= alpha_d, droplet, transition, None),
        None,
    )
