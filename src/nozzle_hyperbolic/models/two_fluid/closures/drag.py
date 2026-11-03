"""Signed interphase resistance with Alberto's bubbly/droplet conventions.

Port reference: Alberto revision faafff704e54fd866abff95fcb88fc0a5217405c.
See docs/notes/two_fluid_migration.md for notation and retained policies.
"""

import jax
import jax.numpy as jnp

from .regimes import regime_weight


DRAG_LAWS = ("schiller_naumann", "ishii_zuber")


def ishii_zuber_mixture_viscosity(mu_c, mu_d, alpha_disp, alpha_max=0.9):
    """Evaluate the viscous-regime Ishii-Zuber mixture viscosity.

    Parameters
    ----------
    mu_c, mu_d : scalar
        Continuous and dispersed dynamic viscosities in Pa s.
    alpha_disp : scalar
        Dispersed-phase volume fraction.
    alpha_max : float, optional
        Crowding cap, used because inactive blended laws may be evaluated.

    Returns
    -------
    scalar
        Mixture viscosity in Pa s.

    Notes
    -----
    ``mu_star=(mu_d+0.4*mu_c)/(mu_d+mu_c)`` and
    ``mu_m=mu_c*(1-min(alpha_disp,alpha_max))**(-2.5*mu_star)``.
    Only the viscous regime is implemented; distorted/Newton regimes are absent.
    """
    mu_star = (mu_d + 0.4 * mu_c) / (mu_d + mu_c)
    return mu_c * (1.0 - jnp.minimum(alpha_disp, alpha_max)) ** (-2.5 * mu_star)


def calculate_drag_force_one_component(
    alpha1,
    alpha2,
    rho1,
    rho2,
    mu1,
    mu2,
    u1,
    u2,
    d,
    alpha_b=0.3,
    alpha_d=0.7,
    blend="linear",
    drag_bubbly="schiller_naumann",
    drag_droplet="schiller_naumann",
):
    """Evaluate signed interphase resistance across bubbly and droplet regimes.

    Parameters
    ----------
    alpha1, alpha2 : scalar
        Gas and liquid volume fractions.
    rho1, rho2 : scalar
        Gas and liquid densities in kg/m**3.
    mu1, mu2 : scalar
        Gas and liquid dynamic viscosities in Pa s.
    u1, u2 : scalar
        Gas and liquid velocities in m/s.
    d : scalar
        Positive dispersed-structure diameter in m.
    alpha_b, alpha_d : float, optional
        Upper bubbly and lower droplet gas-fraction thresholds.
    blend : {'linear', 'smoothstep'}, optional
        Linear interpolation of endpoint forces, or smoothstep of current-state laws.
    drag_bubbly, drag_droplet : {'schiller_naumann', 'ishii_zuber'}, optional
        Drag law for each regime; Ishii-Zuber includes its viscous regime only.

    Returns
    -------
    scalar
        Resistance in N/m**3 with the sign of ``u1-u2``. The source assembler
        negates it to obtain force on the gas.

    Notes
    -----
    The Stokes singularity is canceled analytically, preserving a finite
    value and derivative at zero slip without a velocity floor. The linear
    transition freezes bubbly/droplet laws at their respective thresholds;
    smoothstep evaluates both laws at the actual volume fraction.
    """
    if drag_bubbly not in DRAG_LAWS or drag_droplet not in DRAG_LAWS:
        raise ValueError(
            f"drag laws {DRAG_LAWS}; got {drag_bubbly!r}, {drag_droplet!r}"
        )
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    def force(law, rho_c, mu_c, mu_d, alpha_disp):
        # Cancel |du| against Cd's 1/Re analytically before evaluation.
        # The remaining odd power has value/derivative zero at rest. Its
        # inactive branch uses 1 only to avoid 0**fractional AD; this is not
        # a velocity floor and changes no nonzero-slip value.
        if law == "ishii_zuber":
            viscosity = ishii_zuber_mixture_viscosity(mu_c, mu_d, alpha_disp)
            factor, exponent = 0.1, 0.75
        else:
            viscosity = mu_c
            factor, exponent = 0.15, 0.687
        reynolds_per_speed = rho_c * d / viscosity
        safe_abs = jnp.where(u_rel == 0.0, 1.0, u_rel_abs)
        odd_power = jnp.where(u_rel == 0.0, 0.0, u_rel * safe_abs**exponent)
        return (
            18.0
            * viscosity
            * alpha_disp
            / d**2
            * (u_rel + factor * reynolds_per_speed**exponent * odd_power)
        )

    def bubbly(_):
        return force(drag_bubbly, rho2, mu2, mu1, alpha1)

    def droplet(_):
        return force(drag_droplet, rho1, mu1, mu2, alpha2)

    def transition(_):
        F_b = force(drag_bubbly, rho2, mu2, mu1, alpha_b)
        F_d = force(drag_droplet, rho1, mu1, mu2, 1.0 - alpha_d)
        w = (alpha1 - alpha_b) / (alpha_d - alpha_b)
        return F_b + (F_d - F_b) * w

    if blend != "linear":
        s = regime_weight(alpha1, alpha_b, alpha_d, blend)
        return (1.0 - s) * bubbly(None) + s * droplet(None)

    return jax.lax.cond(
        alpha1 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha1 >= alpha_d, droplet, transition, None),
        None,
    )
