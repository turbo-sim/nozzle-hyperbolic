"""Exact perfect-gas Sod-pattern Riemann reference, independent of the FVM.

The supported wave pattern is a left rarefaction, a contact and a right shock.
This is not a general Riemann solver: reversed patterns and vacuum require
other sampling formulas and are rejected explicitly.

Source: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Springer (2009), chapter 4, *The Riemann Problem for the Euler
Equations*, pp. 115-162, https://doi.org/10.1007/b79761.
The pressure functions, star-state density relations and self-similar fan
sampling below are the classical perfect-gas exact-solver formulas from that
chapter, implemented here in NumPy. No formula is inferred from FVM results.
"""

import numpy as np
from scipy.optimize import brentq


def _pressure_function(pressure, initial_pressure, sound_speed, density, gamma):
    """Wave velocity change f_K(p): shock Hugoniot or isentropic rarefaction."""
    if pressure > initial_pressure:
        coefficient = 2 / ((gamma + 1) * density)
        offset = (gamma - 1) / (gamma + 1) * initial_pressure
        return (pressure - initial_pressure) * np.sqrt(
            coefficient / (pressure + offset)
        )
    exponent = (gamma - 1) / (2 * gamma)
    return (
        2 * sound_speed / (gamma - 1) * ((pressure / initial_pressure) ** exponent - 1)
    )


def _star_density(pressure, initial_pressure, density, gamma):
    """Density on either side of the contact from a shock or an isentrope."""
    ratio = pressure / initial_pressure
    if pressure > initial_pressure:
        coefficient = (gamma - 1) / (gamma + 1)
        return density * (ratio + coefficient) / (coefficient * ratio + 1)
    return density * ratio ** (1 / gamma)


def _sample_sod(
    x,
    t,
    x0,
    rhoL,
    uL,
    pL,
    rhoR,
    uR,
    pR,
    gamma,
    cL,
    cR,
    *,
    lower_factor,
    upper_factor,
    absolute_tolerance,
    inclusive
):
    """Common pressure root and wave sampling for both public reference APIs."""
    x = np.asarray(x, dtype=float)
    if (
        t < 0
        or gamma <= 1
        or min(rhoL, rhoR, pL, pR, cL, cR) <= 0
        or not np.all(np.isfinite([t, x0, rhoL, uL, pL, rhoR, uR, pR, gamma, cL, cR]))
        or np.any(~np.isfinite(x))
    ):
        raise ValueError(
            "Need finite states, t>=0, gamma>1 and positive density/pressure/sound speed"
        )
    if t == 0:
        # Define the initial discontinuity's point value using its right limit.
        return tuple(
            np.where(x < x0, left, right)
            for left, right in (
                (rhoL, rhoR),
                (uL, uR),
                (pL, pR),
            )
        )

    def left_function(pressure):
        return _pressure_function(pressure, pL, cL, rhoL, gamma)

    def right_function(pressure):
        return _pressure_function(pressure, pR, cR, rhoR, gamma)

    p_star = brentq(
        lambda p: left_function(p) + right_function(p) + uR - uL,
        lower_factor * pL,
        upper_factor * max(pL, pR),
        xtol=absolute_tolerance,
    )
    u_star = 0.5 * (uL + uR) + 0.5 * (right_function(p_star) - left_function(p_star))
    tolerance = 1e-10 * max(pL, pR)
    if p_star > pL + tolerance or p_star < pR - tolerance:
        raise ValueError(
            "This reference supports only a left rarefaction and right shock (Sod pattern)"
        )
    rho_left_star = _star_density(p_star, pL, rhoL, gamma)
    rho_right_star = _star_density(p_star, pR, rhoR, gamma)
    c_left_star = np.sqrt(gamma * p_star / rho_left_star)

    rarefaction_head = uL - cL
    rarefaction_tail = u_star - c_left_star
    right_shock = uR + cR * np.sqrt(
        (gamma + 1) / (2 * gamma) * p_star / pR + (gamma - 1) / (2 * gamma)
    )
    similarity = (x - x0) / t
    compare = np.less_equal if inclusive else np.less
    left = compare(similarity, rarefaction_head)
    fan = ~left & compare(similarity, rarefaction_tail)
    left_star = ~left & ~fan & compare(similarity, u_star)
    right_star = ~left & ~fan & ~left_star & compare(similarity, right_shock)
    rho = np.full_like(x, rhoR)
    velocity = np.full_like(x, uR)
    pressure = np.full_like(x, pR)
    rho[left], velocity[left], pressure[left] = rhoL, uL, pL
    rho[left_star], velocity[left_star], pressure[left_star] = (
        rho_left_star,
        u_star,
        p_star,
    )
    rho[right_star], velocity[right_star], pressure[right_star] = (
        rho_right_star,
        u_star,
        p_star,
    )

    # Evaluate the fan only where it applies: no fractional powers of negative
    # extrapolated fan sound speeds in inactive branches.
    fan_velocity = 2 / (gamma + 1) * (cL + (gamma - 1) / 2 * uL + similarity[fan])
    fan_sound = cL + (gamma - 1) / 2 * (uL - fan_velocity)
    fan_pressure = pL * (fan_sound / cL) ** (2 * gamma / (gamma - 1))
    rho[fan] = gamma * fan_pressure / fan_sound**2
    velocity[fan], pressure[fan] = fan_velocity, fan_pressure
    return rho, velocity, pressure


def sod_exact(x_arr, t, x0, rhoL, uL, pL, rhoR, uR, pR, gamma):
    """Return pointwise ``(rho,u,p)`` for a calorically perfect-gas Sod pattern.

    Coordinates x/x0 are in metres and time t in seconds; density, velocity and
    pressure use kg/m3, m/s and Pa. Initial states occupy x<x0 and x>=x0.
    ``gamma>1`` is constant and sound speeds are ``sqrt(gamma*p/rho)``.
    At t=0 the initial discontinuity is returned exactly (right value at x0).
    At positive time the contact and shock planes take their left limits.
    The result is valid before waves interact with finite-domain boundaries.
    It is a point sample, not a finite-volume cell average.
    """
    return _sample_sod(
        x_arr,
        t,
        x0,
        rhoL,
        uL,
        pL,
        rhoR,
        uR,
        pR,
        gamma,
        np.sqrt(gamma * pL / rhoL),
        np.sqrt(gamma * pR / rhoR),
        lower_factor=1e-8,
        upper_factor=10,
        absolute_tolerance=1e-10,
        inclusive=True,
    )


def sod_exact_with_sound_speeds(x, t, x0, rhoL, uL, pL, rhoR, uR, pR, gamma, cL, cR):
    """Compatibility reference with supplied initial sound speeds cL/cR [m/s].

    All other wave relations still assume constant-gamma perfect gas. Supplied
    EOS sound speeds therefore do not make this an exact non-ideal-EOS result.
    Exactness requires ``cK**2=gamma*pK/rhoK``. This entry point retains the
    historical pressure bracket/tolerance and right-sided discontinuity values
    used by the older vectorized driver; all positive-time formulas are shared
    with :func:`sod_exact`. Its previously incorrect t=0 sampling is corrected.
    """
    return _sample_sod(
        x,
        t,
        x0,
        rhoL,
        uL,
        pL,
        rhoR,
        uR,
        pR,
        gamma,
        cL,
        cR,
        lower_factor=1e-6,
        upper_factor=10,
        absolute_tolerance=2e-12,
        inclusive=False,
    )
