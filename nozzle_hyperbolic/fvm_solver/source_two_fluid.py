"""
Source terms for two-fluid model
"""

import functools
import math
import jax
import jax.numpy as jnp

# Weight laws across the bubbly -> droplet transition (alpha_b < alpha < alpha_d),
# used by the drag and by the regime-dependent heat transfer.
REGIME_BLENDS = ("linear", "smoothstep")


def regime_weight(alpha1, alpha_b, alpha_d, blend="linear"):
    """
    Droplet-regime weight s in [0, 1]: 0 for alpha1 <= alpha_b, 1 for alpha1 >= alpha_d.

    linear     : s = w,              w = (alpha1 - alpha_b) / (alpha_d - alpha_b);
                 the slope of a blended closure jumps at alpha_b and alpha_d.
    smoothstep : s = w^2 (3 - 2 w);  zero slope at both ends, so a closure blended
                 as (1 - s) * bubbly + s * droplet is C1 across the transition.
    """
    w = jnp.clip((alpha1 - alpha_b) / (alpha_d - alpha_b), 0.0, 1.0)
    if blend == "smoothstep":
        return w * w * (3.0 - 2.0 * w)
    if blend == "linear":
        return w
    raise ValueError(f"unknown regime blend {blend!r}; choose from {REGIME_BLENDS}")

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
    alpha_b=0.3, alpha_d=0.7, blend="linear",
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
    blend          : transition between alpha_b and alpha_d (regime_weight).
                     'linear' (default): linear interpolation between the bubbly
                     force frozen at alpha_b and the droplet force frozen at alpha_d;
                     continuous, but its slope jumps at both ends.
                     'smoothstep': (1 - s) * bubbly + s * droplet, both laws at the
                     actual alpha, s = smoothstep -> C1 across the transition.
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

    if blend != "linear":
        s = regime_weight(alpha1, alpha_b, alpha_d, blend)
        return (1.0 - s) * bubbly(None) + s * droplet(None)

    return jax.lax.cond(
        alpha1 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha1 >= alpha_d, droplet, transition, None),
        None,
    )


def calculate_heat_transfer_one_component(
    alpha1, state1, state2, u1, u2, d,
    Nu1_int=120.0, Nu2_int=2.0 * jnp.pi**2 / 3.0,
    alpha_b=0.3, alpha_d=0.7, blend="linear",
):
    """
    Regime-dependent interfacial heat transfer coefficients for the
    one-component case, blended across the transition as in
    calculate_drag_force_one_component.

    Convention: phase 1 = vapor (gas), phase 2 = liquid.
    On each side of the interface:
      - continuous phase: Ranz-Marshall, Nu = 2 + 0.6 Re^1/2 Pr^1/3
        (continuous-phase properties, actual slip |u1 - u2|)
      - dispersed phase:  constant internal Nusselt, htc = k * Nu_int / d

      bubbly  (alpha1 <= alpha_b): vapor dispersed in liquid
          htc1 = k1 * Nu1_int / d,              htc2 = RM(liquid continuous)
      droplet (alpha1 >= alpha_d): liquid dispersed in vapor
          htc1 = RM(vapor continuous),          htc2 = k2 * Nu2_int / d
      transition: linear blend in alpha1 of the two regimes, both evaluated
          at the current d and slip (continuous at alpha_b and alpha_d).

    Parameters
    ----------
    alpha1         : vapor volume fraction
    state1, state2 : dicts with "rho", "mu", "k", "cp" of vapor / liquid
                     (same format as ranz_marshall_htc_two_fluid)
    u1, u2         : velocities (vapor, liquid)
    d              : characteristic diameter of dispersed structures
    Nu1_int        : Nusselt number inside a bubble (vapor side, bubbly regime)
    Nu2_int        : Nusselt number inside a droplet (liquid side, droplet regime);
                     default 2*pi^2/3 ≈ 6.58 is the conduction-limited sphere value
    alpha_b        : upper bound of bubbly regime (vapor fraction)
    alpha_d        : lower bound of droplet regime (vapor fraction)
    blend          : transition weight (regime_weight): 'linear' (default) or
                     'smoothstep' (zero slope at alpha_b and alpha_d -> C1 HTCs)

    Returns
    -------
    htc1, htc2 : interfacial HTCs of vapor and liquid [W/(m^2 K)]
    """
    u_rel_abs = jnp.abs(u1 - u2)

    def bubbly(_):
        # Vapor dispersed in liquid: internal Nu on the vapor side, liquid continuous
        htc1 = state1["k"] * Nu1_int / d
        htc2 = ranz_marshall_htc_two_fluid(state2, u_rel_abs, d)
        return htc1, htc2

    def droplet(_):
        # Liquid dispersed in vapor: vapor continuous, internal Nu on the liquid side
        htc1 = ranz_marshall_htc_two_fluid(state1, u_rel_abs, d)
        htc2 = state2["k"] * Nu2_int / d
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

def interfacial_area_original(
    alpha,
    alpha_b=0.3, 
    alpha_d=0.7,
    Nb=1e10,
    Nd=1e10,
    d_min_b=1e-5,
    d_min_d=1e-5,
    d_min=1e-5,
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

def dispersed_fraction(alpha, eps=0.1):
    """
    Smooth version of min(alpha, 1 - alpha), the volume fraction of the dispersed phase.

    Uses |x| ~ sqrt(x^2 + eps^2) with x = 2 alpha - 1, then shifts and rescales so the
    result is exactly 0 at alpha = 0 and 1 and exactly 0.5 at alpha = 0.5. eps -> 0
    recovers the sharp minimum; larger eps rounds the corner at alpha = 0.5 over a wider band.
    """
    def raw(x):
        return 0.5 * (1.0 - jnp.sqrt(x**2 + eps**2))

    lo, mid = raw(1.0), raw(0.0)
    return 0.5 * (raw(2.0 * alpha - 1.0) - lo) / (mid - lo)


def interfacial_area_poly(alpha, d_ctrl=(160e-6, 0.0, 320e-6, 0.0, 1455e-6), eps=0.1, d_min=1e-5,
        alpha_b=0.3,  
        alpha_d=0.7,
        Nb=1e10,
        Nd=1e10,
        d_min_b=1e-5,
        d_min_d=1e-5,):
    """
    Polynomial interfacial area density, parametrised by its diameter.

    d = P(alpha) * (2 alpha_disp)^(1/3), with P a Bernstein polynomial of degree N = len(d_ctrl) - 1:
      P = sum_k d_ctrl[k] * C(N, k) alpha^k (1 - alpha)^(N - k)
    The cube-root factor is the fixed-number-density law of the prior (interfacial_area):
    bubbles/droplets shrink as the dispersed phase vanishes, so d -> 0 at alpha = 0 and 1,
    and ai = 6 alpha_disp / d ~ alpha_disp^(2/3) -> 0 there too. The factor equals 1 at alpha = 0.5.
    P is the diameter the structures would have at alpha_disp = 0.5, i.e. (3 / (pi N))^(1/3) with a
    number density N(alpha); d <= max(d_ctrl) everywhere.

    Default d_ctrl [m] is a least-squares fit (relative error in d over 0.1 < alpha < 0.9) to
    interfacial_area_truth: max error 8% in d there, 6% of the peak in ai.

    Nucleation / breakup floor (as in interfacial_area): d never drops below d_min. Below it the
    structures keep their number density n = 3 / (pi P^3) but stop shrinking, so
    ai = n pi d_min^2 = 3 d_min^2 / P^3: d(0), d(1) = d_min and ai(0), ai(1) > 0. Continuous where
    d reaches d_min; there ai = 6 alpha_disp / d stops holding (as in the prior's floors).

    Returns
    -------
    ai : interfacial area density [1/m]
    d : characteristic diameter [m]
    """
    N = len(d_ctrl) - 1
    P = sum(c * math.comb(N, k) * alpha**k * (1.0 - alpha) ** (N - k) for k, c in enumerate(d_ctrl))
    alpha_disp = dispersed_fraction(alpha, eps)
    # = max(P (2 alpha_disp)^(1/3), d_min), floored inside the cbrt so its gradient at alpha_disp = 0 stays finite
    d = P * jnp.cbrt(jnp.maximum(2.0 * alpha_disp, (d_min / P) ** 3))
    ai = 3.0 * d**2 / P**3  # = n pi d^2 = 6 alpha_disp / d above the floor
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


def interfacial_area_para(alpha, a_min=100.0, a_max=20000.0, eps=0.1, d_min=1e-5,
                    alpha_b=0.3, 
                    alpha_d=0.7,
                    Nb=10e12, 
                    Nd=10e10, 
                    d_min_b=5e-6):
    """
    Interfacial area density for data assimilation: inverted parabola in alpha.

    ai(0) = ai(1) = a_min and ai(0.5) = a_max.

    The diameter follows from the monodisperse-sphere relation ai = 6 alpha_disp / d,
    where alpha_disp is the volume fraction of the dispersed phase, a smooth version of
    min(alpha, 1 - alpha) (see dispersed_fraction; eps sets the rounding at alpha = 0.5):
      alpha < 0.5 : bubbly  (vapor dispersed), alpha_disp ~ alpha
      alpha > 0.5 : misty   (liquid dispersed), alpha_disp ~ 1 - alpha

    Nucleation / breakup floor (as in interfacial_area): the diameter never drops below
    d_min, so d(0) = d(1) = d_min and ai(0) = ai(1) = a_min are both finite and non-zero.
    The floor is active for alpha_disp < a_min * d_min / 6, where ai = 6 alpha_disp / d no
    longer holds (minimum-size bubbles/droplets, as in the prior's floor regions).

    Returns
    -------
    ai : interfacial area density [1/m]
    d : characteristic diameter [m]
    """
    ai = a_min + (a_max - a_min) * 4.0 * alpha * (1.0 - alpha)
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d


def interfacial_area_segments(alpha,
                              alpha_pts=(0.0, 0.05, 0.25, 0.7, 0.9, 1.0),
                              ai_pts=(100.0, 3900.0, 15000.0, 24000.0, 20000.0, 100.0),
                              corner=0.0, eps=0.1, d_min=1e-5,
                              alpha_b=0.3,
                              alpha_d=0.7,
                              Nb=10e12,
                              Nd=10e10,
                              d_min_b=5e-6):
    """
    Interfacial area density for data assimilation: piecewise linear in alpha.

    ai is linear between the control points (alpha_pts[k], ai_pts[k]); alpha_pts must be
    strictly increasing and span [0, 1], ai_pts > 0.  Any shape, e.g. non-symmetric.
    corner > 0 rounds every interior corner over a width ~corner in alpha (each slope
    change becomes a softplus ramp, C-infinity; the segments are unchanged away from the
    corners); corner = 0 is exact linear interpolation.

    The diameter follows from ai = 6 alpha_disp / d with the nucleation / breakup floor
    d >= d_min, as in interfacial_area_para (alpha_disp: dispersed_fraction, eps).

    Default points: a shape skewed to the vapor side (peak 24000 1/m at alpha = 0.7,
    parabola-like area up to alpha ~ 0.25) that runs the case-4 ladder of
    main_ladder.py / settings_ladder.yaml unattended to N = 400, inlet void 1e-5
    (tested 2026-10-16, see LADDER_HISTORY.md).

    Returns
    -------
    ai : interfacial area density [1/m]
    d : characteristic diameter [m]
    """
    xs = tuple(float(v) for v in alpha_pts)
    ys = tuple(float(v) for v in ai_pts)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("interfacial_area_segments: alpha_pts and ai_pts need the same length >= 2")
    if any(b <= a for a, b in zip(xs, xs[1:])) or xs[0] > 0.0 or xs[-1] < 1.0:
        raise ValueError("interfacial_area_segments: alpha_pts must increase strictly from 0 to 1")
    if min(ys) <= 0.0:
        raise ValueError("interfacial_area_segments: ai_pts must be > 0")
    if corner > 0.0:
        slopes = [(y1 - y0) / (x1 - x0) for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:])]
        ai = ys[0] + slopes[0] * (alpha - xs[0])
        for xk, s0, s1 in zip(xs[1:-1], slopes, slopes[1:]):
            ai = ai + (s1 - s0) * corner * jax.nn.softplus((alpha - xk) / corner)
        ai = jnp.maximum(ai, min(ys))
    else:
        ai = jnp.interp(alpha, jnp.asarray(xs), jnp.asarray(ys))
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d


# Interfacial area laws selectable by name (settings: sources.interfacial_area).
# Every law takes (alpha, alpha_b=, alpha_d=, Nb=, Nd=, d_min_b=, ...) and returns (ai, d).
INTERFACIAL_AREA_LAWS = {
    "original": interfacial_area_original,     # TA-BD number-density law (uses Nb / Nd)
    "polynomial": interfacial_area_poly,       # Bernstein-polynomial diameter (ignores Nb / Nd)
    "parabola": interfacial_area_para,         # inverted parabola in alpha (ignores Nb / Nd)
    "segments": interfacial_area_segments,     # piecewise linear in alpha (ignores Nb / Nd)
}


def get_interfacial_area(name="polynomial", params=None):
    """
    Interfacial area law `name` with its keyword arguments `params` fixed, e.g.
    get_interfacial_area("polynomial", {"d_min": 6.5e-5}).  Returns a callable
    with the same signature as the laws above.
    """
    if name not in INTERFACIAL_AREA_LAWS:
        raise ValueError(f"unknown interfacial area law {name!r}; "
                         f"choose from {sorted(INTERFACIAL_AREA_LAWS)}")
    law = INTERFACIAL_AREA_LAWS[name]
    return functools.partial(law, **params) if params else law
