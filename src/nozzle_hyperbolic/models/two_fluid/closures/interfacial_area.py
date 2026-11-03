"""Alberto's interfacial-area laws and their dispersed-structure diameters.

Port reference: Alberto revision faafff704e54fd866abff95fcb88fc0a5217405c.
See docs/notes/two_fluid_migration.md for notation and retained policies.
"""

import functools
import math

import jax
import jax.numpy as jnp


def interfacial_area_original(
    alpha,
    alpha_b=0.3,
    alpha_d=0.7,
    Nb=1e10,
    Nd=1e10,
    d_min_b=1e-5,
    d_min_d=1e-5,
    d_min=1e-5,
    ai_scale=1.0,
):
    """TA-BD interfacial area density with nucleation and breakup floors.

    Parameters
    ----------
    alpha : scalar
        Gas volume fraction in [0, 1].
    alpha_b : float, optional
        Upper bubbly threshold; used by the original law.
    alpha_d : float, optional
        Lower droplet threshold; used by the original law.
    Nb : float, optional
        Bubble number density in 1/m**3; used by the original law.
    Nd : float, optional
        Droplet number density in 1/m**3; used by the original law.
    d_min_b : float, optional
        Minimum bubble diameter in m; used by the original law.
    d_min_d : float, optional
        Minimum droplet diameter in m; used by the original law.
    d_min : float, optional
        Unused common-protocol keyword; this law uses d_min_b and d_min_d.
    ai_scale : float, optional
        Positive area multiplier, applied before recovering diameter.

    Returns
    -------
    ai : scalar
        Interfacial area density in 1/m.
    d : scalar
        Dispersed-structure diameter in m after the inverse area scaling.

    Notes
    -----
    TA-BD interfacial area density with nucleation and breakup floors.

    Five regions as a function of vapor volume fraction alpha:
      1. alpha <= alpha_b_star         : nucleation floor (constant ai, d = d_min_b)
      2. alpha_b_star < alpha <= 0.3   : bubbly regime
      3. 0.3 < alpha < 0.7             : linear transition
      4. 0.7 <= alpha < alpha_d_star   : droplet regime
      5. alpha >= alpha_d_star         : breakup floor (constant ai, d = d_min_d)

    The final correction is ai -> ai_scale*ai and d -> d/ai_scale. The
    unscaled bubble/droplet floors therefore scale with the diameter too;
    unlike the other area laws, no diameter floor is reapplied afterwards.
    This retains Alberto's correlation exactly. Unused protocol keywords
    retain his common area-law signature.
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

    return ai_scale * ai, d / ai_scale


def dispersed_fraction(alpha, eps=0.1):
    """Return a smooth approximation to the dispersed-phase volume fraction.

    Parameters
    ----------
    alpha : scalar
        Gas volume fraction.
    eps : float, optional
        Positive rounding width around alpha=0.5.

    Returns
    -------
    scalar
        Exactly zero at alpha=0 and 1, and exactly 0.5 at alpha=0.5.

    Notes
    -----
    The shifted/rescaled square-root approximation tends to
    ``min(alpha,1-alpha)`` as eps tends to zero.
    """

    def raw(x):
        return 0.5 * (1.0 - jnp.sqrt(x**2 + eps**2))

    lo, mid = raw(1.0), raw(0.0)
    return 0.5 * (raw(2.0 * alpha - 1.0) - lo) / (mid - lo)


def interfacial_area_poly(
    alpha,
    ai_ctrl=(
        100.0,
        12342.0,
        23401.0,
        26272.0,
        20992.0,
        14931.0,
        12818.0,
        12545.0,
        10153.0,
        5255.0,
        100.0,
    ),
    eps=0.1,
    d_min=1e-5,
    alpha_b=0.3,
    alpha_d=0.7,
    Nb=1e10,
    Nd=1e10,
    d_min_b=1e-5,
    d_min_d=1e-5,
    ai_scale=1.0,
):
    """Evaluate Alberto's Bernstein-polynomial interfacial area and diameter.

    Parameters
    ----------
    alpha : scalar
        Gas volume fraction in [0, 1].
    ai_ctrl : sequence of float, optional
        Positive Bernstein area controls in 1/m.
    eps : float, optional
        Positive smoothing width of the dispersed-phase fraction.
    d_min : float, optional
        Positive diameter floor in m.
    alpha_b : float, optional
        Upper bubbly threshold; used by the original law.
    alpha_d : float, optional
        Lower droplet threshold; used by the original law.
    Nb : float, optional
        Bubble number density in 1/m**3; used by the original law.
    Nd : float, optional
        Droplet number density in 1/m**3; used by the original law.
    d_min_b : float, optional
        Minimum bubble diameter in m; used by the original law.
    d_min_d : float, optional
        Minimum droplet diameter in m; used by the original law.
    ai_scale : float, optional
        Positive area multiplier, applied before recovering diameter.

    Returns
    -------
    ai : scalar
        Interfacial area density in 1/m.
    d : scalar
        Dispersed-structure diameter in m, respecting the selected floor.

    Notes
    -----
    Bernstein area law fitted to historical segment points
    (alpha_pts 0, 0.05, 0.25, 0.7, 0.9, 1; ai_pts 100, 6500, 22000, 12000, 5000, 100).

    Before scaling, ai is a Bernstein polynomial of degree N=len(ai_ctrl)-1::

        ai_unscaled = sum_k ai_ctrl[k] * C(N, k) * alpha^k * (1-alpha)^(N-k)

    Its endpoints are the first/last controls and its value lies between
    the minimum/maximum controls. The actual area is ai_scale*ai_unscaled.
    Diameter follows d=6*alpha_disp/ai with d>=d_min, as in the parabola
    and segment laws.

    The default Bernstein controls retain Alberto's historical fit; they are
    not a fit to the current segment-law defaults.

    ai_scale multiplies ai (field-inversion correction); d follows from the scaled ai.

    Unused protocol keywords retain Alberto's common area-law signature.
    """
    N = len(ai_ctrl) - 1
    ai = ai_scale * sum(
        c * math.comb(N, k) * alpha**k * (1.0 - alpha) ** (N - k)
        for k, c in enumerate(ai_ctrl)
    )
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d


def interfacial_area_para(
    alpha,
    a_min=100.0,
    a_max=20000.0,
    eps=0.1,
    d_min=1e-5,
    alpha_b=0.3,
    alpha_d=0.7,
    Nb=10e12,
    Nd=10e10,
    d_min_b=5e-6,
    ai_scale=1.0,
):
    """Interfacial area density for data assimilation: inverted parabola in alpha.

    Parameters
    ----------
    alpha : scalar
        Gas volume fraction in [0, 1].
    a_min : float, optional
        Endpoint area density in 1/m.
    a_max : float, optional
        Area density at alpha=0.5 in 1/m.
    eps : float, optional
        Positive smoothing width of the dispersed-phase fraction.
    d_min : float, optional
        Positive diameter floor in m.
    alpha_b : float, optional
        Upper bubbly threshold; used by the original law.
    alpha_d : float, optional
        Lower droplet threshold; used by the original law.
    Nb : float, optional
        Bubble number density in 1/m**3; used by the original law.
    Nd : float, optional
        Droplet number density in 1/m**3; used by the original law.
    d_min_b : float, optional
        Minimum bubble diameter in m; used by the original law.
    ai_scale : float, optional
        Positive area multiplier, applied before recovering diameter.

    Returns
    -------
    ai : scalar
        Interfacial area density in 1/m.
    d : scalar
        Dispersed-structure diameter in m, respecting the selected floor.

    Notes
    -----
    Interfacial area density for data assimilation: inverted parabola in alpha.

    Before ai_scale is applied, ai(0)=ai(1)=a_min and ai(0.5)=a_max.

    The diameter follows from the monodisperse-sphere relation ai = 6 alpha_disp / d,
    where alpha_disp is the volume fraction of the dispersed phase, a smooth version of
    min(alpha, 1-alpha). See dispersed_fraction; eps sets the rounding at
    alpha=0.5. The two limiting regimes are::

        alpha < 0.5 : bubbly (vapor dispersed), alpha_disp ~ alpha
        alpha > 0.5 : misty  (liquid dispersed), alpha_disp ~ 1-alpha

    Nucleation / breakup floor (as in interfacial_area): the diameter never drops below
    d_min, so d(0)=d(1)=d_min and ai(0)=ai(1)=ai_scale*a_min remain finite.
    The floor is active for alpha_disp < ai*d_min/6, where ai=6*alpha_disp/d no
    longer holds (minimum-size bubbles/droplets, as in the prior's floor regions).

    ai_scale multiplies ai (field-inversion correction); d follows from the scaled ai.

    Unused protocol keywords retain Alberto's common area-law signature.
    """
    ai = ai_scale * (a_min + (a_max - a_min) * 4.0 * alpha * (1.0 - alpha))
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d


def interfacial_area_segments(
    alpha,
    alpha_pts=(0.0, 0.05, 0.25, 0.7, 0.9, 1.0),
    ai_pts=(100.0, 3900.0, 15000.0, 24000.0, 20000.0, 100.0),
    corner=0.0,
    eps=0.1,
    d_min=1e-5,
    alpha_b=0.3,
    alpha_d=0.7,
    Nb=10e12,
    Nd=10e10,
    d_min_b=5e-6,
    ai_scale=1.0,
):
    """Interfacial area density for data assimilation: piecewise linear in alpha.

    Parameters
    ----------
    alpha : scalar
        Gas volume fraction in [0, 1].
    alpha_pts : sequence of float, optional
        Strictly increasing gas-fraction knots spanning [0, 1].
    ai_pts : sequence of float, optional
        Positive area density at each knot, in 1/m.
    corner : float, optional
        Nonnegative softplus corner width; zero gives exact segments.
    eps : float, optional
        Positive smoothing width of the dispersed-phase fraction.
    d_min : float, optional
        Positive diameter floor in m.
    alpha_b : float, optional
        Upper bubbly threshold; used by the original law.
    alpha_d : float, optional
        Lower droplet threshold; used by the original law.
    Nb : float, optional
        Bubble number density in 1/m**3; used by the original law.
    Nd : float, optional
        Droplet number density in 1/m**3; used by the original law.
    d_min_b : float, optional
        Minimum bubble diameter in m; used by the original law.
    ai_scale : float, optional
        Positive area multiplier, applied before recovering diameter.

    Returns
    -------
    ai : scalar
        Interfacial area density in 1/m.
    d : scalar
        Dispersed-structure diameter in m, respecting the selected floor.

    Notes
    -----
    Interfacial area density for data assimilation: piecewise linear in alpha.

    ai is linear between the control points (alpha_pts[k], ai_pts[k]); alpha_pts must be
    strictly increasing and span [0, 1], ai_pts > 0.  Any shape, e.g. non-symmetric.
    corner > 0 rounds every interior corner over a width ~corner in alpha (each slope
    change becomes a softplus ramp, C-infinity; the segments are unchanged away from the
    corners); corner = 0 is exact linear interpolation.

    The diameter follows from ai = 6 alpha_disp / d with the nucleation / breakup floor
    d >= d_min, as in interfacial_area_para (alpha_disp: dispersed_fraction, eps).

    The retained default points peak at 24000 1/m for alpha=0.7.
    They are correlation parameters, not a guarantee of solver convergence.

    ai_scale multiplies ai (field-inversion correction); d follows from the scaled ai.

    Unused protocol keywords retain Alberto's common area-law signature.
    """
    xs = tuple(float(v) for v in alpha_pts)
    ys = tuple(float(v) for v in ai_pts)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(
            "interfacial_area_segments: alpha_pts and ai_pts need the same length >= 2"
        )
    if any(b <= a for a, b in zip(xs, xs[1:])) or xs[0] > 0.0 or xs[-1] < 1.0:
        raise ValueError(
            "interfacial_area_segments: alpha_pts must increase strictly from 0 to 1"
        )
    if min(ys) <= 0.0:
        raise ValueError("interfacial_area_segments: ai_pts must be > 0")
    if corner > 0.0:
        slopes = [
            (y1 - y0) / (x1 - x0) for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:])
        ]
        ai = ys[0] + slopes[0] * (alpha - xs[0])
        for xk, s0, s1 in zip(xs[1:-1], slopes, slopes[1:]):
            ai = ai + (s1 - s0) * corner * jax.nn.softplus((alpha - xk) / corner)
        ai = jnp.maximum(ai, min(ys))
    else:
        ai = jnp.interp(alpha, jnp.asarray(xs), jnp.asarray(ys))
    ai = ai_scale * ai
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d


INTERFACIAL_AREA_LAWS = {
    "original": interfacial_area_original,  # TA-BD number-density law (uses Nb / Nd)
    "polynomial": interfacial_area_poly,  # Bernstein-polynomial diameter (ignores Nb / Nd)
    "parabola": interfacial_area_para,  # inverted parabola in alpha (ignores Nb / Nd)
    "segments": interfacial_area_segments,  # piecewise linear in alpha (ignores Nb / Nd)
}


def get_interfacial_area(name="polynomial", params=None):
    """Bind optional parameters to a named interfacial-area correlation.

    Parameters
    ----------
    name : {'original', 'polynomial', 'parabola', 'segments'}, optional
        Correlation registered in ``INTERFACIAL_AREA_LAWS``.
    params : dict or None, optional
        Keyword arguments fixed with ``functools.partial``.

    Returns
    -------
    callable
        Law accepting gas fraction and returning ``(area_density, diameter)``.

    Notes
    -----
    The model dispatcher passes explicit per-call keywords. Those override
    keywords bound by the partial; ``two_fluid_closure`` merges overrides first.
    """
    if name not in INTERFACIAL_AREA_LAWS:
        raise ValueError(
            f"unknown interfacial area law {name!r}; "
            f"choose from {sorted(INTERFACIAL_AREA_LAWS)}"
        )
    law = INTERFACIAL_AREA_LAWS[name]
    return functools.partial(law, **params) if params else law
