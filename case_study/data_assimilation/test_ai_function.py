import jax
import jax.numpy as jnp
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nozzle_hyperbolic.fvm_solver.source_two_fluid import interfacial_area_para, interfacial_area_segments


def interfacial_area(
    alpha,
    alpha_b=0.3, 
    alpha_d=0.7,
    Nb=1e12,
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


def interfacial_area_da(alpha, a_min=100.0, a_max=20000.0, eps=0.1, d_min=1e-5):
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


def interfacial_area_da_asymmetric(alpha, a_min=100.0, a_max=20000.0, alpha_peak=0.3, sharpness=2.0, eps=0.1, d_min=1e-5):
    """
    Asymmetric interfacial area density for data assimilation (beta-shaped bump).

    ai = a_min + (a_max - a_min) * alpha^p (1 - alpha)^q / [alpha_peak^p (1 - alpha_peak)^q]
    with p = sharpness * alpha_peak and q = sharpness * (1 - alpha_peak).

    ai(0) = ai(1) = a_min and ai(alpha_peak) = a_max, with the maximum located exactly at
    alpha_peak. sharpness = 2 and alpha_peak = 0.5 recovers the symmetric parabola;
    larger sharpness gives a narrower bump.

    The diameter follows the same relation as interfacial_area_da, d = 6 alpha_disp / ai,
    with the same floor d >= d_min (finite, non-zero d and ai at alpha = 0 and 1).

    Returns
    -------
    ai : interfacial area density [1/m]
    d : characteristic diameter [m]
    """
    p = sharpness * alpha_peak
    q = sharpness * (1.0 - alpha_peak)
    bump = alpha**p * (1.0 - alpha) ** q / (alpha_peak**p * (1.0 - alpha_peak) ** q)
    ai = a_min + (a_max - a_min) * bump
    alpha_disp = dispersed_fraction(alpha, eps)
    d = jnp.maximum(6.0 * alpha_disp / ai, d_min)
    return ai, d

def interfacial_area_truth(
    alpha,
    d_b=5.13e-3,
    d_d=7.2e-4,
    alpha_b_cr=0.15,
    alpha_d_cr=0.15,
    m=1.86,
    n=1.93,
):
    """
    Two-field interfacial area density: bubbly field + droplet field.

    Bubbly field (gas dispersed in liquid), present over the whole range:
      ai_b = 6 alpha G / d_b,  G = min(1, ((1 - alpha) / (1 - alpha_b_cr))^m)
      G is the fraction of the gas carried as bubbles: G = 1 in the pure bubbly
      field (alpha <= alpha_b_cr), then it decays to 0 at alpha = 1.

    Droplet field (liquid dispersed in gas), born at alpha_b_cr:
      ai_d = 6 (1 - alpha) F / d_d,  F = x^n,  x = clip((alpha - alpha_b_cr) / (1 - alpha_d_cr - alpha_b_cr), 0, 1)
      F is the fraction of the liquid carried as droplets: F = 0 below alpha_b_cr,
      F = 1 in the pure droplet field (alpha >= 1 - alpha_d_cr).

    Default d_b, d_d, m, n are a least-squares fit to points read by eye from the
    reference figure (bubbly peak ~250 1/m at alpha ~0.3, droplet peak ~1620 1/m at
    alpha ~0.72); alpha_b_cr = alpha_d_cr = 0.15 are taken from the figure.

    Returns
    -------
    ai : total interfacial area density ai_b + ai_d [1/m]
    ai_b : bubbly-field contribution [1/m]
    ai_d : droplet-field contribution [1/m]
    """

    alpha = 1.0 - alpha
    G = jnp.minimum(1.0, ((1.0 - alpha) / (1.0 - alpha_b_cr)) ** m)
    x = jnp.clip((alpha - alpha_b_cr) / (1.0 - alpha_d_cr - alpha_b_cr), 0.0, 1.0)
    F = x**n
    ai_b = 6.0 * alpha * G / d_b
    ai_d = 6.0 * (1.0 - alpha) * F / d_d
    alpha_disp = dispersed_fraction(alpha, 0.1)

    ai = (ai_b + ai_d) * 10
    d = 6.0 * alpha_disp / ai

    return ai, d


def interfacial_area_poly(alpha, d_ctrl=(160e-6, 0.0, 320e-6, 0.0, 1455e-6), eps=0.1, d_min=1e-5):
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


def interfacial_area_poly_prior(alpha, d_ctrl=(10e-6, 114e-6, 0.0, 318e-6, 718e-6, 10e-6), eps=0.1,
                                Nb=1e12, Nd=1e10):
    """
    Pure-polynomial interfacial area density, shaped like the prior (interfacial_area).

    d is a Bernstein polynomial of degree N = len(d_ctrl) - 1:
      d = sum_k d_ctrl[k] * C(N, k) alpha^k (1 - alpha)^(N - k)
    so d(0) = d_ctrl[0], d(1) = d_ctrl[-1], and min(d_ctrl) <= d <= max(d_ctrl) everywhere.
    ai follows from ai = 6 alpha_disp / d (see dispersed_fraction).

    Nucleation / breakup floor (as in interfacial_area): d already stays finite, so the floor is on
    the dispersed fraction used for ai, alpha_disp >= pi Nb d(0)^3 / 6 near alpha = 0 and
    >= pi Nd d(1)^3 / 6 near alpha = 1. Then ai(0) = Nb pi d(0)^2 and ai(1) = Nd pi d(1)^2,
    the prior's floor values, instead of 0.

    Default d_ctrl [m]: ends fixed at the prior's minimum diameters (d_min_b = d_min_d = 10 um),
    inner points a least-squares fit to the prior diameter over 0.02 < alpha < 0.98.
    d is linear near the ends (the prior goes as a cube root), so ai climbs from 0 to ~1e4 1/m
    within alpha ~ 0.02 of alpha = 0, and to ~2e3 1/m near alpha = 1.

    Returns
    -------
    ai : interfacial area density [1/m]
    d : characteristic diameter [m]
    """
    N = len(d_ctrl) - 1
    d = sum(c * math.comb(N, k) * alpha**k * (1.0 - alpha) ** (N - k) for k, c in enumerate(d_ctrl))
    alpha_star = jnp.where(alpha < 0.5, jnp.pi * Nb * d_ctrl[0]**3 / 6.0, jnp.pi * Nd * d_ctrl[-1]**3 / 6.0)
    ai = 6.0 * jnp.maximum(dispersed_fraction(alpha, eps), alpha_star) / d
    return ai, d


def interfacial_area_two_field(
    alpha,
    d_b=5.13e-3,
    d_d=7.2e-4,
    alpha_b_cr=0.15,
    alpha_d_cr=0.15,
    m=1.86,
    n=1.93,
):
    """
    Two-field interfacial area density: bubbly field + droplet field.

    Bubbly field (gas dispersed in liquid), present over the whole range:
      ai_b = 6 alpha G / d_b,  G = min(1, ((1 - alpha) / (1 - alpha_b_cr))^m)
      G is the fraction of the gas carried as bubbles: G = 1 in the pure bubbly
      field (alpha <= alpha_b_cr), then it decays to 0 at alpha = 1.

    Droplet field (liquid dispersed in gas), born at alpha_b_cr:
      ai_d = 6 (1 - alpha) F / d_d,  F = x^n,  x = clip((alpha - alpha_b_cr) / (1 - alpha_d_cr - alpha_b_cr), 0, 1)
      F is the fraction of the liquid carried as droplets: F = 0 below alpha_b_cr,
      F = 1 in the pure droplet field (alpha >= 1 - alpha_d_cr).

    Default d_b, d_d, m, n are a least-squares fit to points read by eye from the
    reference figure (bubbly peak ~250 1/m at alpha ~0.3, droplet peak ~1620 1/m at
    alpha ~0.72); alpha_b_cr = alpha_d_cr = 0.15 are taken from the figure.

    Returns
    -------
    ai : total interfacial area density ai_b + ai_d [1/m]
    ai_b : bubbly-field contribution [1/m]
    ai_d : droplet-field contribution [1/m]
    """
    G = jnp.minimum(1.0, ((1.0 - alpha) / (1.0 - alpha_b_cr)) ** m)
    x = jnp.clip((alpha - alpha_b_cr) / (1.0 - alpha_d_cr - alpha_b_cr), 0.0, 1.0)
    F = x**n
    ai_b = 6.0 * alpha * G / d_b
    ai_d = 6.0 * (1.0 - alpha) * F / d_d
    return ai_b + ai_d, ai_b, ai_d


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    alpha = np.linspace(0.0, 1.0, 1001)
    a = jnp.asarray(alpha)
    ai_p, d_p = (np.asarray(x) for x in jax.vmap(interfacial_area)(a))
    ai_da, d_da = (np.asarray(x) for x in interfacial_area_da(a))
    ai_as, d_as = (np.asarray(x) for x in interfacial_area_da_asymmetric(a))
    ai_tr, d_tr = (np.asarray(x) for x in interfacial_area_truth(a))
    ai_po, d_po = (np.asarray(x) for x in interfacial_area_poly(a))
    ai_pp, d_pp = (np.asarray(x) for x in interfacial_area_poly_prior(a))
    ai_par, d_par = (np.asarray(x) for x in interfacial_area_para(a, d_min=1e-5))
    ai_seg, d_seg = (np.asarray(x) for x in interfacial_area_segments(a, d_min=1e-5))

    # --- Diameter checks: d = 6 alpha_disp / ai in the two dispersed regimes ---
    alpha_hard = np.minimum(alpha, 1.0 - alpha)
    alpha_disp = np.asarray(dispersed_fraction(a))
    bubbly = (alpha > 1e-3) & (alpha <= 0.3)
    misty = (alpha >= 0.7) & (alpha < 1.0 - 1e-3)

    # # Smooth minimum: exact at 0, 0.5, 1 and within ~eps/2 of the sharp minimum elsewhere
    # assert abs(alpha_disp[0]) < 1e-6 and abs(alpha_disp[-1]) < 1e-6 and abs(alpha_disp[500] - 0.5) < 1e-6, "smooth min: not exact at 0, 0.5, 1"
    # assert np.max(np.abs(alpha_disp - alpha_hard)) < 0.05, "smooth min: too far from sharp min"

    # # DA: the relation must hold everywhere, and d must be continuous at alpha = 0.5
    # assert np.allclose(ai_da * d_da, 6.0 * alpha_disp, rtol=1e-4, atol=1e-6), "DA: ai*d != 6*alpha_disp"
    # d_lo, d_hi = (float(interfacial_area_da(jnp.asarray(x))[1]) for x in (0.5 - 1e-6, 0.5 + 1e-6))
    # assert abs(d_lo - d_hi) / d_lo < 1e-4, "DA: d discontinuous at alpha = 0.5"

    # # Asymmetric DA: same relation, continuity, pinned extremes, peak at alpha_peak
    # assert np.allclose(ai_as * d_as, 6.0 * alpha_disp, rtol=1e-4, atol=1e-6), "DA asym: ai*d != 6*alpha_disp"
    # d_lo, d_hi = (float(interfacial_area_da_asymmetric(jnp.asarray(x))[1]) for x in (0.5 - 1e-6, 0.5 + 1e-6))
    # assert abs(d_lo - d_hi) / d_lo < 1e-4, "DA asym: d discontinuous at alpha = 0.5"
    # assert abs(ai_as[0] - 100.0) < 1e-3 and abs(ai_as[-1] - 100.0) < 1e-3, "DA asym: extremes not pinned"
    # assert abs(alpha[np.argmax(ai_as)] - 0.3) < 2e-3 and abs(ai_as.max() - 20000.0) < 1.0, "DA asym: peak misplaced"

    # Prior: same relation in its pure bubbly and pure droplet regimes
    # for name, mask in (("bubbly", bubbly), ("misty", misty)):
    #     assert np.allclose(ai_p[mask] * d_p[mask], 6.0 * alpha_hard[mask], rtol=1e-4), f"prior {name}: ai*d != 6*alpha_disp"
    # print("Diameter checks passed (prior bubbly/misty, DA everywhere, DA continuity at 0.5)")

    # --- Plots ---
    def make_figure(title, curves, filename):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, style, ai_c, d_c in curves:
            axes[0].plot(alpha, ai_c, style, label=label)
            axes[1].plot(alpha, d_c * 1e6, style, label=label)
        axes[0].set_ylabel(r"$a_i$ [1/m]")
        axes[0].set_title("Interfacial area")
        axes[1].set_ylabel(r"$d$ [$\mu$m]")
        axes[1].set_title("Diameter")
        for ax in axes:
            ax.set_xlabel(r"$\alpha$ [-]")
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            # ax.grid(alpha=0.3)
            ax.legend()
        # fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(os.path.join(here, filename), dpi=200)

    # make_figure("Prior", [("Prior", "-", ai_p, d_p)], "prior.png")
    # make_figure("DA", [("Prior", "-", ai_p, d_p), ("DA", "--", ai_po, d_po)], "DA.png") #("DA asymmetric", ":", ai_as, d_as)], "DA.png")
    # make_figure("DA", [("Truth", "--", ai_tr, d_tr), ("Polynomial", "-", ai_po, d_po)], "truth.png")
    # make_figure("Prior polynomial", [("Prior", "-", ai_p, d_p), ("Polynomial", "--", ai_pp, d_pp)], "prior_poly.png")
    make_figure("Parabola", [("Parabola", "-", ai_par, d_par)], "parabola.png")
    make_figure("DA", [("Truth", "--", ai_par, d_par), ("Segments", "-", ai_seg, d_seg)], "DA_test.png")

    # --- Two-field interfacial area (bubbly + droplet field) vs. points read by eye from the reference figure ---
    ai_tf, ai_tf_b, ai_tf_d = (np.asarray(x) for x in interfacial_area_two_field(a))
    ref_b = np.array([[0.10, 120], [0.20, 215], [0.28, 248], [0.40, 235], [0.50, 210], [0.60, 170], [0.75, 100], [0.90, 35]])
    ref_d = np.array([[0.25, 200], [0.375, 600], [0.50, 1050], [0.625, 1450], [0.72, 1620], [0.80, 1500], [0.85, 1200], [0.90, 800]])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alpha, ai_tf_b, "-", label="bubbly field")
    ax.plot(alpha, ai_tf_d, "--", label="droplet field")
    ax.plot(alpha, ai_tf, "k-", lw=2, label="sum")
    ax.plot(*ref_b.T, "o", color="C0", mfc="none", label="reference (by eye)")
    ax.plot(*ref_d.T, "s", color="C1", mfc="none")
    for ac in (0.15, 0.85):
        ax.axvline(ac, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$\alpha$ [-]")
    ax.set_ylabel(r"$a_i$ [1/m]")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(here, "two_field.png"), dpi=200)
    plt.show()
