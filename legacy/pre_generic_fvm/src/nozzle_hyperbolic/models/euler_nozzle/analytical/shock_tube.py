"""Exact Sod shock-tube solution for Euler nozzle verification (Toro 2009, section 4.3)."""

import numpy as np
from scipy.optimize import brentq


def sod_exact(x_arr, t, x0, rhoL, uL, pL, rhoR, uR, pR, gamma):
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)
    AL = 2.0 / ((gamma + 1) * rhoL);  BL = (gamma - 1) / (gamma + 1) * pL
    AR = 2.0 / ((gamma + 1) * rhoR);  BR = (gamma - 1) / (gamma + 1) * pR

    def fK(p, pK, cK, AK, BK, rhoK):
        if p > pK:
            return (p - pK) * np.sqrt(AK / (p + BK))
        else:
            return 2 * cK / (gamma - 1) * ((p / pK) ** ((gamma - 1) / (2 * gamma)) - 1)

    # Solve for p_star
    f_eq = lambda p: (fK(p, pL, cL, AL, BL, rhoL)
                    + fK(p, pR, cR, AR, BR, rhoR)
                    + (uR - uL))
    p_star = brentq(f_eq, 1e-8 * pL, 10 * max(pL, pR), xtol=1e-10)
    u_star = 0.5 * (uL + uR) + 0.5 * (fK(p_star, pR, cR, AR, BR, rhoR)
                                       - fK(p_star, pL, cL, AL, BL, rhoL))

    # Densities in star regions
    rhoL_star = (rhoL * (p_star / pL + (gamma - 1) / (gamma + 1))
                 / ((gamma - 1) / (gamma + 1) * p_star / pL + 1)) if p_star > pL \
                else rhoL * (p_star / pL) ** (1 / gamma)
    rhoR_star = (rhoR * (p_star / pR + (gamma - 1) / (gamma + 1))
                 / ((gamma - 1) / (gamma + 1) * p_star / pR + 1)) if p_star > pR \
                else rhoR * (p_star / pR) ** (1 / gamma)

    cL_star = np.sqrt(gamma * p_star / rhoL_star)
    cR_star = np.sqrt(gamma * p_star / rhoR_star)

    # Wave speeds
    # Left: rarefaction (p_star < pL in Sod)
    S_HL = uL - cL                   # head of left rarefaction
    S_TL = u_star - cL_star          # tail of left rarefaction
    # Right: shock
    S_R  = uR + cR * np.sqrt((gamma + 1) / (2 * gamma) * p_star / pR
                              + (gamma - 1) / (2 * gamma))

    rho_ex = np.empty_like(x_arr)
    u_ex   = np.empty_like(x_arr)
    p_ex   = np.empty_like(x_arr)

    for i, xi in enumerate(x_arr):
        s = (xi - x0) / t   # similarity variable

        if s <= S_HL:
            rho_ex[i], u_ex[i], p_ex[i] = rhoL, uL, pL
        elif s <= S_TL:                            # inside left rarefaction
            u_ex[i]   = 2 / (gamma + 1) * (cL + (gamma - 1) / 2 * uL + s)
            c_fan     = cL + (gamma - 1) / 2 * (uL - u_ex[i])
            p_ex[i]   = pL * (c_fan / cL) ** (2 * gamma / (gamma - 1))
            rho_ex[i] = gamma * p_ex[i] / c_fan**2
        elif s <= u_star:                          # left star region
            rho_ex[i], u_ex[i], p_ex[i] = rhoL_star, u_star, p_star
        elif s <= S_R:                             # right star region
            rho_ex[i], u_ex[i], p_ex[i] = rhoR_star, u_star, p_star
        else:                                      # undisturbed right state
            rho_ex[i], u_ex[i], p_ex[i] = rhoR, uR, pR

    return rho_ex, u_ex, p_ex



def sod_exact_with_sound_speeds(x, t, x0, rhoL, uL, pL, rhoR, uR, pR,
                                gamma, cL, cR):
    """Sod reference with sound speeds supplied by the caller's EOS.

    Retains the vectorized comparison used by the development driver, including
    its root bracket and zero-time sampling. Assumes a left rarefaction and a
    right shock; for a non-ideal EOS this remains an ideal-gas approximation.
    """
    AL = 2 / ((gamma + 1) * rhoL);  BL = (gamma - 1) / (gamma + 1) * pL
    AR = 2 / ((gamma + 1) * rhoR);  BR = (gamma - 1) / (gamma + 1) * pR

    def _fK(p, pK, cK, AK, BK):
        return ((p - pK) * np.sqrt(AK / (p + BK)) if p > pK
                else 2 * cK / (gamma - 1) * ((p / pK) ** ((gamma - 1) / (2 * gamma)) - 1))

    p_star = brentq(lambda p: _fK(p, pL, cL, AL, BL)
                                  + _fK(p, pR, cR, AR, BR)
                                  + (uR - uL), 1e-6 * pL, 10 * pL)
    u_star = (0.5 * (uL + uR)
              + 0.5 * (_fK(p_star, pR, cR, AR, BR) - _fK(p_star, pL, cL, AL, BL)))
    rhoL_star = (rhoL * (p_star / pL + (gamma-1)/(gamma+1))
                 / ((gamma-1)/(gamma+1) * p_star / pL + 1)) if p_star > pL \
                else rhoL * (p_star / pL) ** (1 / gamma)
    rhoR_star = (rhoR * (p_star / pR + (gamma-1)/(gamma+1))
                 / ((gamma-1)/(gamma+1) * p_star / pR + 1)) if p_star > pR \
                else rhoR * (p_star / pR) ** (1 / gamma)
    cL_star = np.sqrt(gamma * p_star / rhoL_star)
    S_HL = uL - cL;  S_TL = u_star - cL_star
    S_R  = uR + cR * np.sqrt((gamma+1)/(2*gamma)*p_star/pR + (gamma-1)/(2*gamma))

    xi_s = (x - x0) / t if t > 0 else np.zeros_like(x)
    gm1 = gamma - 1.0;  gp1 = gamma + 1.0
    # rarefaction fan: u, p, rho as functions of xi_s
    u_fan   = 2/gp1 * (cL + gm1/2*uL + xi_s)
    c_fan   = cL + gm1/2*(uL - u_fan)
    p_fan   = pL * (c_fan/cL)**(2*gamma/gm1)
    rho_fan = gamma * p_fan / c_fan**2
    rho_ex = np.where(xi_s < S_HL, rhoL,
             np.where(xi_s < S_TL, rho_fan,
             np.where(xi_s < u_star, rhoL_star,
             np.where(xi_s < S_R,   rhoR_star, rhoR))))
    u_ex   = np.where(xi_s < S_HL, uL,
             np.where(xi_s < S_TL, u_fan,
             np.where(xi_s < S_R,  u_star, uR)))
    p_ex   = np.where(xi_s < S_HL, pL,
             np.where(xi_s < S_TL, p_fan,
             np.where(xi_s < S_R,  p_star, pR)))
    return rho_ex, u_ex, p_ex
