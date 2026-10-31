"""Retained mixture correlations, independent of an equation of state.

Arguments are physical SI values. Densities and viscosities must be positive,
and quality ``x`` is vapor mass fraction. These empirical laws are available to
user models; their presence does not make the Euler models two-fluid models.
No input is silently clipped into an empirical correlation's domain.
"""

import jax.numpy as jnp


def viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, x):
    """Beattie mixture viscosity; pure-phase limits are exactly recovered."""
    void_fraction = rho_L*x/(rho_L*x+rho_v*(1.0-x))
    return mu_L*(1.0-void_fraction)*(1.0+2.5*void_fraction)+mu_v*void_fraction


def viscosity_MacAdams(mu_L, mu_v, x):
    """McAdams harmonic mixture viscosity (historical spelling preserved)."""
    return (x/mu_v+(1.0-x)/mu_L)**-1


def LM_gronnerud(rho_l, rho_v, mu_l, mu_v, x, G_l, d, g=9.81):
    """Grønnerud two-phase multiplier, requiring nonzero liquid mass flux.

    ``G_l`` is the liquid-reference mass flux; its zero-flux limit is singular.
    Invalid nonpositive Froude numbers return NaN, rather than a clipped law.
    """
    Fr_l = jnp.asarray(G_l)**2/(g*d*rho_l**2)
    safe_fr = jnp.where(Fr_l > 0, Fr_l, 1.0)
    low_fr = safe_fr**0.3+0.0055*jnp.log(1.0/safe_fr)**2
    f_fr = jnp.where(Fr_l >= 1.0, 1.0, low_fr)
    dp_dz_fr = f_fr*(x+4.0*(x**1.8-x**10*jnp.sqrt(f_fr)))
    result = 1.0+dp_dz_fr*((rho_l/rho_v)/(mu_l/mu_v)**0.25-1.0)
    return jnp.where(Fr_l > 0, result, jnp.nan)


def LM_condensation(Re, rho_L, mu_L, rho_V, mu_V, D_h, u, x, G):
    """Retained condensation multiplier and phase shear stresses.

    Returns ``(phi_L, phi_V, tau_L, tau_V)`` with Darcy factors, so that each
    reference shear is ``f_D*G_phase**2/(8*rho_phase)``. This historical formula
    requires ``Re>0``, ``u>0`` and ``0<x<1``. Its pure-phase multipliers are
    singular; invalid inputs return NaN. Definitions and transition at 2300
    match the former nozzle correlation without changing its interpretation.
    """
    Re, u, x = jnp.asarray(Re), jnp.asarray(u), jnp.asarray(x)
    valid = (Re > 0)&(u > 0)&(x > 0)&(x < 1)
    speed = jnp.where(u > 0, u, 1.0)
    quality = jnp.where((x > 0)&(x < 1), x, 0.5)
    Re_L, Re_V = rho_L*speed*D_h/mu_L, rho_V*speed*D_h/mu_V
    f_L = jnp.where(Re < 2300.0, 64.0/Re_L, 0.316/Re_L**0.25)
    f_V = jnp.where(Re < 2300.0, 64.0/Re_V, 0.184/Re_V**0.2)
    X = f_L*rho_V*(1.0-quality)**2/(rho_L*f_V*quality**2)
    phi_L, phi_V = 1.0+20.0/X+1.0/X**2, 1.0+20.0*X+X**2
    tau_L = f_L*G**2*(1.0-quality)**2/(8.0*rho_L)
    tau_V = f_V*G**2*quality**2/(8.0*rho_V)
    return tuple(jnp.where(valid, value, jnp.nan) for value in (phi_L, phi_V, tau_L, tau_V))


def f_HEM_purdue(Re):
    """Purdue homogeneous-equilibrium **Fanning** factor (Darcy is fourfold).

    The laminar law is 16/Re. At zero Reynolds number the factor is infinity;
    evaluate wall shear through a finite limiting law rather than infinity*0.
    Negative Reynolds numbers are outside the correlation and return NaN.
    """
    Re = jnp.asarray(Re)
    positive_re = jnp.where(Re > 0, Re, 1.0)
    factor = jnp.where(Re < 2000.0, 16.0/positive_re,
                       jnp.where(Re < 20000.0, 0.079*positive_re**-0.25,
                                 0.046*positive_re**-0.2))
    return jnp.where(Re > 0, factor, jnp.where(Re == 0, jnp.inf, jnp.nan))


def richardson(x, rho_L, rho_v):
    """Richardson multiplier; the pure-vapor limit is singular (infinity)."""
    x = jnp.asarray(x)
    void_fraction = rho_L*x/(rho_L*x+rho_v*(1.0-x))
    return (1.0-void_fraction)**-1.75
