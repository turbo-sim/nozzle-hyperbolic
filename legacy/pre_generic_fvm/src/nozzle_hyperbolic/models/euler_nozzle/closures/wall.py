"""Unchanged definitions relocated from fvm_solver.fvm."""
import math
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# JAX-compiled real-gas step
# ---------------------------------------------------------------------------

def _haaland_jnp(Re, roughness, D_h):
    eps_D = roughness / D_h
    Re_s  = jnp.maximum(Re, 1.0)
    term  = (eps_D / 3.7) ** 1.11 + 6.9 / Re_s
    f_t   = (-1.81 * jnp.log10(jnp.maximum(term, 1e-300))) ** (-2)
    return jnp.where(Re < 2300.0, 64.0 / Re_s, f_t)


# ---------------------------------------------------------------------------
# Friction and two-phase correlations
# ---------------------------------------------------------------------------

def Haaland(Re, epsilon, D):
    if Re <= 0.0:
        return 0.0
    term = (epsilon / (3.7 * D))**1.11 + 6.9 / Re
    if term <= 0.0:
        return 0.0
    return (-1.81 * math.log10(term))**-2


def viscosity_Beattie(mu_L, mu_v, rho_L, rho_v, x):
    y   = rho_L * x / (rho_L * x + rho_v * (1.0 - x))
    return mu_L * (1.0 - y) * (1.0 + 2.5 * y) + mu_v * y


def viscosity_MacAdams(mu_L, mu_v, x):
    return (x / mu_v + (1.0 - x) / mu_L) ** -1


def LM_gronnerud(rho_l, rho_v, mu_l, mu_v, x, G_l, d, g=9.81):
    Fr_l = G_l**2 / (g * d * rho_l**2)
    f_fr = 1.0 if Fr_l >= 1.0 else Fr_l**0.3 + 0.0055 * (math.log(1.0 / Fr_l))**2
    dp_dz_fr = f_fr * (x + 4.0 * (x**1.8 - x**10 * f_fr**0.5))
    return 1.0 + dp_dz_fr * ((rho_l / rho_v) / (mu_l / mu_v)**0.25 - 1.0)


def LM_condensation(Re, rho_L, mu_L, rho_V, mu_V, D_h, u, x, G):
    Re_L = rho_L * u * D_h / mu_L
    Re_V = rho_V * u * D_h / mu_V
    if Re < 2300:
        f_L = 64.0 / Re_L
        f_V = 64.0 / Re_V
    else:
        f_L = 0.316 / Re_L**0.25
        f_V = 0.184 / Re_V**0.2
    X     = (f_L * rho_V * (1.0 - x)**2) / (rho_L * f_V * x**2)
    phi_L = 1.0 + 20.0 / X + 1.0 / X**2
    phi_V = 1.0 + 20.0 * X + X**2
    tau_L = f_L * G**2 * (1.0 - x)**2 / (8.0 * rho_L)
    tau_V = f_V * G**2 * x**2 / (8.0 * rho_V)
    return phi_L, phi_V, tau_L, tau_V


def f_HEM_purdue(Re):
    if Re < 2000:
        return 16.0 / Re
    elif Re < 20000:
        return 0.079 * Re**-0.25
    return 0.046 * Re**-0.2


def richardson(x, rho_L, rho_v):
    gamma = rho_L * x / (rho_L * x + rho_v * (1.0 - x))
    return (1.0 - gamma)**-1.75
