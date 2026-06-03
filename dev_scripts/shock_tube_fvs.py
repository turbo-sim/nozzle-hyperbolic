# %%
"""
Sod shock tube validation for the FVS flux.

Tests _jax_F_fvs_stadtke independently of:
  - nozzle geometry  (constant area A=1)
  - Newton implicit  (uses explicit Euler + MUSCL Van Leer)
  - nozzle BCs       (reflective walls; no inlet/outlet forcing)

Reference: Toro (2009), Chapter 4, Test 1.
Initial condition (dimensional):
  Left:  rho=1.0 kg/m^3,  p=1e5 Pa,  u=0
  Right: rho=0.125 kg/m^3, p=1e4 Pa, u=0
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jaxprop as jxp
from nozzle_hyperbolic.fvm_solver.fvm import _jax_F_fvs_stadtke
import nozzle_hyperbolic as hyp
hyp.set_plot_options(grid=False)

# ============================================================
# 1.  Gas — air via FluidBicubic (JIT-compatible, same as main solver)
# ============================================================
gamma = 1.4
R     = 287.058
cp    = gamma * R / (gamma - 1)

# Table bounds: generous margins around Sod states (TL=348K, TR=278K)
fluid = jxp.FluidBicubic(
    fluid_name = "Air",
    backend    = "HEOS",
    h_min      = 100e3,   # J/kg  →  T ~ 100 K
    h_max      = 500e3,   # J/kg  →  T ~ 500 K
    p_min      = 1e3,     # Pa    →  well below pR=1e4
    p_max      = 5e5,     # Pa    →  above pL=1e5
    N_h        = 100,
    N_p        = 60,
    table_dir  = "fluid_tables",
)

# ============================================================
# 2.  Sod initial condition
# ============================================================
rhoL, pL, uL = 1.000, 1.0e5, 1.0   # uL=1 m/s avoids K-matrix singularity at u=0
rhoR, pR, uR = 0.125, 1.0e4, 1.0   # (Mach ~0.003, negligible effect on solution)
TL = pL / (rhoL * R);  hL = cp * TL
TR = pR / (rhoR * R);  hR = cp * TR

# ============================================================
# 3.  Grid
# ============================================================
N   = 800
x   = np.linspace(0.0, 1.0, N)
dx  = x[1] - x[0]
x0  = 0.5            # diaphragm
# Show solution at t* = 0.25 in Toro's units; c_L = sqrt(gamma*pL/rhoL)
cL    = np.sqrt(gamma * pL / rhoL)
t_end = 0.2 / cL     # ≈ 5.3e-4 s — developed wave structure, no wall reflections

# ============================================================
# 4.  Initial (p, u, h) field
# ============================================================
U0          = np.zeros((N, 3))
U0[x <= x0] = [pL, uL, hL]
U0[x >  x0] = [pR, uR, hR]
U_ph = jnp.asarray(U0)

# ============================================================
# 5.  Ideal-gas conversions  (exact, no EOS call needed)
# ============================================================
def puh_to_Q(U):
    """Primitive (N,3) → conservative (3,N)  — ideal gas."""
    p, u, h = U[:, 0], U[:, 1], U[:, 2]
    rho = gamma * p / ((gamma - 1) * h)
    e   = p / ((gamma - 1) * rho)
    return jnp.stack([rho, rho * u, rho * (e + 0.5 * u**2)], axis=0)

def Q_to_puh(Q):
    """Conservative (3,N) → primitive (N,3)  — ideal gas."""
    rho = Q[0];  u = Q[1] / Q[0]
    p   = (gamma - 1) * (Q[2] - 0.5 * Q[1]**2 / Q[0])
    h   = gamma / (gamma - 1) * p / rho
    return jnp.stack([p, u, h], axis=1)

# ============================================================
# 6.  JIT-compiled explicit Euler step with FVS flux
# ============================================================
CFL_factor = 0.3

@jax.jit
def fvs_step(U_ph):
    # Zero-gradient BCs: copy interior state to ghost cells.
    # Reflective BCs (negate u) would give u_avg=0 at the wall face → K singular.
    # Zero-gradient works here because waves never reach the walls during t_end.
    U = U_ph.at[0].set(U_ph[1])
    U = U.at[-1].set(U_ph[-2])

    # First-order (constant) reconstruction.
    # MUSCL + FVS is not TVD for strong discontinuities (10:1 pressure ratio here)
    # and produces oscillations. First-order gives monotone results.
    # Add MUSCL only after the base flux is validated.
    U_L = U[:-1]
    U_R = U[1:]

    # FVS interface flux  (N-1, 3)  — A=1 everywhere, no area factor
    F_hat = _jax_F_fvs_stadtke(U_L, U_R, fluid)

    # CFL dt  (ideal gas: a = sqrt((gamma-1)*h))
    a  = jnp.sqrt((gamma - 1) * jnp.maximum(U[:, 2], 1.0))
    dt = CFL_factor * dx / jnp.max(jnp.abs(U[:, 1]) + a)

    # Conservative update for interior cells  (flux divergence, A=1 -> no weighting)
    Q     = puh_to_Q(U)
    dF    = F_hat[1:] - F_hat[:-1]   # (N-2, 3):  F_{i+1/2} - F_{i-1/2}
    Q_new = Q.at[:, 1:-1].add(-dt / dx * dF.T)   # dQ/dt = -(1/dx)(F_{i+1/2} - F_{i-1/2})

    return Q_to_puh(Q_new), dt

# ============================================================
# 7.  Time integration
# ============================================================
print(f"Running FVS Sod tube: N={N}, t_end={t_end:.3e} s ...")
t = 0.0;  n = 0
while t < t_end:
    U_ph, dt_j = fvs_step(U_ph)
    dt          = float(dt_j)
    t          += min(dt, t_end - t)
    n          += 1

print(f"Done: {n} steps, t={t:.4e} s")
U_final = np.asarray(U_ph)

# ============================================================
# 8.  Exact Sod solution  (Toro 2009, §4.3)
# ============================================================
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

rho_ex, u_ex, p_ex = sod_exact(x, t, x0,
                                rhoL, uL, pL, rhoR, uR, pR, gamma)

# ============================================================
# 9.  Extract numerical results
# ============================================================
p_num   = U_final[:, 0]
u_num   = U_final[:, 1]
h_num   = U_final[:, 2]
rho_num = gamma * p_num / ((gamma - 1) * h_num)   # ideal gas

# ============================================================
# 10. Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"Sod shock tube — FVS flux,  N={N},  t={t:.3e} s", fontsize=11)

pairs = [
    (axes[0], r"$\rho$ [kg/m³]", rho_num, rho_ex),
    (axes[1], r"$u$ [m/s]",      u_num,   u_ex),
    (axes[2], r"$p$ [Pa]",       p_num,   p_ex),
]
for ax, ylabel, num, ex in pairs:
    ax.plot(x, ex,  "k-",  lw=1.5, label="Exact")
    ax.plot(x, num, "r--", lw=1.0, label="FVS")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================
# 11. L1 errors
# ============================================================
print("\n  L1 errors (interior cells, vs exact):")
print(f"  density  : {np.mean(np.abs(rho_num[1:-1] - rho_ex[1:-1])):.4e} kg/m³")
print(f"  velocity : {np.mean(np.abs(u_num[1:-1]   - u_ex[1:-1])):.4e} m/s")
print(f"  pressure : {np.mean(np.abs(p_num[1:-1]   - p_ex[1:-1])):.4e} Pa")
