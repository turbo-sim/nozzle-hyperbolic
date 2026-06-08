# %%
"""
Demo script for isentropic_initialization.

Runs the initializer for any YAML case and plots the resulting
primitive variables alongside the expected analytical solution.
Lets you inspect what the solver starts from before running the full solve.
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import pathlib
import yaml
import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic as hyp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.fvm import isentropic_initialization

hyp.set_plot_options(grid=False)

# ── Settings file ────────────────────────────────────────────────────────────
# Change this line to test different cases:
SETTINGS_FILE = "settings_subsonic_pipe.yaml"
# SETTINGS_FILE = "settings_nozzle_validation.yaml"

with open(SETTINGS_FILE) as f:
    case_data = yaml.safe_load(f)

# ── Fluid ────────────────────────────────────────────────────────────────────
perfect_gas = case_data["fluid"]["perfect_gas"]
fluid_name  = case_data["fluid"]["fluid_name"]
use_lut     = case_data["fluid"].get("use_lookup_table", True)
lut_cfg     = case_data["fluid"].get("lut", {})

if perfect_gas:
    fluid = jxp.Fluid(name=fluid_name)
elif use_lut:
    fluid = jxp.FluidBicubic(
        fluid_name = fluid_name,
        backend    = lut_cfg.get("backend",   "HEOS"),
        h_min      = float(lut_cfg.get("h_min",   50e3)),
        h_max      = float(lut_cfg.get("h_max",  850e3)),
        p_min      = float(lut_cfg.get("p_min",    1e4)),
        p_max      = float(lut_cfg.get("p_max",  300e5)),
        N_h        = int(lut_cfg.get("N_h",        200)),
        N_p        = int(lut_cfg.get("N_p",        200)),
        table_dir  = lut_cfg.get("table_dir", "fluid_tables"),
    )
else:
    fluid = jxp.Fluid(name=fluid_name)

# ── Grid ─────────────────────────────────────────────────────────────────────
Nx = case_data["numerical_settings"]["Nx"]
x1 = case_data["numerical_settings"]["x1"]
x2 = case_data["numerical_settings"]["x2"]
x  = np.linspace(x1, x2, Nx)
dx = x[1] - x[0]

# ── Boundary conditions ──────────────────────────────────────────────────────
p_stagnation  = float(case_data["boundary_conditions"]["p_stagnation"])
T_stagnation  = float(case_data["boundary_conditions"]["T_stagnation"])
pT_stagnation = np.array([p_stagnation, T_stagnation])
p_out         = float(case_data["boundary_conditions"]["p_out"])

# ── Geometry ─────────────────────────────────────────────────────────────────
geo_cfg     = case_data["geometry"].copy()
geo_fn_name = geo_cfg.pop("function")
roughness   = float(geo_cfg.pop("roughness", 0.0))
geo_fn      = getattr(_geom, geo_fn_name)
geo         = geo_fn(x, roughness=roughness, **{k: v for k, v in geo_cfg.items()})
A_nodes     = np.asarray(geo.A)

# ── Parameters dict (minimal — only what initializer needs) ─────────────────
parameters = {
    "perfect_gas":   perfect_gas,
    "fluid_name":    fluid_name,
    "fluid":         fluid,
    "pT_stagnation": pT_stagnation,
    "p_out":         p_out,
    "N_eq":          3,
}
if perfect_gas:
    cp, cv = hyp.fvm_solver.specific_heat2()
    parameters["gamma"] = cp / cv

# ── Run initializer ──────────────────────────────────────────────────────────
print("Running isentropic_initialization ...")
Q0 = isentropic_initialization(x, A_nodes, parameters)
print("Done.")

# ── Decode conservative Q0 → primitive (ρ, u, p, T, M, h) ───────────────────
rho0 = Q0[0]
vel0 = Q0[1] / Q0[0]
if perfect_gas:
    cp, cv = hyp.fvm_solver.specific_heat2()
    p0   = (cp / cv - 1.0) * (Q0[2] - 0.5 * Q0[1]**2 / Q0[0])
    T0   = p0 / ((cp - cv) * rho0)
    a0   = np.sqrt((cp / cv) * p0 / rho0)
    h0_  = cp * T0
    s0_  = np.full_like(p0, np.nan)
else:
    e_int = Q0[2] / rho0 - 0.5 * vel0**2
    st    = hyp.fvm_solver.state_from_rho_u_batch(fluid, rho0, e_int)
    p0    = np.asarray(st.pressure,    dtype=float)
    T0    = np.asarray(st.T,           dtype=float)
    a0    = np.asarray(st.speed_sound, dtype=float)
    h0_   = np.asarray(st.hmass,       dtype=float)
    s0_   = np.asarray(st.entropy,     dtype=float)

Ma0  = vel0 / a0
mdot0 = rho0 * vel0 * A_nodes

# ── Print diagnostics ────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"  Settings file : {SETTINGS_FILE}")
print(f"  p_stagnation  : {p_stagnation:.3e} Pa")
print(f"  T_stagnation  : {T_stagnation:.1f} K")
print(f"  p_out         : {p_out:.3e} Pa  (p_out/p0 = {p_out/p_stagnation:.4f})")
print(f"  A_throat      : {A_nodes.min():.4e} m²   A_inlet = {A_nodes[0]:.4e} m²")
print(f"\n  Init field summary (cell-centred):")
print(f"    p   : {p0.min():.3e} … {p0.max():.3e} Pa")
print(f"    u   : {vel0.min():.1f} … {vel0.max():.1f} m/s")
print(f"    T   : {T0.min():.1f} … {T0.max():.1f} K")
print(f"    M   : {Ma0.min():.4f} … {Ma0.max():.4f}")
print(f"    ṁ   : {mdot0.min():.4e} … {mdot0.max():.4e} kg/s")
mdot_rms = np.sqrt(np.mean((mdot0 - mdot0.mean())**2)) / (abs(mdot0.mean()) + 1e-300)
print(f"    ṁ RMS variation : {mdot_rms*100:.3f} %  (0 % = perfect conservation)")
if not perfect_gas:
    s_var = s0_.max() - s0_.min()
    print(f"    s   : {s0_.min():.1f} … {s0_.max():.1f} J/(kg·K)  (Δs = {s_var:.2f})")
print(f"{'─'*60}\n")

# ── Analytical comparison (isentropic, ideal-gas approx for real fluids) ─────
# For isentropic 1D flow:   A * u * ρ = const  and  h0 = h + u²/2 = const
# At each x: M(x) from area-Mach relation, then p, T, u from isentropic.
st_stag   = fluid.get_state(jxp.PT_INPUTS, p_stagnation, T_stagnation)
h_stag    = float(st_stag.hmass)
s_stag    = float(st_stag.smass)

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
fig.suptitle(f"Initialization check — {SETTINGS_FILE}\n"
             f"p0={p_stagnation/1e5:.2f} bar, T0={T_stagnation:.0f} K, "
             f"p_out={p_out/1e5:.2f} bar")

ax = axes[0, 0]
ax.plot(x, p0 / 1e5, 'go-', ms=4, mfc='w', lw=0.8)
ax.axhline(p_stagnation / 1e5, color='k', ls='--', lw=0.8, label='p₀')
ax.axhline(p_out / 1e5,        color='r', ls='--', lw=0.8, label='p_out')
ax.set_ylabel('p [bar]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Pressure')

ax = axes[0, 1]
ax.plot(x, vel0, 'go-', ms=4, mfc='w', lw=0.8)
ax.set_ylabel('u [m/s]');  ax.set_xlabel('x [m]')
ax.set_title('Velocity')

ax = axes[0, 2]
ax.plot(x, T0, 'go-', ms=4, mfc='w', lw=0.8)
ax.axhline(T_stagnation, color='k', ls='--', lw=0.8, label='T₀')
ax.set_ylabel('T [K]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Temperature')

ax = axes[1, 0]
ax.plot(x, Ma0, 'go-', ms=4, mfc='w', lw=0.8)
ax.axhline(1.0, color='r', ls='--', lw=0.8, label='M=1')
ax.set_ylabel('M');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Mach number')

ax = axes[1, 1]
ax.plot(x, mdot0 * 1e3, 'go-', ms=4, mfc='w', lw=0.8)
ax.set_ylabel('ṁ [g/s]');  ax.set_xlabel('x [m]')
ax.set_title(f'Mass flux  (RMS var = {mdot_rms*100:.3f} %)')

ax = axes[1, 2]
ax.plot(x, A_nodes * 1e4, 'k-', lw=1.2)
ax.set_ylabel('A [cm²]');  ax.set_xlabel('x [m]')
ax.set_title('Geometry')

fig.tight_layout()
plt.savefig("init_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → init_check.png")
