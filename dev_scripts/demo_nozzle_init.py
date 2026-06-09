# %%
"""
Demo: compare nozzle initializations for the Städtke Newton solver.

Shows side-by-side:
  - Simple init  (p linear, u from continuity, h = h0 - u²/2)
  - Isentropic init (branch-selected isentrope — used as analytical reference)

Change SETTINGS_FILE to test different nozzle cases.
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)

import yaml
import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp
import nozzle_hyperbolic as hyp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.fvm import (
    stadtke_puh_initialization_simple,
    stadtke_puh_initialization,
    stadtke_puh_initialization_subsonic,
)

hyp.set_plot_options(grid=False)

SETTINGS_FILE = "settings_nozzle_validation.yaml"

# ── Load settings ─────────────────────────────────────────────────────────────
with open(SETTINGS_FILE) as f:
    case_data = yaml.safe_load(f)

perfect_gas = case_data["fluid"]["perfect_gas"]
fluid_name  = case_data["fluid"]["fluid_name"]
use_lut     = case_data["fluid"].get("use_lookup_table", True)
lut_cfg     = case_data["fluid"].get("lut", {})

if perfect_gas:
    fluid = jxp.Fluid(name=fluid_name)
elif use_lut:
    fluid = jxp.FluidBicubic(
        fluid_name = fluid_name,
        backend    = lut_cfg.get("backend",  "HEOS"),
        h_min      = float(lut_cfg.get("h_min",  50e3)),
        h_max      = float(lut_cfg.get("h_max", 850e3)),
        p_min      = float(lut_cfg.get("p_min",   1e4)),
        p_max      = float(lut_cfg.get("p_max", 300e5)),
        N_h        = int(lut_cfg.get("N_h",       200)),
        N_p        = int(lut_cfg.get("N_p",       200)),
        table_dir  = lut_cfg.get("table_dir", "fluid_tables"),
    )
else:
    fluid = jxp.Fluid(name=fluid_name)

# ── Grid ──────────────────────────────────────────────────────────────────────
Nx = case_data["numerical_settings"]["Nx"]
x1 = case_data["numerical_settings"]["x1"]
x2 = case_data["numerical_settings"]["x2"]
x  = np.linspace(x1, x2, Nx)

# ── Geometry ──────────────────────────────────────────────────────────────────
geo_cfg     = case_data["geometry"].copy()
geo_fn_name = geo_cfg.pop("function")
roughness   = float(geo_cfg.pop("roughness", 0.0))
geo_fn      = getattr(_geom, geo_fn_name)
geo         = geo_fn(x, roughness=roughness, **{k: v for k, v in geo_cfg.items()})
A_nodes     = np.asarray(geo.A)

# ── Boundary conditions ───────────────────────────────────────────────────────
p_stag = float(case_data["boundary_conditions"]["p_stagnation"])
T_stag = float(case_data["boundary_conditions"]["T_stagnation"])
p_out  = float(case_data["boundary_conditions"]["p_out"])

st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
h0      = float(st_stag.hmass)

parameters = {
    "perfect_gas":   perfect_gas,
    "fluid_name":    fluid_name,
    "fluid":         fluid,
    "pT_stagnation": np.array([p_stag, T_stag]),
    "p_out":         p_out,
    "N_eq":          3,
}
if perfect_gas:
    cp, cv = hyp.fvm_solver.specific_heat2()
    parameters["gamma"] = cp / cv

# ── Run both initializations ──────────────────────────────────────────────────
print("Running simple initialization ...")
U_simple = stadtke_puh_initialization_subsonic(x, A_nodes, parameters)

print("Running isentropic initialization ...")
U_isentr = stadtke_puh_initialization(x, A_nodes, parameters)
print("Done.")

def decode(U):
    p, u, h = U[:, 0], U[:, 1], U[:, 2]
    st   = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    a    = np.asarray(st["a"],   dtype=float)
    rho  = np.asarray(st["rho"], dtype=float)
    M    = u / np.maximum(a, 1e-300)
    mdot = rho * u * A_nodes
    h0_field = h + 0.5 * u**2
    return p, u, h, a, rho, M, mdot, h0_field

p_s, u_s, h_s, a_s, rho_s, M_s, mdot_s, h0s_s = decode(U_simple)
p_i, u_i, h_i, a_i, rho_i, M_i, mdot_i, h0s_i = decode(U_isentr)

def rms_var(arr):
    return np.sqrt(np.mean((arr - arr.mean())**2)) / (abs(arr.mean()) + 1e-300)

# ── Diagnostics ───────────────────────────────────────────────────────────────
print(f"\n{'─'*62}")
print(f"  Settings : {SETTINGS_FILE}")
print(f"  p0={p_stag/1e5:.2f} bar   T0={T_stag:.0f} K   "
      f"p_out={p_out/1e5:.4f} bar  (ratio={p_out/p_stag:.4f})")
print(f"  A_inlet={A_nodes[0]:.3e} m²   A_throat={A_nodes.min():.3e} m²   "
      f"AR={A_nodes[0]/A_nodes.min():.2f}")
print(f"\n  {'':22s}  {'simple':>12s}  {'isentropic':>12s}")
print(f"  {'p  range [bar]':22s}  "
      f"{p_s.min()/1e5:.3f}–{p_s.max()/1e5:.3f}  "
      f"{p_i.min()/1e5:.3f}–{p_i.max()/1e5:.3f}")
print(f"  {'u  range [m/s]':22s}  "
      f"{u_s.min():.1f}–{u_s.max():.1f}  "
      f"{u_i.min():.1f}–{u_i.max():.1f}")
print(f"  {'h  range [kJ/kg]':22s}  "
      f"{h_s.min()/1e3:.1f}–{h_s.max()/1e3:.1f}  "
      f"{h_i.min()/1e3:.1f}–{h_i.max()/1e3:.1f}")
print(f"  {'M  range':22s}  "
      f"{M_s.min():.3f}–{M_s.max():.3f}  "
      f"{M_i.min():.3f}–{M_i.max():.3f}")
print(f"  {'h0 range [kJ/kg]':22s}  "
      f"{h0s_s.min()/1e3:.2f}–{h0s_s.max()/1e3:.2f}  "
      f"{h0s_i.min()/1e3:.2f}–{h0s_i.max()/1e3:.2f}  (should = {h0/1e3:.2f})")
print(f"  {'ṁ RMS var [%]':22s}  "
      f"{rms_var(mdot_s)*100:>12.3f}  "
      f"{rms_var(mdot_i)*100:>12.3f}")
print(f"{'─'*62}\n")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
fig.suptitle(f"Nozzle initialization — {SETTINGS_FILE}\n"
             f"p0={p_stag/1e5:.2f} bar  T0={T_stag:.0f} K  "
             f"p_out={p_out/1e5:.4f} bar  AR={A_nodes[0]/A_nodes.min():.1f}")

# Mark which strategy is the active solver init
active = case_data["numerical_settings"].get("initialization", "simple")
kw_s = dict(color='tab:blue',   lw=1.5,
            label=f'simple{"  ← active" if active=="simple" else ""}')
kw_i = dict(color='tab:orange', lw=1.5,
            label=f'isentropic{"  ← active" if active=="isentropic" else ""}')

ax = axes[0, 0]
ax.plot(x, p_s/1e5, **kw_s);  ax.plot(x, p_i/1e5, **kw_i)
ax.axhline(p_stag/1e5, color='grey', ls=':', lw=0.8, label='p₀')
ax.axhline(p_out/1e5,  color='red',  ls=':', lw=0.8, label='p_out')
ax.set_ylabel('p [bar]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Pressure')

ax = axes[0, 1]
ax.plot(x, u_s, **kw_s);  ax.plot(x, u_i, **kw_i)
ax.set_ylabel('u [m/s]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Velocity')

ax = axes[0, 2]
ax.plot(x, h_s/1e3, **kw_s);  ax.plot(x, h_i/1e3, **kw_i)
ax.axhline(h0/1e3, color='grey', ls=':', lw=0.8, label='h₀')
ax.set_ylabel('h [kJ/kg]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Static enthalpy')

ax = axes[1, 0]
ax.plot(x, M_s, **kw_s);  ax.plot(x, M_i, **kw_i)
ax.axhline(1.0, color='red', ls='--', lw=0.8, label='M=1')
ax.set_ylabel('M');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title('Mach number')

ax = axes[1, 1]
ax.plot(x, mdot_s*1e3, **kw_s);  ax.plot(x, mdot_i*1e3, **kw_i)
ax.set_ylabel('ṁ [g/s]');  ax.set_xlabel('x [m]');  ax.legend(fontsize='x-small')
ax.set_title(f'Mass flux  (RMS simple={rms_var(mdot_s)*100:.2f}%  '
             f'isentr={rms_var(mdot_i)*100:.2f}%)')

ax = axes[1, 2]
ax.plot(x, h0s_s/1e3, **kw_s);  ax.plot(x, h0s_i/1e3, **kw_i)
ax.axhline(h0/1e3, color='grey', ls='--', lw=0.8, label='h₀ exact')
ax.set_ylabel('h₀ = h + u²/2 [kJ/kg]');  ax.set_xlabel('x [m]')
ax.legend(fontsize='x-small');  ax.set_title('Total enthalpy conservation')

fig.tight_layout()
plt.savefig("nozzle_init_check.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → nozzle_init_check.png")
