"""Quick diagnostic: print Mach number profile of the simple initialization."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.fvm import (
    stadtke_puh_initialization,
    stadtke_puh_initialization_subsonic,
)

SETTINGS_FILE = "settings_nozzle_validation.yaml"

with open(SETTINGS_FILE) as f:
    case_data = yaml.safe_load(f)

lut_cfg = case_data["fluid"]["lut"]
fluid = jxp.FluidBicubic(
    fluid_name = case_data["fluid"]["fluid_name"],
    backend    = lut_cfg.get("backend", "HEOS"),
    h_min      = float(lut_cfg["h_min"]), h_max = float(lut_cfg["h_max"]),
    p_min      = float(lut_cfg["p_min"]), p_max = float(lut_cfg["p_max"]),
    N_h        = int(lut_cfg["N_h"]),     N_p   = int(lut_cfg["N_p"]),
    table_dir  = lut_cfg["table_dir"])

Nx  = case_data["numerical_settings"]["Nx"]
x   = np.linspace(0.0, float(case_data["numerical_settings"]["x2"]), Nx)
geo_cfg  = case_data["geometry"].copy()
geo_fn   = getattr(_geom, geo_cfg.pop("function"))
geo      = geo_fn(x, roughness=float(geo_cfg.pop("roughness", 0.0)),
                  **{k: float(v) for k, v in geo_cfg.items()})
A_nodes  = np.asarray(geo.A)

p_stag = float(case_data["boundary_conditions"]["p_stagnation"])
T_stag = float(case_data["boundary_conditions"]["T_stagnation"])
p_out  = float(case_data["boundary_conditions"]["p_out"])
parameters = {
    "pT_stagnation": np.array([p_stag, T_stag]),
    "p_out": p_out, "fluid": fluid, "fluid_name": "Air",
    "N_eq": 3, "perfect_gas": False,
}

print(f"p_stag = {p_stag/1e5:.3f} bar   p_out = {p_out/1e5:.4f} bar   "
      f"ratio = {p_out/p_stag:.4f}")

U0 = stadtke_puh_initialization_subsonic(x, A_nodes, parameters)

p, u, h = U0[:, 0], U0[:, 1], U0[:, 2]
st  = fluid.get_state(jxp.HmassP_INPUTS, h, p)
a   = np.asarray(st["a"],   dtype=float)
rho = np.asarray(st["rho"], dtype=float)
M    = u / a
mdot = rho * u * A_nodes
i_thr = int(np.argmin(A_nodes))

print(f"p     : {p.min()/1e5:.4f} – {p.max()/1e5:.4f} bar")
print(f"u     : {u.min():.2f} – {u.max():.2f} m/s")
print(f"M     : {M.min():.4f} – {M.max():.4f}  (max should be < 0.9)")
print(f"M at throat (cell {i_thr:3d}): {M[i_thr]:.4f}")
print(f"M at outlet (cell  -2): {M[-2]:.4f}")
print(f"h0 - h - u²/2 max residual: {np.max(np.abs(h + 0.5*u**2 - (h[0] + 0.5*u[0]**2))):.2f} J/kg")
print(f"mdot : min={mdot.min():.4f}  max={mdot.max():.4f} kg/s  "
      f"(RMS var = {np.sqrt(np.mean((mdot-mdot.mean())**2))/(abs(mdot.mean())+1e-300)*100:.2f} %)")

# Print profile every 10 cells
print(f"\n{'cell':>5}  {'x':>6}  {'A':>8}  {'p[bar]':>8}  {'u[m/s]':>8}  "
      f"{'a[m/s]':>8}  {'M':>7}  {'mdot':>8}")
step = max(1, Nx // 15)
for i in range(0, Nx, step):
    print(f"{i:5d}  {x[i]:6.4f}  {A_nodes[i]:8.4f}  {p[i]/1e5:8.4f}  "
          f"{u[i]:8.2f}  {a[i]:8.2f}  {M[i]:7.4f}  {mdot[i]:8.4f}")
