"""
Diagnostic: trace first N outer steps of the Städtke Newton solver
and print Mach profile + outlet BC state at each step.
Goal: find at which step the solver jumps to the supersonic solution.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as _geom
from nozzle_hyperbolic.fvm_solver.fvm import (
    stadtke_puh_initialization_subsonic,
    make_jax_stadtke_step,
)

SETTINGS_FILE   = "settings_nozzle_validation.yaml"
MAX_TRACE_STEPS = 20    # only trace this many steps

with open(SETTINGS_FILE) as f:
    case_data = yaml.safe_load(f)

lut_cfg = case_data["fluid"]["lut"]
fluid = jxp.FluidBicubic(
    fluid_name = case_data["fluid"]["fluid_name"],
    backend    = lut_cfg.get("backend", "HEOS"),
    h_min=float(lut_cfg["h_min"]), h_max=float(lut_cfg["h_max"]),
    p_min=float(lut_cfg["p_min"]), p_max=float(lut_cfg["p_max"]),
    N_h=int(lut_cfg["N_h"]),       N_p=int(lut_cfg["N_p"]),
    table_dir=lut_cfg["table_dir"])

ns    = case_data["numerical_settings"]
Nx    = ns["Nx"]
CFL   = float(ns["CFL"])
x     = np.linspace(0.0, float(ns["x2"]), Nx)

geo_cfg = case_data["geometry"].copy()
geo_fn  = getattr(_geom, geo_cfg.pop("function"))
geo     = geo_fn(x, roughness=float(geo_cfg.pop("roughness", 0.0)),
                 **{k: float(v) for k, v in geo_cfg.items()})
A_nodes = np.asarray(geo.A)
A_face  = 0.5 * (A_nodes[:-1] + A_nodes[1:])
dx      = x[1] - x[0]

p_stag = float(case_data["boundary_conditions"]["p_stagnation"])
T_stag = float(case_data["boundary_conditions"]["T_stagnation"])
p_out  = float(case_data["boundary_conditions"]["p_out"])

st_stag = fluid.get_state(jxp.PT_INPUTS, p_stag, T_stag)
h0 = float(st_stag.hmass);  s0 = float(st_stag.smass)

params = {"pT_stagnation": np.array([p_stag, T_stag]), "p_out": p_out,
          "fluid": fluid, "fluid_name": "Air", "N_eq": 3, "perfect_gas": False}

U0 = stadtke_puh_initialization_subsonic(x, A_nodes, params)

dAdx_nodes = np.gradient(A_nodes, x)
_step, _compute_dt = make_jax_stadtke_step(
    fluid, A_nodes, A_face, dAdx_nodes,
    np.full(Nx, 2*np.sqrt(A_nodes/np.pi)),   # D_h
    0.0, dx, CFL, p_out, h0, s0,
    wall_friction = case_data["numerical_settings"].get("wall_friction", False),
    recon         = ns.get("reconstruction", "muscl_van_leer"),
    delta_factor  = float(ns.get("delta_factor", 1e-6)),
    newton_iter   = int(ns.get("newton_iter", 50)),
)

print("Compiling ...", flush=True)
U = jnp.asarray(U0, dtype=jnp.float64)
dt_j, _ = _compute_dt(U)
_U_w, _ = _step(U, dt_j)
jax.block_until_ready(_U_w)
print("Done. Tracing steps ...\n")

i_thr = int(np.argmin(A_nodes))

def mach_profile(U_arr):
    p, u, h = np.asarray(U_arr[:,0]), np.asarray(U_arr[:,1]), np.asarray(U_arr[:,2])
    a = np.asarray(fluid.get_state(jxp.HmassP_INPUTS, h, p)["a"], dtype=float)
    return u / a

def summarise(step_n, U_arr, dt_val, ssr_val):
    M  = mach_profile(U_arr)
    p  = np.asarray(U_arr[:, 0])
    M_thr = M[i_thr]
    M_out = M[-2]     # last interior cell (index used by outlet BC)
    M_max = M[1:-1].max()
    M_idx = int(np.argmax(M[1:-1])) + 1
    p_out_ghost = float(U_arr[-1, 0])
    bc_active   = "p_out imposed" if M_out < 1.0 else "EXTRAPOLATED (supersonic)"
    print(f"Step {step_n:3d}  dt={dt_val:.2e}  SSR={ssr_val:.2e}"
          f"  M_thr={M_thr:.4f}  M_out(interior)={M_out:.4f}"
          f"  M_max={M_max:.4f}@cell{M_idx}"
          f"  p_ghost={p_out_ghost/1e5:.4f}bar  BC:{bc_active}")

print(f"{'step':>4}  {'dt':>10}  {'SSR':>10}  {'M_thr':>7}  "
      f"{'M_out(int)':>11}  {'M_max':>7}  {'p_ghost':>10}  BC")
print("-"*90)
summarise(0, np.asarray(U), float(dt_j), float('nan'))

DT_GROW = 1.20
dt = float(dt_j)
for n in range(1, MAX_TRACE_STEPS + 1):
    dt_max, a_j = _compute_dt(U)
    dt = min(dt * DT_GROW, float(dt_max))

    U_new, ssr = _step(U, jnp.float64(dt))
    jax.block_until_ready(U_new)

    summarise(n, np.asarray(U_new), dt, float(ssr))
    U = U_new

    M = mach_profile(np.asarray(U_new))
    if M[1:-1].max() > 1.5:
        print(f"\n*** Solver went supersonic at step {n}. M_max = {M[1:-1].max():.3f} ***")
        break
