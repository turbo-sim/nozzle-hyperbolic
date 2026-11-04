import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic.fvm_solver.geometry as geom
from nozzle_hyperbolic.fvm_solver.two_fluid.solver import make_two_fluid_step, W_from_U_h_batch
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure, D_VE_algebric, D_VE_algebric_two_component, D_VE_area)
from typing import NamedTuple

OUT = sys.argv[1]
spec = json.load(open(f"{OUT}/spec.json"))
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"

# 1. State and geometry on the old layout (ghost cells at x1, x2)
data = np.genfromtxt(spec["reference"], delimiter=",", names=True)
x = data["x"]; dx = float(x[1] - x[0])
U_h = np.column_stack([data[k] for k in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])
geo = getattr(geom, spec["geometry"])(x, roughness=0.0)
A = np.asarray(geo.A, float); dAdx = np.array(geo.dAdx, float)
A_face = 0.5 * (A[:-1] + A[1:])
dAdx[1:-1] = (A_face[1:] - A_face[:-1]) / dx

# 2. Fluids
fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB)
fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)
W = W_from_U_h_batch(jnp.asarray(U_h), fg, fl)
if spec.get("ghost_copy"):
    W = W.at[0].set(W[1]).at[-1].set(W[-2])
np.savez(f"{OUT}/state.npz", x=x, dx=dx, A=A, A_face=A_face, dAdx=dAdx, U_h=U_h, W=np.asarray(W))

class AreaClosure(NamedTuple):
    dlnA_dx: jnp.ndarray

# 3. One residual per test configuration
results = {}
for t in spec["tests"]:
    cl = t["closure"]
    def sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l):
        D = jnp.zeros(6)
        if cl is not None:
            c = two_fluid_closure(u_h, eos_g, eos_l, fg, fl,
                                  T_sat=sat_g["temperature"], h_sat_g=sat_g["enthalpy"],
                                  h_sat_l=sat_l["enthalpy"], dlnA_dx=dlnA, **cl)
            asm = D_VE_algebric if cl["mass_transfer"] else D_VE_algebric_two_component
            D = D + asm(u_h, cq, c, eos_g, eos_l, 0.5)
        if t["area_source"]:
            D = D + D_VE_area(u_h, cq, AreaClosure(dlnA_dx=dlnA), eos_g, eos_l)
        return D
    ret = make_two_fluid_step(
        fg, fl, A, A_face, dAdx, dx, k=0.5, delta_factor=1e-6, recon=t["recon"],
        sources_fn=sources_fn, lowmach_diss=1.0, lowmach_diss_bc=t.get("lowmach_diss_bc"),
        du_blend_lo=5.0, du_blend_hi=20.0, nc_form=t["nc_form"],
        eps_void=t["eps_void"], eps_void_scale="frozen", eps_void_uwidth=12.0,
        eps_ghost=t.get("eps_ghost", 0.0), pin_ghost=t.get("pin_ghost", False),
        use_saturation=bool(cl and cl["mass_transfer"]) or t.get("pin_ghost", False),
        va_eps_mode="magnitude", va_mean_stop_gradient=True,
        bc_inlet="transmissive", bc_outlet="transmissive", return_debug=True)
    residual = jax.jit(ret[2])
    Wf = jnp.asarray(W)
    R = residual(Wf[1:-1].ravel(), Wf.ravel(), 1.0).reshape(-1, 6)
    results[t["name"]] = np.asarray(R)
    print(t["name"], "done", float(jnp.max(jnp.abs(R))))
np.savez(f"{OUT}/R_old.npz", **results)
