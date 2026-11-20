"""alberto_branch interior residual with per-cell correction fields (reference side).

Run with alberto_branch first on the path (see README.md in this folder):
    PYTHONPATH="<alberto_branch>;<alberto_branch>/nozzle_hyperbolic/fvm_solver" python -B fields_residual_old.py <state_dir> <out_dir>
<state_dir> is an alberto_branch port_comparison folder (spec.json, state.npz). Fields are the runtime
``ai_scale`` argument of the alberto_branch step with four columns (beta_ai, beta_cd, beta_htc_g,
beta_htc_l), passed by sources_fn to two_fluid_closure. Writes R_old.npz (with and without fields).
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.fvm_solver.two_fluid.solver import make_two_fluid_step
from nozzle_hyperbolic.fvm_solver.two_fluid.sources import (
    two_fluid_closure, D_VE_algebric, D_VE_algebric_two_component, D_VE_area)

STATE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TESTS = sys.argv[3].split(",")
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
OUT.mkdir(parents=True, exist_ok=True)


class AreaClosure(NamedTuple):
    dlnA_dx: jnp.ndarray


# 1. Saved state and old-layout geometry, fluids, correction fields (ghost rows included)
spec = json.loads((STATE / "spec.json").read_text())
s = np.load(STATE / "state.npz")
A, A_face, dAdx, dx, W = s["A"], s["A_face"], s["dAdx"], float(s["dx"]), jnp.asarray(s["W"])
fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB)
fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)
xi = np.linspace(0.0, 1.0, len(A))
fields = np.column_stack((1.0 + 0.4 * np.sin(2 * np.pi * xi), 1.3 + 0.5 * np.cos(3 * np.pi * xi),
                          0.8 + 0.3 * xi, 1.2 - 0.4 * xi**2))
np.save(OUT / "fields.npy", fields)

# 2. Interior residual per test, with and without the fields
results = {}
for t in [t for t in spec["tests"] if t["name"] in TESTS]:
    cl = t["closure"]

    def sources_fn(u_h, eos_g, eos_l, cq, dlnA, sat_g, sat_l, factor=None):
        D = jnp.zeros(6)
        if cl is not None:
            extra = {} if factor is None else dict(ai_scale=factor[0], beta_cd=factor[1],
                                                   beta_htc_g=factor[2], beta_htc_l=factor[3])
            c = two_fluid_closure(u_h, eos_g, eos_l, fg, fl, T_sat=sat_g["temperature"],
                                  h_sat_g=sat_g["enthalpy"], h_sat_l=sat_l["enthalpy"],
                                  dlnA_dx=dlnA, **cl, **extra)
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
    plain = residual(W[1:-1].ravel(), W.ravel(), 1.0).reshape(-1, 6)
    corrected = residual(W[1:-1].ravel(), W.ravel(), 1.0, None, jnp.asarray(fields)).reshape(-1, 6)
    results[t["name"]] = np.asarray(plain)
    results[t["name"] + "__fields"] = np.asarray(corrected)
    print(t["name"], "max |R|", float(jnp.max(jnp.abs(plain))), "with fields", float(jnp.max(jnp.abs(corrected))))
np.savez(OUT / "R_old.npz", **results)
