"""main interior residual with the same per-cell correction fields (run-time parameters).

Run from the repository root (main first on the path):
    PYTHONPATH="<nozzle-hyperbolic>/src;<nozzle-hyperbolic>" python -B fields_residual_new.py <state_dir> <out_dir> <tests>
Mesh, traces and settings mirror alberto_branch dev_scripts/port_comparison/res_new.py (old mesh layout,
transmissive traces copied from the old ghosts). Fields come from fields_residual_old.py (fields.npy).
Writes R_new.npz (with and without fields, plus the conservative term used as scale).
"""

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    ClosureConfig, CompatibilityConfig, PhasePinConfig, StabilizationConfig)
from nozzle_hyperbolic.solvers.fvm.operator import Operator

STATE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TESTS = sys.argv[3].split(",")
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"

# 1. Old-layout mesh, fluids, transmissive traces and the fields of the old run
spec = json.loads((STATE / "spec.json").read_text())
s = np.load(STATE / "state.npz")
x, dx, A, A_face, dAdx, W = s["x"], float(s["dx"]), s["A"], s["A_face"], s["dAdx"], s["W"]
mesh = Mesh(faces=0.5 * (x[:-1] + x[1:]), centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=A_face,
            area_centers=A[1:-1], darea_centers=dAdx[1:-1], area_derivative_integrals=np.diff(A_face))
fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB)
fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)
full = np.array(W)
full[0], full[-1] = W[1], W[-2]
fields = np.load(OUT / "fields.npy")[1:-1]
params = dict(beta_ai=fields[:, 0], beta_cd=fields[:, 1], beta_htc_g=fields[:, 2], beta_htc_l=fields[:, 3])

# 2. Interior residual per test, with and without the fields
results = {}
for t in [t for t in spec["tests"] if t["name"] in TESTS]:
    cl = t["closure"]
    model = hyp.TwoFluidModel(
        mesh, fg, fl, initial_state=full,
        closures=None if cl is None else ClosureConfig(**cl),
        compatibility=CompatibilityConfig(nc_form=t["nc_form"], lowmach_diss=1.0,
                                          lowmach_diss_boundary=t.get("lowmach_diss_bc"),
                                          du_blend_lo=5.0, du_blend_hi=20.0, delta_factor=1e-6),
        stabilization=StabilizationConfig(void_strength=t["eps_void"], speed_scale="frozen",
                                          void_velocity_width=12.0,
                                          thin_phase_strength=t.get("eps_ghost", 0.0)),
        pins=PhasePinConfig(saturation_energy=t.get("pin_ghost", False)),
        virtual_mass=0.5, area_source=t["area_source"], runtime_inputs=True)
    op = Operator(model, hyp.Scheme(flux="model_compatibility", reconstruction=t["recon"],
                                    time_integrator="euler-implicit", boundary_treatment="coupled"))
    state = jnp.asarray(full)
    for label, values in (("", None), ("__fields", params if cl is not None else None)):
        R = op.physical_spatial_residual(state, params=values)
        R = model.transform_cell_equations(state, R, dt=1.0)
        results[t["name"] + label] = np.asarray(R)
    results[t["name"] + "__conservative"] = np.asarray(op.residual_contributions(state)["conservative"])
    print(t["name"], "done")
np.savez(OUT / "R_new.npz", **results)
