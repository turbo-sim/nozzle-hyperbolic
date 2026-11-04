import json, sys, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    ClosureConfig, CompatibilityConfig, StabilizationConfig)
from nozzle_hyperbolic.solvers.fvm.operator import Operator

OUT = sys.argv[1]
spec = json.load(open(f"{OUT}/spec.json"))
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
s = np.load(f"{OUT}/state.npz")
x, dx, A, A_face, dAdx, W = s["x"], float(s["dx"]), s["A"], s["A_face"], s["dAdx"], s["W"]

# 1. Mesh replicating the old layout: old interior cells, old averaged face areas, A*dx volumes
faces = 0.5 * (x[:-1] + x[1:])
mesh = Mesh(faces=faces, centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=A_face,
            area_centers=A[1:-1], darea_centers=dAdx[1:-1],
            area_derivative_integrals=np.diff(A_face))

# 2. Fluids (same tables as the old run)
fg = jxp.FluidBicubic(**spec["eos"]["gas"], table_dir=TAB)
fl = jxp.FluidBicubic(**spec["eos"]["liquid"], table_dir=TAB)

# 3. Traces = old transmissive ghosts (copies of the first/last interior cell)
full = np.array(W)
full[0], full[-1] = W[1], W[-2]

results = {}
for t in spec["tests"]:
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
        pins=hyp.models.two_fluid.PhasePinConfig(saturation_energy=t.get("pin_ghost", False)),
        virtual_mass=0.5, area_source=t["area_source"])
    op = Operator(model, hyp.Scheme(flux="model_compatibility", reconstruction=t["recon"],
                                    time_integrator="euler-implicit", boundary_treatment="coupled"))
    R = op.physical_spatial_residual(jnp.asarray(full))
    R = model.transform_cell_equations(jnp.asarray(full), R, dt=1.0)
    parts = op.residual_contributions(jnp.asarray(full))
    results[t["name"]] = np.asarray(R)
    for k, v in parts.items():
        results[t["name"] + "__" + k] = np.asarray(v)
    print(t["name"], "done", float(jnp.max(jnp.abs(R))))
np.savez(f"{OUT}/R_new.npz", **results)
