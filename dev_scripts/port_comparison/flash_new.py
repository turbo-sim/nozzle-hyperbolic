import sys, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    BoundaryConfig, ClosureConfig, CompatibilityConfig, StabilizationConfig, PhasePinConfig)
from nozzle_hyperbolic.models.two_fluid.constrained_state import primitives
from cases.NICFD2026.scripts.case import reservoir_states, geometry_arrays

out, cfl, newton = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
REF = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/single_component/supersonic_adapted/reference.csv"

# 1. Old-layout mesh: Nx=600 points incl. ghosts, averaged face areas, A*dx volumes
x = np.linspace(0.0, 1.2, 600); dx = x[1] - x[0]
A, _ = geometry_arrays(tuple(x), "single_component")
Af = 0.5 * (A[:-1] + A[1:])
mesh = Mesh(faces=0.5 * (x[:-1] + x[1:]), centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=Af,
            area_centers=A[1:-1], darea_centers=np.diff(Af) / dx, area_derivative_integrals=np.diff(Af))

# 2. Fluids (old-branch tables), reservoir and boundary data as in the NICFD case
eos = dict(fluid_name="water", backend="HEOS", p_min=1e4, p_max=1.3e7, N_h=90, N_p=90, N_p_sat=150,
           gradient_method="forward", table_dir=TAB)
fg = jxp.FluidBicubic(h_min=2.2e6, h_max=3.4e6, metastable_phase="vapor", **eos)
fl = jxp.FluidBicubic(h_min=1e5, h_max=1.8e6, metastable_phase="liquid", **eos)
sg, sl = reservoir_states("single_component", 1.2e7, 578.95, 578.95, fg, fl)
a0, slip = 1e-3, 1.0
quality = a0 * sg["rho"] * slip / (a0 * sg["rho"] * slip + (1 - a0) * sl["rho"])
bc = BoundaryConfig(inlet="stagnation", outlet="characteristic", quality=quality, slip=slip,
                    H_g0=sg["h"], H_l0=sl["h"], s0=sl["s"], dominant_phase="liquid",
                    back_pressure=0.8e5, pressure_relaxation=1.0)

# 3. Same restart profile as the old run
d = np.genfromtxt(REF, delimiter=",", names=True)
U0 = np.column_stack([np.interp(mesh.points, d["x"], d[k]) for k in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])

model = hyp.TwoFluidModel(
    mesh, fg, fl, initial_primitives=U0, boundaries=bc,
    closures=ClosureConfig(Nb=1e11, Nd=1e10, mass_transfer=True, area_law="original"),
    compatibility=CompatibilityConfig(nc_form="wb", lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                                      du_blend_lo=5.0, du_blend_hi=20.0, delta_factor=1e-6),
    stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_width=12.0,
                                      alpha_min=1e-6, thin_phase_strength=0.05),
    pins=PhasePinConfig(saturation_energy=True),
    virtual_mass=0.5)
scheme = hyp.Scheme(flux="model_compatibility", reconstruction="muscl_van_albada", time_integrator="euler-implicit",
                    boundary_treatment="coupled", newton_backtracking=True, newton_iterations=newton)
t0 = time.time()
sol = hyp.solve_single_grid(model, scheme, hyp.SteadySettings(cfl=cfl, cfl_growth=1.2, max_steps=4000))
print("success", sol.success, "reason", sol.reason, "steps", len(sol.history), f"wall {time.time()-t0:.1f}s")
print("metrics", {k: v for k, v in sol.metrics.items() if np.ndim(v) == 0})
for h in sol.history[-5:]:
    print("hist", {k: v for k, v in (h.items() if isinstance(h, dict) else vars(h).items()) if np.ndim(v) == 0})
U = np.asarray(primitives(jnp.asarray(sol.state)))
np.savez(out, x=mesh.points, U=U, success=sol.success, reason=str(sol.reason))
