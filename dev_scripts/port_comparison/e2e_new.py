import sys, time, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    BoundaryConfig, ClosureConfig, CompatibilityConfig, StabilizationConfig)
from nozzle_hyperbolic.models.two_fluid.constrained_state import primitives
import nozzle_hyperbolic.geometry as _g
from cases.NICFD2026.scripts.case import reservoir_states, geometry_arrays

recon, out = sys.argv[1], sys.argv[2]
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
REF = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/two_component/subsonic/reference.csv"

# 1. Old-layout mesh: Nx=200 points incl. ghosts, averaged face areas, A*dx volumes
x = np.linspace(0.0, 0.31, 200); dx = x[1] - x[0]
A, _ = geometry_arrays(tuple(x), "two_component")
Af = 0.5 * (A[:-1] + A[1:])
mesh = Mesh(faces=0.5 * (x[:-1] + x[1:]), centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=Af,
            area_centers=A[1:-1], darea_centers=np.diff(Af) / dx, area_derivative_integrals=np.diff(Af))

# 2. Fluids, reservoir and boundary data (as in the NICFD case)
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward", table_dir=TAB)
fg = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
fl = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
sg, sl = reservoir_states("two_component", 2e6, 295.15, 295.15, fg, fl)
bc = BoundaryConfig(inlet="stagnation", outlet="characteristic", quality=0.5, slip=1.0,
                    H_g0=sg["h"], H_l0=sl["h"], s0=sg["s"], dominant_phase="gas",
                    back_pressure=1.6e6, pressure_relaxation=1.0)

# 3. Same restart profile as the old run, interpolated at cells and traces
d = np.genfromtxt(REF, delimiter=",", names=True)
U0 = np.column_stack([np.interp(mesh.points, d["x"], d[k]) for k in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])

model = hyp.TwoFluidModel(
    mesh, fg, fl, initial_primitives=U0, boundaries=bc,
    closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
    compatibility=CompatibilityConfig(nc_form="wb", lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                                      du_blend_lo=5.0, du_blend_hi=20.0, delta_factor=1e-6),
    stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_width=12.0),
    virtual_mass=0.5)
scheme = hyp.Scheme(flux="model_compatibility", reconstruction=recon, time_integrator="euler-implicit",
                    boundary_treatment="coupled", newton_backtracking=True, newton_iterations=20)
t0 = time.time()
sol = hyp.solve_single_grid(model, scheme, hyp.SteadySettings(cfl=2.0, cfl_growth=1.2, max_steps=4000))
print("success", sol.success, "reason", sol.reason, "steps", len(sol.history), f"wall {time.time()-t0:.1f}s")
print("metrics", {k: v for k, v in sol.metrics.items() if np.ndim(v) == 0})
U = np.asarray(primitives(jnp.asarray(sol.state)))
np.savez(out, x=mesh.points, U=U, success=sol.success, reason=str(sol.reason))
