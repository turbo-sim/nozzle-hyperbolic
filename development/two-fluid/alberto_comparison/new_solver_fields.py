"""main steady solve of the two-component subsonic case with run-time fields and back pressure.

Mirrors alberto_branch dev_scripts/port_comparison/e2e_new.py (old mesh layout with 200 points including
ghosts, Van Leer, same closures and boundaries). Run from the repository root:
    PYTHONPATH="<nozzle-hyperbolic>/src;<nozzle-hyperbolic>" python -B new_solver_fields.py <out.npz> [back_pressure_Pa] [fields]
Writes outputs/fields_N200.npy (Nx rows including ghosts) for the alberto_branch run; with "fields" the
interior rows are passed as beta_ai, beta_cd, beta_htc_g, beta_htc_l.
"""

import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

import nozzle_hyperbolic as hyp
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    BoundaryConfig, ClosureConfig, CompatibilityConfig, StabilizationConfig, primitives)
from cases.NICFD2026.scripts.case import reservoir_states, geometry_arrays

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
BACK_PRESSURE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6e6
USE_FIELDS = len(sys.argv) > 3 and sys.argv[3] == "fields"
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"
REF = "C:/Users/alber/projects/nozzle-hyperbolic/cases/NICFD2026/two_component/subsonic/reference.csv"

# 1. Old-layout mesh: Nx=200 points incl. ghosts, averaged face areas, A*dx volumes; fields on all points
x = np.linspace(0.0, 0.31, 200)
dx = x[1] - x[0]
A, _ = geometry_arrays(tuple(x), "two_component")
Af = 0.5 * (A[:-1] + A[1:])
mesh = Mesh(faces=0.5 * (x[:-1] + x[1:]), centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=Af,
            area_centers=A[1:-1], darea_centers=np.diff(Af) / dx, area_derivative_integrals=np.diff(Af))
xi = np.linspace(0.0, 1.0, len(x))
fields = np.column_stack((1.0 + 0.4 * np.sin(2 * np.pi * xi), 1.3 + 0.5 * np.cos(3 * np.pi * xi),
                          0.8 + 0.3 * xi, 1.2 - 0.4 * xi**2))
np.save(HERE / "outputs" / "fields_N200.npy", fields)

# 2. Fluids, reservoir and boundary data (configured back pressure 16 bar, as alberto_branch's settings)
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward", table_dir=TAB)
fg = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
fl = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
sg, sl = reservoir_states("two_component", 2e6, 295.15, 295.15, fg, fl)
bc = BoundaryConfig(inlet="stagnation", outlet="characteristic", quality=0.5, slip=1.0,
                    H_g0=sg["h"], H_l0=sl["h"], s0=sg["s"], dominant_phase="gas",
                    back_pressure=1.6e6, pressure_relaxation=1.0)
d = np.genfromtxt(REF, delimiter=",", names=True)
U0 = np.column_stack([np.interp(mesh.points, d["x"], d[k]) for k in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")])

# 3. Model, scheme and run-time parameters
model = hyp.TwoFluidModel(
    mesh, fg, fl, initial_primitives=U0, boundaries=bc,
    closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
    compatibility=CompatibilityConfig(nc_form="wb", lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                                      du_blend_lo=5.0, du_blend_hi=20.0, delta_factor=1e-6),
    stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_width=12.0),
    virtual_mass=0.5, runtime_inputs=True)
scheme = hyp.Scheme(flux="model_compatibility", reconstruction="muscl_van_leer", time_integrator="euler-implicit",
                    boundary_treatment="coupled", newton_backtracking=True, newton_iterations=20)
parameters = dict(back_pressure=BACK_PRESSURE)
if USE_FIELDS:
    parameters.update(beta_ai=fields[1:-1, 0], beta_cd=fields[1:-1, 1],
                      beta_htc_g=fields[1:-1, 2], beta_htc_l=fields[1:-1, 3])

# 4. Solve and save the primitives at points (cells and traces)
t0 = time.time()
sol = hyp.solve_single_grid(model, scheme, hyp.SteadySettings(cfl=2.0, cfl_growth=1.2, max_steps=4000),
                            parameters=parameters)
print("success", sol.success, "reason", sol.reason, "steps", len(sol.history), f"wall {time.time() - t0:.1f}s")
U = np.asarray(primitives(jnp.asarray(sol.state)))
np.savez(OUT, x=mesh.points, U=U, success=sol.success, reason=str(sol.reason), back_pressure=BACK_PRESSURE,
         fields=USE_FIELDS)
