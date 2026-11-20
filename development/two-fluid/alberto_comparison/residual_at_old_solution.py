"""Evaluate main's interior equations at alberto_branch's converged steady states.

Run from the repository root (main first on the path):
    PYTHONPATH="<nozzle-hyperbolic>/src;<nozzle-hyperbolic>" python -B residual_at_old_solution.py
Same model/mesh as new_solver_fields.py. The old ghost cells are inserted as traces; cells next to the
traces are reported separately because the two codes close the boundaries differently. A state that also
solves main's interior equations (relative to the flux term) means the steady difference comes from the
boundary formulations, not from the interior equations or the run-time inputs.
"""

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

OUT = Path(__file__).resolve().parent / "outputs"
TAB = "C:/Users/alber/projects/nozzle-hyperbolic_alberto_branch/dev_scripts/fluid_tables"

# 1. Model and mesh exactly as in new_solver_fields.py
x = np.linspace(0.0, 0.31, 200)
dx = x[1] - x[0]
A, _ = geometry_arrays(tuple(x), "two_component")
Af = 0.5 * (A[:-1] + A[1:])
mesh = Mesh(faces=0.5 * (x[:-1] + x[1:]), centers=x[1:-1], volumes=A[1:-1] * dx, area_faces=Af,
            area_centers=A[1:-1], darea_centers=np.diff(Af) / dx, area_derivative_integrals=np.diff(Af))
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward", table_dir=TAB)
fg = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
fl = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
sg, sl = reservoir_states("two_component", 2e6, 295.15, 295.15, fg, fl)
bc = BoundaryConfig(inlet="stagnation", outlet="characteristic", quality=0.5, slip=1.0,
                    H_g0=sg["h"], H_l0=sl["h"], s0=sg["s"], dominant_phase="gas",
                    back_pressure=1.6e6, pressure_relaxation=1.0)
fields = np.load(OUT / "fields_N200.npy")[1:-1]

# 2. Interior residual at each converged state, alberto_branch and main
for case, back, use_fields in (("base", 1.6e6, False), ("varied", 1.5e6, True)):
    params = dict(back_pressure=back)
    if use_fields:
        params.update(beta_ai=fields[:, 0], beta_cd=fields[:, 1], beta_htc_g=fields[:, 2], beta_htc_l=fields[:, 3])
    for label, path, key in (("alberto_branch", OUT / f"old_{case}_20000.npz", "W"), ("main", OUT / f"new_{case}.npz", "U")):
        data = np.load(path)
        if key == "W":
            state = jnp.asarray(data["W"])
        else:
            state = hyp.models.two_fluid.from_primitives(jnp.asarray(data["U"]), fg, fl)
        model = hyp.TwoFluidModel(
            mesh, fg, fl, initial_state=state, boundaries=bc,
            closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
            compatibility=CompatibilityConfig(nc_form="wb", lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                                              du_blend_lo=5.0, du_blend_hi=20.0, delta_factor=1e-6),
            stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_width=12.0),
            virtual_mass=0.5, runtime_inputs=True)
        op = hyp.solvers.fvm.Operator(model, hyp.Scheme(flux="model_compatibility", reconstruction="muscl_van_leer",
                                                          time_integrator="euler-implicit", boundary_treatment="coupled"))
        R = np.asarray(op.physical_spatial_residual(state, params=params))
        scale = np.max(np.abs(np.asarray(op.residual_contributions(state, params=params)["conservative"])), axis=0)
        rel = np.abs(R) / scale
        print(f"{case:6s} {label:15s} max |R|/flux: interior (cells 3..N-4) {rel[3:-3].max():.1e}, "
              f"first 3 cells {rel[:3].max():.1e}, last 3 cells {rel[-3:].max():.1e}")
