"""Check TwoFluidModel k_free/smooth against the four two_fluid_test_v2 prototype arms.

Run from the repository root: PYTHONPATH=. poetry run python -B development/two-fluid/check_k_free_model.py
States are the saved K-inverse solutions in outputs/k_inverse_N100 (100 cells, NICFD setup of run.py).
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("two_fluid_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import (TwoFluidModel, BoundaryConfig, ClosureConfig, CompatibilityConfig,
                                                StabilizationConfig)
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from development.two_fluid_test_v2.scripts.numerics import adapt

ARMS = {"baseline": (False, False), "inverse_old_sonic": (True, False), "sonic_only": (False, True), "both": (True, True)}
PRESSURES = {"subsonic": 16e5, "shocked": 12e5, "adapted": 98600.0}

# 1. Fluids and mesh (same as run.py)
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward",
           table_dir=str(HERE / "outputs" / "fluid_tables"))
gas = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
liquid = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
sg, sl = gas.get_state(jxp.PT_INPUTS, 2e6, 295.15), liquid.get_state(jxp.PT_INPUTS, 2e6, 295.15)
mesh = run.mesh_for(100)

# 2. Residual contributions of every arm: src switches vs prototype AblationModel
worst = {}
for regime, pressure in PRESSURES.items():
    W = jnp.asarray(np.load(HERE / "outputs" / "k_inverse_N100" / f"{regime}.npz")["full_state"])
    for reconstruction in ("constant", "muscl_van_albada"):
        scheme = Scheme(flux="model_compatibility", reconstruction=reconstruction, time_integrator="euler-implicit",
                        boundary_treatment="coupled", newton_backtracking=True, newton_iterations=20)
        for arm, (k_free, smooth) in ARMS.items():
            kwargs = dict(
                initial_state=W,
                boundaries=BoundaryConfig(inlet="stagnation", outlet="characteristic", H_g0=float(sg["h"]),
                                          H_l0=float(sl["h"]), s0=float(sg["s"]), dominant_phase="gas", quality=0.5,
                                          slip=1.0, back_pressure=pressure, pressure_relaxation=1.0),
                closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
                stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_cutoff=None,
                                                  void_velocity_width=12.0),
                virtual_mass=0.5,
            )
            base_cfg = dict(nc_form="wb", delta_factor=1e-6, lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                            du_blend_lo=5.0, du_blend_hi=20.0)
            new = TwoFluidModel(mesh, gas, liquid, compatibility=CompatibilityConfig(**base_cfg, k_free=k_free,
                                                                                     smooth=smooth), **kwargs)
            old = adapt(TwoFluidModel(mesh, gas, liquid, compatibility=CompatibilityConfig(**base_cfg), **kwargs),
                        scheme, arm)
            c_new, c_old = Operator(new, scheme).residual_contributions(W), Operator(old, scheme).residual_contributions(W)
            row = []
            for key in ("conservative", "stabilization", "nonconservative", "source"):
                diff = float(jnp.max(jnp.abs(c_new[key] - c_old[key])))
                rel = diff / max(float(jnp.max(jnp.abs(c_old[key]))), 1e-300)
                worst[arm] = max(worst.get(arm, 0.0), rel)
                row.append(f"{key[:5]} {rel:.1e}")
            print(f"{regime:9s} {reconstruction:17s} {arm:18s} mode={new.nonconservative_mode:15s} " + "  ".join(row))
print("worst relative difference per arm:", {k: f"{v:.1e}" for k, v in worst.items()})

# 3. Colored vs dense Jacobian with k_free + smooth under MUSCL (dependency radius of the path mode)
W = jnp.asarray(np.load(HERE / "outputs" / "k_inverse_N100" / "shocked.npz")["full_state"])
scheme = Scheme(flux="model_compatibility", reconstruction="muscl_van_albada", time_integrator="euler-implicit",
                boundary_treatment="coupled", newton_backtracking=True, newton_iterations=20)
model = TwoFluidModel(mesh, gas, liquid, initial_state=W, compatibility=CompatibilityConfig(
    nc_form="wb", delta_factor=1e-6, lowmach_diss=1.0, lowmach_diss_boundary=0.0, k_free=True, smooth=True),
    closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
    stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_cutoff=None),
    boundaries=BoundaryConfig(inlet="stagnation", outlet="characteristic", H_g0=float(sg["h"]), H_l0=float(sl["h"]),
                              s0=float(sg["s"]), dominant_phase="gas", quality=0.5, slip=1.0, back_pressure=12e5))
op = Operator(model, scheme)
args = (W.ravel(), W.ravel(), 1e-6, 0.0)
dense, colored = np.asarray(op.dense_jacobian(*args)), np.asarray(op.jacobian(*args))
scale = np.tile(model.variable_scales, W.shape[0])
print(f"k_free+smooth MUSCL colored vs dense Jacobian: max rel diff "
      f"{np.max(np.abs((colored - dense) * scale)) / np.max(np.abs(dense * scale)):.2e}, finite {np.all(np.isfinite(dense))}")

# 4. Validation: k_free with mass_upwind_cutoff is refused
try:
    TwoFluidModel(mesh, gas, liquid, initial_state=W, compatibility=CompatibilityConfig(k_free=True),
                  stabilization=StabilizationConfig(mass_upwind_cutoff=1e-3))
    print("k_free + mass_upwind_cutoff: NOT refused")
except ValueError as error:
    print(f"k_free + mass_upwind_cutoff refused: {error}")
