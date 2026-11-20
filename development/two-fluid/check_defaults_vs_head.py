"""Bit-identity of default two-fluid runs: current code vs a reference checkout (e.g. the last commit).

Run twice, with each code first on the path, then compare:
    PYTHONPATH="<checkout>/src;<checkout>" python -B check_defaults_vs_head.py write <out.npz>
    python -B check_defaults_vs_head.py compare <reference.npz> <current.npz>
States: saved NICFD nitrogen-water (shocked, K-inverse and k_free, MUSCL) and Super Moby Dick flashing
(subsonic, mass transfer and ghost-enthalpy pin) solutions of this folder. Defaults only: no run-time
inputs, no new closure options.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODE = sys.argv[1]

if MODE == "compare":
    reference, current = np.load(sys.argv[2]), np.load(sys.argv[3])
    for name in reference.files:
        identical = np.array_equal(reference[name], current[name])
        print(f"{name:45s} {'IDENTICAL' if identical else 'DIFFERENT, max ' + f'{np.max(np.abs(reference[name] - current[name])):.2e}'}")
    sys.exit(0)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp

from nozzle_hyperbolic.models.two_fluid import (
    TwoFluidModel, BoundaryConfig, ClosureConfig, CompatibilityConfig, StabilizationConfig)
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from cases.NICFD2026.scripts.case import build_case, prepare_fluids, specification

results = {}


def record(label, model, scheme, state):
    op = Operator(model, scheme)
    results[label + "/spatial"] = np.asarray(op.spatial_residual(state))
    for key, value in op.residual_contributions(state).items():
        results[f"{label}/{key}"] = np.asarray(value)
    dt = jnp.float64(op.compute_dt(state, 2.0))
    new, _ = op.step(state, dt, jnp.float64(0.0))
    results[label + "/step"] = np.asarray(new)


# 1. Nitrogen-water shocked case (setup of run.py), K-inverse and k_free
eos = dict(backend="HEOS", p_min=1e4, p_max=6e6, N_h=145, N_p=200, gradient_method="forward",
           table_dir=str(HERE / "outputs" / "fluid_tables"))
gas = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
liquid = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
sg, sl = gas.get_state(jxp.PT_INPUTS, 2e6, 295.15), liquid.get_state(jxp.PT_INPUTS, 2e6, 295.15)
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("two_fluid_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
arguments, sys.argv = sys.argv, sys.argv[:1]  # run.py reads its own optional arguments
spec.loader.exec_module(run)
sys.argv = arguments
mesh = run.mesh_for(100)
scheme = Scheme(flux="model_compatibility", reconstruction="muscl_van_albada", time_integrator="euler-implicit",
                boundary_treatment="coupled", newton_backtracking=True, newton_iterations=20)
for label, k_free in (("n2_shocked_k_inverse", False), ("n2_shocked_k_free", True)):
    W = jnp.asarray(np.load(HERE / "outputs" / f"{'k_free' if k_free else 'k_inverse'}_N100" / "shocked.npz")["full_state"])
    model = TwoFluidModel(
        mesh, gas, liquid, initial_state=W,
        boundaries=BoundaryConfig(inlet="stagnation", outlet="characteristic", H_g0=float(sg["h"]), H_l0=float(sl["h"]),
                                  s0=float(sg["s"]), dominant_phase="gas", quality=0.5, slip=1.0, back_pressure=12e5,
                                  pressure_relaxation=1.0),
        closures=ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original"),
        compatibility=CompatibilityConfig(nc_form="wb", delta_factor=1e-6, lowmach_diss=1.0, lowmach_diss_boundary=0.0,
                                          du_blend_lo=5.0, du_blend_hi=20.0, **({"k_free": True} if k_free else {})),
        stabilization=StabilizationConfig(void_strength=0.05, speed_scale="frozen", void_velocity_cutoff=None,
                                          void_velocity_width=12.0),
        virtual_mass=0.5)
    record(label, model, scheme, W)

# 2. Super Moby Dick flashing, subsonic (mass transfer, ghost-enthalpy pin), K-inverse
config = specification("single_component", "subsonic")
fluids, _ = prepare_fluids(config, "single_component")
base, flash_scheme, _, _ = build_case(config, fluids, 100, family="single_component", initialization="reference",
                                      newton_iterations=20)
W = jnp.asarray(np.load(HERE / "outputs" / "flashing" / "k_free_N100" / "subsonic.npz")["state"])
model = TwoFluidModel(base.mesh, base.fluid_g, base.fluid_l, initial_state=W, boundaries=base.boundaries,
                      closures=base.closures, compatibility=base.compatibility, stabilization=base.stabilization,
                      pins=base.pins, virtual_mass=base.virtual_mass, source_scale=base.source_scale,
                      area_source=base.area_source)
record("smd_subsonic_k_inverse", model, flash_scheme, W)

np.savez(sys.argv[2], **results)
print(f"saved {len(results)} arrays to {sys.argv[2]}")
