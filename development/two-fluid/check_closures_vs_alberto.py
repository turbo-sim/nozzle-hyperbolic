"""Compare the two-fluid closure (two_fluid_closure) of main with alberto_branch.

Run from the repository root: poetry run python -B development/two-fluid/check_closures_vs_alberto.py
alberto_branch is read from ../nozzle-hyperbolic_alberto_branch and loaded under its own package
name, so both codes run with the same JAX/JAXProp. Fluids are perfect-gas Air with synthetic saturation
data (kernel check, not a physical case). States have finite slip and lie inside alberto's table clip
bounds, where main's two deliberate port differences (no table clipping, analytic zero-slip drag) give
the same values. main's beta_ai is alberto's ai_scale.
"""

import importlib.util
import sys
import types
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid.matrices import phase_props
from nozzle_hyperbolic.models.two_fluid.sources import two_fluid_closure

ALBERTO = Path(__file__).resolve().parents[3] / "nozzle-hyperbolic_alberto_branch" / "nozzle_hyperbolic" / "fvm_solver"

# 1. Load alberto_branch two_fluid/sources.py as package "alberto_tf" (its source_two_fluid is a top-level module)
package = types.ModuleType("alberto_tf")
package.__path__ = [str(ALBERTO / "two_fluid")]
sys.modules["alberto_tf"] = package
for name in ("matrices", "sources"):
    spec = importlib.util.spec_from_file_location(f"alberto_tf.{name}", ALBERTO / "two_fluid" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
alberto_closure = sys.modules["alberto_tf.sources"].two_fluid_closure
alberto_props = sys.modules["alberto_tf.matrices"].phase_props

# 2. Option sets: defaults, every new option, and earlier options combined with them
fluid = jxp.FluidPerfectGas(name="Air")
new = dict(nu_l_int=30.0, drag_scale_bubbly=0.4, drag_scale_droplet=2.5, beta_cd=1.7, beta_htc_g=0.6, beta_htc_l=1.3)
cases = [dict()]
for mass_transfer in (True, False):
    for regime_htc in (False, True):
        for blend in ("linear", "smoothstep"):
            cases.append(dict(mass_transfer=mass_transfer, regime_htc=regime_htc, regime_blend=blend, **new))
cases += [dict(htc_liquid_bubbly="sum", **new), dict(htc_liquid_bubbly="sum", htc_liquid_droplet="sum", regime_htc=True, **new),
          dict(htc_liquid_bubbly="max", htc_liquid_droplet="jakob", regime_htc=True, **new),
          dict(drag_bubbly="ishii_zuber", drag_droplet="ishii_zuber", **new),
          dict(area_law="original", Nb=1e11, Nd=1e11, beta_ai=0.5, **new), dict(area_law="parabola", beta_ai=2.0, **new),
          dict(area_law="segments", beta_ai=0.8, smooth_slip=True, **new), dict(beta_ai=1.6, nu_g_int=60.0, **new)]

# 3. Evaluate both codes at bubbly, transition and droplet states with finite slip
worst = 0.0
for options in cases:
    alberto_options = {("ai_scale" if k == "beta_ai" else k): v for k, v in options.items()}
    for alpha in (0.15, 0.3, 0.5, 0.85):
        for slip in (12.0, -7.0, 0.8):
            U = jnp.array((2e5, 20.0 + slip, 20.0, alpha, 3.1e5, 2.8e5))
            mine = two_fluid_closure(U, phase_props(fluid, U[0], U[4]), phase_props(fluid, U[0], U[5]),
                                     fluid, fluid, 260.0, 5e5, 2.5e5, **options)
            theirs = alberto_closure(U, alberto_props(fluid, U[0], U[4]), alberto_props(fluid, U[0], U[5]),
                                     fluid, fluid, 260.0, 5e5, 2.5e5, **alberto_options)
            a, b = np.array(mine, dtype=float), np.array(theirs, dtype=float)
            rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
            worst = max(worst, rel)
            if rel > 1e-12:
                print(f"DIFFERENCE {rel:.2e} options {options} alpha {alpha} slip {slip}")
print(f"{len(cases)} option sets x 4 void fractions x 3 slips: largest relative difference main vs alberto_branch {worst:.2e}")
