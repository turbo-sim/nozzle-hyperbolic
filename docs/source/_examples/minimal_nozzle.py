"""Small steady air-nozzle solve used by the getting-started page."""

import jax

jax.config.update("jax_enable_x64", True)

import jaxprop as jxp
import matplotlib.pyplot as plt

from nozzle_hyperbolic import NozzleModel, Scheme, SteadySettings, solve_single_grid
from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry

geometry = SymmetricNozzleGeometry(
    length=0.2, inlet_area=3.0e-4, throat_area=1.5e-4
)
model = NozzleModel(
    geometry.build_mesh(cells=24),
    jxp.FluidPerfectGas("Air"),
    p0=2.0e5,
    T0=300.0,
    back_pressure=1.9e5,
)
scheme = Scheme(flux="hll", reconstruction="constant")
solution = solve_single_grid(model, scheme, SteadySettings())
print(f"success={solution.success}, reason={solution.reason}")
if not solution.success:
    raise RuntimeError(f"Nozzle solve failed: {solution.reason}")

fields = model.fields(solution.state)
fig, ax = plt.subplots()
ax.plot(model.mesh.centers, fields["p"] / 1.0e5, "o-")
ax.set(xlabel="Axial coordinate [m]", ylabel="Pressure [bar]")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("minimal_nozzle.png", dpi=180)
plt.close(fig)
