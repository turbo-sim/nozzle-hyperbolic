"""Scalar conservative Burgers physics for an isolated Riemann problem.

No eigensystem, EOS, reconstruction or nonlinear solver is supplied here.
Rusanov/HLL use the shared engine and the physical bounds u_L/u_R. End states
are fixed at inflow and extrapolated at outflow; reference comparisons must
stop before the Riemann waves reach the boundaries.
"""

import jax.numpy as jnp
import numpy as np

from ...numerics.mesh import Mesh
from .analytical import burgers_riemann


class BurgersModel:
    nvar = 1
    name = "burgers"
    equation_names = ("Scalar transport",)
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, cells, *, left=1.0, right=-0.5, length=1.0, location=0.5):
        if (
            not isinstance(cells, (int, np.integer))
            or isinstance(cells, bool)
            or cells < 2
        ):
            raise ValueError("Use at least two integer cells")
        if (
            not np.all(np.isfinite([left, right, length, location]))
            or not 0 < location < length
        ):
            raise ValueError(
                "Finite Riemann data and an interior jump location are required"
            )
        self.left, self.right = float(left), float(right)
        self.location, self.length = float(location), float(length)
        faces = np.linspace(0.0, length, cells + 1)
        centers = 0.5 * (faces[:-1] + faces[1:])
        self.mesh = Mesh(
            faces=faces,
            centers=centers,
            volumes=np.diff(faces),
            area_faces=np.ones_like(faces),
        )
        scale = max(abs(left), abs(right), 1.0)
        self.variable_scales = np.array([scale])
        self.conservative_scales = np.array([scale])
        self.reference_time = length / scale
        self.global_flux_scales = np.array([scale**2])

    def profile(self, coordinate, time=0.0):
        return burgers_riemann(
            coordinate, time, left=self.left, right=self.right, location=self.location
        )

    def initial_state(self):
        return self.profile(self.mesh.points)[:, None]

    def fields(self, full):
        return {"q": full[1:-1, 0]}

    def reference(self, time=0.0):
        return {"q": self.profile(self.mesh.centers, time)}

    def cell_conserved(self, physical):
        return physical

    filter_conserved = cell_conserved

    def filter_signal_speed(self, full):
        speed = jnp.abs(full[:, 0])
        return jnp.maximum(speed[:-1], speed[1:])

    def face_conserved(self, state):
        return state

    def face_physical_flux(self, state):
        return 0.5 * state**2

    def face_speed_bounds(self, state):
        return state[..., 0], state[..., 0]

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return jnp.zeros_like(full[1:-1])

    def boundary_residual(self, full, time=0.0):
        left = full[0] - self.left if self.left > 0 else full[0] - full[1]
        right = full[-1] - self.right if self.right < 0 else full[-1] - full[-2]
        return jnp.stack((left, right)) / self.variable_scales

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )

    def wave_speed(self, physical):
        return jnp.abs(physical[..., 0])
