"""Signed scalar advection: no EOS or Euler-shaped dummy variables."""

import jax.numpy as jnp
import numpy as np

from ...numerics.mesh import Mesh


class AdvectionModel:
    """u_t + c u_x = 0 with exact inflow and coupled extrapolated outflow."""

    nvar = 1
    equation_names = ("Scalar transport",)
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, cells, speed=1.0, length=1.0):
        if (
            not isinstance(cells, (int, np.integer))
            or isinstance(cells, bool)
            or cells < 2
        ):
            raise ValueError("Use at least two integer cells")
        self.speed = float(speed)
        self.length = float(length)
        if not np.isfinite(self.speed) or not np.isfinite(length) or length <= 0:
            raise ValueError("Advection needs finite speed and positive length")
        faces = np.linspace(0.0, length, int(cells) + 1)
        centers = 0.5 * (faces[:-1] + faces[1:])
        self.mesh = Mesh(
            faces=faces,
            centers=centers,
            volumes=np.diff(faces),
            area_faces=np.ones_like(faces),
            geometry={"kind": "unit-area straight domain"},
        )
        self.variable_scales = jnp.ones(1)
        self.conservative_scales = jnp.ones(1)
        speed_scale = max(abs(self.speed), 1.0)
        self.reference_time = length / speed_scale
        self.global_flux_scales = jnp.array([speed_scale])

    def profile(self, coordinate, time=0.0):
        phase = 2 * jnp.pi * (jnp.asarray(coordinate) - self.speed * time) / self.length
        return jnp.sin(phase) + 0.25 * jnp.cos(2 * phase)

    def reference(self, time=0.0):
        return {"q": self.profile(self.mesh.centers, time)}

    def exact(self, time=0.0):
        return self.reference(time)

    def initial_state(self):
        return self.profile(self.mesh.points)[:, None]

    def cell_conserved(self, physical):
        return physical

    filter_conserved = cell_conserved

    def filter_signal_speed(self, full):
        return jnp.full((self.mesh.n_cells + 1,), abs(self.speed))

    def face_conserved(self, face_state):
        return face_state

    def face_physical_flux(self, face_state):
        return self.speed * face_state

    def face_speed_bounds(self, face_state):
        speed = jnp.full(face_state.shape[:-1], self.speed)
        return speed, speed

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        return jnp.zeros_like(full[1:-1])

    def boundary_residual(self, full, time=0.0):
        if self.speed > 0:
            left = full[0] - self.profile(self.mesh.faces[0], time)
            right = full[-1] - full[-2]
        elif self.speed < 0:
            left = full[0] - full[1]
            right = full[-1] - self.profile(self.mesh.faces[-1], time)
        else:
            left, right = full[0] - full[1], full[-1] - full[-2]
        return jnp.stack([left, right])

    def admissible(self, full, left, right):
        return (
            jnp.all(jnp.isfinite(full))
            & jnp.all(jnp.isfinite(left))
            & jnp.all(jnp.isfinite(right))
        )

    def wave_speed(self, physical):
        return jnp.full(physical.shape[0], abs(self.speed))

    def fields(self, full):
        return {"q": full[1:-1, 0]}
