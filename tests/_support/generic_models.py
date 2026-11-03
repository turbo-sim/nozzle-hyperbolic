"""Independent user-defined models exercise the shared numerical contracts."""

import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Mesh


def _mesh(count=8):
    faces = np.linspace(0.0, 1.0, count + 1) ** 1.2
    return Mesh(
        faces, (faces[:-1] + faces[1:]) / 2.0, np.diff(faces), np.ones(count + 1)
    )


class _UserBalance:
    """A non-Euler two-equation law with nonlinear storage and a wide source.

    This class intentionally has no EOS or characteristic methods. Its
    artificial source exercises the declaration of a radius-three stencil.
    """

    nvar = 2
    equation_names = ("first_content", "second_content")
    conservative_scales = np.asarray([1.0, 2.0])
    variable_scales = np.asarray([1.0, 2.0])
    global_flux_scales = np.asarray([1.0, 2.0])
    reference_time = 0.7
    source_dependency_radius = 3
    boundary_dependency_radius = 3
    boundary_coupling = "local"

    def __init__(self):
        self.mesh = _mesh()

    def cell_conserved(self, physical):
        return physical + 0.03 * physical**3

    face_conserved = cell_conserved

    def face_physical_flux(self, physical):
        return self.cell_conserved(physical) * jnp.asarray([0.7, -0.4])

    def face_speed_bounds(self, physical):
        return (jnp.full(physical.shape[:-1], -0.4), jnp.full(physical.shape[:-1], 0.7))

    def flux_to_conserved_frame(self, flux):
        return flux

    def integrated_source(self, full):
        indices = jnp.arange(1, self.mesh.n_cells + 1)
        left = full[jnp.maximum(indices - 3, 0)]
        right = full[jnp.minimum(indices + 3, self.mesh.n_cells + 1)]
        return (
            0.04
            * jnp.asarray(self.mesh.volumes)[:, None]
            * (left - 2.0 * full[1:-1] + right)
        )

    def boundary_residual(self, full, time):
        return jnp.stack(
            (full[0] ** 2 - 0.2 * full[3] - 1.0 - time, full[-1] - 0.3 * full[-4] - 0.8)
        )

    def wave_speed(self, physical):
        return jnp.full(physical.shape[:-1], 0.7)

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full)) & jnp.all(full > 0)

    def initial_state(self):
        return jnp.linspace(1.0, 1.5, (self.mesh.n_cells + 2) * self.nvar).reshape(
            -1, 2
        )


class _UserAdvection:
    nvar = 1
    equation_names = ("advected_content",)
    conservative_scales = variable_scales = global_flux_scales = np.ones(1)
    reference_time = 1.0
    source_dependency_radius = 0
    boundary_dependency_radius = 1
    boundary_coupling = "local"

    def __init__(self, speed, count=16):
        self.mesh, self.speed = _mesh(count), speed

    def cell_conserved(self, physical):
        return physical

    face_conserved = cell_conserved
    flux_to_conserved_frame = cell_conserved

    def face_physical_flux(self, physical):
        return self.speed * physical

    def face_speed_bounds(self, physical):
        speed = jnp.full(physical.shape[:-1], self.speed)
        return speed, speed

    def wave_speed(self, physical):
        return jnp.full(physical.shape[:-1], abs(self.speed))

    def integrated_source(self, full):
        return jnp.zeros_like(full[1:-1])

    @staticmethod
    def exact(x, time, speed):
        return 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * (x - speed * time))

    def boundary_residual(self, full, time):
        if self.speed > 0:
            return jnp.stack(
                (full[0] - self.exact(0.0, time, self.speed), full[-1] - full[-2])
            )
        return jnp.stack(
            (full[0] - full[1], full[-1] - self.exact(1.0, time, self.speed))
        )

    def admissible(self, full, left, right):
        return jnp.all(jnp.isfinite(full))

    def initial_state(self):
        return self.exact(jnp.asarray(self.mesh.points), 0.0, self.speed)[:, None]


class _UserFilteredAdvection(_UserAdvection):
    filter_conserved = _UserAdvection.cell_conserved

    def filter_signal_speed(self, full):
        return jnp.full((self.mesh.n_cells + 1,), abs(self.speed))
