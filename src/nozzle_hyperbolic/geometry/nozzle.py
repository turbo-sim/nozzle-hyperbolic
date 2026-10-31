"""Smooth nozzle/tube geometry evaluated independently of a computational mesh.

Face areas and derivatives come directly from the physical geometry. Neither
uses neighboring cell samples. The parabolic nozzle is the established physical
verification shape, with an analytic antiderivative for cell volumes.
"""

from dataclasses import dataclass

import numpy as np

from ..numerics.mesh import Mesh


def _faces(length, cells, faces):
    if faces is None:
        if (
            not isinstance(cells, (int, np.integer))
            or isinstance(cells, bool)
            or cells < 2
        ):
            raise ValueError("Use at least two integer cells")
        return np.linspace(0.0, length, cells + 1)
    values = np.asarray(faces, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 3
        or not np.all(np.isfinite(values))
        or not np.all(np.diff(values) > 0)
        or values[0] != 0.0
        or values[-1] != length
    ):
        raise ValueError(
            "faces must increase strictly from zero to the physical length"
        )
    return values


@dataclass(frozen=True)
class SymmetricNozzleGeometry:
    length: float = 0.2
    inlet_area: float = 3.0e-4
    throat_area: float = 1.5e-4

    def __post_init__(self):
        if (
            not np.all(np.isfinite([self.length, self.inlet_area, self.throat_area]))
            or min(self.length, self.inlet_area, self.throat_area) <= 0
            or self.throat_area > self.inlet_area
        ):
            raise ValueError(
                "Nozzle length/areas must be positive, with throat_area <= inlet_area"
            )

    def sample(self, coordinate):
        """Return exact smooth-model area and derivative at supplied positions."""
        x = np.asarray(coordinate, dtype=float)
        fraction = x / self.length
        difference = self.inlet_area - self.throat_area
        area = self.inlet_area - 4 * difference * fraction * (1 - fraction)
        derivative = 4 * difference / self.length * (2 * fraction - 1)
        return dict(
            area=area,
            area_derivative=derivative,
            hydraulic_diameter=2 * np.sqrt(area / np.pi),
        )

    def area_integral(self, coordinate):
        x = np.asarray(coordinate, dtype=float)
        difference = self.inlet_area - self.throat_area
        return (
            self.inlet_area * x
            - 2 * difference * x * x / self.length
            + 4 * difference * x * x * x / (3 * self.length * self.length)
        )

    def cell_volumes(self, faces):
        """Integrate the smooth area exactly over each physical cell."""
        values = self.area_integral(faces)
        return np.diff(values)

    def build_mesh(self, cells=None, *, faces=None):
        faces = _faces(self.length, cells, faces)
        centers = 0.5 * (faces[:-1] + faces[1:])
        sampled = self.sample(centers)
        return Mesh(
            faces=faces,
            centers=centers,
            volumes=self.cell_volumes(faces),
            area_faces=self.sample(faces)["area"],
            area_centers=sampled["area"],
            darea_centers=sampled["area_derivative"],
            area_derivative_integrals=sampled["area_derivative"] * np.diff(faces),
            geometry=self,
        )


@dataclass(frozen=True)
class ConstantAreaGeometry:
    length: float = 1.0
    area: float = 1.0

    def __post_init__(self):
        if (
            not np.all(np.isfinite([self.length, self.area]))
            or min(self.length, self.area) <= 0
        ):
            raise ValueError("Duct length and area must be positive and finite")

    def sample(self, coordinate):
        x = np.asarray(coordinate, dtype=float)
        return dict(
            area=np.full_like(x, self.area),
            area_derivative=np.zeros_like(x),
            hydraulic_diameter=np.full_like(x, 2 * np.sqrt(self.area / np.pi)),
        )

    def cell_volumes(self, faces):
        return self.area * np.diff(np.asarray(faces, dtype=float))

    def build_mesh(self, cells=None, *, faces=None):
        faces = _faces(self.length, cells, faces)
        centers = 0.5 * (faces[:-1] + faces[1:])
        return Mesh(
            faces=faces,
            centers=centers,
            volumes=self.cell_volumes(faces),
            area_faces=np.full_like(faces, self.area),
            area_centers=np.full_like(centers, self.area),
            darea_centers=np.zeros_like(centers),
            area_derivative_integrals=np.zeros_like(centers),
            geometry=self,
        )
