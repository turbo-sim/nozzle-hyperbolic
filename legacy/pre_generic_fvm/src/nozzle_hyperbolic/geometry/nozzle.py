"""Smooth nozzle/tube geometry evaluated independently of a computational mesh.

Face areas and derivatives come directly from the physical geometry. Neither
uses neighboring cell samples. The parabolic nozzle is the established physical
verification shape, with an analytic antiderivative for cell volumes.
"""

from dataclasses import dataclass

import numpy as np

from nozzle_hyperbolic.geometry.duct import get_symmetric_nozzle


@dataclass(frozen=True)
class SymmetricNozzleGeometry:
    length: float = 0.2
    inlet_area: float = 3.0e-4
    throat_area: float = 1.5e-4

    def sample(self, coordinate):
        """Return exact smooth-model area and derivative at supplied positions."""
        x = np.asarray(coordinate, dtype=float)
        value = get_symmetric_nozzle(x, L=self.length, A_inlet=self.inlet_area,
                                    A_throat=self.throat_area)
        return dict(area=np.asarray(value.A), area_derivative=np.asarray(value.dAdx))

    def area_integral(self, coordinate):
        x = np.asarray(coordinate, dtype=float)
        difference = self.inlet_area-self.throat_area
        return (self.inlet_area*x-2*difference*x*x/self.length
                +4*difference*x*x*x/(3*self.length*self.length))

    def cell_volumes(self, faces):
        """Integrate the smooth area exactly over each physical cell."""
        values = self.area_integral(faces)
        return np.diff(values)

    def build_mesh(self, cells):
        from ..numerics.mesh import Mesh
        faces=np.linspace(0.0,self.length,int(cells)+1)
        centers=0.5*(faces[:-1]+faces[1:])
        sampled=self.sample(centers)
        return Mesh(faces=faces,centers=centers,volumes=self.cell_volumes(faces),
                    area_faces=self.sample(faces)["area"],area_centers=sampled["area"],
                    darea_centers=sampled["area_derivative"],
                    area_derivative_integrals=sampled["area_derivative"]*np.diff(faces),
                    geometry=self)


@dataclass(frozen=True)
class ConstantAreaGeometry:
    length: float = 1.0
    area: float = 1.0

    def sample(self, coordinate):
        x = np.asarray(coordinate, dtype=float)
        return dict(area=np.full_like(x, self.area), area_derivative=np.zeros_like(x))

    def cell_volumes(self, faces):
        return self.area*np.diff(np.asarray(faces, dtype=float))

    def build_mesh(self,cells):
        from ..numerics.mesh import Mesh
        faces=np.linspace(0.0,self.length,int(cells)+1)
        centers=0.5*(faces[:-1]+faces[1:])
        return Mesh(faces=faces,centers=centers,volumes=self.cell_volumes(faces),
                    area_faces=np.full_like(faces,self.area),area_centers=np.full_like(centers,self.area),
                    darea_centers=np.zeros_like(centers),area_derivative_integrals=np.zeros_like(centers),
                    geometry=self)
