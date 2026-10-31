"""Annular geometry data sampled from the original smooth NURBS definitions.

Face areas and cell-center derivatives are direct smooth-model evaluations.
The integrated pressure-source coefficient is quadrature of the analytical
area derivative, not a neighboring-cell finite-difference estimate.
"""

import numpy as np

from ..numerics.mesh import Mesh


def smooth_area_derivative_integrals(geometry, faces, *, order=16):
    """Integrate the smooth analytical dA/dm over each physical cell.

    Split at all geometry knot distances. The fundamental theorem provides an
    independent audit against directly evaluated face areas, but those area
    differences are not used to calculate the derivative or its integral.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integrals = []
    for lower, upper in zip(faces[:-1], faces[1:]):
        splits = [lower, *(point for point in geometry.knot_distances
                           if lower < point < upper), upper]
        integral = 0.0
        for left, right in zip(splits[:-1], splits[1:]):
            points = left+(right-left)*(nodes+1)/2
            derivative = np.asarray(geometry.evaluate(points).dA_dm)
            integral += (right-left)/2*float(weights@derivative)
        integrals.append(integral)
    return np.asarray(integrals)


def build_annular_mesh(geometry, cells, *, faces=None):
    """Return the common mesh and native physical geometry snapshot."""
    native = (geometry.mesh(cells) if faces is None else
              geometry.mesh(m_faces=np.asarray(faces)))
    derivative_integrals = smooth_area_derivative_integrals(geometry, native.m_faces)
    exact_area_change = native.A_face[1:]-native.A_face[:-1]
    scale = max(float(np.max(np.abs(native.A_face))), 1e-30)
    if not np.allclose(derivative_integrals, exact_area_change,
                       rtol=2e-10, atol=2e-13*scale):
        raise ValueError("Smooth area-derivative quadrature failed its fundamental-theorem audit")
    mesh = Mesh(faces=native.m_faces, centers=native.m_centers,
                volumes=native.volumes, area_faces=native.A_face,
                area_centers=native.A_center, darea_centers=native.dA_dm,
                area_derivative_integrals=derivative_integrals, geometry=geometry)
    return mesh, native
