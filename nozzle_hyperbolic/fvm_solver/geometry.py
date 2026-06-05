"""
Geometry definitions for the 1D FVM solver.

Each function returns a Geometry object containing:
  A       — cell-centre cross-sectional areas  (N,)  [m²]
  dAdx    — analytical dA/dx                   (N,)  [m]
  D_h     — hydraulic diameter                 (N,)  [m]
  roughness — wall roughness                   scalar [m]

All functions accept (x, roughness=0.0, **kwargs) and return Geometry.
x must be a 1-D NumPy array of cell-centre positions.
"""

import numpy as np
from typing import NamedTuple


class Geometry(NamedTuple):
    A:         np.ndarray  # cell-centre areas  (N,)  [m²]
    dAdx:      np.ndarray  # dA/dx              (N,)  [m]
    D_h:       np.ndarray  # hydraulic diameter (N,)  [m]
    roughness: float       # wall roughness           [m]


def get_pipe(x, roughness=0.0, radius=0.005):
    """Constant-area circular pipe."""
    x   = np.asarray(x, dtype=float)
    N   = len(x)
    A   = np.full(N, np.pi * radius**2)
    dAdx = np.zeros(N)
    D_h  = np.full(N, 2.0 * radius)
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_shock_tube(x, roughness=0.0, area=1.0):
    """Constant-area shock tube (circular equivalent diameter)."""
    x   = np.asarray(x, dtype=float)
    N   = len(x)
    A   = np.full(N, float(area))
    dAdx = np.zeros(N)
    D_h  = np.full(N, 2.0 * np.sqrt(float(area) / np.pi))
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_symmetric_nozzle(x, roughness=0.0, L=0.2, A_inlet=0.30, A_throat=0.15):
    """
    Symmetric parabolic converging-diverging nozzle.

    Area profile:  A(x) = A_inlet - 4*(A_inlet - A_throat) * (x/L) * (1 - x/L)
    Derivative:   dA/dx = (4*(A_inlet - A_throat) / L) * (2*x/L - 1)
    """
    x      = np.asarray(x, dtype=float)
    xi     = x / L
    A      = A_inlet - 4.0 * (A_inlet - A_throat) * xi * (1.0 - xi)
    dAdx   = (4.0 * (A_inlet - A_throat) / L) * (2.0 * xi - 1.0)
    D_h    = 2.0 * np.sqrt(A / np.pi)
    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))


def get_lettieri_nozzle(x, roughness=0.0, file_path="Lettieri_nozzle.xlsx",
                         sheet_name="Sheet1", nozzle_type="axisymmetric",
                         width=6.48e-3):
    """
    Nozzle geometry from Excel spline fit (Lettieri et al.).

    nozzle_type : 'axisymmetric' or 'planar'
    width       : channel depth [m] (planar only)
    """
    import nozzle_hyperbolic as hyp
    x = np.asarray(x, dtype=float)

    _, _, spline = hyp.fit_nozzle_shape(file_path, sheet_name, degree=10)

    x_min = -0.03201
    r = np.array([spline(xi) if xi >= x_min else 0.006365 for xi in x])

    nozzle_type = str(nozzle_type).lower()
    if nozzle_type == "axisymmetric":
        A   = np.pi * r**2
        D_h = 2.0 * r
    elif nozzle_type == "planar":
        width = float(width)
        A   = 2.0 * r * width
        D_h = 4.0 * A / (2.0 * (width + 2.0 * r))
    else:
        raise ValueError(f"Unknown nozzle_type '{nozzle_type}'. "
                         f"Choose 'axisymmetric' or 'planar'.")

    # dAdx from central differences on the spline-derived A
    dx   = x[1] - x[0]
    dAdx = np.gradient(A, dx)

    return Geometry(A=A, dAdx=dAdx, D_h=D_h, roughness=float(roughness))
