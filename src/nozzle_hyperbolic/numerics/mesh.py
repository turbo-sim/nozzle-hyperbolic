"""One physical-cell/face layout, independent of equation count and physics."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mesh:
    """Fixed one-dimensional finite-volume geometry with read-only arrays.

    Parameters
    ----------
    faces : array_like, shape (N + 1,)
        Strictly increasing physical face coordinates, with at least two cells.
    centers : array_like, shape (N,)
        Cell coordinates strictly inside their corresponding face intervals.
    volumes : array_like, shape (N,)
        Positive physical control-volume measures, including cross section.
    area_faces : array_like, shape (N + 1,)
        Positive face measures applied once to unweighted numerical fluxes.
    area_centers : array_like, shape (N,), optional
        Positive physical cross sections used by model-specific geometry laws.
    darea_centers : array_like, shape (N,), optional
        Physical area derivatives at cell centers.
    area_derivative_integrals : array_like, shape (N,), optional
        Integrated physical area derivatives for geometric source assembly.
    geometry : object, optional
        Opaque model geometry metadata; not interpreted by the shared engine.

    Notes
    -----
    Coordinate and measure arrays are copied to finite host float arrays and
    marked read-only at construction. Geometry preprocessing is host-side;
    differentiation through mesh construction is not provided.
    """

    faces: np.ndarray
    centers: np.ndarray
    volumes: np.ndarray
    area_faces: np.ndarray
    # Optional physical-geometry metadata. Reconstruction/assembly use only
    # coordinates, volumes and face measures; scalar laws need no dummy area
    # derivatives or source integrals.
    area_centers: np.ndarray | None = None
    darea_centers: np.ndarray | None = None
    area_derivative_integrals: np.ndarray | None = None
    geometry: Any = None

    def __post_init__(self):
        arrays = (
            "faces",
            "centers",
            "volumes",
            "area_faces",
            "area_centers",
            "darea_centers",
            "area_derivative_integrals",
        )
        for name in arrays:
            original = getattr(self, name)
            if original is None and name in arrays[4:]:
                continue
            value = np.array(original, dtype=float, copy=True)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        n = len(self.centers)
        if n < 2 or len(self.faces) != n + 1 or np.any(np.diff(self.faces) <= 0):
            raise ValueError("Use at least two physical cells and increasing faces")
        if np.any(self.centers <= self.faces[:-1]) or np.any(
            self.centers >= self.faces[1:]
        ):
            raise ValueError("Every center must lie inside its physical faces")
        for name in arrays[2:]:
            expected = n + 1 if name == "area_faces" else n
            value = getattr(self, name)
            if value is not None and len(value) != expected:
                raise ValueError(f"Wrong {name} shape")
        if any(
            np.any(getattr(self, name) <= 0)
            for name in ("volumes", "area_faces", "area_centers")
            if getattr(self, name) is not None
        ):
            raise ValueError("Volumes and areas must be positive")

    @property
    def n_cells(self):
        """int: Number of physical cells, excluding endpoint traces."""
        return len(self.centers)

    @property
    def widths(self):
        """numpy.ndarray: Face-to-face cell widths, shape ``(N,)``."""
        return np.diff(self.faces)

    @property
    def points(self):
        """numpy.ndarray: Endpoint/center coordinates, shape ``(N + 2,)``."""
        return np.r_[self.faces[0], self.centers, self.faces[-1]]
