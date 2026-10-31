"""One physical-cell/face layout, independent of equation count and physics."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mesh:
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
        return len(self.centers)

    @property
    def widths(self):
        return np.diff(self.faces)

    @property
    def points(self):
        return np.r_[self.faces[0], self.centers, self.faces[-1]]
