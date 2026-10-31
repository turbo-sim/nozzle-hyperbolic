"""Small scalar models demonstrating the public conservative model interface."""

from .advection import AdvectionModel
from .burgers import BurgersModel

__all__ = ["AdvectionModel", "BurgersModel"]
