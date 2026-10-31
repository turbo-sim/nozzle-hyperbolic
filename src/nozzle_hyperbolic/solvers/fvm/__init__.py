"""Shared conservative FVM engine; physical models are callback providers."""

from .config import Scheme, SteadySettings, Solution, TimestepInfo
from .contracts import ConservativeModel, FusedConservativeModel
from .operator import Operator
from .drivers import solve_single_grid, solve_grid_sequence, solve_transient, sampler
from ..residuals import ResidualPolicy
from ...numerics.mesh import Mesh

__all__ = [
    "Mesh",
    "Scheme",
    "SteadySettings",
    "Solution",
    "ResidualPolicy",
    "Operator",
    "solve_single_grid",
    "solve_grid_sequence",
    "solve_transient",
    "sampler",
    "TimestepInfo",
    "ConservativeModel",
    "FusedConservativeModel",
]
