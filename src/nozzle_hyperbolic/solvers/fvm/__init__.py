"""Shared conservative FVM engine; physical models are callback providers.

Scheme defaults to corrected Stadtke; disabling both custom flags recovers its
original contraction. Rusanov, HLL, Roe and Osher are also explicit Scheme
options. Osher quadrature order and integration policy are separate settings.
Optional structural capabilities require no model inheritance.
"""

from .config import Scheme, SteadySettings, Solution, TimestepInfo
from .contracts import (
    ConservativeModel,
    FusedConservativeModel,
    CharacteristicModel,
    FusedCharacteristicModel,
    FaceStatePathModel,
    RoeModel,
    FusedRoeModel,
    BalanceModel,
    ConstrainedCellModel,
    FluxDifferenceNonconservativeModel,
    CompatibilityFluxModel,
)
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
    "CharacteristicModel",
    "FusedCharacteristicModel",
    "FaceStatePathModel",
    "RoeModel",
    "FusedRoeModel",
    "BalanceModel",
    "ConstrainedCellModel",
    "FluxDifferenceNonconservativeModel",
    "CompatibilityFluxModel",
]
