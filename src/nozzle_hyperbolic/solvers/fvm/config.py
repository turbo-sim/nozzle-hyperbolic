"""Validated numerical settings and shared solve results."""

from dataclasses import dataclass, field
import operator as index_operator
from typing import NamedTuple, Any
import numpy as np
from ..residuals import ResidualPolicy


@dataclass(frozen=True)
class Scheme:
    """Numerical choices with two independent Stadtke correction flags.

    The default is corrected ``stadtke`` (both custom corrections enabled).
    ``custom_jump_correction`` replaces finite physical-flux-jump dissipation
    by mean-characteristic conservative-jump viscosity.
    ``custom_endpoint_envelope`` adds the characteristic viscosity increment
    max(mean regularized absolute speed, absolute endpoint family speeds)
    minus the mean regularized speed. The flags are independent; disable both
    to recover the original Stadtke contraction. Neither supplies a general
    monotonicity theorem. Other flux families ignore the flags. No alternative
    public name selects Stadtke.
    For ``flux='osher'``, ``quadrature_points`` selects 3, 5 (default) or 9
    Gauss points. ``integration='gauss'`` applies one unsplit rule;
    ``'sonic_partitioned'`` applies that rule on panels split at sampled sonic
    crossings. These settings are independent of ``time_integrator`` and
    ignored by other fluxes. Osher requires an admissible path and supported
    storage differentiation, distinct from Stadtke's custom flags.
    """

    reconstruction: str = "constant"
    flux: str = "stadtke"
    delta_factor: float = 0.1
    newton_iterations: int = 10
    newton_backtracking: bool = False
    newton_update_tolerance: float = 1e-10
    newton_residual_tolerance: float = 1e-12
    max_backtracks: int = 20
    jacobian_mode: str = "colored"
    linear_solver: str = "banded"
    time_integrator: str = "euler-implicit"
    boundary_treatment: str = "auto"
    filter_strength: float = 0.0
    smooth_limiter_coefficient: float = 1.0
    custom_jump_correction: bool = True
    custom_endpoint_envelope: bool = True
    quadrature_points: int = 5
    integration: str = "gauss"

    def __post_init__(self):
        if not isinstance(self.newton_backtracking, bool):
            raise ValueError("newton_backtracking must be a boolean")
        if self.reconstruction not in (
            "constant",
            "muscl_van_leer",
            "muscl_van_leer_smooth",
            "muscl_minmod",
            "muscl_superbee",
            "muscl_van_albada",
        ):
            raise ValueError("Unsupported reconstruction")
        if self.flux not in ("stadtke", "rusanov", "hll", "roe", "osher", "model_compatibility"):
            raise ValueError("Unsupported numerical flux")
        if type(self.quadrature_points) is not int or self.quadrature_points not in (3, 5, 9):
            raise ValueError("quadrature_points must be 3, 5 or 9")
        if not isinstance(self.integration, str) or self.integration not in ("gauss", "sonic_partitioned"):
            raise ValueError("integration must be gauss or sonic_partitioned")
        for name in ("custom_jump_correction", "custom_endpoint_envelope"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.time_integrator not in (
            "euler-implicit",
            "euler-explicit",
            "runge-kutta-3",
            "runge-kutta-4",
        ):
            raise ValueError("Unsupported time integrator")
        if self.boundary_treatment not in ("auto", "coupled", "segregated"):
            raise ValueError("boundary_treatment must be auto, coupled or segregated")
        if (
            self.boundary_treatment == "coupled"
            and self.time_integrator != "euler-implicit"
        ):
            raise ValueError(
                "Explicit time integrators require segregated boundary treatment"
            )
        if not np.isfinite(self.delta_factor) or self.delta_factor <= 0:
            raise ValueError("Positive characteristic width required")
        if self.jacobian_mode not in ("dense", "colored") or self.linear_solver not in (
            "dense",
            "banded",
        ):
            raise ValueError("Unsupported Jacobian or linear solver")
        if self.jacobian_mode == "dense" and self.linear_solver == "banded":
            raise ValueError(
                "The banded backend uses colored AD; select colored/banded or dense/dense"
            )
        for name in ("newton_iterations", "max_backtracks"):
            value = getattr(self, name)
            if isinstance(value, bool) or index_operator.index(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("newton_update_tolerance", "newton_residual_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.newton_update_tolerance == 0 and self.newton_residual_tolerance == 0:
            raise ValueError("At least one Newton convergence criterion is required")
        if not np.isfinite(self.filter_strength) or self.filter_strength < 0:
            raise ValueError("filter_strength must be finite and nonnegative")
        if (
            not np.isfinite(self.smooth_limiter_coefficient)
            or self.smooth_limiter_coefficient <= 0
        ):
            raise ValueError("smooth_limiter_coefficient must be positive and finite")

    @property
    def effective_boundary_treatment(self):
        """Resolve auto without changing the historical integrator defaults."""
        if self.boundary_treatment == "auto":
            return (
                "coupled" if self.time_integrator == "euler-implicit" else "segregated"
            )
        return self.boundary_treatment


@dataclass(frozen=True)
class SteadySettings:
    residual: ResidualPolicy = field(default_factory=ResidualPolicy)
    cfl: float = 0.2
    cfl_growth: float = 1.2
    cfl_shrink: float = 0.5
    cfl_max: float = 1e6
    max_steps: int = 4000
    max_retries: int = 14
    stagnation_steps: int = 12

    def __post_init__(self):
        if not isinstance(self.residual, ResidualPolicy):
            raise ValueError("residual must be a ResidualPolicy")
        for name in ("cfl", "cfl_max", "cfl_growth", "cfl_shrink"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            self.cfl > self.cfl_max
            or self.cfl_growth < 1
            or not 0 < self.cfl_shrink < 1
        ):
            raise ValueError(
                "Require cfl <= cfl_max, cfl_growth >= 1 and 0 < cfl_shrink < 1"
            )
        for name in ("max_steps", "max_retries", "stagnation_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass
class Solution:
    state: np.ndarray
    success: bool
    reason: str
    history: list
    metrics: dict
    time: float = 0.0
    elapsed_s: float = 0.0
    operator: object = None


class TimestepInfo(NamedTuple):
    """Actual dt and independent wave/source/filter limits, all in time units."""

    dt: Any
    wave_dt: Any
    source_dt: Any
    filter_dt: Any
    valid: Any
