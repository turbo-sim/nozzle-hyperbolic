"""Validated numerical settings and shared solve results."""

from dataclasses import dataclass, field
import operator as index_operator
from typing import NamedTuple, Any
import numpy as np
from ..residuals import ResidualPolicy


@dataclass(frozen=True)
class Scheme:
    """Numerical choices with two independent Stadtke correction flags.

    Parameters
    ----------
    reconstruction : str, optional
        Piecewise-constant or named MUSCL reconstruction. Default is
        ``"constant"``; see the model interface guide for supported limiters.
    flux : str, optional
        Numerical flux: ``"stadtke"`` (default), ``"rusanov"``, ``"hll"``,
        ``"roe"``, ``"osher"`` or the model-owned ``"model_compatibility"``.
        The model must supply the selected flux's required capabilities.
    delta_factor : float, optional
        Positive characteristic regularization coefficient, default 0.1.
    newton_iterations : int, optional
        Maximum Newton corrections per inner solve, default 10.
    newton_backtracking : bool, optional
        Enable Armijo backtracking for flow Newton corrections. Default False.
    newton_update_tolerance : float, optional
        Maximum scaled update for inner convergence, default 1e-10.
        Zero disables this criterion.
    newton_residual_tolerance : float, optional
        Maximum scaled residual RMS for inner convergence, default 1e-12.
        Zero disables this criterion; at least one criterion must remain active.
    max_backtracks : int, optional
        Maximum trial halvings after the initial Newton trial, default 20.
    jacobian_mode : {"colored", "dense"}, optional
        Automatic-differentiation assembly, default ``"colored"``.
    linear_solver : {"banded", "dense"}, optional
        Newton linear solver, default ``"banded"``. The banded solver requires
        colored assembly; globally coupled boundaries require dense/dense.
    time_integrator : str, optional
        ``"euler-implicit"`` (default), ``"euler-explicit"``,
        ``"runge-kutta-3"`` or ``"runge-kutta-4"``.
    boundary_treatment : {"auto", "coupled", "segregated"}, optional
        Algebraic trace solution strategy, default ``"auto"``. Auto selects
        coupled implicit boundaries and segregated explicit stage boundaries.
    filter_strength : float, optional
        Nonnegative conservative fourth-difference coefficient, default zero.
    smooth_limiter_coefficient : float, optional
        Positive width coefficient for smooth Van Leer reconstruction,
        default 1.0. Other reconstructions ignore this setting.
    custom_jump_correction : bool, optional
        Select conservative-jump Stadtke viscosity, default True.
    custom_endpoint_envelope : bool, optional
        Bound Stadtke viscosity by endpoint family speeds, default True.
    quadrature_points : {3, 5, 9}, optional
        Gauss nodes per Osher integration panel, default 5.
    integration : {"gauss", "sonic_partitioned"}, optional
        Osher path integration policy, default ``"gauss"``.

    Notes
    -----
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
    """Fixed stopping policy and pseudo-time CFL controller settings.

    Parameters
    ----------
    residual : ResidualPolicy, optional
        Independent equation, boundary, constraint and global-balance targets.
        A new default policy is created for each settings instance.
    cfl : float, optional
        Initial requested CFL, default 0.2.
    cfl_growth : float, optional
        Accepted-step multiplier, at least one; default 1.2.
    cfl_shrink : float, optional
        Rejected-step multiplier in the open interval (0, 1), default 0.5.
    cfl_max : float, optional
        Maximum requested CFL, default 1e6. Explicit steady integration also
        caps the requested CFL at 0.45.
    max_steps : int, optional
        Maximum accepted pseudo-time steps, default 4000.
    max_retries : int, optional
        Additional attempts after the first trial of each step, default 14.
    stagnation_steps : int, optional
        Consecutive bitwise-unchanged accepted states before reporting
        stagnation, default 12. Zero disables this check.
    """

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
    """Host-side state, outcome and diagnostics from a shared solver driver.

    Attributes
    ----------
    state : numpy.ndarray, shape (N + 2, V)
        Physical cell states with independent endpoint traces in the first
        and last rows, using the model's iteration variables.
    success : bool
        Whether the driver's convergence or requested-time checks and any
        configured model audit passed.
    reason : str
        Machine-readable termination reason.
    history : list of dict
        Initial/accepted states and rejected attempts recorded by the driver.
        Rejected attempts do not advance the accepted-step count or time.
    metrics : dict
        Final equation, boundary, constraint and conservation diagnostics.
    time : float
        Reached physical time; zero for steady pseudo-time solves.
    elapsed_s : float
        Driver wall time in seconds. Includes compilation occurring inside the
        timed solve; operator construction may occur before the timer starts.
    operator : Operator or None
        Bound spatial operator, retained for diagnostics and continuation.
    """

    state: np.ndarray
    success: bool
    reason: str
    history: list
    metrics: dict
    time: float = 0.0
    elapsed_s: float = 0.0
    operator: object = None


class TimestepInfo(NamedTuple):
    """Timestep selection and its independent stability diagnostics.

    Attributes
    ----------
    dt : jax.Array
        Selected time increment after applying requested CFL and, for explicit
        integration, combining wave, source and filter rates.
    wave_dt : jax.Array
        Wave time scale at unit CFL, before multiplication by requested CFL.
        The actual accepted wave CFL is ``dt / wave_dt``.
    source_dt, filter_dt : jax.Array
        Absolute explicit source/filter limits, or infinity when inactive.
    valid : jax.Array
        Scalar boolean indicating valid completion and stability data.
    """

    dt: Any
    wave_dt: Any
    source_dt: Any
    filter_dt: Any
    valid: Any
