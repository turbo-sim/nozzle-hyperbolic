"""Explicit, model-owned two-fluid physical and stabilization policies."""

from dataclasses import dataclass, fields
from inspect import signature
import math
from numbers import Real


def _finite_fields(instance):
    for item in fields(instance):
        value = getattr(instance, item.name)
        if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"{item.name} must be finite")


@dataclass(frozen=True)
class ClosureConfig:
    """Frozen Alberto closure parameters, with SI units.

    Nb/Nd are number densities [1/m3]; slip_u0 [m/s], p_mt_max [Pa].
    Interfacial-area params retain the selected correlation's parameter names.
    No wall friction law was implemented upstream and none is supplied here.
    """
    Nb: float = 1e10
    Nd: float = 1e10
    alpha_b: float = 0.3
    alpha_d: float = 0.7
    mass_transfer: bool = True
    regime_htc: bool = False
    regime_blend: str = "linear"
    gate_dT: float = 1e-3
    gate_dh: float = 0.5
    p_mt_max: float | None = None
    area_law: str = "polynomial"
    area_params: dict | None = None
    ai_scale: float = 1.0
    smooth_slip: bool = False
    smooth_slip_u0: float = 2.0
    nu_g_int: float | None = None
    htc_liquid_bubbly: str = "ranz_marshall"
    htc_liquid_droplet: str = "internal"
    drag_bubbly: str = "schiller_naumann"
    drag_droplet: str = "schiller_naumann"

    def __post_init__(self):
        _finite_fields(self)
        from ._alberto_closures import (
            DRAG_LAWS, INTERFACIAL_AREA_LAWS, LIQUID_HTC_BUBBLY,
            LIQUID_HTC_DROPLET, REGIME_BLENDS,
        )
        if not (0 < self.alpha_b < self.alpha_d < 1):
            raise ValueError("Require 0 < alpha_b < alpha_d < 1")
        if min(self.Nb, self.Nd, self.ai_scale, self.gate_dT, self.gate_dh) <= 0:
            raise ValueError("Closure number densities, area scale and gate widths must be positive")
        if not self.mass_transfer and (self.htc_liquid_bubbly, self.htc_liquid_droplet) != ("ranz_marshall", "internal"):
            raise ValueError("Jakob/Liao liquid HTCs require mass_transfer=True in the frozen model")
        if self.smooth_slip_u0 < 0 or (self.smooth_slip and self.smooth_slip_u0 == 0) or (self.nu_g_int is not None and self.nu_g_int <= 0):
            raise ValueError("Invalid slip regularization or internal Nusselt number")
        if self.p_mt_max is not None and self.p_mt_max <= 0:
            raise ValueError("p_mt_max must be positive")
        for value, options in ((self.drag_bubbly, DRAG_LAWS), (self.drag_droplet, DRAG_LAWS),
                               (self.htc_liquid_bubbly, LIQUID_HTC_BUBBLY),
                               (self.htc_liquid_droplet, LIQUID_HTC_DROPLET),
                               (self.regime_blend, REGIME_BLENDS),
                               (self.area_law, INTERFACIAL_AREA_LAWS)):
            if value not in options:
                raise ValueError(f"Unsupported closure {value!r}; choose from {tuple(options)}")
        if self.area_params is not None:
            if not isinstance(self.area_params, dict):
                raise ValueError("area_params must be a dictionary")
            parameters = signature(INTERFACIAL_AREA_LAWS[self.area_law]).parameters
            unknown = set(self.area_params)-set(parameters)
            if unknown or "alpha" in self.area_params:
                raise ValueError(f"Unsupported interfacial-area parameters: {sorted(unknown or {'alpha'})}")
            for name, value in self.area_params.items():
                if name in ("ai_ctrl", "ai_pts", "alpha_pts") and not isinstance(value, (tuple, list)):
                    raise ValueError(f"Interfacial-area {name} must be a sequence")
                values = value if isinstance(value, (tuple, list)) else (value,)
                if not values or not all(isinstance(x, Real) and math.isfinite(x) for x in values):
                    raise ValueError(f"Interfacial-area {name} must contain finite numbers")
                if name == "alpha_pts":
                    if values[0] != 0 or values[-1] != 1 or any(a >= b for a,b in zip(values,values[1:])):
                        raise ValueError("alpha_pts must increase strictly from 0 to 1")
                elif name == "corner":
                    if value < 0:
                        raise ValueError("corner must be nonnegative")
                elif any(x <= 0 for x in values):
                    raise ValueError(f"Interfacial-area {name} must be positive")
            selected = {name:self.area_params.get(name,item.default) for name,item in parameters.items()}
            if not 0 < selected["alpha_b"] < selected["alpha_d"] < 1:
                raise ValueError("Interfacial-area thresholds require 0 < alpha_b < alpha_d < 1")
            if self.area_law == "segments" and len(selected["alpha_pts"]) != len(selected["ai_pts"]):
                raise ValueError("alpha_pts and ai_pts must have matching lengths")
            if self.area_law == "polynomial" and len(selected["ai_ctrl"]) < 2:
                raise ValueError("ai_ctrl must have at least two controls")
            if self.area_law == "parabola" and selected["a_max"] < selected["a_min"]:
                raise ValueError("Parabolic area requires a_max >= a_min")


@dataclass(frozen=True)
class CompatibilityConfig:
    """Operational frozen FVS/NC choices, independent of generic solver settings."""
    nc_form: str = "cell"
    lowmach_diss: float = 1.0
    lowmach_diss_boundary: float | None = None
    du_blend_lo: float = 5.0
    du_blend_hi: float = 20.0
    delta_factor: float = 1e-6

    def __post_init__(self):
        _finite_fields(self)
        if self.nc_form not in ("cell", "wb", "staedtke"):
            raise ValueError("nc_form must be cell, wb, or staedtke")
        if not (0 <= self.lowmach_diss <= 1 and 0 < self.du_blend_lo < self.du_blend_hi
                and self.delta_factor > 0):
            raise ValueError("Invalid compatibility flux parameters")
        if self.lowmach_diss_boundary is not None and not 0 <= self.lowmach_diss_boundary <= 1:
            raise ValueError("lowmach_diss_boundary must lie in [0,1]")


@dataclass(frozen=True)
class StabilizationConfig:
    """Independent, off-by-default filters; no physical phase clipping.

    alpha_min is an admissibility threshold, never a projection. Pure phases
    cannot be represented by eight independent variables and are rejected.
    """
    alpha_min: float = 0.0
    void_strength: float = 0.0
    void_velocity_cutoff: float | None = None
    void_velocity_width: float = 12.0
    speed_scale: str = "frozen"
    void_mass_cutoff: float | None = None
    thin_phase_strength: float = 0.0
    thin_phase_cutoff: float = 3e-3
    thin_phase_lower: float = 1e-5
    thin_phase_log_width: float = 0.25
    thin_phase_second_order: float = 0.0
    mass_upwind_cutoff: float | None = None

    def __post_init__(self):
        _finite_fields(self)
        if not (0 <= self.alpha_min < 0.5):
            raise ValueError("alpha_min must be in [0,0.5)")
        if min(self.void_strength, self.thin_phase_strength, self.thin_phase_second_order) < 0:
            raise ValueError("Filter strengths must be nonnegative")
        if self.speed_scale not in ("wood", "frozen"):
            raise ValueError("speed_scale must be wood or frozen")
        if self.void_velocity_width <= 0 or self.thin_phase_log_width <= 0:
            raise ValueError("Filter widths must be positive")
        if not (0 < self.thin_phase_lower < self.thin_phase_cutoff < 1):
            raise ValueError("Require 0 < thin_phase_lower < thin_phase_cutoff < 1")
        for value in (self.void_mass_cutoff, self.mass_upwind_cutoff):
            if value is not None and not 0 < value < 1:
                raise ValueError("Phase cutoffs must lie in (0,1)")


@dataclass(frozen=True)
class BoundaryConfig:
    """Algebraic trace laws; numerical treatment is selected in Scheme.

    quality is inlet vapor mass-flow fraction, slip=u_g/u_l. Stagnation
    specifies H_g0, H_l0 and the dominant-phase reservoir entropy s0.
    """
    inlet: str = "transmissive"
    outlet: str = "transmissive"
    quality: float = 0.5
    slip: float = 1.0
    H_g0: float | None = None
    H_l0: float | None = None
    s0: float | None = None
    dominant_phase: str = "gas"
    alpha0: float | None = None
    inlet_state: tuple | None = None
    back_pressure: float | None = None
    pressure_relaxation: float = 1.0
    mach_width: float = 0.05

    def __post_init__(self):
        _finite_fields(self)
        if self.inlet not in ("transmissive", "stagnation", "total_state", "static_state", "subsonic_inflow"):
            raise ValueError("Unsupported inlet; interior ramping is initialization, not a boundary law")
        if self.outlet not in ("transmissive", "pressure", "characteristic", "characteristic_old"):
            raise ValueError("Unsupported two-fluid outlet")
        if self.inlet == "stagnation":
            if any(v is None for v in (self.H_g0, self.H_l0, self.s0)):
                raise ValueError("Stagnation inlet needs H_g0, H_l0, s0")
            if not (0 < self.quality < 1 and self.slip > 0):
                raise ValueError("Stagnation inlet requires 0<quality<1 and positive slip")
        if self.inlet == "total_state" and any(v is None for v in (self.H_g0, self.H_l0, self.alpha0)):
            raise ValueError("total_state inlet needs H_g0, H_l0 and alpha0")
        if self.alpha0 is not None and not 0 < self.alpha0 < 1:
            raise ValueError("alpha0 must lie in (0,1)")
        if self.inlet in ("static_state", "subsonic_inflow"):
            if self.inlet_state is None or len(self.inlet_state) != 6:
                raise ValueError(f"{self.inlet} inlet needs six primitive values")
            if not all(math.isfinite(value) for value in self.inlet_state):
                raise ValueError("inlet_state values must be finite")
            if self.inlet_state[0] <= 0 or not 0 < self.inlet_state[3] < 1:
                raise ValueError("inlet_state requires positive pressure and void fraction in (0,1)")
        if self.dominant_phase not in ("gas", "liquid"):
            raise ValueError("dominant_phase must be gas or liquid")
        if self.outlet != "transmissive" and (self.back_pressure is None or (not callable(self.back_pressure) and self.back_pressure <= 0)):
            raise ValueError("Pressure outlets require positive back_pressure")
        if not (0 < self.pressure_relaxation <= 1 and self.mach_width > 0):
            raise ValueError("Invalid pressure relaxation or Mach blending width")


@dataclass(frozen=True)
class PhasePinConfig:
    """Explicit equation-changing upstream phase pins, disabled by default.

    Saturation energy pin uses the binary evaporation gate. Momentum pin
    requires the upstream dt-consistent finite relaxation time [s]; the old
    dt-dependent algebraic momentum blend is deliberately unsupported.
    These policies are restricted to implicit steady solution, not physical
    transients or explicit integration.
    """
    saturation_energy: bool = False
    momentum: bool = False
    momentum_alpha: float = 1e-5
    momentum_time: float | None = None
    pressure_cutoff: float | None = None

    def __post_init__(self):
        _finite_fields(self)
        if not 0 < self.momentum_alpha < 1:
            raise ValueError("momentum_alpha must lie in (0,1)")
        if self.momentum and (self.momentum_time is None or self.momentum_time <= 0):
            raise ValueError("Momentum pin requires positive finite momentum_time; dt-dependent blend unsupported")
        if self.pressure_cutoff is not None and self.pressure_cutoff <= 0:
            raise ValueError("pressure_cutoff must be positive")
