"""Compatibility imports for the model-owned two-fluid analytical profiles."""

from nozzle_hyperbolic.models.two_fluid.analytical import (
    _area_mach,
    _am_ratio,
    isentropic_ic,
    isentropic_shock_ic,
    isentropic_subsonic_ic,
)

__all__ = ["isentropic_ic", "isentropic_shock_ic", "isentropic_subsonic_ic"]
