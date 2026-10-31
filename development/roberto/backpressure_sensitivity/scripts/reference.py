"""Compatibility import for the canonical model-owned nozzle operating map.

Pass reservoir and geometry parameters explicitly when changing the baseline;
this module no longer imports a numerical study to obtain physical defaults.
"""

from nozzle_hyperbolic.models.euler_nozzle.analytical.perfect_gas import PerfectGasReference

__all__ = ["PerfectGasReference"]
