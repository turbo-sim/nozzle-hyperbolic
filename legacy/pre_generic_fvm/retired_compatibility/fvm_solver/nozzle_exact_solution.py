"""Compatibility import for :mod:`nozzle_hyperbolic.models.euler_nozzle.analytical.nozzle_exact_solution`."""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("nozzle_hyperbolic.models.euler_nozzle.analytical.nozzle_exact_solution")
