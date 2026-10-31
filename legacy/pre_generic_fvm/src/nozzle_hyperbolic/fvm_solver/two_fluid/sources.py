"""Compatibility import for :mod:`nozzle_hyperbolic.models.two_fluid.sources`."""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("nozzle_hyperbolic.models.two_fluid.sources")
