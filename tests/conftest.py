"""Shared precision and execution categories for the maintained test suite."""

import importlib.util
import os
from pathlib import Path

import jax
import pytest


# Configure precision before collection imports model modules or creates arrays.
# A session fixture would run too late for module-level JAX constants.
jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    """Make the optional integration job fail if its dependency is absent."""
    if os.environ.get("NOZZLE_REQUIRE_NOZZLEX") == "1":
        if importlib.util.find_spec("nozzlex") is None:
            raise pytest.UsageError("The optional integration job requires NozzleX")


def pytest_collection_modifyitems(items):
    """Classify cost/dependencies without changing numerical assertions."""
    slow_modules = {
        "test_annular.py",  # Independent NURBS/arclength quadrature.
        "test_boundary_treatment.py",
        "test_euler_jacobians.py",
        "test_general_eos_outlet.py",
        "test_osher.py",
        "test_osher_binding.py",
        "test_port.py",
        "test_roe_integration.py",
        "test_roe_physics.py",
    }
    for item in items:
        path = Path(item.path)
        relative = path.relative_to(Path(__file__).parent)
        if relative.parts[0] == "regression":
            item.add_marker(pytest.mark.regression)
        if path.name in slow_modules:
            item.add_marker(pytest.mark.slow)
        if "nozzlex" in path.name:
            item.add_marker(pytest.mark.optional)
