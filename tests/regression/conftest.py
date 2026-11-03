"""Reuse physical solves across accuracy and saved-state comparisons."""

import gc

import jax
import pytest

from tests._support.regression import STEADY_CASES, solve_steady_case


@pytest.fixture(scope="session", params=STEADY_CASES, ids=STEADY_CASES)
def steady_run(request):
    """One cold numerical hierarchy per physical case, without analytical seeds."""
    # Independent physical cases bind many shape-specific operators. Release
    # earlier compilation caches at this case boundary, while keeping both
    # accuracy and snapshot assertions on the same solved hierarchy. This never
    # clears caches inside a solve or inside the compilation-reuse tests.
    jax.clear_caches()
    yield solve_steady_case(request.param)
    jax.clear_caches()
    gc.collect()
