"""Complete Euler solves checked against independently projected exact states."""

import numpy as np
import pytest

from tests._support.regression import (
    GRIDS,
    assert_steady_accuracy,
    assert_steady_converged,
)

pytestmark = pytest.mark.slow


def test_cold_hierarchy_converges_on_the_intended_physical_branch(steady_run):
    assert [stage["cells"] for stage in steady_run.stages] == list(GRIDS)
    assert all(stage["success"] for stage in steady_run.stages)
    assert not any(stage["attempts"] for stage in steady_run.stages)
    assert_steady_converged(steady_run.solution)
    assert_steady_accuracy(steady_run)
    if steady_run.problem.shock_position is None:
        coarse = max(steady_run.errors_by_grid[64])
        fine = max(steady_run.errors_by_grid[128])
        # A halving of spacing must improve the smooth MUSCL result substantially.
        assert fine < 0.65 * coarse, (coarse, fine)
        assert np.isfinite((coarse, fine)).all()
