"""Regression comparisons must reject materially wrong, incomplete or NaN data."""

import numpy as np
import pytest

from tests._support.baselines import read_snapshot
from tests._support.regression import assert_saved_state


@pytest.mark.parametrize("fault", ["mass", "momentum", "energy", "nan", "missing-cell"])
def test_saved_profile_guard_rejects_injected_faults(fault):
    arrays, _, _ = read_snapshot("euler_nozzle/shock_35")
    expected = arrays["conserved"]
    damaged = expected.copy()
    if fault == "missing-cell":
        damaged = damaged[:-1]
    elif fault == "nan":
        damaged[10, 0] = np.nan
    else:
        column = {"mass": 0, "momentum": 1, "energy": 2}[fault]
        damaged[10:20, column] += 0.01 * arrays["conservative_scales"][column]
    with pytest.raises(AssertionError):
        assert_saved_state(damaged, expected, arrays["conservative_scales"])
