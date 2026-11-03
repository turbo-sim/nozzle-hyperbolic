"""Physical area derivatives for tabulated nozzle geometry helpers."""

import sys
from types import ModuleType

import numpy as np
import pytest

from nozzle_hyperbolic.geometry.duct import get_lettieri_nozzle


@pytest.mark.parametrize("nozzle_type", ["axisymmetric", "planar"])
def test_lettieri_area_derivative_uses_nonuniform_coordinates(monkeypatch, nozzle_type):
    # Isolate workbook loading: the known straight wall defines an independent
    # analytical area derivative without an Excel file or optional I/O engine.
    radius, slope, width = 0.01, 0.002, 0.02
    fitter = ModuleType("nozzle_hyperbolic.fit_nozzle_shape")
    fitter.fit_nozzle_shape = lambda *args, **kwargs: (
        None,
        None,
        lambda x: radius + slope * x,
    )
    monkeypatch.setitem(sys.modules, fitter.__name__, fitter)
    x = np.array([0.0, 0.1, 0.3, 0.6, 0.9])
    geometry = get_lettieri_nozzle(x, nozzle_type=nozzle_type, width=width)
    expected = (
        2 * np.pi * (radius + slope * x) * slope
        if nozzle_type == "axisymmetric"
        else np.full_like(x, 2 * width * slope)
    )
    # Interior differences are exact for the quadratic/linear area profiles;
    # the helper deliberately retains first-order one-sided endpoint slopes.
    np.testing.assert_allclose(geometry.dAdx[1:-1], expected[1:-1], rtol=2e-14)
