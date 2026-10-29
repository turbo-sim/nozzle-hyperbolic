"""Strict portable export of convergence histories and failed trial diagnostics."""

import json
import io

import numpy as np
import pytest

from nozzle_hyperbolic.solvers.convergence_output import ConvergenceTable, strict_json


def test_strict_json_retains_nonfinite_kind_and_path_without_modifying_input():
    array = np.array([1., np.nan, np.inf, -np.inf])
    original = dict(history=[dict(accepted=False, residual=array)], count=np.int64(2))
    exported = strict_json(original)
    decoded = json.loads(exported, parse_constant=lambda token: pytest.fail(f"Nonstandard JSON: {token}"))
    assert decoded["count"] == 2
    assert decoded["history"][0]["residual"] == [1., None, None, None]
    assert decoded["nonfinite_values"] == [
        dict(path="$.history[0].residual[1]", classification="nan"),
        dict(path="$.history[0].residual[2]", classification="positive_infinity"),
        dict(path="$.history[0].residual[3]", classification="negative_infinity"),
    ]
    assert np.isnan(array[1]) and np.isposinf(array[2]) and np.isneginf(array[3])


def test_strict_json_keeps_previous_classifications_when_exported_again():
    artifact = json.loads(strict_json(dict(value=float("inf"))))
    second = json.loads(strict_json(artifact))
    assert second == artifact


def test_strict_json_requires_result_mapping():
    with pytest.raises(TypeError, match="mapping"):
        strict_json([1., 2.])


@pytest.mark.parametrize("count", [3, 4, 6])
def test_compact_table_bounds_width_and_reports_overall_without_integral_columns(count):
    stream = io.StringIO()
    table = ConvergenceTable(["equation_"+str(i) for i in range(count)],
                             ["Cp_mean", "mdot_out"], stream=stream)
    row = dict(step=10000, cfl=1e6, dt=2e-3, inner_iterations=8,
               equation_relative=[1e-8]*count, relative_active=[True]*count,
               equation_max_ratio=10., boundary_max=1e-14, conservation_max=2e-13,
               monitors={"Cp_mean": {"value": .5}, "mdot_out": {"value": .2}})
    table(row)
    table(dict(step=10001, accepted=False, cfl=float("inf")))
    output = stream.getvalue()
    assert max(map(len, output.splitlines())) <= 100
    assert "Cp_mean" not in output and "mdot_out" not in output
    assert "1.00e-08" in output and "R" in output and f"R{count}" in output
    assert "1.00e+01" not in output  # R is a reduction, not a tolerance ratio.
    assert "retry" in output and "+inf" in output
