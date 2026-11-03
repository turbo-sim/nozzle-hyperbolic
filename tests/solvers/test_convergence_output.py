"""Strict portable export of convergence histories and failed trial diagnostics."""

import json
import io

import numpy as np
import pytest

from nozzle_hyperbolic.solvers.convergence_output import ConvergenceTable, strict_json


def test_strict_json_retains_nonfinite_kind_and_path_without_modifying_input():
    array = np.array([1.0, np.nan, np.inf, -np.inf])
    original = dict(history=[dict(accepted=False, residual=array)], count=np.int64(2))
    exported = strict_json(original)
    decoded = json.loads(
        exported, parse_constant=lambda token: pytest.fail(f"Nonstandard JSON: {token}")
    )
    assert decoded["count"] == 2
    assert decoded["history"][0]["residual"] == [1.0, None, None, None]
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
        strict_json([1.0, 2.0])


def test_solution_summary_separates_updates_rejections_and_physical_defects():
    from types import SimpleNamespace
    from nozzle_hyperbolic.solvers.convergence_output import solution_summary

    result = SimpleNamespace(
        success=False,
        reason="retry_limit",
        time=0.1,
        elapsed_s=2.0,
        history=[
            dict(phase="initial", accepted=True),
            dict(phase="accepted", accepted=True, inner_iterations=3, cfl=0.4),
            dict(
                phase="rejected",
                accepted=False,
                inner_iterations=7,
                rejected_newton_trials=2,
            ),
        ],
        metrics=dict(
            equation_norm=np.array([2e-4]),
            boundary_max=1e-12,
            constraint_max=3e-12,
            global_defect=np.array([4e-5, 2e-5]),
            conservation_budget=np.array([1e-13, -2e-13]),
        ),
    )
    summary = solution_summary(result)
    assert summary["accepted_steps"] == summary["rejected_steps"] == 1
    assert summary["inner_iterations"] == 10
    assert summary["accepted_inner_iterations"] == 3
    assert summary["rejected_inner_iterations"] == 7
    assert summary["rejected_newton_trials"] == 2
    assert summary["global_balance_max"] == 4e-5
    assert summary["conservation_budget_max"] == 2e-13
    result.metrics["global_defect"][0] = np.nan
    assert np.isnan(solution_summary(result)["global_balance_max"])


def test_transient_table_labels_time_inner_work_and_content_budget():
    stream = io.StringIO()
    table = ConvergenceTable(
        ["mass", "momentum", "energy"], stream=stream, mode="transient"
    )
    table(
        dict(
            step=12,
            time_s=0.03,
            cfl=0.4,
            dt=1e-4,
            inner_iterations=6,
            inner_residual=1e-13,
            budget=np.array([1e-12, -2e-12]),
        )
    )
    output = stream.getvalue()
    assert "t [s]" in output and "Inner" in output and "Budget" in output
    assert "R1" not in output and "2.00e-12" in output
    assert max(map(len, output.splitlines())) <= 100


@pytest.mark.parametrize(
    "defects, expected",
    [([1e-13, 2e-13], "2.00e-13"), ([np.nan, 1e-13], "nan"), ([np.inf], "+inf")],
)
def test_table_reports_solver_global_defect_vector(defects, expected):
    stream = io.StringIO()
    table = ConvergenceTable(["mass", "energy"], stream=stream)
    table(dict(step=1, global_defect=np.asarray(defects), conservation_max=9.0))
    # Inspect the G column, including when another compatibility key is present.
    assert stream.getvalue().splitlines()[-1].split()[-2] == expected


@pytest.mark.parametrize("count", [3, 4, 6])
def test_compact_table_bounds_width_and_reports_overall_without_integral_columns(count):
    stream = io.StringIO()
    table = ConvergenceTable(
        ["equation_" + str(i) for i in range(count)],
        ["Cp_mean", "mdot_out"],
        stream=stream,
    )
    row = dict(
        step=10000,
        cfl=1e6,
        dt=2e-3,
        inner_iterations=8,
        equation_relative=[1e-8] * count,
        relative_active=[True] * count,
        equation_max_ratio=10.0,
        boundary_max=1e-14,
        conservation_max=2e-13,
        monitors={"Cp_mean": {"value": 0.5}, "mdot_out": {"value": 0.2}},
    )
    table(row)
    table(dict(step=10001, accepted=False, cfl=float("inf")))
    output = stream.getvalue()
    assert max(map(len, output.splitlines())) <= 100
    assert "Cp_mean" not in output and "mdot_out" not in output
    assert "1.00e-08" in output and "R" in output and f"R{count}" in output
    assert "2.00e-13" in output  # Scalar compatibility key still supplies G.
    assert "1.00e+01" not in output  # R is a reduction, not a tolerance ratio.
    assert "retry" in output and "+inf" in output
