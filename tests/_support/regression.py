"""Small physical problems and independent acceptance checks for regressions.

Physical definitions come from the installed verification API. Numerical choices
are fixed here, independently of demos and development benchmark clients.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import (
    Scheme,
    SteadySettings,
    sampler,
    solve_grid_sequence,
)
from nozzle_hyperbolic.verification import build_case


STEADY_CASES = (
    "euler_nozzle/subsonic",
    "euler_nozzle/shock_35",
    "euler_nozzle/adapted",
    "euler_annular/conical_subsonic",
    "euler_annular/shock35",
    "euler_annular/supersonic_expansion",
)
GRIDS = (32, 64, 128)
STEADY_SCHEME = Scheme(
    flux="hll", reconstruction="muscl_van_leer", newton_backtracking=True
)
STEADY_SETTINGS = SteadySettings(max_steps=4000, cfl_max=1e4, stagnation_steps=64)
TRANSIENT_SCHEME = Scheme(
    flux="hll", reconstruction="muscl_van_leer", time_integrator="runge-kutta-3"
)


@dataclass
class SteadyRun:
    """One cold hierarchy, final solution and independently measured grid errors."""

    problem: object
    solution: object
    stages: list
    errors_by_grid: dict


def conservative_error(problem, state):
    """Physical-volume mean absolute error, normalized by fixed density scales."""
    actual = np.asarray(problem.model.cell_conserved(jnp.asarray(state[1:-1])))
    expected = problem.cell_reference()["q"]
    difference = np.abs((actual - expected) / problem.model.conservative_scales)
    return np.average(difference, weights=problem.mesh.volumes, axis=0)


def solve_steady_case(case_id):
    """Solve a model-owned cold start; retain every grid's outcome and error."""
    problems = {}
    errors = {}

    def model_factory(count):
        problems[count] = build_case(case_id, count)
        return problems[count].model

    def assess_stage(model, solution, stages):
        assert_steady_converged(solution)
        errors[model.mesh.n_cells] = conservative_error(
            problems[model.mesh.n_cells], solution.state
        )

    solution, stages = solve_grid_sequence(
        model_factory,
        GRIDS,
        STEADY_SCHEME,
        STEADY_SETTINGS,
        on_stage=assess_stage,
        stop_on_failure=True,
        retry_cold=False,
    )
    return SteadyRun(problems[GRIDS[-1]], solution, stages, errors)


def assert_steady_converged(solution):
    """Audit original equations independently of the solver success flag."""
    assert solution.success, (solution.reason, solution.metrics)
    assert solution.reason == "converged"
    initial_norms = solution.history[0]["equation_norm"]
    fresh = sampler(solution.operator, STEADY_SETTINGS.residual)(
        jnp.asarray(solution.state), initial_norms
    )
    for check in (
        "finite",
        "admissible",
        "all_equations_passed",
        "boundary_passed",
        "constraints_passed",
        "global_passed",
    ):
        assert fresh[check], (check, fresh)
    assert solution.operator.model.solution_audit(jnp.asarray(solution.state))
    np.testing.assert_allclose(
        fresh["equation_norm"],
        solution.metrics["equation_norm"],
        rtol=2e-12,
        atol=1e-15,
    )


def shock_position(problem, state):
    """Locate a compressive interior Mach crossing, independently of its target."""
    fields = problem.numerical_fields(state)
    mach = np.asarray(fields.get("M_m", fields.get("M")))
    crossings = np.flatnonzero((mach[:-1] > 1) & (mach[1:] <= 1))
    assert len(crossings) == 1, f"Expected one interior shock, found {crossings}"
    index = crossings[0]
    assert fields["p"][index + 1] > fields["p"][index]
    return float(np.mean(problem.mesh.centers[index : index + 2]))


def assert_steady_accuracy(run):
    """Check branch and projected accuracy at the explicitly fixed 128-cell grid.

    Smooth cases must meet 1e-3 conservative normalized L1. Shock cases allow
    one percent integrated error and require position within two cell widths;
    a pointwise maximum norm is inappropriate inside a captured discontinuity.
    """
    problem, solution = run.problem, run.solution
    errors = conservative_error(problem, solution.state)
    limit = 1e-3 if problem.shock_position is None else 1e-2
    assert np.all(np.isfinite(errors)) and np.max(errors) < limit, errors
    fields = problem.numerical_fields(solution.state)
    mach = np.asarray(fields.get("M_m", fields.get("M")))
    assert np.all(fields["velocity"] > 0)
    if problem.shock_position is not None:
        measured = shock_position(problem, solution.state)
        cell_error = abs(measured - problem.shock_position) / max(problem.mesh.widths)
        assert cell_error <= 2, cell_error
    elif "subsonic" in problem.case_id:
        assert np.all(mach < 1), mach
    else:
        assert mach[-1] > 1, mach


def cell_averages(faces, profile, cuts=()):
    """Integrate an independent profile, splitting at known wave fronts."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    averages = []
    for lower, upper in zip(faces[:-1], faces[1:]):
        edges = sorted([lower, *(x for x in cuts if lower < x < upper), upper])
        total = None
        for left, right in zip(edges[:-1], edges[1:]):
            positions = (left + right) / 2 + (right - left) * nodes / 2
            integral = (right - left) / 2 * (weights @ np.asarray(profile(positions)))
            total = integral if total is None else total + integral
        averages.append(total / (upper - lower))
    return np.asarray(averages)


def assert_transient_budget(model, initial, solution, tolerance=3e-10):
    """Check actual inventory change against accepted integrated transfers."""
    assert solution.success, (solution.reason, solution.metrics)
    assert solution.metrics["boundary_passed"]
    assert solution.metrics["constraints_passed"]
    scale = sum(model.mesh.volumes) * np.asarray(model.conservative_scales)
    initial_total = np.asarray(solution.operator.storage(initial[1:-1])).sum(axis=0)
    final_total = np.asarray(solution.operator.storage(solution.state[1:-1])).sum(
        axis=0
    )
    accepted = [row for row in solution.history if row["accepted"]]
    transfer = sum(
        (np.asarray(row["integrated_transfer"]) for row in accepted),
        np.zeros_like(scale),
    )
    np.testing.assert_allclose(
        (final_total - initial_total - transfer) / scale, 0, rtol=0, atol=tolerance
    )
    np.testing.assert_allclose(
        solution.metrics["conservation_budget"], 0, rtol=0, atol=tolerance
    )


def assert_saved_state(actual, expected, scales):
    """Compare normalized fields with explicit refactoring-regression tolerances."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
    np.testing.assert_allclose(actual / scales, expected / scales, rtol=5e-7, atol=5e-8)
