"""Generic Newton/PTC contracts using tiny systems and the real custom backend."""

import numpy as np
import pysolver_view as psv
import pytest

from development_new.newton_solve.scripts.ptc import solve_ptc


class LinearProblem:
    """Count exact linear equations without supplying an implicit timestepper."""

    def __init__(self, matrix, root):
        self.matrix = np.asarray(matrix, dtype=float)
        self.root = np.asarray(root, dtype=float)
        self.residual_calls = self.jacobian_calls = 0

    def residual(self, point):
        """Return actual physical equations rather than an affine-inner metric."""
        self.residual_calls += 1
        return self.matrix @ (np.asarray(point) - self.root)

    def gradient(self, point):
        """Return the exact physical Jacobian and count every request."""
        self.jacobian_calls += 1
        return self.matrix.copy()

    def step(self, *args, **kwargs):
        """Fail if a purported single-loop method invokes hidden time marching."""
        raise AssertionError("PTC must not call a physical implicit timestepper")

    @property
    def initial_state(self):
        """Reject initialization from anything other than the supplied x0."""
        raise AssertionError("PTC must use the caller's x0")

    @property
    def reference(self):
        """Reject use of a reference solution during generic convergence."""
        raise AssertionError("PTC must not access a reference solution")


class ConstantStorage:
    """Supply an independently known, possibly nonsymmetric storage matrix."""

    def __init__(self, matrix, characteristic_time=1.0):
        self.value = np.asarray(matrix, dtype=float)
        self.time = float(characteristic_time)
        self.matrix_calls = self.time_calls = 0

    def matrix(self, point):
        """Return the scaled storage matrix without doing physical evolution."""
        self.matrix_calls += 1
        return self.value.copy()

    def characteristic_dt(self, point):
        """Return the exact CFL-one time used by the independent step oracle."""
        self.time_calls += 1
        return self.time


@pytest.mark.parametrize("fraction", [1.0, 0.25])
def test_one_ptc_correction_matches_independent_shifted_linear_solve(fraction):
    """One custom affine solve must implement the stated mass shift and damping."""
    problem = LinearProblem([[2.0, 0.0], [0.3, 3.0]], [1.0, 2.0])
    storage = ConstantStorage([[2.0, 0.4], [0.1, 1.5]], characteristic_time=0.5)
    initial = np.array([0.5, 0.75])
    residual = problem.matrix @ (initial - problem.root)
    dt = 2.0 * storage.time
    expected = initial - fraction * np.linalg.solve(
        problem.matrix + storage.value / dt, residual
    )
    result = solve_ptc(
        problem,
        storage,
        initial,
        max_iterations=1,
        initial_cfl=2.0,
        maximum_cfl=2.0,
        first_step_fraction=fraction,
        step_growth_factor=1.0,
        max_relative_step=10.0,
        newton_switch_ratio=0.0,
    )
    np.testing.assert_allclose(result["x"], expected, rtol=1e-13, atol=1e-13)
    assert result["iterations"] == 1
    assert len(result["trial_history"]) == 1
    assert result["history"][-1]["rejected_trials"] == 0
    inner = result["trial_history"][0]["backend"]
    assert inner["library"] == "pysolver"
    assert inner["method"] == "newton_raphson"
    assert inner["updates"] == inner["trials"] == 1
    assert inner["affine_residual_calls"] == 2
    assert inner["affine_jacobian_calls"] == 1
    assert inner["effective_options"]["backtracking"] is None
    assert inner["effective_options"]["first_step_fraction"] == fraction
    assert result["nfev"] == problem.residual_calls == 2
    assert result["njev"] == problem.jacobian_calls == 1
    assert not result["root_success"]
    assert not result["native_success"]


def test_rejected_increment_trials_reuse_jacobian_and_storage_at_same_anchor():
    """Retries change the shift without recomputing unchanged local matrices."""
    problem = LinearProblem([[1.0]], [1.0])
    storage = ConstantStorage([[1.0]])
    initial = np.array([0.1])
    result = solve_ptc(
        problem,
        storage,
        initial,
        max_iterations=1,
        initial_cfl=1.0,
        first_step_fraction=1.0,
        max_relative_step=0.1,
        max_retries=12,
        newton_switch_ratio=0.0,
    )
    trials = result["trial_history"]
    assert len(trials) > 1
    assert any(row["reason"] == "relative_step_limit" for row in trials)
    assert sum(row["accepted"] for row in trials) == 1
    assert problem.jacobian_calls == result["njev"] == 1
    assert storage.matrix_calls == storage.time_calls == 1
    assert problem.residual_calls == result["nfev"] == 2
    assert result["x"][0] > initial[0]
    assert np.max(np.abs(result["x"] - initial) / (1 + np.abs(initial))) <= 0.1
    assert len(result["history"]) == 2
    assert result["backend"]["rejected_trials"] == len(trials) - 1
    assert result["history"][-1]["rejected_trials"] == len(trials) - 1


def test_cold_generic_nonsymmetric_storage_reaches_original_root_without_time_steps():
    """The generic single loop solves from x0 without a reference or timestepper."""
    problem = LinearProblem([[2.0, 0.0], [0.3, 3.0]], [1.0, 2.0])
    storage = ConstantStorage([[2.0, 0.4], [0.1, 1.5]])
    result = solve_ptc(
        problem,
        storage,
        np.array([0.05, 0.1]),
        residual_rms_tol=1e-9,
        max_iterations=200,
        initial_cfl=1.0,
    )
    assert result["root_success"] and result["native_success"], result["message"]
    independent = problem.matrix @ (result["x"] - problem.root)
    assert np.linalg.norm(independent) / np.sqrt(independent.size) <= 1e-9
    assert result["nfev"] == problem.residual_calls
    assert result["njev"] == problem.jacobian_calls
    assert result["backend"]["library"] == "pysolver"
    assert result["backend"]["method"] == "newton_raphson"
    assert result["backend"]["affine_solves"] == len(result["trial_history"])
    assert len(result["history"]) == result["iterations"] + 1
    assert all(row["backend"]["updates"] <= 1 for row in result["trial_history"])


def test_bounded_true_residual_increase_is_not_physical_root_success():
    """Affine success can coexist with an allowed rise in the actual residual."""
    problem = LinearProblem([[-1.0]], [1.0])
    result = solve_ptc(
        problem,
        ConstantStorage([[2.0]]),
        [2.0],
        max_iterations=1,
        initial_cfl=1.0,
        first_step_fraction=1.0,
        max_relative_step=10.0,
        residual_growth_limit=2.01,
        newton_switch_ratio=0.0,
    )
    np.testing.assert_allclose(result["x"], [3.0], rtol=0.0, atol=1e-14)
    assert result["trial_history"][0]["accepted"]
    assert result["trial_history"][0]["residual_increased"]
    assert result["trial_history"][0]["backend"]["affine_native_success"]
    assert result["history"][-1]["residual_rms"] == pytest.approx(2.0)
    assert not result["root_success"]
    assert not result["native_success"]


class LogDomainProblem:
    """Require domain failures to remain actual failed evaluations."""

    def __init__(self):
        self.residual_calls = self.jacobian_calls = 0

    def residual(self, point):
        """Return an actual log equation or its explicit domain failure."""
        self.residual_calls += 1
        if point[0] <= 0:
            raise psv.DomainError("Log state must be positive")
        return np.array([np.log(point[0]) + 10.0])

    def gradient(self, point):
        """Count exact derivatives at the unchanged positive anchor."""
        self.jacobian_calls += 1
        return np.array([[1.0 / point[0]]])


def test_domain_retries_keep_failed_trials_and_reuse_physical_jacobian():
    """PTC must retry real invalid candidates rather than inventing residuals."""
    problem = LogDomainProblem()
    result = solve_ptc(
        problem,
        ConstantStorage([[1.0]]),
        [1.0],
        max_iterations=1,
        first_step_fraction=1.0,
        max_relative_step=10.0,
        max_retries=12,
        newton_switch_ratio=0.0,
    )
    failures = [
        row for row in result["trial_history"] if row["reason"] == "physical_domain"
    ]
    assert failures
    assert all(not row["accepted"] and row["residual_rms"] is None for row in failures)
    assert result["iterations"] == 1
    assert result["x"][0] > 0
    assert result["njev"] == problem.jacobian_calls == 1
    assert result["nfev"] == problem.residual_calls
    assert not result["root_success"]


def test_already_converged_seed_needs_no_affine_or_storage_work():
    """Only the original residual determines whether initial work is needed."""
    problem = LinearProblem([[2.0]], [1.0])
    storage = ConstantStorage([[1.0]])
    result = solve_ptc(problem, storage, [1.0])
    assert result["root_success"] and result["native_success"]
    assert result["iterations"] == 0
    assert result["trial_history"] == []
    assert result["nfev"] == problem.residual_calls == 1
    assert result["njev"] == problem.jacobian_calls == 0
    assert storage.matrix_calls == storage.time_calls == 0
    assert result["backend"]["affine_solves"] == 0


def test_well_modeled_plateau_probe_obeys_configured_cfl_growth_cap():
    """A growth probe must honor a caller's cap even when it is below 1.05."""
    result = solve_ptc(
        LinearProblem([[1.0]], [1.0]),
        ConstantStorage([[1.0]]),
        [0.5],
        max_iterations=3,
        initial_cfl=1e-5,
        first_step_fraction=1.0,
        step_growth_factor=1.0,
        cfl_growth_limit=1.01,
        newton_switch_ratio=0.0,
    )
    assert result["iterations"] == 3
    cfls = [row["cfl"] for row in result["history"]]
    assert all(
        after <= before * 1.01 * (1 + 1e-12) for before, after in zip(cfls, cfls[1:])
    )


def test_unrepresentable_affine_step_stops_without_more_regularization_retries():
    """Increasing the shift cannot restore a correction already lost to rounding."""
    problem = LinearProblem([[1.0]], [1.0])
    result = solve_ptc(
        problem,
        ConstantStorage([[1e20]]),
        [0.5],
        max_iterations=3,
        first_step_fraction=1.0,
        max_retries=12,
        newton_switch_ratio=0.0,
    )
    assert not result["root_success"] and not result["native_success"]
    assert result["iterations"] == 0
    assert len(result["trial_history"]) == 1
    trial = result["trial_history"][0]
    assert trial["reason"] == "affine_step_stagnation"
    assert not trial["accepted"]
    assert trial["backend"]["updates"] == 0
    assert "stagnation" in trial["backend"]["affine_message"].lower()
    assert "stagnation" in result["message"].lower()
    assert result["nfev"] == problem.residual_calls == 1
    assert result["njev"] == problem.jacobian_calls == 1
    np.testing.assert_array_equal(result["x"], [0.5])
