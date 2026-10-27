"""Custom Newton identity, root and accepted-history contracts, without PDEs."""

import numpy as np

from development_new.newton_solve.scripts.direct_solvers import solve_newton


class CountedLinearProblem:
    """Expose an independently specified root and count actual model work."""

    def __init__(self):
        self.matrix = np.array([[3.0, 0.5], [0.25, 2.0]])
        self.root = np.array([1.0, 2.0])
        self.residual_points = []
        self.jacobian_calls = 0

    def residual(self, point):
        """Evaluate the exact linear equations and retain a detached request."""
        self.residual_points.append(np.array(point, dtype=float, copy=True))
        return self.matrix @ (np.asarray(point) - self.root)

    def gradient(self, point):
        """Return the exact Jacobian without evaluating the residual."""
        self.jacobian_calls += 1
        return self.matrix.copy()


def test_custom_newton_reaches_root_with_truthful_accepted_history():
    """Backend identity, work accounting, and root audits must agree independently."""
    problem = CountedLinearProblem()
    result = solve_newton(
        problem, np.array([0.5, 0.5]), residual_rms_tol=1e-9, max_iterations=200
    )
    assert result["backend"]["library"] == "pysolver"
    assert result["backend"]["method"] == "newton_raphson"
    assert result["native_success"], result["message"]
    assert result["nfev"] == len(problem.residual_points)
    assert result["njev"] == problem.jacobian_calls
    np.testing.assert_allclose(result["x"], problem.root, rtol=0.0, atol=1e-8)
    independent = problem.matrix @ (result["x"] - problem.root)
    assert np.linalg.norm(independent) / np.sqrt(independent.size) <= 1e-9
    history = result["history"]
    assert len(history) > 1
    assert [row["accepted_index"] for row in history] == list(range(len(history)))
    np.testing.assert_allclose(
        history[-1]["residual_rms"],
        np.linalg.norm(independent) / np.sqrt(independent.size),
        rtol=0.0,
        atol=1e-15,
    )
    assert all(row["elapsed_s"] <= result["wall_s"] for row in history)
    assert all(a["elapsed_s"] <= b["elapsed_s"] for a, b in zip(history, history[1:]))
    assert result["iterations"] == len(history) - 1
    assert result["backend"]["ntrials"] >= result["iterations"]
    np.testing.assert_allclose(
        result["backend"]["effective_options"]["residual_tol"], 1e-9 * np.sqrt(2)
    )
