"""Unshifted, coupled steady Newton with the installed custom PySolverView.

Every cell and both algebraic boundary traces remain independent unknowns.
The equations are exactly ``Operator.steady_residual``: no pseudo-time term,
boundary elimination, equation clipping, or modified thermodynamic closure.
The fixed variable scaling is centered at the supplied initial field so the
first physical evaluation preserves that field bit for bit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pysolver_view as psv

from nozzle_hyperbolic.solvers.fvm.drivers import sampler


def state_hash(full):
    """Hash a canonical little-endian float64, C-order physical state."""
    return sha256(np.asarray(full, dtype="<f8", order="C").tobytes()).hexdigest()


class DirectProblem:
    """NumPy interface to exact AD of the original, fixed-scaled steady rows."""

    def __init__(self, operator, initial_full):
        if not jax.config.x64_enabled:
            raise ValueError("Direct Newton requires jax_enable_x64")
        self.operator = operator
        self.shape = operator.shape
        self.initial_full = np.array(initial_full, dtype=float, copy=True)
        if self.initial_full.shape != self.shape:
            raise ValueError(f"Expected full cell-and-boundary state {self.shape}")
        self.scales = np.broadcast_to(
            np.asarray(operator.model.variable_scales, dtype=float), self.shape
        ).copy()
        self.initial_state_sha256 = state_hash(self.initial_full)
        self.first_evaluated_state_sha256 = None
        self.domain_failures = []
        self.model_residual_calls = 0
        self.model_jacobian_calls = 0
        reference, scales = jnp.asarray(self.initial_full), jnp.asarray(self.scales)

        def residual(coordinates):
            full = reference + coordinates.reshape(self.shape) * scales
            return operator.steady_residual(full.ravel(), 0.0)

        self._residual = jax.jit(residual)
        self._gradient = jax.jit(jax.jacfwd(residual))
        self._jvp = jax.jit(lambda z, v: jax.jvp(residual, (z,), (v,))[1])

    @property
    def x0(self):
        return np.zeros(self.initial_full.size, dtype=float)

    def full_state(self, coordinates):
        """Recover physical variables without altering signs or table bounds."""
        point = np.asarray(coordinates, dtype=float)
        if point.shape != (self.initial_full.size,):
            raise ValueError("Expected one coordinate per full-state variable")
        return self.initial_full + point.reshape(self.shape) * self.scales

    def _checked(self, coordinates, kind):
        full = self.full_state(coordinates)
        # Check cells, traces and reconstructed faces before calling the
        # physical residual/Jacobian. Zero and signed velocities are allowed
        # by the model; no positivity condition is imposed on velocity here.
        valid = np.isfinite(full).all() and bool(
            self.operator.is_admissible(jnp.asarray(full))
        )
        if not valid:
            reason = "State, boundary trace or reconstructed face outside model domain"
            self.domain_failures.append(dict(
                kind=kind, reason=reason,
                coordinates=np.array(coordinates, dtype=float, copy=True),
                physical_state=full.copy(),
            ))
            raise psv.DomainError(reason)
        return full

    def residual(self, coordinates):
        self.model_residual_calls += 1
        full = self._checked(coordinates, "residual")
        if self.first_evaluated_state_sha256 is None:
            self.first_evaluated_state_sha256 = state_hash(full)
            if self.first_evaluated_state_sha256 != self.initial_state_sha256:
                raise RuntimeError("The first direct-Newton state differs from its supplied start")
        values = np.asarray(self._residual(jnp.asarray(coordinates)), dtype=float)
        if not np.isfinite(values).all():
            raise psv.DomainError("Nonfinite original steady residual")
        return values.copy()

    def gradient(self, coordinates):
        self.model_jacobian_calls += 1
        self._checked(coordinates, "jacobian")
        values = np.asarray(self._gradient(jnp.asarray(coordinates)), dtype=float)
        if not np.isfinite(values).all():
            raise psv.DomainError("Nonfinite exact AD Jacobian of original steady residual")
        return values.copy()

    def compile(self):
        """Compile the three used graphs separately from solve timing."""
        started = perf_counter()
        self._checked(self.x0, "initial_compilation")
        residual = np.asarray(self._residual(jnp.asarray(self.x0)))
        jacobian = np.asarray(self._gradient(jnp.asarray(self.x0)))
        if not (np.isfinite(residual).all() and np.isfinite(jacobian).all()):
            raise psv.DomainError("Nonfinite initial residual or exact AD Jacobian")
        return perf_counter() - started


def _metrics_from_rows(rows, operator, policy, initial=None):
    """Host reductions of the backend's existing, fixed-scaled equation values."""
    values = np.asarray(rows, dtype=float).reshape(operator.shape)
    cell = values[1:-1]
    volumes = np.asarray(operator.mesh.volumes, dtype=float)
    weights = (volumes / volumes.sum() if policy.weighting == "volume"
               else np.full(volumes.size, 1.0 / volumes.size))
    absolute = np.abs(cell)
    largest = absolute.max(axis=0)
    safe = np.where(largest > 0, largest, 1.0)
    l2 = largest * np.sqrt(np.sum(weights[:, None] * (cell / safe)**2, axis=0))
    l1 = np.sum(weights[:, None] * absolute, axis=0)
    current = l2 if policy.norm == "l2" else l1
    initial = current.copy() if initial is None else np.asarray(initial, dtype=float)
    active = initial > policy.initial_floor
    relative = np.divide(current, initial, out=np.zeros_like(current), where=initial > 0)
    absolute_ok = current <= policy.absolute_tolerance
    relative_ok = active & (relative <= policy.relative_tolerance)
    absolute_ratio = current / policy.absolute_tolerance
    relative_ratio = relative / policy.relative_tolerance
    if policy.criterion == "either":
        passed = absolute_ok | relative_ok
        ratio = np.where(active, np.minimum(absolute_ratio, relative_ratio), absolute_ratio)
    elif policy.criterion == "both":
        passed = absolute_ok & (~active | relative_ok)
        ratio = np.maximum(absolute_ratio, np.where(active, relative_ratio, 0.0))
    elif policy.criterion == "relative":
        passed = np.where(active, relative_ok, absolute_ok)
        ratio = np.where(active, relative_ratio, absolute_ratio)
    else:
        passed, ratio = absolute_ok, absolute_ratio
    model = operator.model
    signed_global = np.sum(
        cell * volumes[:, None] * np.asarray(model.conservative_scales)[None, :]
        / float(model.reference_time), axis=0
    ) / np.asarray(model.global_flux_scales)
    boundary_max = float(np.max(np.abs(values[[0, -1]])))
    finite = bool(np.isfinite(values).all() and np.isfinite(initial).all())
    return dict(
        equation_norm=current, equation_l1=l1, equation_l2=l2,
        equation_linf=largest, worst_cell=np.argmax(absolute, axis=0),
        equation_relative=relative, relative_active=active,
        equation_passed=passed & finite,
        equation_max_ratio=float(np.max(ratio)),
        all_equations_passed=bool(passed.all() and finite),
        absolute_satisfied=absolute_ok & finite, relative_satisfied=relative_ok & finite,
        overall_relative=float(np.max(np.where(active, relative, 0.0))),
        criterion=policy.criterion, boundary_max=boundary_max,
        global_defect=np.abs(signed_global), signed_global_defect=signed_global,
        finite=finite, admissible=True,
        boundary_passed=bool(boundary_max <= policy.boundary_tolerance),
        global_passed=bool(np.max(np.abs(signed_global)) <= policy.global_tolerance),
    )


def _common_passed(metrics):
    return bool(all(metrics[key] for key in (
        "finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed"
    )))


def _native_tolerance(operator, policy):
    """Conservative fallback; accepted callbacks apply the actual common policy.

    A two-norm bound implies each absolute weighted equation bound, every
    boundary row, and every global balance through Cauchy--Schwarz. It is not
    used in place of the relative-or-absolute shared stopping policy.
    """
    model = operator.model
    volumes = np.asarray(operator.mesh.volumes, dtype=float)
    weights = (volumes / volumes.sum() if policy.weighting == "volume"
               else np.full(volumes.size, 1.0 / volumes.size))
    global_mapping = (volumes[:, None] * np.asarray(model.conservative_scales)[None, :]
                      / float(model.reference_time) / np.asarray(model.global_flux_scales)[None, :])
    mapping_norm = np.max(np.linalg.norm(global_mapping, axis=0))
    return float(min(policy.absolute_tolerance / np.sqrt(weights.max()),
                     policy.boundary_tolerance, policy.global_tolerance / mapping_norm))


def solve_direct(context, initial_full, policy, options=None, *, wall_seconds=600):
    """Solve the unshifted square system; keep native and common success separate.

    ``context`` supplies ``operator``. The solve timer includes backend work
    and reductions of available residual rows, including rejected trials. JIT
    compilation and the independent final original-equation audit are separate.
    The wall budget is checked at accepted points; a running Jacobian or a
    line-search sequence is completed before a callback can stop it.
    """
    operator = context.operator
    problem = DirectProblem(operator, initial_full)
    native_tolerance = _native_tolerance(operator, policy)
    requested_options = dict(
        first_step_fraction=0.25, step_growth_factor=1.5,
        history_level="full", backtracking={"step_limits": None},
    )
    requested_options.update(deepcopy(options or {}))
    max_iterations = int(requested_options.pop("max_iterations", 300))
    max_accepted_steps = requested_options.pop("max_accepted_steps", 400)
    if (isinstance(max_accepted_steps, (bool, np.bool_))
            or not isinstance(max_accepted_steps, (int, np.integer))
            or max_accepted_steps < 1):
        raise ValueError("max_accepted_steps must be a positive integer")
    max_accepted_steps = int(max_accepted_steps)
    if requested_options.get("history_level") != "full":
        raise ValueError("This study retains full accepted and trial histories")
    if "residual_tol" in requested_options:
        raise ValueError("Native fallback tolerance is derived from the shared policy")
    compile_seconds = problem.compile()
    initial_norm = None
    history = []
    stop_reason = None
    last_accepted = problem.x0.copy()
    independent_sample = sampler(operator, policy)
    monitor_original_evaluations = 0

    def accepted(point, _callback_count):
        nonlocal initial_norm, stop_reason, last_accepted, monitor_original_evaluations
        entry = solver.iteration_history[-1]
        metrics = _metrics_from_rows(entry["fun"], operator, policy, initial_norm)
        if initial_norm is None:
            # The native driver defines its baseline from exactly this initial
            # field. Retain it unchanged across all accepted and rejected work.
            initial_norm = np.asarray(metrics["equation_norm"]).copy()
        elapsed = perf_counter() - started
        row = dict(
            accepted=True, step=int(entry["iteration"]),
            native_iteration=int(entry["iteration"]),
            phase="initial" if entry["iteration"] == 0 else "accepted",
            accepted_index=len(history), elapsed_seconds=elapsed, elapsed_s=elapsed,
            dt=0.0, **metrics,
        )
        for key in ("step_fraction", "backtracks", "step_norm", "scaled_step_norm", "nfev", "njev"):
            if key in entry:
                row[key] = deepcopy(entry[key])
        row["common_success"] = _common_passed(metrics)
        last_accepted = np.array(point, dtype=float, copy=True)
        if row["common_success"]:
            # The standalone spatial graph may differ in final-bit roundoff
            # from a fused steady graph. Verify its actual shared checks before
            # stopping; otherwise let Newton continue with the same targets.
            independent = independent_sample(jnp.asarray(problem.full_state(point)), initial_norm)
            monitor_original_evaluations += 1
            row["original_check_before_stop"] = independent
            row["common_success"] = _common_passed(independent)
        history.append(row)
        if row["common_success"]:
            stop_reason = "common_target_reached"
            return False
        if int(entry["iteration"]) >= max_accepted_steps:
            stop_reason = f"newton_step_limit_{max_accepted_steps}"
            return False
        if wall_seconds is not None and elapsed >= wall_seconds:
            stop_reason = "wall_limit_at_accepted_point"
            return False
        return None

    solver = psv.NonlinearSystemSolver(
        problem, library="pysolver", method="newton_raphson",
        tolerance=native_tolerance, max_iterations=max_iterations,
        options=requested_options, update_on="iteration", callback_func=accepted,
        print_convergence=False, plot_convergence=False,
    )
    exception = None
    started = perf_counter()
    try:
        solver.solve(problem.x0)
    except Exception as error:
        # Preserve the accepted iterate and classify unexpected failures. Never
        # replace them with a different direction, physical problem or seed.
        exception = dict(type=type(error).__name__, message=str(error))
    elapsed_seconds = perf_counter() - started
    backend = solver.backend_result
    point = np.asarray(solver.x_final if solver.x_final is not None else last_accepted, dtype=float)
    full = problem.full_state(point)
    if initial_norm is None:
        initial_norm = np.asarray(independent_sample(jnp.asarray(initial_full))["equation_norm"])
    final_audit_started = perf_counter()
    metrics = independent_sample(jnp.asarray(full), initial_norm)
    unshifted = np.asarray(operator.steady_residual(jnp.asarray(full).ravel(), 0.0))
    raw = np.asarray(operator.spatial_residual(jnp.asarray(full)))
    boundary = np.asarray(operator.boundary_residual(jnp.asarray(full), 0.0))
    original_success = _common_passed(metrics)
    model_audit = getattr(operator.model, "solution_audit", None)
    selected_regime_passed = bool(model_audit(jnp.asarray(full))) if model_audit is not None else True
    final_audit_seconds = perf_counter() - final_audit_started
    native_success = bool(solver.success) if solver.success is not None else False
    native_status = int(backend.status) if backend is not None else None
    native_message = str(solver.message) if solver.message is not None else "Backend raised before returning a result"
    reason = stop_reason or ("converged" if original_success else native_message)
    counters = {}
    if backend is not None:
        for key in (
            "nfev", "njev", "nit", "nupdates", "ntrials", "nrejected", "ninvalid", "nbacktracks",
            "derivative_nfev", "failed_njev", "linear_algebra", "numerical_rank", "linear_residual_norm",
            "max_trial_step_scaled", "step_limit", "column_scaling_min", "column_scaling_max",
        ):
            if key in backend:
                counters[key] = deepcopy(backend[key])
    counters.update(
        model_residual_calls=problem.model_residual_calls,
        model_jacobian_calls=problem.model_jacobian_calls,
        explicitly_recorded_domain_failures=len(problem.domain_failures),
        original_monitor_evaluations=monitor_original_evaluations,
    )
    effective_options = (deepcopy(backend.effective_options)
                         if backend is not None else asdict(solver.options))
    # The accepted-update cap is enforced by this adapter, not an unsupported
    # option passed to the native backend. Record both layers explicitly.
    effective_options["max_accepted_steps"] = max_accepted_steps
    return dict(
        full_state=full.copy(), state=full.copy(), metrics=metrics, initial_norm=initial_norm,
        success=bool(original_success and selected_regime_passed and exception is None),
        common_success=bool(original_success), selected_regime_passed=selected_regime_passed,
        reason=reason, native_success=native_success, native_status=native_status,
        native_message=native_message, exception=exception,
        options=effective_options,
        limits=dict(max_accepted_steps=max_accepted_steps,
                    max_trial_attempts=effective_options["max_iterations"],
                    max_residual_evaluations=effective_options["max_nfev"],
                    wall_seconds=wall_seconds),
        backend=dict(library=solver.library, method=solver.method, version=psv.__version__,
                     linear_solver="dense pivoted QR with strict full-rank Newton; no rank truncation",
                     jacobian="exact dense forward AD in fixed centered scaled coordinates"),
        history=history,
        native_history=deepcopy(solver.iteration_history),
        trials=deepcopy(backend.get("trials", [])) if backend is not None else [],
        evaluations=deepcopy(backend.get("evaluations", [])) if backend is not None else [],
        domain_failures=problem.domain_failures,
        counters=counters, elapsed_seconds=elapsed_seconds, compile_seconds=compile_seconds,
        final_audit_seconds=final_audit_seconds,
        initial_state_sha256=problem.initial_state_sha256,
        actual_initial_state_sha256=problem.first_evaluated_state_sha256,
        original_steady_residual=unshifted, original_integrated_spatial_residual=raw,
        original_boundary_residual=boundary,
        native_tolerance=native_tolerance,
        stopping_definition="shared policy at accepted states; native callback status is separate from physical root checks",
        coordinate_definition="full = initial_full + z * model.variable_scales; z0 is exactly zero",
        wall_budget=dict(seconds=wall_seconds, enforcement="accepted-point callback; not a hard interruption"),
    )
