"""Single-loop Newton/pseudo-transient continuation using custom PySolverView.

The original steady residual is never modified. Each trial delegates exactly
one Newton correction of a frozen affine model to PySolverView, then audits
the actual nonlinear residual. No backward-Euler timestep problem is solved.
The storage provider supplies the consistently scaled physical mass matrix
and a mesh/wave-speed characteristic time; the controller has no nozzle or
shock-location knowledge.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from time import perf_counter

import numpy as np
import pysolver_view as psv


class _AffineCorrection:
    """Exact affine equations for one shifted correction, not the physical PDE."""

    def __init__(self, anchor, residual, matrix):
        self.anchor = anchor
        self.value = residual
        self.matrix = matrix
        self.residual_calls = self.jacobian_calls = 0

    def residual(self, point):
        self.residual_calls += 1
        return self.value + self.matrix @ (point - self.anchor)

    def gradient(self, point):
        self.jacobian_calls += 1
        return self.matrix


def _one_custom_correction(anchor, residual, matrix, fraction):
    """Use the public custom Newton backend for one affine correction only."""
    model = _AffineCorrection(anchor, residual, matrix)
    solver = psv.NonlinearSystemSolver(
        model,
        library="pysolver",
        method="newton_raphson",
        tolerance=max(1e-14, 1e-10 * float(np.linalg.norm(residual))),
        max_iterations=1,
        options={
            "max_nfev": 2,
            "backtracking": None,
            "first_step_fraction": fraction,
            "step_growth_factor": 1.0,
            "x_scale": 1.0,
            "jacobian_scaling": True,
            "history_level": "full",
        },
        update_on="iteration",
        print_convergence=False,
        plot_convergence=False,
    )
    solver.solve(anchor.copy())
    if solver.library != "pysolver" or solver.method != "newton_raphson":
        raise RuntimeError("PTC requires the custom PySolverView Newton backend")
    native = solver.backend_result
    candidate = np.array(solver.x_final, dtype=float, copy=True)
    updates = int(native.nit)
    trials = int(native.get("ntrials", updates))
    if updates > 1 or trials > 1:
        raise RuntimeError("The affine backend performed more than one correction")
    step = candidate - anchor
    expected = (1.0 - fraction) * residual
    linear_defect = residual + matrix @ step - expected
    return candidate, dict(
        library=solver.library,
        method=solver.method,
        pysolver_view_version=psv.__version__,
        updates=updates,
        trials=trials,
        affine_native_success=bool(solver.success),
        affine_message=str(solver.message),
        affine_status=int(native.status),
        affine_residual_calls=model.residual_calls,
        affine_jacobian_calls=model.jacobian_calls,
        linear_relative_defect=float(
            np.linalg.norm(linear_defect)
            / max(fraction * np.linalg.norm(residual), 1e-300)
        ),
        effective_options=deepcopy(native.effective_options),
    )


def solve_ptc(
    problem,
    storage,
    x0,
    *,
    residual_rms_tol=1e-8,
    max_iterations=3000,
    initial_cfl=1.0,
    minimum_cfl=1e-6,
    maximum_cfl=1e8,
    first_step_fraction=0.25,
    step_growth_factor=1.5,
    ser_exponent=0.7,
    cfl_growth_limit=1.5,
    cfl_shrink_limit=0.5,
    max_relative_step=0.15,
    nonmonotone_window=10,
    residual_growth_limit=2.0,
    model_error_limit=1.0,
    max_retries=12,
    newton_switch_ratio=1e-3,
    progress=None,
):
    """Solve one fixed steady system with adaptive physical-storage shifts.

    problem supplies residual(z), gradient(z) and optionally _validate(z).
    storage supplies matrix(z) in the SAME scaled units as gradient(z), and
    characteristic_dt(z), the positive time for CFL=1. Every trial solves
    [J+M/dt] dz=-R with the stated direction fraction. J and M are refreshed
    after acceptance and reused on rejection at the unchanged anchor.

    The controller permits bounded nonmonotone true-residual growth. Domain,
    relative-change, nonlinear-model and affine-solve audits can reject a
    candidate and decrease CFL. SER updates CFL after acceptance, with a
    small growth probe on a well-modeled plateau. Near a root, a zero-shift
    trial is allowed after three well-modeled steps; unsuccessful trials
    return to finite regularization. Only original residual RMS determines
    success. Inner affine success never establishes a physical steady root.

    max_iterations counts accepted physical corrections; max_retries limits
    trials per anchor. Histories distinguish physical accepted states from
    all trial candidates and affine backend observations. Timings include
    construction, failed trials and monitoring, but exclude caller JIT warmup.
    """
    point = np.array(x0, dtype=float, copy=True)
    if point.ndim != 1 or not point.size or not np.isfinite(point).all():
        raise ValueError("x0 must be a finite nonempty vector")
    for name, value in (
        ("residual_rms_tol", residual_rms_tol),
        ("minimum_cfl", minimum_cfl),
        ("initial_cfl", initial_cfl),
        ("maximum_cfl", maximum_cfl),
        ("max_relative_step", max_relative_step),
        ("model_error_limit", model_error_limit),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not minimum_cfl <= initial_cfl <= maximum_cfl:
        raise ValueError("Require minimum_cfl <= initial_cfl <= maximum_cfl")
    if (
        not 0 < first_step_fraction <= 1
        or not np.isfinite(step_growth_factor)
        or step_growth_factor < 1
    ):
        raise ValueError("Use a direction fraction in (0,1] and growth >=1")
    if not 0 < cfl_shrink_limit < 1 < cfl_growth_limit:
        raise ValueError("CFL shrink/growth limits must surround one")
    if not np.isfinite(ser_exponent) or ser_exponent < 0:
        raise ValueError("ser_exponent must be finite and nonnegative")
    if not np.isfinite(residual_growth_limit) or residual_growth_limit < 1:
        raise ValueError("residual_growth_limit must be finite and >=1")
    for name, value in (
        ("max_iterations", max_iterations),
        ("max_retries", max_retries),
        ("nonmonotone_window", nonmonotone_window),
    ):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(newton_switch_ratio) or newton_switch_ratio < 0:
        raise ValueError("newton_switch_ratio must be finite and nonnegative")

    started = perf_counter()
    physical_residual_calls = physical_jacobian_calls = storage_calls = 0
    affine_residual_calls = affine_jacobian_calls = 0
    history, trial_history = [], []
    rejection_counts = Counter()
    cfl = float(initial_cfl)
    fraction = float(first_step_fraction)
    well_modeled = 0
    newton_mode = False
    first_backend = None

    def residual_at(value):
        nonlocal physical_residual_calls
        physical_residual_calls += 1
        if hasattr(problem, "_validate"):
            problem._validate(value)
        result = np.asarray(problem.residual(value), dtype=float)
        if result.shape != point.shape:
            raise ValueError("Incorrect physical residual shape")
        if not np.isfinite(result).all():
            raise psv.DomainError("Non-finite physical residual")
        return result

    def rms(value):
        return float(np.linalg.norm(value) / np.sqrt(point.size))

    residual = residual_at(point)
    initial_rms = current_rms = rms(residual)
    history.append(
        dict(
            accepted_index=0,
            elapsed_s=perf_counter() - started,
            residual_rms=current_rms,
            cfl=cfl,
            dt=None,
            shift=None,
            step_fraction=0.0,
        )
    )
    message = "Maximum accepted PTC iterations reached"
    for iteration in range(1, int(max_iterations) + 1):
        if current_rms <= residual_rms_tol:
            message = "Original steady residual target reached"
            break
        physical_jacobian_calls += 1
        jacobian = np.asarray(problem.gradient(point), dtype=float)
        storage_calls += 1
        mass = np.asarray(storage.matrix(point), dtype=float)
        if jacobian.shape != (point.size, point.size) or mass.shape != jacobian.shape:
            raise ValueError("Incorrect Jacobian or physical-storage matrix shape")
        if not np.isfinite(jacobian).all() or not np.isfinite(mass).all():
            raise psv.DomainError("Non-finite Jacobian or storage matrix")
        characteristic_time = float(storage.characteristic_dt(point))
        if not np.isfinite(characteristic_time) or characteristic_time <= 0:
            raise ValueError("characteristic_dt must be positive and finite")
        accepted = False
        try_newton = newton_mode or (
            newton_switch_ratio > 0
            and current_rms <= initial_rms * newton_switch_ratio
            and well_modeled >= 3
        )
        for retry in range(int(max_retries)):
            dt = cfl * characteristic_time
            shift = 0.0 if try_newton else 1.0 / dt
            matrix = jacobian if try_newton else jacobian + shift * mass
            candidate, backend = _one_custom_correction(
                point, residual, matrix, fraction
            )
            affine_residual_calls += backend["affine_residual_calls"]
            affine_jacobian_calls += backend["affine_jacobian_calls"]
            if first_backend is None:
                first_backend = deepcopy(backend)
            step = candidate - point
            relative_step = float(np.max(np.abs(step) / (1.0 + np.abs(point))))
            trial = dict(
                trial_index=len(trial_history),
                anchor_index=len(history) - 1,
                retry=retry,
                elapsed_s=perf_counter() - started,
                cfl=cfl,
                dt=None if try_newton else dt,
                shift=shift,
                step_fraction=fraction,
                relative_step=relative_step,
                residual_rms=None,
                model_error=None,
                residual_increased=None,
                accepted=False,
                backend=backend,
            )
            candidate_residual = None
            reason = None
            if backend["updates"] != 1:
                reason = (
                    "affine_step_stagnation"
                    if "stagnation" in backend["affine_message"].lower()
                    else "affine_backend_zero_updates"
                )
            elif backend["linear_relative_defect"] > 1e-6:
                reason = "inaccurate_affine_correction"
            elif relative_step > max_relative_step:
                reason = "relative_step_limit"
            else:
                try:
                    candidate_residual = residual_at(candidate)
                    candidate_rms = rms(candidate_residual)
                    linear_prediction = residual + jacobian @ step
                    model_error = float(
                        np.linalg.norm(candidate_residual - linear_prediction)
                        / max(
                            np.linalg.norm(residual),
                            np.linalg.norm(linear_prediction),
                            1e-300,
                        )
                    )
                    window_max = max(
                        row["residual_rms"]
                        for row in history[-int(nonmonotone_window) :]
                    )
                    trial.update(
                        residual_rms=candidate_rms,
                        model_error=model_error,
                        residual_increased=bool(candidate_rms > current_rms),
                    )
                    if candidate_rms > residual_growth_limit * window_max:
                        reason = "nonmonotone_residual_limit"
                    elif model_error > model_error_limit:
                        reason = "nonlinear_model_error"
                    elif try_newton and candidate_rms >= current_rms:
                        reason = "zero_shift_no_residual_reduction"
                except psv.DomainError as error:
                    reason = "physical_domain"
                    trial["domain_message"] = str(error)
            trial["reason"] = reason or "accepted"
            if reason is not None:
                rejection_counts[reason] += 1
                trial_history.append(trial)
                if try_newton:
                    try_newton = newton_mode = False
                    well_modeled = 0
                    continue
                if reason == "affine_step_stagnation":
                    # A larger shift would make an already unrepresentable
                    # correction still smaller. Retain the unchanged anchor.
                    message = "PTC rejected: affine step stagnation above original residual tolerance"
                    break
                if cfl <= minimum_cfl * (1 + 1e-12):
                    message = f"PTC rejected at minimum CFL: {reason}"
                    break
                cfl = max(minimum_cfl, 0.5 * cfl)
                continue
            accepted = True
            trial["accepted"] = True
            trial_history.append(trial)
            old_rms = current_rms
            point, residual = candidate, candidate_residual
            current_rms = rms(residual)
            well_modeled = well_modeled + 1 if trial["model_error"] < 0.05 else 0
            newton_mode = bool(try_newton)
            history.append(
                dict(
                    accepted_index=len(history),
                    elapsed_s=perf_counter() - started,
                    residual_rms=current_rms,
                    cfl=cfl,
                    dt=None if try_newton else dt,
                    shift=shift,
                    step_fraction=fraction,
                    relative_step=relative_step,
                    model_error=trial["model_error"],
                    residual_increased=trial["residual_increased"],
                    rejected_trials=sum(rejection_counts.values()),
                )
            )
            ratio = old_rms / max(current_rms, 1e-300)
            factor = float(
                np.clip(ratio**ser_exponent, cfl_shrink_limit, cfl_growth_limit)
            )
            if (
                0.98 < ratio < 1.02
                and trial["model_error"] < 0.05
                and relative_step < 0.5 * max_relative_step
            ):
                factor = min(cfl_growth_limit, max(factor, 1.05))
            cfl = float(np.clip(cfl * factor, minimum_cfl, maximum_cfl))
            fraction = min(1.0, fraction * step_growth_factor)
            if progress is not None:
                progress(history[-1])
            break
        if not accepted:
            if not message.startswith("PTC rejected"):
                message = "PTC trial budget exhausted at unchanged anchor"
            break
    success = bool(current_rms <= residual_rms_tol)
    if success:
        message = "Original steady residual target reached"
    return dict(
        x=point.copy(),
        history=history,
        trial_history=trial_history,
        history_started_at_s=started,
        wall_s=perf_counter() - started,
        root_success=success,
        native_success=success,
        message=message,
        iterations=len(history) - 1,
        nfev=physical_residual_calls,
        njev=physical_jacobian_calls,
        residual_rms=current_rms,
        backend=dict(
            library="pysolver",
            method="newton_raphson",
            algorithm="single_loop_newton_ptc",
            pysolver_view_version=psv.__version__,
            native_success_meaning="outer audit of original spatial residual; inner success is affine-model only",
            first_affine_backend=first_backend,
            affine_solves=len(trial_history),
            affine_residual_calls=affine_residual_calls,
            affine_jacobian_calls=affine_jacobian_calls,
            physical_residual_calls=physical_residual_calls,
            physical_jacobian_calls=physical_jacobian_calls,
            storage_matrix_calls=storage_calls,
            rejected_trials=sum(rejection_counts.values()),
            rejection_counts=dict(rejection_counts),
        ),
    )
