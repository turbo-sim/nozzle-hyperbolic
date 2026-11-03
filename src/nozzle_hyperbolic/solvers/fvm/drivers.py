"""Common single-grid, coarse-to-fine and physical-time solution drivers."""

from .config import SteadySettings, Solution
from dataclasses import dataclass, field, replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.residuals import reduce_residuals
from nozzle_hyperbolic.solvers.monitoring import MonitorTracker

from .operator import Operator, Scheme
from ..residuals import ResidualPolicy, assess_residuals


def sampler(operator, policy):
    """Build a host-side sampler of the original spatial and algebraic rows.

    Parameters
    ----------
    operator : Operator
        Bound model and numerical scheme with fixed physical scales.
    policy : ResidualPolicy
        Norm, weighting and independent stopping targets.

    Returns
    -------
    sample : callable
        ``sample(state, initial=None, time=0.0)`` returns a metrics dictionary.
        State has shape ``(N + 2, V)``; initial is a fixed vector of transported
        equation norms, defaulting to the current norms. Time is physical time.

    Notes
    -----
    Sampling transfers diagnostics to the host and synchronizes JAX work.
    Retain this callable for repeated samples of the same operator and policy;
    its reduction kernel is compiled lazily and reused.
    """
    if not isinstance(policy, ResidualPolicy):
        raise ValueError("policy must be a ResidualPolicy")
    model = operator.model

    @jax.jit
    def reduce(raw, bc):
        reductions = reduce_residuals(
            raw,
            jnp.asarray(operator.mesh.volumes),
            jnp.asarray(model.conservative_scales),
            model.reference_time,
            weighting=policy.weighting,
        )
        return (
            reductions,
            jnp.max(jnp.abs(bc)),
            jnp.abs(jnp.sum(raw, axis=0) / jnp.asarray(model.global_flux_scales)),
        )

    def sample(state, initial=None, time=0.0):
        # Keep the original spatial graph separate from reductions: recompiling
        # it inside a large fused monitor changes roundoff near strict targets.
        raw = operator.spatial_residual(state)
        bc = operator.boundary_residual(state, time)
        constraints = np.asarray(operator.cell_constraints(state[1:-1]))
        constraint_norm = np.max(np.abs(constraints), axis=0)
        constraint_max = float(np.max(constraint_norm, initial=0.0))
        reductions, boundary, global_defect = jax.device_get(reduce(raw, bc))
        norms = reductions[policy.norm]
        initial = norms if initial is None else initial
        assessment = jax.device_get(assess_residuals(norms, initial, policy))
        relative = np.asarray(assessment["equation_relative"])
        active = np.asarray(assessment["relative_active"])
        valid = bool(operator.is_admissible(state))
        metrics = dict(
            equation_norm=np.asarray(norms),
            equation_relative=relative,
            equation_l1=np.asarray(reductions["l1"]),
            equation_l2=np.asarray(reductions["l2"]),
            equation_linf=np.asarray(reductions["linf"]),
            worst_cell=np.asarray(reductions["worst_cell"]),
            equation_max_ratio=float(assessment["equation_max_ratio"]),
            criterion=policy.criterion,
            overall_relative=float(np.max(np.where(active, relative, 0.0))),
            global_defect=np.asarray(global_defect),
            boundary_max=float(boundary),
            boundary_equation_max=np.max(np.abs(np.asarray(bc)), axis=0),
            constraint_norm=constraint_norm,
            constraint_max=constraint_max,
            constraints_passed=bool(np.all(np.isfinite(constraints)) and constraint_max <= policy.constraint_tolerance),
            finite=bool(assessment["finite"])
            and np.isfinite(boundary)
            and np.isfinite(global_defect).all()
            and np.isfinite(constraints).all(),
            # Algebraic cells have fixed scales and an independent absolute
            # infinity-norm check; no phase weights or initial-relative test.
            admissible=valid,
            all_equations_passed=bool(assessment["all_equations_passed"]),
            absolute_satisfied=np.asarray(assessment["absolute_satisfied"]),
            relative_satisfied=np.asarray(assessment["relative_satisfied"]),
            boundary_passed=bool(boundary <= policy.boundary_tolerance),
            global_passed=bool(np.max(global_defect) <= policy.global_tolerance),
            relative_active=np.asarray(assessment["relative_active"]),
        )
        if operator.equation_policy_active:
            original = operator.physical_spatial_residual(state)
            unmodified, _, unmodified_global = jax.device_get(reduce(original, bc))
            metrics.update(
                equation_policy_active=True,
                unmodified_equation_norm=np.asarray(unmodified[policy.norm]),
                unmodified_equation_linf=np.asarray(unmodified["linf"]),
                unmodified_global_defect=np.asarray(unmodified_global),
            )
            metrics["finite"] = metrics["finite"] and bool(unmodified["finite"])
        if hasattr(model, "invariant_projections"):
            physical = operator.physical_spatial_residual(state) if operator.equation_policy_active else raw
            signed = np.asarray(model.invariant_projections) @ np.asarray(physical).sum(axis=0)
            signed = signed / np.asarray(model.invariant_flux_scales)
            metrics.update(invariant_names=tuple(model.invariant_names),
                           invariant_defect=signed,
                           invariants_passed=bool(np.all(np.abs(signed) <= policy.global_tolerance)))
            metrics["finite"] = metrics["finite"] and bool(np.all(np.isfinite(signed)))
        return metrics

    return sample


@dataclass(frozen=True)
class SteadyWorkspace:
    """Reusable kernels for one model, scheme and residual policy.

    Binding does not advance a solution. Repeated solves retain compiled
    functions while resetting the initial field, controller, histories and
    residual baselines. Model identity is intentional: bound callbacks and
    physical scales must not be replaced underneath an existing workspace.

    Parameters
    ----------
    model : BalanceModel
        Exact model instance reused for every solve. Its configuration and
        callback closure data must remain fixed for the workspace lifetime.
    scheme : Scheme, optional
        Fixed numerical choices used when binding the operator.
    residual : ResidualPolicy, optional
        Fixed sampling and stopping policy.

    Attributes
    ----------
    operator : Operator
        Reusable operator, constructed once at workspace initialization.
    sample : callable
        Reusable host-side residual sampler for this operator and policy.
    """

    model: object
    scheme: Scheme = field(default_factory=Scheme)
    residual: ResidualPolicy = field(default_factory=ResidualPolicy)
    operator: Operator = field(init=False, repr=False)
    sample: object = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.residual, ResidualPolicy):
            raise ValueError("workspace residual must be a ResidualPolicy")
        operator = Operator(self.model, self.scheme)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "sample", sampler(operator, self.residual))

    def validate(self, model, scheme, residual):
        """Require identical model identity and equal numerical/stopping settings.

        Parameters
        ----------
        model : BalanceModel
            Model instance requested by the solve.
        scheme : Scheme
            Requested numerical scheme.
        residual : ResidualPolicy
            Requested residual policy.

        Raises
        ------
        ValueError
            If any binding differs. In-place changes to model configuration
            are not detected; callers must keep that configuration fixed.
        """
        if self.model is not model or self.scheme != scheme or self.residual != residual:
            raise ValueError("workspace must match model identity, scheme and residual policy")


def solve_single_grid(
    model,
    scheme=Scheme(),
    settings=SteadySettings(),
    *,
    initial_state=None,
    monitor_specs=(),
    quantity_evaluator=None,
    callback=None,
    workspace=None,
):
    """Solve the original coupled steady equations on exactly one physical grid.

    All models use identical Newton/CFL/acceptance and convergence semantics.
    Model-specific scalar quantities enter only through the supplied evaluator.

    Parameters
    ----------
    model : BalanceModel
        Physical model on a fixed mesh, with ``N`` cells and ``V`` variables.
    scheme : Scheme, optional
        Reconstruction, flux, time integrator and nonlinear solution settings.
    settings : SteadySettings, optional
        Residual stopping policy and pseudo-time CFL controller.
    initial_state : array_like, shape (N + 2, V), optional
        Full iteration state including endpoint traces. Defaults to the
        model's physical initial guess. No analytical warm start is inferred.
    monitor_specs : iterable of MonitorSpec, optional
        Scalar monitor definitions sampled at initial and accepted states.
    quantity_evaluator : callable, optional
        ``quantity_evaluator(state, integrated_face_flux)`` returns a mapping
        keyed by monitor name or quantity. Required when monitors are supplied.
    callback : callable, optional
        Receives each recorded history dictionary, including rejected trials.
    workspace : SteadyWorkspace, optional
        Reuse bound kernels for this exact model, scheme and residual policy.
        Without it, each invocation constructs a new operator and sampler.

    Returns
    -------
    Solution
        Final host state, success/reason, complete attempt history, final
        metrics and bound operator. Failure states remain available for audit.

    Notes
    -----
    Inner Newton success alone does not establish steady convergence. The
    original spatial rows, traces, cell constraints, global balances and
    required monitors are checked separately, followed by any model audit.
    """
    monitor_specs = tuple(monitor_specs)
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable when supplied")
    if monitor_specs and not callable(quantity_evaluator):
        raise ValueError("Configured monitors require a quantity_evaluator callback")
    audit = getattr(model, "solution_audit", None)
    if audit is not None and not callable(audit):
        raise ValueError("solution_audit must be a callback when supplied")
    if workspace is not None:
        if not isinstance(workspace, SteadyWorkspace):
            raise ValueError("workspace must be a SteadyWorkspace")
        workspace.validate(model, scheme, settings.residual)
    operator = Operator(model, scheme) if workspace is None else workspace.operator
    state = jnp.asarray(
        model.initial_state() if initial_state is None else initial_state,
        dtype=jnp.float64,
    )
    if state.shape != operator.shape:
        raise ValueError(f"Expected full physical/boundary state {operator.shape}")
    explicit = scheme.time_integrator != "euler-implicit"
    cfl_cap = min(settings.cfl_max, 0.45) if explicit else settings.cfl_max
    history = []
    start = perf_counter()
    cfl = min(settings.cfl, cfl_cap)
    tracker = MonitorTracker(monitor_specs)
    if not bool(operator.is_admissible(state)):
        return Solution(
            np.asarray(state),
            False,
            "invalid_initial_state",
            [],
            {},
            elapsed_s=perf_counter() - start,
            operator=operator,
        )
    sample = sampler(operator, settings.residual) if workspace is None else workspace.sample
    current = sample(state)
    initial = np.asarray(current["equation_norm"])

    def update_monitors(point):
        values = (
            {}
            if quantity_evaluator is None
            else quantity_evaluator(point, operator.face_flux(point))
        )
        return tracker.update(
            {
                spec.name: (
                    values[spec.name] if spec.name in values else values[spec.quantity]
                )
                for spec in monitor_specs
            }
        )

    current["monitors"] = update_monitors(state)

    def done(metrics):
        return bool(
            metrics["finite"]
            and metrics["admissible"]
            and metrics["all_equations_passed"]
            and metrics["boundary_passed"]
            and metrics["constraints_passed"]
            and metrics["global_passed"]
            and metrics.get("invariants_passed", True)
            and tracker.passed
        )

    def emit(row):
        history.append(row)
        if callback is not None:
            callback(row)

    emit(dict(step=0, phase="initial", accepted=True, cfl=cfl, dt=0.0, **current))
    reason = "max_steps"
    accepted = 0
    unchanged = 0
    while accepted < settings.max_steps and not done(current):
        okay = False
        for retry in range(settings.max_retries + 1):
            limits = jax.device_get(operator.timestep(state, cfl))
            dt = float(limits.dt)
            if not np.isfinite(dt) or dt <= 0:
                reason = "invalid_timestep"
                break
            candidate, info = operator.step(state, jnp.float64(dt), jnp.float64(0.0))
            info = jax.device_get(info)
            okay = bool(info.converged and info.admissible)
            trial = None
            if okay:
                trial = sample(candidate, initial)
                okay = bool(trial["finite"] and trial["admissible"])
            row = dict(
                step=accepted + 1,
                retry=retry,
                phase="accepted" if okay else "rejected",
                accepted=okay,
                cfl=dt / float(limits.wave_dt),
                requested_cfl=cfl,
                dt=dt,
                inner_iterations=int(info.iterations),
                inner_status=int(info.status),
                inner_residual=float(info.residual_norm),
                rejected_newton_trials=int(info.rejected_trials),
            )
            if okay:
                unchanged = (
                    unchanged + 1
                    if np.array_equal(np.asarray(candidate), np.asarray(state))
                    else 0
                )
                state = candidate
                accepted += 1
                current = trial
                current["monitors"] = update_monitors(state)
                emit(dict(**row, **current))
                cfl = min(cfl_cap, cfl * settings.cfl_growth)
                break
            emit(row)
            cfl *= settings.cfl_shrink
        if not okay:
            if reason != "invalid_timestep":
                reason = "retry_limit"
            break
        if (
            settings.stagnation_steps > 0
            and unchanged >= settings.stagnation_steps
            and not done(current)
        ):
            reason = "stagnation"
            break
    final = sample(state, initial)
    final["monitors"] = current["monitors"]
    success = done(final)
    if success and audit is not None and not audit(state):
        success = False
        reason = "solution_audit_failed"
    if success:
        reason = "converged"
    return Solution(
        np.asarray(state),
        success,
        reason,
        history,
        final,
        elapsed_s=perf_counter() - start,
        operator=operator,
    )


def _last_accepted_cfl(solution):
    """Actual accepted CFL; use the initial value for a zero-update success."""
    for phase in ("accepted", "initial"):
        for row in reversed(solution.history):
            if row.get("phase") == phase:
                value = row.get("cfl")
                if value is not None and np.isfinite(value) and value > 0:
                    return float(value)
                return None
    return None


def solve_grid_sequence(
    model_factory,
    cells=(50, 100, 200, 400),
    scheme=Scheme(),
    settings=SteadySettings(),
    *,
    on_stage=None,
    monitor_specs=(),
    initial_solution=None,
    quantity_evaluator_factory=None,
    carry_cfl=True,
    scheme_factory=None,
    workspace_factory=None,
    stop_on_failure=False,
    retry_cold=True,
):
    """Nested-grid initialization, not a defect-correction/FAS multigrid cycle.

    Every grid solves its own original equations and starts its own residual
    normalization. Only successful numerical states are prolonged. A failed
    stage is retained and the next stage starts cold. Prolongation is bounded
    componentwise linear interpolation of the model's declared state variables;
    it assumes identical variable meanings and domain endpoints on every grid.
    Model admissibility is checked before solving, with a cold retry on failure.
    By default a warm transfer also inherits the preceding successful solve's
    last accepted CFL, capped by settings.cfl_max. The timestep is recomputed
    on the new grid. A zero-update success forwards its initial CFL; unavailable
    controller history falls back to settings.cfl. Cold starts/retries always
    use settings.cfl. Set carry_cfl=False to reset CFL on every grid.

    ``scheme_factory(count)`` can explicitly choose a different Scheme on each
    mesh, for example constant reconstruction only on the coarsest mesh.
    ``workspace_factory(model, scheme, settings)`` can reuse bound kernels
    between complete hierarchy replays. Defaults retain cold fallback/retry;
    set ``retry_cold=False, stop_on_failure=True`` for a strict sequence that
    never replaces a failed transfer or solve with an independent start.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(count)`` returns a model with the requested cell count.
    cells : iterable of int, optional
        Strictly increasing cell counts, each at least two.
        Defaults to ``(50, 100, 200, 400)``.
    scheme : Scheme, optional
        Shared numerical scheme unless ``scheme_factory`` is supplied.
    settings : SteadySettings, optional
        Shared stopping/controller policy; inherited CFL affects warm starts.
    on_stage : callable, optional
        ``on_stage(model, solution, stages)`` runs after each completed stage.
    monitor_specs : iterable of MonitorSpec, optional
        Monitor definitions reused on every mesh.
    initial_solution : Solution, optional
        Previous numerical solution on a coarser matching domain.
    quantity_evaluator_factory : callable, optional
        ``quantity_evaluator_factory(model)`` returns the stage's monitor
        evaluator. Defaults to the model's ``quantities`` callback if present.
    carry_cfl : bool, optional
        Inherit the last successful accepted CFL on warm transfers, default True.
    scheme_factory : callable, optional
        ``scheme_factory(count)`` selects an explicit Scheme for each mesh.
    workspace_factory : callable, optional
        ``workspace_factory(model, scheme, settings)`` returns a matching
        SteadyWorkspace for replays of the same hierarchy.
    stop_on_failure : bool, optional
        Stop at the first unsuccessful stage, default False.
    retry_cold : bool, optional
        Permit independent cold attempts after failed warm starts, default True.

    Returns
    -------
    solution : Solution
        Result of the last attempted grid, including an unsuccessful result.
    stages : list of dict
        Stage outcomes, initialization policies, CFL data, preparation costs
        and retained failed warm attempts.
    """
    if not isinstance(carry_cfl, bool):
        raise ValueError("carry_cfl must be a boolean")
    if not isinstance(stop_on_failure, bool) or not isinstance(retry_cold, bool):
        raise ValueError("stop_on_failure and retry_cold must be booleans")
    if scheme_factory is not None and not callable(scheme_factory):
        raise ValueError("scheme_factory must be callable")
    if workspace_factory is not None and not callable(workspace_factory):
        raise ValueError("workspace_factory must be callable")
    cells = tuple(cells)
    monitor_specs = tuple(monitor_specs)
    if (
        not cells
        or any(
            isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer))
            or n < 2
            for n in cells
        )
        or list(cells) != sorted(set(cells))
    ):
        raise ValueError("Use a strictly increasing grid hierarchy")
    previous = initial_solution
    stages = []
    if previous is not None and cells[0] <= previous.operator.mesh.n_cells:
        raise ValueError("Extended grids must be finer than the supplied solved grid")
    for count in cells:
        stage_started = perf_counter()
        setup_started = perf_counter()
        model = model_factory(count)
        stage_scheme = scheme if scheme_factory is None else scheme_factory(count)
        if not isinstance(stage_scheme, Scheme):
            raise ValueError("scheme_factory must return a Scheme")
        workspace = (None if workspace_factory is None else
                     workspace_factory(model, stage_scheme, settings))
        if workspace is not None:
            if not isinstance(workspace, SteadyWorkspace):
                raise ValueError("workspace_factory must return a SteadyWorkspace")
            workspace.validate(model, stage_scheme, settings.residual)
        setup_s = perf_counter() - setup_started
        if model.mesh.n_cells != count:
            raise ValueError(
                "model_factory must return the requested physical cell count"
            )
        initial = None
        preparation = None
        if previous is not None and previous.success:
            old = previous.operator.mesh.points
            new = model.mesh.points
            if previous.state.shape != (len(old), model.nvar) or not np.array_equal(
                old[[0, -1]], new[[0, -1]]
            ):
                raise ValueError(
                    "Prolongation requires matching state dimensions and domain endpoints"
                )
            if getattr(model, "nconstraints", 0):
                from .continuation import prepare_continuation_state
                reuse = {} if workspace is None else {"operator": workspace.operator}
                initial, preparation = prepare_continuation_state(model, previous, stage_scheme, **reuse)
            else:
                initial = np.column_stack(
                    [np.interp(new, old, previous.state[:, j]) for j in range(model.nvar)]
                )
        evaluator = (
            getattr(model, "quantities", None)
            if quantity_evaluator_factory is None
            else quantity_evaluator_factory(model)
        )
        stage_settings = settings
        if carry_cfl and initial is not None:
            inherited = _last_accepted_cfl(previous)
            if inherited is not None:
                stage_settings = replace(settings, cfl=min(inherited, settings.cfl_max))
        reuse = {} if workspace is None else {"workspace": workspace}
        failed_preparation = preparation is not None and not preparation["success"]
        if failed_preparation and not retry_cold:
            candidate = preparation.get("state", model.initial_state())
            solution = Solution(np.asarray(candidate), False, "continuation_preparation_failed",
                                [], {}, elapsed_s=0.0,
                                operator=Operator(model, stage_scheme) if workspace is None else workspace.operator)
        else:
            solution = solve_single_grid(
                model,
                stage_scheme,
                stage_settings,
                initial_state=initial,
                monitor_specs=monitor_specs,
                quantity_evaluator=evaluator,
                **reuse,
            )
        attempts = []
        if retry_cold and initial is not None and not solution.success:
            # Prolongation can make a nine-order target smaller than available
            # floating-point resolution. Preserve that failed solve in full;
            # an independent cold restart must satisfy its own original tests.
            attempts.append(
                dict(
                    initialization="componentwise linear prolongation",
                    success=solution.success,
                    reason=solution.reason,
                    state=solution.state,
                    metrics=solution.metrics,
                    history=solution.history,
                    elapsed_s=solution.elapsed_s,
                    initial_cfl=stage_settings.cfl,
                )
            )
            print(
                f"  N={count}: retained failed warm attempt ({solution.reason}); restarting cold",
                flush=True,
            )
            solution = solve_single_grid(
                model,
                stage_scheme,
                settings,
                monitor_specs=monitor_specs,
                quantity_evaluator=evaluator,
                **reuse,
            )
            stage_settings = settings
        metadata = dict(
            cells=count,
            success=solution.success,
            reason=solution.reason,
            initialization=(
                "failed continuation preparation" if failed_preparation and not retry_cold else
                "cold restart after retained warm failure"
                if attempts
                else (
                    ("transported-density interpolation with constraint recovery"
                     if preparation is not None else "componentwise linear prolongation")
                    if initial is not None
                    else "cold physical guess"
                )
            ),
            attempts=attempts,
            setup_s=setup_s,
            preparation=preparation,
            solve_elapsed_s=solution.elapsed_s + sum(attempt["elapsed_s"] for attempt in attempts),
            elapsed_s=perf_counter() - stage_started,
            residual_normalization="fresh per-equation initial norm on this grid",
            cfl_policy="carry" if carry_cfl else "reset",
            initial_cfl=(solution.history[0]["cfl"] if solution.history
                         else stage_settings.cfl),
            last_accepted_cfl=_last_accepted_cfl(solution),
            reconstruction=stage_scheme.reconstruction,
            flux=stage_scheme.flux,
        )
        stages.append(metadata)
        if on_stage is not None:
            on_stage(model, solution, list(stages))
        previous = solution
        if stop_on_failure and not solution.success:
            break
    return previous, stages


def solve_transient(
    model,
    output_times,
    scheme=Scheme(flux="hll"),
    *,
    cfl=0.45,
    initial_state=None,
    max_steps=100000,
    callback=None,
    operator=None,
):
    """Conservative physical-time integration, starting at time zero.

    ``max_steps`` counts accepted physical steps across all requested outputs.
    History retains rejected attempts without advancing time or conservation
    budgets. Steady spatial/global convergence is not a transient stopping rule;
    algebraic boundary defects must nevertheless meet their independent target.
    The optional model transient_solution_audit applies to each output; when
    absent, solution_audit provides the backward-compatible fallback. A final
    steady branch constraint need not describe every transient intermediate.

    Parameters
    ----------
    model : BalanceModel
        Physical model on a fixed mesh; steady-only equation policies are
        rejected for physical-time integration.
    output_times : iterable of float
        Nonnegative, finite, strictly increasing physical output times. The
        initial physical time is zero; a zero-time output is permitted.
    scheme : Scheme, optional
        Numerical method. Defaults to HLL with implicit Euler integration.
    cfl : float, optional
        Positive requested wave CFL, default 0.45. Explicit source and filter
        stability bounds can further reduce the actual timestep.
    initial_state : array_like, shape (N + 2, V), optional
        Admissible full iteration state, defaulting to the model's initial guess.
    max_steps : int, optional
        Maximum accepted steps across all outputs, default 100000.
    callback : callable, optional
        Receives each accepted/rejected attempt dictionary as it is appended
        to the cumulative history. Accepted rows include physical time and the
        content-change budget. Exceptions from the callback propagate.
    operator : Operator, optional
        Reuse an operator bound to this exact model and scheme, for example
        after completing initial algebraic traces. Otherwise bind a new one.

    Returns
    -------
    list of Solution
        Snapshots at requested times, stopping after the first failed output.
        A failed snapshot records the time actually reached. Histories and
        conservation budgets accumulate from the initial state.

    Notes
    -----
    Timestep retries halve the attempted increment, with at most sixteen
    attempts per accepted step. Each output checks physical admissibility,
    finite diagnostics, boundary and cell-constraint defects, and any model
    audit; spatial steady-state convergence is not required.
    """
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable when supplied")
    if getattr(model, "steady_only", False):
        raise ValueError("The selected model equation policy is steady-only; disable it for physical-time integration")
    targets = np.asarray(tuple(output_times), dtype=float)
    if (
        targets.ndim != 1
        or not targets.size
        or not np.all(np.isfinite(targets))
        or np.any(targets < 0)
        or np.any(np.diff(targets) <= 0)
    ):
        raise ValueError("Use nonnegative, finite, strictly increasing output times")
    if isinstance(cfl, (bool, np.bool_)) or not np.isfinite(cfl) or cfl <= 0:
        raise ValueError("cfl must be positive and finite")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("max_steps must be a nonnegative integer")
    audit = getattr(
        model, "transient_solution_audit", getattr(model, "solution_audit", None)
    )
    if audit is not None and not callable(audit):
        raise ValueError(
            "The transient solution audit must be a callback when supplied"
        )
    if operator is None:
        operator = Operator(model, scheme)
    elif (
        not isinstance(operator, Operator)
        or operator.model is not model
        or operator.scheme != scheme
    ):
        raise ValueError("operator must match model identity and scheme")
    state = jnp.asarray(
        model.initial_state() if initial_state is None else initial_state,
        dtype=jnp.float64,
    )
    if state.shape != operator.shape:
        raise ValueError(f"Expected full physical/boundary state {operator.shape}")
    if not bool(operator.is_admissible(state)):
        raise ValueError("Transient initial state must be admissible")
    sample = sampler(operator, ResidualPolicy())
    initial_totals = np.asarray(operator.storage(state[1:-1])).sum(axis=0)
    influx = np.zeros(operator.ntransport)
    time = 0.0
    history = []
    snapshots = []
    start = perf_counter()
    budget = np.zeros(operator.ntransport)
    accepted = 0

    def emit(row):
        history.append(row)
        if callback is not None:
            callback(row)

    for target in targets:
        target = float(target)
        reason = "time_reached"
        while time < target and accepted < max_steps:
            limits = jax.device_get(operator.timestep(state, cfl, time))
            base_dt = float(limits.wave_dt)
            if not bool(limits.valid) or not np.isfinite(limits.dt) or limits.dt <= 0:
                reason = "invalid_timestep"
                break
            dt = min(float(limits.dt), float(target - time))
            okay = False
            for retry in range(16):
                if not np.isfinite(dt) or dt <= 0 or time + dt <= time:
                    reason = "invalid_timestep"
                    break
                if scheme.time_integrator == "euler-implicit":
                    candidate, info = operator.step(
                        state, jnp.float64(dt), jnp.float64(time + dt)
                    )
                else:
                    candidate, info, step_balance = operator.step_with_balance(
                        state, jnp.float64(dt), jnp.float64(time + dt)
                    )
                info = jax.device_get(info)
                if bool(info.converged and info.admissible):
                    okay = True
                    break
                emit(
                    dict(
                        step=accepted + 1,
                        phase="rejected",
                        accepted=False,
                        time=time,
                        time_s=time,
                        trial_time=time + dt,
                        dt=dt,
                        cfl=dt / base_dt,
                        requested_cfl=cfl,
                        retry=retry,
                        inner_iterations=int(info.iterations),
                        inner_status=int(info.status),
                        inner_residual=float(info.residual_norm),
                        rejected_newton_trials=int(info.rejected_trials),
                    )
                )
                dt *= 0.5
            if not okay:
                if reason != "invalid_timestep":
                    reason = "retry_limit"
                break
            transfer = (
                dt * np.asarray(operator.balance_rate(candidate))
                if scheme.time_integrator == "euler-implicit"
                else np.asarray(step_balance)
            )
            influx += transfer
            state = candidate
            time = min(target, time + dt)
            accepted += 1
            totals = np.asarray(operator.storage(state[1:-1])).sum(axis=0)
            budget = (totals - initial_totals - influx) / (
                np.sum(model.mesh.volumes) * np.asarray(model.conservative_scales)
            )
            emit(
                dict(
                    step=accepted,
                    phase="accepted",
                    accepted=True,
                    time=time,
                    time_s=time,
                    dt=dt,
                    cfl=dt / base_dt,
                    requested_cfl=cfl,
                    retry=retry,
                    budget=budget,
                    integrated_transfer=transfer,
                    conserved_totals=totals,
                    inner_iterations=int(info.iterations),
                    inner_status=int(info.status),
                    inner_residual=float(info.residual_norm),
                    rejected_newton_trials=int(info.rejected_trials),
                )
            )
        if accepted >= max_steps and time < target:
            reason = "max_steps"
        metrics = sample(state, time=time)
        metrics["conservation_budget"] = budget
        metrics["accepted_steps"] = accepted
        metrics["rejected_steps"] = sum(not row["accepted"] for row in history)
        success = bool(
            time == target
            and metrics["admissible"]
            and metrics["finite"]
            and metrics["boundary_passed"]
            and metrics["constraints_passed"]
        )
        if time == target and not success:
            reason = (
                "invalid_final_state"
                if not metrics["admissible"] or not metrics["finite"]
                else ("cell_constraint_defect" if not metrics["constraints_passed"] else "boundary_defect")
            )
        if success and audit is not None and not audit(state):
            success = False
            reason = "solution_audit_failed"
        snapshots.append(
            Solution(
                np.asarray(state),
                success,
                reason,
                list(history),
                metrics,
                time=time,
                elapsed_s=perf_counter() - start,
                operator=operator,
            )
        )
        if not success:
            break
    return snapshots
