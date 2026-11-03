"""Audited physical-parameter continuation and constraint-consistent restarts.

No continuation schedule or physical parameter meaning is inferred here.
One fixed numerical Scheme is used throughout the user-declared stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .config import Scheme, SteadySettings
from .drivers import solve_single_grid
from .operator import Operator


@dataclass(frozen=True)
class RestartState:
    """Portable initialization data; contains no executable EOS or solver object."""

    state: np.ndarray
    densities: np.ndarray
    points: np.ndarray
    centers: np.ndarray
    volumes: np.ndarray
    equation_names: tuple[str, ...]
    success: bool
    time: float = 0.0
    metadata: dict | None = None


def _snapshot(previous):
    if isinstance(previous, RestartState):
        return previous
    operator = previous.operator
    if operator is None:
        raise ValueError("A solution restart needs its source operator")
    return RestartState(
        state=np.asarray(previous.state).copy(),
        densities=np.asarray(operator.model.cell_conserved(jnp.asarray(previous.state[1:-1]))),
        points=np.asarray(operator.mesh.points).copy(),
        centers=np.asarray(operator.mesh.centers).copy(),
        volumes=np.asarray(operator.mesh.volumes).copy(),
        equation_names=tuple(operator.model.equation_names),
        success=bool(previous.success),
        time=float(previous.time),
        metadata={"source_reason": previous.reason},
    )


def _validate_snapshot(snapshot):
    n = len(snapshot.centers)
    if (
        n < 2
        or snapshot.state.ndim != 2
        or snapshot.state.shape[0] != n + 2
        or snapshot.densities.ndim != 2
        or snapshot.densities.shape[0] != n
        or snapshot.points.shape != (n + 2,)
        or snapshot.volumes.shape != (n,)
        or len(snapshot.equation_names) != snapshot.densities.shape[1]
        or not np.all(np.diff(snapshot.points) > 0)
        or not np.array_equal(snapshot.points[1:-1], snapshot.centers)
        or not np.all(snapshot.volumes > 0)
        or not np.isfinite(snapshot.time)
        or not all(np.all(np.isfinite(array)) for array in (
            snapshot.state, snapshot.densities, snapshot.points, snapshot.volumes
        ))
    ):
        raise ValueError("Invalid restart dimensions, coordinates or finite physical data")


def prepare_continuation_state(model, previous, scheme=Scheme(), *, time=0.0, allow_unsuccessful=False,
                               operator=None):
    """Interpolate transported density, recover constraints, then complete traces.

    Returns ``(state_or_None, record)``. A failed preparation is retained in the
    record and can be followed by an independent cold solve. Content and variable
    meanings must match; equation names/counts and domain endpoints are checked.

    Bounded center interpolation is an initialization policy, not conservative
    finite-volume remapping for physical transients. The old and target domain
    inventories are reported explicitly. New cell volumes apply once.
    ``allow_unsuccessful=True`` permits an explicitly requested diagnostic
    restart from an unfinished finite state; its unsuccessful origin is
    recorded and never relabelled converged. Automatic sequences keep False.
    A supplied ``operator`` reuses recovery/trace kernels and must be bound to
    this exact target model and scheme.
    """
    start = perf_counter()
    if not isinstance(allow_unsuccessful, bool):
        raise ValueError("allow_unsuccessful must be an explicit boolean")
    record = dict(
        strategy="interpolated transported densities, cell recovery, algebraic trace completion",
        success=False, reason="not_started", elapsed_s=0.0,
        operator_setup_s=0.0, recovery_s=0.0, boundary_s=0.0,
        source_success=bool(previous.success), allow_unsuccessful=allow_unsuccessful,
    )
    if not bool(previous.success) and not allow_unsuccessful:
        record.update(reason="source_solution_unsuccessful", elapsed_s=perf_counter() - start)
        return None, record
    source = _snapshot(previous)
    _validate_snapshot(source)
    ntransport = int(getattr(model, "ntransport", model.nvar))
    if (
        source.state.shape[1] != model.nvar
        or source.densities.shape[1] != ntransport
        or source.equation_names != tuple(model.equation_names)
        or not np.array_equal(source.points[[0, -1]], model.mesh.points[[0, -1]])
    ):
        raise ValueError("Continuation requires matching state/content meanings and domain endpoints")
    setup = perf_counter()
    record["operator_reused"] = operator is not None
    if operator is None:
        operator = Operator(model, scheme)
    elif not isinstance(operator, Operator) or operator.model is not model or operator.scheme != scheme:
        raise ValueError("operator must match target model identity and scheme")
    record["operator_setup_s"] = perf_counter() - setup
    density = np.column_stack([
        np.interp(model.mesh.centers, source.centers, source.densities[:, j])
        for j in range(ntransport)
    ])
    guess = np.column_stack([
        np.interp(model.mesh.points, source.points, source.state[:, j])
        for j in range(model.nvar)
    ])
    target = np.asarray(model.mesh.volumes)[:, None] * density
    record.update(
        source_cells=len(source.centers), target_cells=model.mesh.n_cells,
        source_inventory=np.sum(source.volumes[:, None] * source.densities, axis=0),
        target_inventory=np.sum(target, axis=0), target_contents=target.copy(),
    )
    candidate = guess
    try:
        recovery_start = perf_counter()
        cells, recovery = operator.recover_storage(jnp.asarray(target), jnp.asarray(guess[1:-1]))
        cells, recovery = jax.device_get((cells, recovery))
        record.update(recovery_s=perf_counter() - recovery_start,
                      recovery_info=dict(recovery._asdict()))
        candidate = guess.copy()
        candidate[1:-1] = cells
        if not bool(recovery.converged and recovery.admissible):
            record["reason"] = "cell_recovery_failed"
        else:
            boundary_start = perf_counter()
            candidate, boundary = operator.complete_boundaries(jnp.asarray(candidate), time)
            candidate, boundary = jax.device_get((candidate, boundary))
            record.update(boundary_s=perf_counter() - boundary_start,
                          boundary_info=dict(boundary._asdict()))
            constraint = np.asarray(operator.cell_constraints(jnp.asarray(candidate[1:-1])))
            bc = np.asarray(operator.boundary_residual(jnp.asarray(candidate), time))
            # Audit the original fixed-scale recovery equations, independently
            # of a Newton solver's small-update termination flag.
            stored = np.asarray(operator.storage(jnp.asarray(candidate[1:-1])))
            scales = np.asarray(model.mesh.volumes)[:, None] * np.asarray(model.conservative_scales)
            storage_defect = float(np.max(np.abs((stored - target) / scales)))
            constraint_defect = float(np.max(np.abs(constraint))) if constraint.size else 0.0
            boundary_defect = float(np.max(np.abs(bc)))
            limit = scheme.newton_residual_tolerance
            success = bool(
                boundary.converged and boundary.admissible
                and operator.is_admissible(jnp.asarray(candidate))
                and np.isfinite(storage_defect + constraint_defect + boundary_defect)
                and max(storage_defect, constraint_defect, boundary_defect) <= limit
            )
            record.update(
                success=success, reason="prepared" if success else "preparation_audit_failed",
                storage_defect=storage_defect, constraint_defect=constraint_defect,
                boundary_defect=boundary_defect,
            )
    except Exception as exc:
        record.update(reason="preparation_exception", error=f"{type(exc).__name__}: {exc}")
    record.update(state=np.asarray(candidate).copy(), elapsed_s=perf_counter() - start)
    return (np.asarray(candidate) if record["success"] else None), record


def _attempt(solution, initialization, elapsed):
    return dict(
        initialization=initialization, success=solution.success, reason=solution.reason,
        state=np.asarray(solution.state).copy(), metrics=deepcopy(solution.metrics),
        history=deepcopy(solution.history), solve_elapsed_s=elapsed,
        driver_elapsed_s=solution.elapsed_s,
    )


def solve_parameter_sequence(
    model_factory, parameter_stages, scheme=Scheme(), settings=SteadySettings(), *,
    initial_solution=None, retry_cold=True, on_stage=None, monitor_specs=(),
    quantity_evaluator_factory=None,
):
    """Solve each declared physical stage and return the final requested attempt.

    ``model_factory(parameters)`` owns parameter meanings and physical setup.
    Every stage uses the same Scheme, so this is not a flux/reconstruction
    method-sequence driver. Every solve establishes fresh residual baselines.
    Failed warm solves remain in ``attempts`` before an optional cold retry;
    failed preparation states and their costs remain in ``preparation``.
    Earlier successful stages never replace an unsuccessful requested endpoint.
    """
    stages = tuple(parameter_stages)
    if not stages or any(not isinstance(stage, Mapping) for stage in stages):
        raise ValueError("Use a nonempty sequence of explicit parameter mappings")
    if not isinstance(retry_cold, bool):
        raise ValueError("retry_cold must be a boolean")
    monitor_specs = tuple(monitor_specs)
    previous, records = initial_solution, []
    for index, parameters in enumerate(stages):
        started = perf_counter()
        parameters = deepcopy(dict(parameters))
        setup = perf_counter()
        model = model_factory(deepcopy(parameters))
        setup_s = perf_counter() - setup
        initial, preparation = None, None
        if previous is not None:
            initial, preparation = prepare_continuation_state(model, previous, scheme)
        evaluator = (getattr(model, "quantities", None) if quantity_evaluator_factory is None
                     else quantity_evaluator_factory(model))
        solve_started = perf_counter()
        solution = solve_single_grid(
            model, scheme, settings, initial_state=initial,
            monitor_specs=monitor_specs, quantity_evaluator=evaluator,
        )
        attempts = [_attempt(solution, "prepared previous stage" if initial is not None
                             else "cold physical guess", perf_counter() - solve_started)]
        if initial is not None and not solution.success and retry_cold:
            solve_started = perf_counter()
            solution = solve_single_grid(model, scheme, settings, monitor_specs=monitor_specs,
                                         quantity_evaluator=evaluator)
            attempts.append(_attempt(solution, "cold retry after warm failure", perf_counter() - solve_started))
        record = dict(
            stage=index, parameters=parameters, cells=model.mesh.n_cells,
            success=solution.success, reason=solution.reason, setup_s=setup_s,
            preparation=preparation, attempts=attempts,
            solve_elapsed_s=sum(attempt["solve_elapsed_s"] for attempt in attempts),
            elapsed_s=perf_counter() - started,
            residual_normalization="fresh per-equation initial norms for every attempt",
            requested_final_stage=index == len(stages) - 1,
        )
        records.append(record)
        if on_stage is not None:
            on_stage(model, solution, list(records))
        previous = solution
    return previous, records


def save_restart(path, solution, *, metadata=None):
    """Write portable arrays and JSON metadata without pickled model objects."""
    source = _snapshot(solution)
    _validate_snapshot(source)
    details = dict(schema_version=1, equation_names=source.equation_names,
                   success=source.success, time=source.time,
                   source_metadata=source.metadata, metadata=metadata or {})
    encoded = json.dumps(details, allow_nan=False)
    # A file handle preserves the exact requested path (NumPy adds no suffix).
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, state=source.state, densities=source.densities,
                            points=source.points, centers=source.centers,
                            volumes=source.volumes, metadata=np.asarray(encoded))


def load_restart(path):
    """Read and validate a version-one snapshot; call preparation before use."""
    with np.load(path, allow_pickle=False) as archive:
        details = json.loads(str(archive["metadata"]))
        if details.get("schema_version") != 1:
            raise ValueError("Unsupported restart schema")
        result = RestartState(
            state=np.array(archive["state"]), densities=np.array(archive["densities"]),
            points=np.array(archive["points"]), centers=np.array(archive["centers"]),
            volumes=np.array(archive["volumes"]), equation_names=tuple(details["equation_names"]),
            success=bool(details["success"]), time=float(details["time"]),
            metadata=details,
        )
    _validate_snapshot(result)
    return result
