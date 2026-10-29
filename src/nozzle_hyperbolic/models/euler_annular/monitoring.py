"""Annular physical adapters for reusable steady convergence diagnostics.

The input is the original signed integrated spatial residual, not the
backward-Euler residual. Norms are never inserted into Newton's equations.
The host driver records the initial state and post-update accepted states;
unevaluated rejected candidates never inherit the previous state's diagnostics.
"""

from dataclasses import asdict
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from ...solvers.monitoring import MonitorSpec, MonitorTracker
from ...solvers.residuals import (DEFINITION, ResidualPolicy, assess_residuals, reduce_residuals,
                                validate_residual_setup)
from .boundaries import boundary_mach


EQUATIONS = ("mass", "meridional_momentum", "angular_momentum", "energy")


def _plain(value):
    """Transfer diagnostics only; retain arrays as simple serializable lists."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


class AnnularMonitoring:
    """Bind fixed mesh/scales and named physical observables before JIT.

    Cp uses the instantaneous inlet trace pressure and a fixed reservoir total
    pressure. A supersonic inlet therefore needs ``pressure_total`` explicitly
    unless the corresponding monitor supplies it in ``parameters``. Mass flow
    always uses the actual numerical mass flux, including reconstruction.
    """

    def __init__(self, solver, residual_policy=None, monitors=(), pressure_total=None):
        self.solver = solver
        self.policy = ResidualPolicy() if residual_policy is None else residual_policy
        if not isinstance(self.policy, ResidualPolicy):
            raise TypeError("residual_policy must be a ResidualPolicy")
        self.monitors = tuple(monitors)
        if any(not isinstance(spec, MonitorSpec) for spec in self.monitors):
            raise TypeError("monitors must contain MonitorSpec instances")
        # Tracker validates unique names and the generic stability policy.
        MonitorTracker(self.monitors)
        volumes = jnp.asarray(solver.mesh.volumes)
        widths = jnp.asarray(solver.mesh.widths)
        length = float(np.sum(solver.mesh.widths))
        q_scale = jnp.asarray(solver.conservative_scales)
        flux_scale = jnp.asarray(solver.spatial_flux_scales)
        global_scale = jnp.asarray(solver.global_flux_scales)
        reference_time = solver.mesh.geometry.length/solver.reference_sound_speed
        validate_residual_setup(solver.mesh.volumes, solver.conservative_scales, reference_time)
        resolved = []
        for spec in self.monitors:
            quantity = {"cp": "pressure_coefficient", "mdot": "mass_flow"}.get(
                spec.quantity, spec.quantity)
            if quantity == "pressure_coefficient":
                if spec.reduction not in ("inlet", "outlet", "length_mean", "length_integral"):
                    raise ValueError(f"Unsupported Cp reduction {spec.reduction!r}")
                p0 = spec.parameters.get("pressure_total", pressure_total)
                if p0 is None:
                    p0 = solver.inlet_data.get("p0")
                if p0 is None or not np.isfinite(p0) or p0 <= 0:
                    raise ValueError("Cp monitoring needs a finite positive pressure_total")
                resolved.append((quantity, spec.reduction, float(p0)))
            elif quantity == "mass_flow":
                if spec.reduction not in ("inlet", "outlet", "mean"):
                    raise ValueError(f"Unsupported mass-flow reduction {spec.reduction!r}")
                resolved.append((quantity, spec.reduction, None))
            else:
                raise ValueError(f"Unknown annular monitor quantity {spec.quantity!r}")

        @jax.jit
        def reduce_rows(physical):
            scaled = physical/flux_scale
            return dict(
                norms=reduce_residuals(physical, volumes, q_scale, reference_time,
                                      weighting=self.policy.weighting),
                spatial_rms=jnp.sqrt(jnp.mean(scaled**2)),
                spatial_max=jnp.max(jnp.abs(scaled)))

        @jax.jit
        def auxiliary(full):
            flux = solver.face_flux(full)
            source = solver.source(full)
            bc = solver.boundary_residual(full)
            balance = flux[-1]-flux[0]-jnp.sum(source, axis=0)
            values = []
            for quantity, reduction, p0 in resolved:
                if quantity == "pressure_coefficient":
                    pin = full[0, 0]
                    denominator = p0-pin
                    safe = jnp.where(denominator > 0, denominator, 1.0)
                    cp = (full[1:-1, 0]-pin)/safe
                    if reduction == "inlet":
                        value = jnp.asarray(0.0)
                    elif reduction == "outlet":
                        value = (full[-1, 0]-pin)/safe
                    elif reduction == "length_mean":
                        value = jnp.sum(widths*cp)/length
                    else:
                        value = jnp.sum(widths*cp)
                    value = jnp.where(denominator > 0, value, jnp.nan)
                elif reduction == "inlet":
                    value = flux[0, 0]
                elif reduction == "outlet":
                    value = flux[-1, 0]
                else:
                    value = jnp.sum(widths*(flux[:-1, 0]+flux[1:, 0])/2)/length
                values.append(value)
            return dict(
                monitor_values=jnp.asarray(values),
                integrated_balance_scaled=balance/global_scale,
                boundary_rms=jnp.sqrt(jnp.mean(bc**2)),
                boundary_max=jnp.max(jnp.abs(bc)),
                conservation_max=jnp.max(jnp.abs(balance/global_scale)),
                boundary_mach=boundary_mach(full, solver.fluid),
                admissible=solver.is_admissible(full),
            )

        def evaluate(full):
            # Keep the canonical signed spatial kernel as an execution boundary.
            # Fusing it into a larger diagnostics graph can change cancellation
            # at roundoff, enough to alter an initial-relative nine-order test.
            # The reducer remains JIT-compiled and reads exactly the public rows.
            rows = solver.spatial_residual(full)
            return dict(**reduce_rows(rows), **auxiliary(full))

        self.evaluate = evaluate
        self.metadata = dict(
            definition=DEFINITION, equation_names=list(EQUATIONS),
            policy=asdict(self.policy), initial_sample="before first pseudo-time update",
            accepted_samples="post-update original spatial residual",
            conservative_scales=np.asarray(q_scale).tolist(),
            spatial_flux_scales=np.asarray(flux_scale).tolist(),
            global_flux_scales=np.asarray(global_scale).tolist(),
            reference_time=reference_time,
            reference_length=float(solver.mesh.geometry.length),
            reference_speed=solver.reference_sound_speed,
            volumes=np.asarray(volumes).tolist(),
            monitors=[dict(**asdict(spec), resolved_quantity=quantity,
                           resolved_pressure_total=p0)
                      for spec, (quantity, _, p0) in zip(self.monitors, resolved)],
            zero_initial_rule="initial norm <= initial_floor: absolute safeguard only",
            controller_metric="fixed growth after accepted strict Newton step",
        )

    def sample(self, state, initial_norms=None):
        raw = _plain(jax.device_get(self.evaluate(state)))
        norms = raw.pop("norms")
        selected = np.asarray(norms[self.policy.norm])
        initial_norms = selected if initial_norms is None else initial_norms
        assessed = _plain(jax.device_get(assess_residuals(selected, initial_norms, self.policy)))
        values = dict(zip((spec.name for spec in self.monitors), raw.pop("monitor_values")))
        raw.update({"equation_"+name: norms[name] for name in ("l1", "l2", "linf", "worst_cell")})
        raw.update(assessed)
        raw["equation_norm"] = selected.tolist()
        raw["finite"] = bool(norms["finite"] and assessed["finite"]
                             and all(np.isfinite(value) for value in values.values())
                             and np.isfinite(raw["boundary_max"])
                             and np.isfinite(raw["conservation_max"]))
        raw["boundary_passed"] = bool(raw["boundary_max"] <= self.policy.boundary_tolerance)
        raw["global_passed"] = bool(raw["conservation_max"] <= self.policy.global_tolerance)
        raw["boundary_ratio"] = raw["boundary_max"]/self.policy.boundary_tolerance
        raw["global_ratio"] = raw["conservation_max"]/self.policy.global_tolerance
        raw["overall_max_ratio"] = max(raw["equation_max_ratio"], raw["boundary_ratio"], raw["global_ratio"])
        return raw, values


def solve_steady_monitored(solver, initial_state=None, *, residual_policy=None,
                           monitors=(), pressure_total=None, max_steps=16000,
                           cfl=0.2, cfl_growth=1.2, cfl_shrink=0.5,
                           cfl_max=1e6, max_retries=12, stagnation_steps=0, callback=None):
    """March strict implicit updates with independent, equation-wise stopping."""
    from .fvm import AnnularSolution

    if (not np.all(np.isfinite([cfl, cfl_growth, cfl_shrink, cfl_max]))
            or cfl <= 0 or cfl_max < cfl or not 1 < cfl_growth <= 2
            or not 0 < cfl_shrink < 1
            or isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer))
            or isinstance(max_retries, bool) or not isinstance(max_retries, (int, np.integer))
            or max_steps < 0 or max_retries < 0
            or isinstance(stagnation_steps, bool)
            or not isinstance(stagnation_steps, (int, np.integer)) or stagnation_steps < 0):
        raise ValueError("Invalid monitored steady CFL/controller settings")
    state = jnp.asarray(solver.initial_state() if initial_state is None else initial_state,
                        dtype=jnp.float64)
    if state.shape != solver.shape:
        raise ValueError(f"Initial state must have shape {solver.shape}")
    adapter = AnnularMonitoring(solver, residual_policy, monitors, pressure_total)
    tracker = MonitorTracker(adapter.monitors)
    history, accepted, attempt, pseudo_time = [], 0, 0, 0.0
    start = perf_counter()
    metadata = dict(adapter.metadata, controller=dict(
        initial_cfl=cfl, growth=cfl_growth, shrink=cfl_shrink,
        cfl_max=cfl_max, max_retries=max_retries, stagnation_steps=stagnation_steps))
    if not bool(solver.is_admissible(state)):
        metadata["reason"] = "invalid_initial_state"
        return AnnularSolution(np.asarray(state), False, "Invalid initial state", history,
                               solver._invalid_metrics(), monitoring=metadata)
    current, values = adapter.sample(state)
    initial_norms = np.asarray(current["equation_norm"])
    metadata["initial_equation_norms"] = initial_norms.tolist()
    current["monitors"] = tracker.update(values)
    current["monitors_passed"] = bool(tracker.passed)

    def emit(row):
        history.append(row)
        if callback is not None:
            callback(row)

    def make_row(metrics, *, phase, retry=0, dt=0.0, step_cfl=cfl, inner=None):
        row = dict(step=accepted, attempt=attempt, retry=retry, phase=phase,
                   accepted=True, dt=dt, cfl=step_cfl, pseudo_time=pseudo_time,
                   elapsed=perf_counter()-start, **metrics)
        if inner is not None:
            row.update(inner_iterations=int(inner.iterations), inner_status=int(inner.status),
                       inner_residual=float(inner.residual_norm),
                       inner_relative_update=float(inner.relative_update),
                       rejected_newton_trials=int(inner.rejected_trials))
        return row

    emit(make_row(current, phase="initial"))

    def converged(metrics):
        return bool(metrics["finite"] and metrics["admissible"]
                    and metrics["all_equations_passed"] and metrics["boundary_passed"]
                    and metrics["global_passed"] and metrics["monitors_passed"])

    reason, unchanged_updates = "max_iterations", 0
    while accepted < max_steps and current["finite"] and not converged(current):
        step_accepted = False
        for retry in range(max_retries+1):
            attempt += 1
            dt = float(solver.compute_dt(state, cfl))
            candidate, info = solver.step(state, dt)
            info = jax.device_get(info)
            okay = bool(info.converged and info.admissible)
            trial = monitor_values = None
            if okay:
                trial, monitor_values = adapter.sample(candidate, initial_norms)
                okay = bool(trial["admissible"] and trial["finite"])
            if okay:
                unchanged = (stagnation_steps > 0 and cfl >= cfl_max
                             and np.array_equal(np.asarray(candidate), np.asarray(state)))
                unchanged_updates = unchanged_updates+1 if unchanged else 0
                state = candidate
                accepted += 1
                pseudo_time += dt
                current = trial
                current["monitors"] = tracker.update(monitor_values)
                current["monitors_passed"] = bool(tracker.passed)
                row = make_row(current, phase="accepted", retry=retry, dt=dt, step_cfl=cfl, inner=info)
                row["next_cfl_factor"] = cfl_growth
                emit(row)
                cfl = min(cfl_max, cfl*cfl_growth)
                step_accepted = True
                break
            # No previous-state metrics are attached to an unevaluated trial.
            row = dict(step=accepted+1, attempt=attempt, retry=retry, phase="rejected",
                       accepted=False, dt=dt, cfl=cfl, pseudo_time=pseudo_time,
                       elapsed=perf_counter()-start, inner_iterations=int(info.iterations),
                       inner_status=int(info.status), rejected_newton_trials=int(info.rejected_trials),
                       inner_residual=float(info.residual_norm),
                       inner_relative_update=float(info.relative_update),
                       diagnostics_evaluated=trial is not None, next_cfl_factor=cfl_shrink)
            if trial is not None:
                row["candidate_diagnostics"] = trial
                row["candidate_monitor_values"] = monitor_values
            emit(row)
            cfl *= cfl_shrink
        if not step_accepted:
            reason = "retry_limit"
            break
        if stagnation_steps and unchanged_updates >= stagnation_steps and not converged(current):
            reason = "stagnation"
            break

    # The returned field is independently reevaluated, without advancing the
    # monitor's window by adding a duplicate final sample.
    final, final_values = adapter.sample(state, initial_norms)
    for name, value in final_values.items():
        if not np.isclose(value, current["monitors"][name]["value"], rtol=1e-12,
                          atol=0.0, equal_nan=True):
            raise RuntimeError("Final monitor audit does not match the returned accepted state")
    final["monitors"] = current["monitors"]
    final["monitors_passed"] = current["monitors_passed"]
    root_ok = converged(final)
    regime_ok = solver._regimes_match(final)
    success = bool(root_ok and regime_ok)
    if success:
        reason = "converged"
    elif root_ok and not regime_ok:
        reason = "regime_mismatch"
    elif not final["finite"]:
        reason = "nonfinite"
    messages = dict(
        converged="Every equation reduction, absolute safeguard, boundary/global and monitor target met",
        regime_mismatch="Residual targets met but selected boundary regimes do not match",
        nonfinite="Nonfinite required monitoring diagnostics",
        retry_limit="Implicit timestep retries exhausted",
        max_iterations="Steady update limit reached",
        stagnation="State unchanged at maximum CFL; requested convergence targets remain unmet")
    metadata.update(reason=reason, accepted_updates=accepted, attempts=attempt,
                    rejected_attempts=attempt-accepted, pseudo_time=pseudo_time,
                    final_cfl=cfl, elapsed=perf_counter()-start, unchanged_updates=unchanged_updates,
                    final_audit="fresh original spatial/boundary/source operators")
    return AnnularSolution(np.asarray(state), success, messages[reason], history, final,
                           monitoring=metadata)
