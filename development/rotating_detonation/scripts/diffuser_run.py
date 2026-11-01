"""Run the angle-only annular diffuser study with fixed reservoir and backpressure."""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml

from nozzle_hyperbolic.models.euler_annular.state import primitive_fields
from nozzle_hyperbolic.solvers.fvm import (
    Operator, ResidualPolicy, Scheme, SteadySettings, Solution, sampler, solve_single_grid,
)
from .diffuser_geometry import analytical_profile, cold_initial, native_meshes
from .diffuser_model import make_model

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
OUT = STUDY / "outputs" / "diffuser"
FIELD_NAMES = ("p", "M", "M_m", "T", "alpha_deg", "mdot", "rho", "v_m", "v_theta", "h0", "s")


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprints():
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src").rglob("*.py")}


def scheme(config, method):
    return Scheme(flux=config["numerics"]["flux"],
                  reconstruction=config["numerics"]["reconstruction"],
                  time_integrator="euler-implicit" if method == "implicit" else "runge-kutta-3",
                  boundary_treatment="coupled" if method == "implicit" else "segregated",
                  newton_iterations=config["numerics"].get("newton_iterations", 15),
                  newton_backtracking=config["numerics"].get("newton_backtracking", True))


def state_diagnostics(model, state, time=0.0):
    """Diagnostics use physical traces and every cell, not exterior targets."""
    values = model.constraint_metrics(state)
    vm = state[:, 1]
    speed = jnp.sqrt(vm**2 + (state[:, 2] / model._r_full)**2)
    start = int(np.searchsorted(model.native_mesh.x_center,
                               model.config["geometry"]["constant_length_m"])) + 1
    end = int(np.searchsorted(model.native_mesh.x_center, model.config["geometry"]["constant_length_m"] + model.config["geometry"]["expansion_length_m"], side="right"))
    values.update(
        expansion_meridional_ratio=vm[end] / vm[start],
        expansion_speed_ratio=speed[end] / speed[start],
        boundary_max=jnp.max(jnp.abs(model.boundary_residual(state, time))),
    )
    return values


def regime_passed(values):
    return (values["normal_mach_min"] > 0 and values["normal_mach_max"] < 1
            and values["meridional_velocity_ratio"] < 1
            and values["expansion_meridional_ratio"] < 1)


def fields_and_boundaries(model, state, time, operator):
    fields = model.fields(state)
    full = primitive_fields(state, model._r_full, model.fluid)
    cp = model.fluid.constants.gamma * model.fluid.constants.R / (model.fluid.constants.gamma - 1)
    gamma = model.fluid.constants.gamma
    flux = operator.face_flux(state)
    return ({key: np.asarray(fields[key]) for key in FIELD_NAMES},
            dict(inlet_M=float(full["M"][0]), inlet_M_m=float(full["M_m"][0]),
                 outlet_M_m=float(full["M_m"][-1]),
                 inlet_p0=float(full["p"][0] * (1 + (gamma - 1)*full["M"][0]**2/2)**(gamma/(gamma-1))),
                 inlet_T0=float(full["h0"][0] / cp),
                 inlet_alpha_deg=float(jnp.rad2deg(jnp.arctan2(full["v_theta"][0], full["v_m"][0]))),
                 inlet_mdot=float(flux[0, 0]), outlet_mdot=float(flux[-1, 0]),
                 inlet_vm=float(full["v_m"][0]), outlet_vm=float(full["v_m"][-1]),
                 inlet_pressure=float(full["p"][0]), outlet_pressure=float(full["p"][-1]),
                 target_outlet=float(model.target_outlet(time))))


def solution_record(solution):
    model = solution.operator.model
    field, boundary = fields_and_boundaries(model, jnp.asarray(solution.state), 0.0, solution.operator)
    return dict(cells=model.mesh.n_cells, x=model.native_mesh.x_center, state=solution.state,
                fields=field, boundary_fields=boundary, success=solution.success,
                reason=solution.reason, metrics=solution.metrics, history=solution.history,
                elapsed_s=solution.elapsed_s,
                constraints={k: float(v) for k, v in state_diagnostics(model, jnp.asarray(solution.state)).items()})


def steady_sequence(config, meshes):
    records, solutions = [], []
    previous = None
    numerical = scheme(config, "implicit")
    settings = SteadySettings(cfl=0.2, cfl_growth=1.2, cfl_max=1e6,
                              max_steps=config["numerics"]["max_steady_steps"])
    for n in config["numerics"]["steady_grids"]:
        native = meshes[n]
        if previous is not None:
            old = previous.operator.mesh.points
            points = np.r_[native.m_faces[0], native.m_centers, native.m_faces[-1]]
            seed = np.column_stack([np.interp(points, old, previous.state[:, k]) for k in range(4)])
        else:
            seed = cold_initial(config, native)
        model = make_model(config, native, initial_state=seed)
        print(f"STEADY pressure N={n}", flush=True)
        solution = solve_single_grid(model, numerical, settings, initial_state=seed)
        records.append(solution_record(solution)); solutions.append(solution)
        save(OUT / "steady_pressure.json", records)
        print("  ", solution.success, solution.reason,
              records[-1]["constraints"], flush=True)
        if not solution.success:
            break
        previous = solution
    return records, solutions


def periodic_assessment(states, frames, scales, forcing, inventory_change):
    """Full-field repeatability at one-cycle and ten-cycle separations."""
    if len(states) < 2*frames + 1:
        return None
    recent = np.asarray(states[-frames:])[:, 1:-1]
    previous = np.asarray(states[-2*frames:-frames])[:, 1:-1]
    change = (recent - previous)/scales
    rms = np.sqrt(np.mean(change**2, axis=(0, 1)))
    maximum = float(np.max(np.abs(change)))
    lag = int(forcing["comparison_lag_cycles"])
    lag_complete = len(states) >= (lag+1)*frames+1
    lag_rms = None
    lag_maximum = None
    if lag_complete:
        earlier = np.asarray(states[-(lag+1)*frames:-lag*frames])[:, 1:-1]
        lag_change = (recent-earlier)/scales
        lag_rms = np.sqrt(np.mean(lag_change**2, axis=(0, 1)))
        lag_maximum = float(np.max(np.abs(lag_change)))
    passed = (lag_complete and max(rms) <= forcing["periodic_tolerance"]
              and maximum <= forcing["periodic_maximum_tolerance"]
              and max(lag_rms) <= forcing["periodic_tolerance"]
              and lag_maximum <= forcing["periodic_maximum_tolerance"]
              and max(abs(inventory_change)) <= forcing["inventory_tolerance"])
    return dict(per_variable_RMS=rms, maximum=float(max(rms)), local_maximum=maximum,
                lag_cycles=lag, lag_complete=lag_complete, lag_per_variable_RMS=lag_rms,
                lag_maximum=None if lag_rms is None else float(max(lag_rms)),
                lag_local_maximum=lag_maximum, inventory_change=inventory_change,
                inventory_maximum=float(max(abs(inventory_change))), passed=bool(passed))


def transient_run(config, native, seed, *, cfl=.4, label="run", start_cycle=0,
                  analysis_role="primary"):
    """Run RK3, verify periodicity, then retain ten further validated cycles.

    A physical regime failure is saved and stops the run. Only genuine inner
    solver/thermodynamic failure can reject and halve a trial timestep.
    """
    run_start = perf_counter()
    seed_digest = sha256(np.asarray(seed, dtype=np.float64).tobytes()).hexdigest()
    model = make_model(config, native, driven=True, initial_state=seed)
    numerical = scheme(config, "rk3")
    operator = Operator(model, numerical)
    forcing = config["forcing"]
    frequency, frames = forcing["frequency_Hz"], forcing["frames_per_cycle"]
    refinement = analysis_role == "time_refinement"
    maximum = forcing["refinement_cycles"] if refinement else forcing["maximum_cycles"]
    minimum = forcing["minimum_cycles"]
    initial_time = start_cycle/frequency
    state, initial_info = operator.complete_boundaries(jnp.asarray(seed), initial_time)
    if not bool(initial_info.converged and initial_info.admissible):
        raise RuntimeError(f"Initial boundary completion failed for {label}: {initial_info}")
    start_index = int(np.searchsorted(native.x_center, config["geometry"]["constant_length_m"]))+1
    end_index = int(np.searchsorted(native.x_center, config["geometry"]["constant_length_m"]
                                    +config["geometry"]["expansion_length_m"], side="right"))

    @jax.jit
    def advance(old, dt, end):
        candidate, info, transfer = operator.step_with_balance(old, dt, end)
        values = model.constraint_metrics(candidate)
        speed = jnp.sqrt(candidate[:, 1]**2+(candidate[:, 2]/model._r_full)**2)
        values.update(expansion_meridional_ratio=candidate[end_index, 1]/candidate[start_index, 1],
                      expansion_speed_ratio=speed[end_index]/speed[start_index],
                      boundary_max=jnp.max(jnp.abs(model.boundary_residual(candidate, end))),
                      outlet_pressure_error=candidate[-1, 0]-model.target_outlet_pressure(end))
        return candidate, info, transfer, values

    initial_totals = np.asarray(operator.storage(state[1:-1])).sum(axis=0)
    totals = initial_totals.copy()
    norm = np.sum(native.volumes)*np.asarray(model.conservative_scales)
    influx = np.zeros(4); budget = np.zeros(4)
    history, rejection_history, metrics, cycles_history = [], [], [], []
    states, times, contents = [np.asarray(state)], [initial_time], [totals.copy()]
    field, boundary = fields_and_boundaries(model, state, initial_time, operator)
    field_series = {k: [v] for k, v in field.items()}
    boundary_series = {k: [v] for k, v in boundary.items()}
    reason = "maximum_cycles"; consecutive = 0; converged = False
    candidate_cycle = None; resets = 0
    initial_values = {k: float(v) for k, v in state_diagnostics(model, state, initial_time).items()}
    initial_values["outlet_pressure_error"] = float(state[-1, 0]-model.target_outlet_pressure(initial_time))
    extrema = {k: [v, v] for k, v in initial_values.items()}
    rejected = 0; violation = None; max_budget = 0.0
    if not regime_passed(initial_values):
        raise ValueError(f"Initial state violates diffuser regime: {initial_values}")
    dt0 = float(operator.timestep(state, cfl, initial_time).dt)
    compile_start = perf_counter()
    _, compile_info, _, _ = jax.device_get(advance(state, jnp.float64(dt0), jnp.float64(initial_time+dt0)))
    compile_s = perf_counter()-compile_start
    start = perf_counter(); time = initial_time; stopped = False
    for local_cycle in range(1, maximum+1):
        cycle = start_cycle+local_cycle
        cycle_start_totals = totals.copy()
        for frame in range(1, frames+1):
            target = ((cycle-1)*frames+frame)/(frequency*frames)
            while time < target:
                if len(history) >= config["numerics"]["max_transient_steps"]:
                    reason = "maximum_steps"; stopped = True; break
                limits = jax.device_get(operator.timestep(state, cfl, time))
                if not bool(limits.valid) or not np.isfinite(limits.dt) or limits.dt <= 0:
                    reason = "invalid_timestep"; stopped = True; break
                dt = min(float(limits.dt), target-time)
                okay = False
                for retry in range(16):
                    if not np.isfinite(dt) or dt <= 0 or time+dt <= time:
                        reason = "invalid_timestep"; break
                    candidate, info, transfer, values = advance(state, jnp.float64(dt), jnp.float64(time+dt))
                    info, transfer, values = jax.device_get((info, transfer, values))
                    if bool(info.converged and info.admissible):
                        okay = True; break
                    rejection_history.append(dict(time=time, dt=dt, retry=retry,
                        inner_residual=float(info.residual_norm), iterations=int(info.iterations)))
                    rejected += 1; dt *= .5
                if not okay:
                    if reason != "invalid_timestep": reason = "retry_limit"
                    stopped = True; break
                state = candidate; time = min(target, time+dt)
                influx += np.asarray(transfer)
                totals = np.asarray(operator.storage(state[1:-1])).sum(axis=0)
                budget = (totals-initial_totals-influx)/norm
                max_budget = max(max_budget, float(max(abs(budget))))
                values = {k: float(v) for k, v in values.items()}
                history.append(dict(time=time, dt=dt, cfl=dt/float(limits.wave_dt),
                                    iterations=int(info.iterations), retry=retry, **values))
                for k, value in values.items():
                    extrema[k][0] = min(extrema[k][0], value)
                    extrema[k][1] = max(extrema[k][1], value)
                if not regime_passed(values):
                    reason = "requested_flow_regime_violated"; violation = dict(time=time, **values)
                    stopped = True; break
                if values["boundary_max"] > 1e-9 or max_budget > 1e-8:
                    reason = "boundary_or_conservation_audit_failed"; stopped = True; break
            if time > times[-1]:
                states.append(np.asarray(state)); times.append(time); contents.append(totals.copy())
                field, boundary = fields_and_boundaries(model, state, time, operator)
                for k, v in field.items(): field_series[k].append(v)
                for k, v in boundary.items(): boundary_series[k].append(v)
                metrics.append(dict(time=time, boundary_max=float(jnp.max(jnp.abs(operator.boundary_residual(state, time)))),
                                    conservation_budget=budget))
            if stopped: break
        if stopped: break
        assessment = periodic_assessment(states, frames, np.asarray(model.variable_scales), forcing,
                                         (totals-cycle_start_totals)/norm)
        if assessment is not None:
            consecutive = consecutive+1 if assessment["passed"] else 0
            if candidate_cycle is not None and not assessment["passed"]:
                candidate_cycle = None; resets += 1
            if candidate_cycle is None and local_cycle >= minimum and consecutive >= forcing["consecutive_cycles"]:
                candidate_cycle = cycle
            cycles_history.append(dict(cycle=cycle, candidate_cycle=candidate_cycle, **assessment))
        print(f"{label}: cycle={cycle} M_in={boundary_series['inlet_M'][-1]:.6f} "
              f"Mm_max={extrema['normal_mach_max'][1]:.6f} "
              f"error={assessment['maximum'] if assessment else None} "
              f"lag={assessment['lag_maximum'] if assessment else None} candidate={candidate_cycle}", flush=True)
        save(OUT/"progress"/f"{label}.json", dict(cycle=cycle, time=time, candidate_cycle=candidate_cycle,
             summary=cycles_history[-1] if cycles_history else None, extrema=extrema,
             accepted_steps=len(history), elapsed_s=perf_counter()-start))
        if candidate_cycle is not None and cycle-candidate_cycle >= forcing["collection_cycles"]:
            reason = "periodic_converged_and_collected"; converged = True; break
    if not stopped and not converged:
        reason = "time_refinement_window_completed" if refinement else "cycle_limit_reached"
    elapsed = perf_counter()-start
    summary = dict(inlet_M_min=extrema["inlet_total_mach"][0], inlet_M_max=extrema["inlet_total_mach"][1],
        inlet_M_m_min=extrema["inlet_normal_mach"][0], inlet_M_m_max=extrema["inlet_normal_mach"][1],
        domain_M_m_min=extrema["normal_mach_min"][0], domain_M_m_max=extrema["normal_mach_max"][1],
        expansion_deceleration_passed=extrema["expansion_meridional_ratio"][1]<1,
        total_speed_deceleration_passed=(extrema["total_speed_ratio"][1]<1 and extrema["expansion_speed_ratio"][1]<1),
        flow_regime_passed=violation is None, maximum_expansion_vm_ratio=extrema["expansion_meridional_ratio"][1],
        maximum_expansion_speed_ratio=extrema["expansion_speed_ratio"][1],
        outlet_pressure_error_max_Pa=max(abs(v) for v in extrema["outlet_pressure_error"]),
        maximum_conservation_defect=max_budget, accepted_steps=len(history), rejected_steps=rejected,
        actual_cfl_min=min((r["cfl"] for r in history), default=0.), actual_cfl_max=max((r["cfl"] for r in history), default=0.),
        elapsed_s=elapsed, compilation_s=compile_s, setup_and_execution_s=perf_counter()-run_start,
        periodic_verified=converged, periodic_candidate_cycle=candidate_cycle,
        collection_start_cycle=candidate_cycle, collection_end_cycle=round(time*frequency) if converged else None,
        collection_cycles_completed=forcing["collection_cycles"] if converged else 0, collection_resets=resets,
        completed_cycles=time*frequency, elapsed_cycles=time*frequency-start_cycle, violation=violation,
        start_time_s=initial_time, end_time_s=time,
        audit_frequency="Every accepted physical timestep, including traces and all cells")
    record = dict(id=label, method="rk3", analysis_role=analysis_role, cfl=cfl, outlet_kind="pressure",
        config=config, scheme=asdict(numerical), success=not stopped, reason=reason,
        times=times, x=native.x_center, states=states, fields=field_series, boundary_fields=boundary_series,
        history=history, rejections=rejection_history, metrics=metrics, contents=contents,
        summary=summary, periodicity=cycles_history, variable_scales=model.variable_scales,
        initial_totals=initial_totals, initial_state_sha256=seed_digest,
        initial_condition_kind="trajectory_checkpoint" if refinement else "converged_steady_state",
        regime_policy="extended_area_two_subsonic_meridional_v5",
        final_totals=totals, integrated_transfer=influx, content_scales=norm,
        discarded_compilation_step_success=bool(compile_info.converged and compile_info.admissible))
    save(OUT/"runs"/f"{label}.json", record)
    print(label, record["success"], reason, summary, flush=True)
    return record


def cached_steady(config, meshes):
    """Reuse a saved root only after reassembling its original equations."""
    path=OUT/'steady_pressure.json'
    if not path.exists():return None
    records=json.loads(path.read_text());solutions=[]
    if [r['cells'] for r in records]!=config['numerics']['steady_grids']:return None
    for record in records:
        if not record['success']:return None
        native=meshes[record['cells']];state=jnp.asarray(record['state'])
        model=make_model(config,native,initial_state=state)
        operator=Operator(model,scheme(config,'implicit'))
        audit=sampler(operator,ResidualPolicy())(state,np.asarray(record['history'][0]['equation_norm']))
        if not (audit['all_equations_passed'] and audit['boundary_passed'] and audit['global_passed'] and model.solution_audit(state)):
            return None
        solutions.append(Solution(np.asarray(state),True,'cached_root_reaudited',record['history'],audit,
                                  elapsed_s=record['elapsed_s'],operator=operator))
    print('Reused pressure steady roots after independent residual audit',flush=True)
    return records,solutions


def temporal_error(record, reference, frames):
    """Compare identical physical times over the final ten-cycle window."""
    cycles = record["config"]["forcing"]["collection_cycles"]
    end = min(record["times"][-1], reference["times"][-1])
    begin = end-cycles/record["config"]["forcing"]["frequency_Hz"]
    a = np.flatnonzero((np.asarray(record["times"]) >= begin-1e-13) & (np.asarray(record["times"]) <= end+1e-13))
    b = np.flatnonzero((np.asarray(reference["times"]) >= begin-1e-13) & (np.asarray(reference["times"]) <= end+1e-13))
    if len(a) != cycles*frames+1 or len(b) != len(a) or not np.allclose(np.asarray(record["times"])[a], np.asarray(reference["times"])[b], rtol=0, atol=1e-13):
        return dict(passed=False, reason="incomplete_matched_ten_cycle_window")
    result = dict(window_cycles=[begin*record["config"]["forcing"]["frequency_Hz"], end*record["config"]["forcing"]["frequency_Hz"]],
                  start_s=begin, end_s=end, comparison="Identical absolute times over the final ten cycles; periodicity is assessed separately")
    for field, scale in [("p", 2e6), ("M", 1.), ("M_m", 1.), ("alpha_deg", 45.), ("v_m", 600.)]:
        delta = np.asarray(record["fields"][field])[a]-np.asarray(reference["fields"][field])[b]
        rms = float(np.sqrt(np.mean(delta**2)))
        result[field] = dict(RMS=rms, normalized_RMS=rms/scale, maximum=float(np.max(abs(delta))))
    result["passed"] = all(result[k]["normalized_RMS"] < 1e-3 for k in ("p", "M", "M_m", "alpha_deg", "v_m"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Audit saved states and regenerate all figures/report without CFD")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        from .diffuser_postprocess import process
        process(OUT / "results.json")
        return
    config = yaml.safe_load((STUDY / "case.yaml").read_text(encoding="utf-8-sig"))
    if config["outlet"]["kind"] != "pressure":
        raise ValueError("This focused case uses constant outlet static pressure")
    before = fingerprints()
    study_before = {p.relative_to(STUDY).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in (STUDY / "scripts").glob("diffuser_*.py")}
    _, meshes, dense = native_meshes(config, config["numerics"]["steady_grids"])
    save(OUT / "geometry.json", dense)
    reference = analytical_profile(config, meshes[config["numerics"]["transient_cells"]])
    if not np.isclose(reference["outlet_pressure_Pa"], config["outlet"]["pressure_Pa"], rtol=1e-10):
        raise ValueError("Nominal reference and fixed backpressure differ; update the independent reference first")
    cached = cached_steady(config, meshes)
    records, solutions = cached if cached else steady_sequence(config, meshes)
    if len(solutions) != len(config["numerics"]["steady_grids"]) or not all(s.success for s in solutions):
        raise RuntimeError("Steady grid sequence failed; see saved original-residual evidence")
    result = dict(config=config, geometry=dense, reference=reference,
                  fluid_constants=dict(gamma=float(solutions[-1].operator.model.fluid.constants.gamma),
                                       R=float(solutions[-1].operator.model.fluid.constants.R)),
                  provenance=dict(source_before=before, study_sources_before=study_before,
                     scope="All study writes are inside development/rotating_detonation. External source edits are permitted.",
                     dtype="float64", jax=jax.__version__, devices=str(jax.devices()),
                     timing_protocol="Single-run validation costs, some executed concurrently on a shared machine. A discarded compiled trial is excluded from execution time. These wall times cannot establish a fair method speed ranking."),
                  steady=dict(implicit=dict(grids=records)), transient={}, temporal_errors={},
                  comparison={}, periodicity={})
    seed = solutions[-1].state
    native = meshes[config["numerics"]["transient_cells"]]
    digest = sha256(np.asarray(seed, dtype=np.float64).tobytes()).hexdigest()
    def run(label, cfl, initial, *, start_cycle=0, role="primary"):
        path = OUT/"runs"/f"{label}.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        digest = sha256(np.asarray(initial, dtype=np.float64).tobytes()).hexdigest()
        if (old and old.get("config") == config and old.get("cfl") == cfl
                and old.get("initial_state_sha256") == digest
                and old.get("regime_policy") == "extended_area_two_subsonic_meridional_v5"):
            record = old
            print("Reusing exact saved campaign record", label, flush=True)
        else:
            record = transient_run(config, native, initial, cfl=cfl, label=label,
                                   start_cycle=start_cycle, analysis_role=role)
        result["transient"][label] = record
        save(OUT/"results.json", result)
        return record

    primary = run("rk3_primary", config["numerics"]["rk3_cfl"], seed)
    result.update(selected_id=primary["id"], startup_id=primary["id"], comparison_reference_id=primary["id"])
    if primary["success"]:
        end_cycle = int(round(primary["times"][-1]*config["forcing"]["frequency_Hz"]))
        start_cycle = end_cycle-config["forcing"]["refinement_cycles"]
        index = int(np.argmin(abs(np.asarray(primary["times"])-start_cycle/config["forcing"]["frequency_Hz"])))
        checkpoint = np.asarray(primary["states"][index])
        refined = run("rk3_refined", config["numerics"]["rk3_refined_cfl"], checkpoint,
                      start_cycle=start_cycle, role="time_refinement")
        if refined["success"]:
            result["temporal_errors"][primary["id"]] = temporal_error(primary, refined, config["forcing"]["frames_per_cycle"])
            result["comparison_reference_id"] = refined["id"]
            if not result["temporal_errors"][primary["id"]]["passed"]:
                finest = run("rk3_finest", config["numerics"]["rk3_refined_cfl"]/2, checkpoint,
                             start_cycle=start_cycle, role="time_refinement")
                if finest["success"]:
                    result["temporal_errors"][refined["id"]] = temporal_error(refined, finest, config["forcing"]["frames_per_cycle"])
                    result.update(selected_id=refined["id"], comparison_reference_id=finest["id"])
    result["periodicity"] = {label: dict(verified=r["summary"]["periodic_verified"], cycles=r["periodicity"],
        criterion="Five consecutive full-field, local, ten-cycle-lag and inventory checks, followed by ten additional cycles with all checks active.")
        for label, r in result["transient"].items()}
    result["provenance"]["source_after"] = fingerprints()
    result["provenance"]["study_sources_after"] = {p.relative_to(STUDY).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in (STUDY/"scripts").glob("diffuser_*.py")}
    save(OUT/"results.json", result)
    from .diffuser_postprocess import process
    process(OUT/"results.json")


if __name__ == "__main__":
    main()
