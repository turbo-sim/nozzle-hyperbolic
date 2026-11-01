"""Audit retained diffuser evidence, compare periodic means, and render media.

All operations use saved states.  This module performs no physical time steps,
steady solves or continuation.  Only the current diffuser model/geometry are
imported for independent equations and native-mesh checks.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / "outputs/diffuser"
ROOT = STUDY.parents[1]
FIELD_NAMES = ("p", "M", "M_m", "T", "alpha_deg", "mdot", "rho", "v_m", "v_theta", "h0", "s")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float)**2)))


def _verified(record):
    return bool(record.get("success") and record.get("summary", {}).get("periodic_verified"))


def _physical_signature(record, fallback):
    config = record.get("config", fallback)
    forcing = {k: config["forcing"].get(k) for k in ("frequency_Hz", "angle_amplitude_deg")}
    inlet = {k: config["inlet"].get(k) for k in
             ("stagnation_temperature_K", "stagnation_pressure_Pa", "angle_deg")}
    outlet = config["outlet"]
    return dict(geometry=config["geometry"], inlet=inlet, outlet=outlet,
                fluid=config["fluid"], forcing=forcing)


def periodic_assessment(record, config):
    """Independently recompute candidate detection and all ten collection cycles."""
    forcing = config["forcing"]
    frames, frequency = int(forcing["frames_per_cycle"]), float(forcing["frequency_Hz"])
    scales = np.asarray(record["variable_scales"], dtype=float)
    times, states = np.asarray(record["times"]), np.asarray(record["states"])
    contents = np.asarray(record["contents"])
    norm = np.asarray(record["content_scales"])
    if states.shape[0] != times.size or contents.shape != (len(times), 4) or scales.shape != (4,):
        raise ValueError("Periodicity requires matching states, times and conserved contents")
    if np.any(np.diff(times) <= 0) or np.any(scales <= 0):
        raise ValueError("Strictly increasing times and positive fixed scales are required")
    start_cycle = int(round(times[0]*frequency))
    final_cycle = int(np.floor(times[-1]*frequency+1e-7))
    lag = int(forcing["comparison_lag_cycles"])
    rows = []; consecutive = 0; candidate = None; resets = 0
    minimum = forcing["minimum_cycles"]

    def indices(requested):
        requested = np.asarray(requested)
        ix = np.searchsorted(times, requested-1e-12/frequency)
        if (np.any(ix >= len(times)) or not np.allclose(times[ix], requested, rtol=0, atol=1e-8/frequency)):
            return None
        return ix

    for cycle in range(start_cycle+2, final_cycle+1):
        phase_times = np.arange((cycle-1)*frames+1, cycle*frames+1)/(frames*frequency)
        recent, earlier = indices(phase_times), indices(phase_times-1/frequency)
        ends = indices([(cycle-1)/frequency, cycle/frequency])
        if recent is None or earlier is None or ends is None: continue
        delta = (states[recent, 1:-1]-states[earlier, 1:-1])/scales
        rms = np.sqrt(np.mean(delta**2, axis=(0, 1)))
        local = float(np.max(abs(delta)))
        lag_index = indices(phase_times-lag/frequency) if cycle-start_cycle >= lag+1 else None
        lag_rms, lag_local = None, None
        if lag_index is not None:
            change = (states[recent, 1:-1]-states[lag_index, 1:-1])/scales
            lag_rms = np.sqrt(np.mean(change**2, axis=(0, 1)))
            lag_local = float(np.max(abs(change)))
        drift = (contents[ends[-1]]-contents[ends[0]])/norm
        passed = bool(lag_rms is not None and max(rms) <= forcing["periodic_tolerance"]
            and local <= forcing["periodic_maximum_tolerance"]
            and max(lag_rms) <= forcing["periodic_tolerance"]
            and lag_local <= forcing["periodic_maximum_tolerance"]
            and max(abs(drift)) <= forcing["inventory_tolerance"])
        consecutive = consecutive+1 if passed else 0
        if candidate is not None and not passed: candidate = None; resets += 1
        if candidate is None and cycle-start_cycle >= minimum and consecutive >= forcing["consecutive_cycles"]:
            candidate = cycle
        rows.append(dict(cycle=cycle, per_variable_RMS=rms, maximum=float(max(rms)), local_maximum=local,
            lag_per_variable_RMS=lag_rms, lag_maximum=None if lag_rms is None else float(max(lag_rms)),
            lag_local_maximum=lag_local, inventory_change=drift, inventory_maximum=float(max(abs(drift))),
            passed=passed, candidate_cycle=candidate))
    complete_end = abs(times[-1]*frequency-final_cycle) <= 1e-7
    collected = final_cycle-candidate if candidate is not None else 0
    regime = (valid_prefix(record, config) == len(times)-1
              and all(_regime(v) for v in record.get("history", [])))
    verified = bool(_verified(record) and complete_end and candidate is not None
                    and collected >= forcing["collection_cycles"] and regime)
    return dict(verified=verified, tolerance=forcing["periodic_tolerance"], primitive_scales=scales,
        end_time_s=float(times[-1]), complete_cycles=final_cycle, cycles=rows,
        candidate_cycle=candidate, collection_cycles=collected if verified else 0, collection_resets=resets,
        collection_start_cycle=candidate, collection_end_cycle=final_cycle if verified else None,
        regime_passed=regime,
        criterion="Five consecutive adjacent-cycle and ten-cycle-lag full-field RMS/local-maximum checks plus conserved-inventory repeatability, followed by ten additional cycles with every check and the meridional regime remaining valid.")


def cycle_window(record, config):
    """Return all samples and both endpoints of the ten verified collection cycles."""
    assessment = periodic_assessment(record, config)
    if not assessment["verified"]:
        raise ValueError(f"{record.get('id')} has no independently verified periodic collection window")
    frames = int(config["forcing"]["frames_per_cycle"])
    frequency = float(config["forcing"]["frequency_Hz"])
    count = int(config["forcing"]["collection_cycles"])*frames+1
    times = np.asarray(record["times"])
    if len(times) < count: raise ValueError("The complete ten-cycle window is required")
    ix = np.arange(len(times)-count, len(times)); selected = times[ix]
    expected = selected[-1]-np.arange(count-1, -1, -1)/(frames*frequency)
    if not np.allclose(selected, expected, rtol=0, atol=1e-8/frequency):
        raise ValueError("Collection samples are not complete phase-aligned periods")
    if abs(selected[-1]*frequency-round(selected[-1]*frequency)) > 1e-7:
        raise ValueError("Collection must end at a forcing-period boundary")
    return ix, selected, assessment


def valid_prefix(record, config):
    """Last retained snapshot satisfying the intended meridional regime."""
    times = np.asarray(record["times"], dtype=float)
    states = np.asarray(record["states"], dtype=float)
    normal = np.asarray(record["fields"]["M_m"], dtype=float)
    x = np.asarray(record["x"])
    start = int(np.searchsorted(x, config["geometry"]["constant_length_m"])) + 1
    end_expansion = int(np.searchsorted(x, config["geometry"]["constant_length_m"]+config["geometry"]["expansion_length_m"], side="right"))
    velocity = states[:, :, 1]
    mask = np.all(np.isfinite(states), axis=(1, 2)) & np.all(np.isfinite(normal), axis=1)
    mask &= (np.min(normal, axis=1) > 0) & (np.max(normal, axis=1) < 1)
    mask &= (velocity[:, -1]/velocity[:, 0] < 1) & (velocity[:, end_expansion]/velocity[:, start] < 1)
    for key in ("inlet_M_m", "outlet_M_m"):
        if key in record.get("boundary_fields", {}):
            values = np.asarray(record["boundary_fields"][key])
            mask &= (values > 0) & (values < 1)
    for key in ("p", "T", "rho"):
        if key in record["fields"]:
            values = np.asarray(record["fields"][key])
            mask &= np.all(np.isfinite(values) & (values > 0), axis=1)
    failed = np.flatnonzero(~mask)
    end = int(failed[0]-1) if len(failed) else len(times)-1
    if end < 0:
        raise ValueError("Initial retained snapshot does not satisfy the meridional regime")
    return end


def sample_fields(record, times):
    original = np.asarray(record["times"], dtype=float)
    if times[0] < original[0]-1e-12 or times[-1] > original[-1]+1e-12:
        raise ValueError("Sampling would extrapolate beyond retained physical time")
    fields = {}
    for key in FIELD_NAMES:
        if key in record["fields"]:
            values = np.asarray(record["fields"][key], dtype=float)
            fields[key] = np.column_stack([np.interp(times, original, values[:, i]) for i in range(values.shape[1])])
    return fields


def mean_window(record, config):
    periodic = periodic_assessment(record, config)
    if periodic["verified"]:
        _, times, _ = cycle_window(record, config)
        return times, periodic, "periodic"
    end = valid_prefix(record, config)
    original = np.asarray(record["times"])
    frequency = config["forcing"]["frequency_Hz"]
    frames = config["forcing"]["frames_per_cycle"]
    last = int(np.floor(original[end]*frequency+1e-8))
    first = int(np.ceil(original[0]*frequency-1e-8))
    cycles = min(config["forcing"]["collection_cycles"], last-first)
    if cycles >= 2:
        return np.arange((last-cycles)*frames, last*frames+1)/(frequency*frames), periodic, "finite_transient"
    if end < 1: raise ValueError("Two valid output samples are required")
    return original[:end+1], periodic, "finite_transient"


def means(record, config):
    times, periodic, kind = mean_window(record, config)
    duration = times[-1] - times[0]
    values, average, envelope, quadrature = {}, {}, {}, {}
    for name, field in sample_fields(record, times).items():
        if not np.all(np.isfinite(field)):
            raise ValueError(f"Nonfinite saved {name} samples")
        values[name] = field
        average[name] = np.trapezoid(field, times, axis=0)/duration
        envelope[name] = dict(min=np.min(field, axis=0), max=np.max(field, axis=0))
        coarse_indices = np.unique(np.r_[np.arange(0, len(times), 2), len(times)-1])
        coarse = np.trapezoid(field[coarse_indices], times[coarse_indices], axis=0)/duration
        quadrature[name] = dict(RMS_change=_rms(average[name]-coarse),
                                maximum_absolute_change=float(max(abs(average[name]-coarse))),
                                outlet_change=float(average[name][-1]-coarse[-1]))
    boundary_mean = {}
    for name, field in record.get("boundary_fields", {}).items():
        field = np.asarray(field, dtype=float)
        if field.shape == (len(record["times"]),):
            boundary_mean[name] = float(np.trapezoid(np.interp(times, record["times"], field), times)/duration)
    return dict(times=times, periodicity=periodic, kind=kind, samples=values,
                mean=average, envelope=envelope, quadrature_sensitivity=quadrature,
                boundary_mean=boundary_mean)


def comparisons(data):
    """Compare verified ten-cycle means and phase-resolved periodic waveforms."""
    selected_id = data["selected_id"]
    selected = data["transient"][selected_id]
    config = selected.get("config", data["config"])
    baseline = data["steady"]["implicit"]["grids"][-1]
    periodicity = {name: periodic_assessment(record, record.get("config", data["config"]))
                   for name, record in data["transient"].items()}
    averages = {}
    for name, record in data["transient"].items():
        try: averages[name] = means(record, record.get("config", data["config"]))
        except ValueError:
            if name == selected_id: raise
    current = averages[selected_id]
    summary = {}
    for name, average in current["mean"].items():
        steady = np.asarray(baseline["fields"][name]); change = average-steady
        summary[name] = dict(steady_outlet=float(steady[-1]), mean_outlet=float(average[-1]),
            outlet_difference=float(change[-1]),
            outlet_percent_difference=float(100*change[-1]/steady[-1]) if abs(steady[-1]) > 1e-15 else None,
            spatial_RMS_difference=_rms(change), maximum_absolute_difference=float(max(abs(change))),
            averaging_quadrature_RMS_change=current["quadrature_sensitivity"][name]["RMS_change"])
    reference_id = data.get("comparison_reference_id", selected_id)
    temporal = dict(reference_id=reference_id, selected_id=selected_id, comparisons={},
        criterion="Compare independently verified ten-cycle means and 24 forcing-phase averages on the same mesh. Integer-cycle shifts between validated collection windows are permitted; unverified windows are compared only at identical absolute times.")
    ref = averages.get(reference_id)
    for name, average in averages.items():
        if name == reference_id or ref is None: continue
        record, reference = data["transient"][name], data["transient"][reference_id]
        if _physical_signature(record, data["config"]) != _physical_signature(reference, data["config"]):
            temporal["comparisons"][name] = dict(comparable=False, reason="Different physical forcing or load")
            continue
        if not np.array_equal(np.asarray(record["x"]), np.asarray(reference["x"])):
            raise ValueError("Temporal comparison requires the same mesh")
        frames = config["forcing"]["frames_per_cycle"]
        cycles = config["forcing"]["collection_cycles"]
        periodic = average["periodicity"]["verified"] and ref["periodicity"]["verified"]
        windows = {}
        if periodic:
            fields = {}
            for key in FIELD_NAMES:
                first = average["samples"][key][-cycles*frames:].reshape(cycles, frames, -1).mean(axis=0)
                second = ref["samples"][key][-cycles*frames:].reshape(cycles, frames, -1).mean(axis=0)
                delta = first-second; mean_delta = average["mean"][key]-ref["mean"][key]
                fields[key] = dict(phase_spatial_RMS_difference=_rms(delta),
                    outlet_phase_RMS_difference=_rms(delta[:, -1]),
                    maximum_absolute_difference=float(np.max(abs(delta))),
                    mean_RMS_difference=_rms(mean_delta), mean_outlet_difference=float(mean_delta[-1]))
            windows["verified_collection"] = dict(
                record_window_s=[float(average["times"][0]), float(average["times"][-1])],
                reference_window_s=[float(ref["times"][0]), float(ref["times"][-1])],
                periodic_state_claimed=True, cycles=cycles, phase_samples=frames, fields=fields)
        else:
            frequency = config["forcing"]["frequency_Hz"]
            end = int(np.floor(min(record["times"][valid_prefix(record, config)], reference["times"][valid_prefix(reference, config)])*frequency+1e-8))
            begin = end-config["forcing"]["collection_cycles"]
            if begin/frequency >= max(record["times"][0], reference["times"][0])-1e-12:
                times = np.arange(begin*frames, end*frames+1)/(frequency*frames)
                first, second = sample_fields(record, times), sample_fields(reference, times)
                fields = {}
                for key in FIELD_NAMES:
                    delta = first[key]-second[key]
                    mean_delta = np.trapezoid(delta, times, axis=0)/(times[-1]-times[0])
                    fields[key] = dict(time_spatial_RMS_difference=_rms(delta),
                        outlet_time_RMS_difference=_rms(delta[:, -1]), maximum_absolute_difference=float(np.max(abs(delta))),
                        mean_RMS_difference=_rms(mean_delta), mean_outlet_difference=float(mean_delta[-1]))
                windows["matched_transient"] = dict(window_start_s=float(times[0]), window_end_s=float(times[-1]),
                    periodic_state_claimed=False, samples=len(times), fields=fields)
        temporal["comparisons"][name] = dict(comparable=bool(windows), method=record["method"], cfl=record["cfl"],
            windows=windows, reason=None if windows else "No complete common-time window")
    boundary = current["boundary_mean"]
    matching = dict(steady_boundary=baseline.get("boundary_fields", {}), mean_window_boundary=boundary,
        interpretation="Both solutions use the same fixed reservoir state and constant outlet pressure. Cycle-averaged mass imbalance measures residual duct-inventory drift; a persistent mean-flow claim additionally requires verified periodicity and temporal/spatial resolution.")
    if "inlet_mdot" in boundary and "outlet_mdot" in boundary:
        matching["mean_inlet_minus_outlet_kg_s"] = boundary["inlet_mdot"]-boundary["outlet_mdot"]
    window_cycles = (current["times"][-1]-current["times"][0])*config["forcing"]["frequency_Hz"]
    verified = periodicity[selected_id]["verified"]
    comparison = dict(selected_id=selected_id, reference_id=reference_id, verified_periodic=verified,
        averaging_kind=current["kind"], x=selected["x"], steady=baseline["fields"], mean=current["mean"],
        envelope=current["envelope"], summary=summary,
        window_start_s=float(current["times"][0]), window_end_s=float(current["times"][-1]),
        complete_cycles=round(window_cycles), averaging_samples=dict(times=current["times"], fields=current["samples"]),
        quadrature=f"Trapezoidal integration across {len(current['times'])} phase outputs spanning {window_cycles:.6g} complete forcing periods. Nonlinear diagnostic fields are averaged directly. Window kind: {current['kind']}.",
        quadrature_sensitivity=current["quadrature_sensitivity"], matching_conditions=matching)
    source_id = data.get("startup_id", selected_id)
    source = data["transient"][source_id]
    frequency, frames = config["forcing"]["frequency_Hz"], config["forcing"]["frames_per_cycle"]
    end_cycle = int(round(current["times"][-1]*frequency))
    sensitivity = {}
    if not verified:
        for cycles in (20, 50):
            begin = (end_cycle-cycles)/frequency
            if begin < source["times"][0]-1e-12 or current["times"][-1] > source["times"][-1]+1e-12:
                continue
            times = np.arange((end_cycle-cycles)*frames, end_cycle*frames+1)/(frequency*frames)
            fields = sample_fields(source, times)
            checks = {}
            for key, values in fields.items():
                average = np.trapezoid(values, times, axis=0)/(times[-1]-times[0])
                difference = average-current["mean"][key]
                checks[key] = dict(mean_last_cell=float(average[-1]),
                    last_cell_difference_from_ten_cycle_mean=float(difference[-1]),
                    spatial_RMS_difference_from_ten_cycle_mean=_rms(difference))
            sensitivity[f"last_{cycles}_cycles"] = dict(source_id=source_id,
                window_start_s=begin, window_end_s=float(times[-1]), cycles=cycles, fields=checks)
    comparison["window_sensitivity"] = sensitivity
    return comparison, periodicity, temporal


def reflection_analysis(data):
    """Separate local acoustic pressure amplitudes in the constant-area tail."""
    record = data["transient"][data["selected_id"]]
    config = data["config"]
    times, periodic, window_kind = mean_window(record, config)
    ix = np.searchsorted(np.asarray(record["times"]), times-1e-13)
    if not np.allclose(np.asarray(record["times"])[ix], times, rtol=0, atol=1e-12):
        raise ValueError("Reflection diagnostics require exact stored phase outputs")
    constants = data["fluid_constants"]
    gamma, gas_constant = constants["gamma"], constants["R"]
    cp = gamma*gas_constant/(gamma-1)
    duration = times[-1]-times[0]
    omega = 2*np.pi*config["forcing"]["frequency_Hz"]
    tail_start = config["geometry"]["constant_length_m"]+config["geometry"]["expansion_length_m"]
    length = config["geometry"]["extension_length_m"]
    x = np.asarray(record["x"])
    probes = []; signals = dict(times=times)
    locations = [(f"tail_{number}", tail_start+fraction*length, False)
                 for number, fraction in enumerate((.25, .5, .75), 1)]
    locations.append(("outlet_trace", tail_start+length, True))
    for name, position, trace in locations:
        if trace:
            state = np.asarray(record["states"])[ix, -1]
            pressure, velocity = state[:, 0], state[:, 1]
            temperature = state[:, 3]/cp
            rho = pressure/(gas_constant*temperature)
            location = position
        else:
            cell = int(np.argmin(abs(x-position)))
            pressure = np.asarray(record["fields"]["p"])[ix, cell]
            velocity = np.asarray(record["fields"]["v_m"])[ix, cell]
            temperature = np.asarray(record["fields"]["T"])[ix, cell]
            rho = np.asarray(record["fields"]["rho"])[ix, cell]
            location = float(x[cell])
        impedance = float(np.trapezoid(rho*np.sqrt(gamma*gas_constant*temperature), times)/duration)
        pp = pressure-np.trapezoid(pressure, times)/duration
        vp = velocity-np.trapezoid(velocity, times)/duration
        outgoing, reflected = .5*(pp+impedance*vp), .5*(pp-impedance*vp)
        first = 2*np.trapezoid(outgoing*np.exp(-1j*omega*times), times)/duration
        second = 2*np.trapezoid(reflected*np.exp(-1j*omega*times), times)/duration
        ratio = None if abs(first) < 1e-10 else float(abs(second/first))
        phase = None if ratio is None else float(np.angle(second/first, deg=True))
        probes.append(dict(id=name, x_m=location, at_boundary=trace, mean_acoustic_impedance=impedance,
            outgoing_amplitude_Pa=float(abs(first)), reflected_amplitude_Pa=float(abs(second)),
            amplitude_ratio=ratio, phase_difference_deg=phase))
        signals[name] = dict(w_plus=outgoing, w_minus=reflected)
    return dict(available=True, periodic_verified=periodic["verified"], window_kind=window_kind,
        window_start_s=float(times[0]), window_end_s=float(times[-1]),
        definition="Local linear acoustic pressure components: w_plus=(p_prime+mean(rho*a)*vm_prime)/2 and w_minus=(p_prime-mean(rho*a)*vm_prime)/2 in the straight constant-area tail. Prime denotes the mean over the analyzed ten-cycle window removed. Complex amplitudes are finite-window forcing-frequency Fourier coefficients, not proof of an asymptotic periodic state. Interior phase includes propagation delay and slow modulation can affect these diagnostics; the fixed-pressure trace ideally gives pressure-amplitude ratio 1 and phase 180 degrees.",
        probes=probes, signals=signals)


def _regime(values):
    keys = ("normal_mach_min", "normal_mach_max", "meridional_velocity_ratio", "expansion_meridional_ratio")
    if any(key not in values for key in keys):
        return None
    return bool(values["normal_mach_min"] > 0 and values["normal_mach_max"] < 1
                and values["meridional_velocity_ratio"] < 1
                and values["expansion_meridional_ratio"] < 1)


def independent_audit(data):
    """Re-evaluate exact BCs/current storage and original steady residuals."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, sampler, ResidualPolicy
    from .diffuser_geometry import native_meshes
    from .diffuser_model import make_model

    counts = {np.asarray(r["states"]).shape[1]-2 for r in data.get("transient", {}).values() if r.get("states")}
    counts.update(r["cells"] for group in data.get("steady", {}).values() for r in group.get("grids", []))
    _, meshes, _ = native_meshes(data["config"], sorted(counts))
    if data["config"]["outlet"]["kind"] != "pressure":
        raise ValueError("The clean angle-forcing case requires a constant pressure outlet")
    audit = dict(steady=[], transient={}, scope="No new solves. Current exact boundary equations and storage are independently recomputed at every retained snapshot. Stored every-accepted-step diagnostics are reclassified independently; intermediate step states are not retained.")

    def scheme(config, method, stored=None):
        if stored:
            return Scheme(**stored)
        return Scheme(flux=config["numerics"]["flux"], reconstruction=config["numerics"]["reconstruction"],
                      time_integrator="euler-implicit" if method == "implicit" else "runge-kutta-3",
                      boundary_treatment="coupled" if method == "implicit" else "segregated")

    for name, group in data.get("steady", {}).items():
        for record in group.get("grids", []):
            state = jnp.asarray(record["state"])
            model = make_model(data["config"], meshes[record["cells"]], initial_state=state)
            operator = Operator(model, scheme(data["config"], "implicit", group.get("scheme")))
            history = record.get("history", [])
            reference = next((v["equation_norm"] for v in history if "equation_norm" in v), None)
            residual = sampler(operator, ResidualPolicy())(state, jnp.asarray(reference) if reference is not None else None)
            constraints = {k: float(v) for k, v in model.constraint_metrics(state).items()}
            passed = bool(residual["all_equations_passed"] and residual["boundary_passed"]
                          and residual["global_passed"] and residual["admissible"] and model.solution_audit(state))
            audit["steady"].append(dict(method=name, cells=record["cells"], passed=passed,
                original_residual=plain(jax.device_get(residual)), constraints=constraints))

    for name, record in data.get("transient", {}).items():
        if not record.get("states"):
            continue
        config = record.get("config", data["config"])
        states = jnp.asarray(record["states"])
        times = jnp.asarray(record["times"])
        native = meshes[states.shape[1]-2]
        model = make_model(config, native, driven=True, initial_state=states[0])
        operator = Operator(model, scheme(config, record["method"], record.get("scheme")))
        start = int(np.searchsorted(native.x_center, config["geometry"]["constant_length_m"])) + 1
        end_expansion = int(np.searchsorted(native.x_center, config["geometry"]["constant_length_m"]+config["geometry"]["expansion_length_m"], side="right"))

        def one(state, time):
            values = model.constraint_metrics(state)
            speed = jnp.sqrt(state[:, 1]**2 + (state[:, 2]/model._r_full)**2)
            values.update(expansion_meridional_ratio=state[end_expansion, 1]/state[start, 1],
                          expansion_speed_ratio=speed[end_expansion]/speed[start],
                          conserved_contents=jnp.sum(operator.storage(state[1:-1]), axis=0),
                          boundary_max=jnp.max(jnp.abs(model.boundary_residual(state, time))),
                          numerical_outlet_mass_flow=operator.face_flux(state)[-1, 0],
                          outlet_pressure=state[-1, 0],
                          target_outlet=model.target_outlet(time))
            return values

        arrays = {k: np.asarray(v) for k, v in jax.device_get(jax.jit(jax.vmap(one))(states, times)).items()}
        snapshot_rows = [dict(time=float(record["times"][i]), **{k: float(v[i]) for k, v in arrays.items()
                         if k != "conserved_contents"}) for i in range(len(times))]
        for row in snapshot_rows:
            row["regime_passed"] = _regime(row)
        current_initial = np.asarray(operator.storage(states[0, 1:-1])).sum(axis=0)
        current_final = np.asarray(operator.storage(states[-1, 1:-1])).sum(axis=0)
        initial, final, transfer, scales = [np.asarray(record[k], dtype=float) for k in
            ("initial_totals", "final_totals", "integrated_transfer", "content_scales")]
        budget = (final - initial - transfer)/scales
        recomputed_budget = (current_final - current_initial - transfer)/scales
        contents_agreement = float(np.max(abs((arrays["conserved_contents"]-np.asarray(record["contents"]))/scales)))
        history = record.get("history", [])
        classified = [(i, _regime(v)) for i, v in enumerate(history)]
        failed_steps = [dict(index=i, time=history[i].get("time")) for i, passed in classified if passed is False]
        bcmax = float(max(arrays["boundary_max"]))
        endpoint_agreement = max(float(max(abs((current_initial-initial)/scales))), float(max(abs((current_final-final)/scales))))
        conservation_passed = bool(max(abs(budget)) <= 1e-8 and max(abs(recomputed_budget)) <= 1e-8
                                   and endpoint_agreement <= 1e-10 and contents_agreement <= 1e-10)
        item = dict(success=record.get("success"), snapshots=snapshot_rows,
            snapshot_regimes_passed=all(v["regime_passed"] for v in snapshot_rows),
            every_accepted_step_saved_metrics_passed=not failed_steps and all(v is not None for _, v in classified),
            failed_accepted_steps=failed_steps, accepted_step_count=len(history),
            boundary_max=bcmax, boundary_passed=bcmax <= 1e-9,
            conservation=dict(normalized_defect=budget, current_storage_normalized_defect=recomputed_budget,
                maximum_absolute_defect=float(max(abs(budget))), current_endpoint_agreement=endpoint_agreement,
                maximum_saved_inventory_disagreement=contents_agreement,
                passed=conservation_passed, definition="(final totals − initial totals − integrated stage/step-weighted flux and source transfers) / fixed integrated conservative scales"))
        item["deceleration_from_saved_accepted_steps"] = dict(
            expansion_meridional_passed=all(v["expansion_meridional_ratio"] < 1 for v in history),
            inlet_to_outlet_meridional_passed=all(v["meridional_velocity_ratio"] < 1 for v in history),
            maximum_expansion_vm_ratio=max((v["expansion_meridional_ratio"] for v in history), default=None),
            maximum_total_speed_ratio=max((v.get("total_speed_ratio", 0) for v in history), default=None),
            maximum_expansion_total_speed_ratio=max((v.get("expansion_speed_ratio", 0) for v in history), default=None),
            interpretation="Meridional deceleration is classified independently of the Mach regime. Instantaneous total-speed ratios are diagnostics, not acceptance gates.")
        pressure_error = arrays["outlet_pressure"]-arrays["target_outlet"]
        item["outlet_pressure"] = dict(maximum_absolute_error_Pa=float(max(abs(pressure_error))),
            maximum_scaled_error=float(max(abs(pressure_error))/model.boundary_scales["p"]),
            passed=bool(max(abs(pressure_error))/model.boundary_scales["p"] <= 1e-9))
        item["numerical_checks_passed"] = bool(item["boundary_passed"] and conservation_passed
            and item["outlet_pressure"]["passed"])
        item["requested_meridional_regime_passed"] = bool(item["snapshot_regimes_passed"]
            and item["every_accepted_step_saved_metrics_passed"])
        item["passed"] = bool(item["numerical_checks_passed"] and item["requested_meridional_regime_passed"])
        audit["transient"][name] = item
        print(f"Saved-state audit {name}: BC={bcmax:.3g}, budget={max(abs(budget)):.3g}, pass={item['passed']}", flush=True)
    audit["selected_passed"] = audit["transient"].get(data["selected_id"], {}).get("passed", False)
    audit["source_fingerprints_at_audit"] = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), STUDY/"scripts/diffuser_model.py", STUDY/"scripts/diffuser_geometry.py",
                  ROOT/"src/nozzle_hyperbolic/models/euler_annular/model.py",
                  ROOT/"src/nozzle_hyperbolic/solvers/fvm/operator.py"] if p.exists()}
    return plain(audit)


def coordinates(geometry, cells):
    """Map native uniform-arclength faces through the saved normal offsets."""
    g = {key: np.asarray(geometry[key], dtype=float) for key in ("m", "x", "r", "b", "tip_x", "tip_r")}
    m = np.linspace(g["m"][0], g["m"][-1], cells + 1)
    x, radius, height, tx, tr = [np.interp(m, g["m"], g[key]) for key in ("x", "r", "b", "tip_x", "tip_r")]
    rp, xp = 2*(x-tx)/height, 2*(tr-radius)/height
    eta = np.linspace(-.5, .5, 9)[:, None]
    X, Y = x[None, :] - eta*height*rp, radius[None, :] + eta*height*xp
    alignment = np.min(abs(g["m"][None, :] - m[:, None]), axis=1)
    return X, Y, float(max(alignment))


def render_media(data, geometry, output_dir, digest):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from PIL import Image

    selected = data["transient"][data["selected_id"]]
    startup_id = data.get("startup_id", data["selected_id"])
    startup = data["transient"][startup_id]
    config = selected.get("config", data["config"])
    baseline = data["steady"]["implicit"]["grids"][-1]
    cells = len(baseline["x"])
    X, Y, alignment = coordinates(geometry, cells)
    panels = [("M", "Absolute Mach number", 1), ("M_m", "Meridional Mach number", 1),
              ("v_m", "Meridional velocity [m/s]", 1), ("p", "Static pressure [bar]", 1e-5)]

    def draw(ax, values, norm, label):
        mesh = ax.pcolormesh(X, Y, np.tile(values, (X.shape[0]-1, 1)), shading="flat",
                             cmap="plasma", norm=norm, rasterized=True)
        ax.plot(X[0], Y[0], color="#30253d", lw=1)
        ax.plot(X[-1], Y[-1], color="#30253d", lw=1)
        ax.set(xlabel="Axial coordinate x [m]", ylabel="Radius [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(X.min())-.002, float(X.max())+.002)
        ax.set_ylim(float(Y.min())-.004, float(Y.max())+.004)
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        return mesh

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.2), layout="constrained")
    for ax, (key, label, scale) in zip(axes, panels):
        values = np.asarray(baseline["fields"][key])*scale
        mesh = draw(ax, values, Normalize(float(min(values)), float(max(values))), label)
        fig.colorbar(mesh, ax=ax, pad=.015, shrink=.85, aspect=15)
    fig.suptitle("Steady annular transition duct · meridional diffuser branch", fontweight="bold")
    fig.savefig(output_dir/"steady_contours.png", dpi=180)
    fig.savefig(output_dir/"steady_contours.pdf")
    plt.close(fig)
    times = np.asarray(selected["times"])
    startup_times = np.asarray(startup["times"])
    frames_per_cycle = int(config["forcing"]["frames_per_cycle"])
    startup_end = valid_prefix(startup, startup.get("config", data["config"]))
    startup_indices = np.arange(min(3*frames_per_cycle+1, startup_end+1))
    periodic = data["periodicity"][data["selected_id"]]
    if periodic["verified"]:
        late_indices, _, _ = cycle_window(selected, config)
        collection_window = [float(times[late_indices[0]]), float(times[late_indices[-1]])]
        late_indices = late_indices[-(2*frames_per_cycle+1):]
        late_file, late_title = "periodic.gif", "Verified periodic response · final two of ten collected cycles"
    else:
        analysis_times, _, _ = mean_window(selected, config)
        collection_window = [float(analysis_times[0]), float(analysis_times[-1])]
        late_end = valid_prefix(selected, config)
        late_indices = np.arange(max(0, late_end-2*frames_per_cycle), late_end+1)
        late_file, late_title = "late_transient.gif", "Final two of ten analyzed cycles · not periodic"
    # Shared limits include the steady state and every frame of this trajectory.
    limits = {}
    for key, _, scale in panels:
        values = np.asarray(selected["fields"][key])[late_indices]*scale
        startup_values = np.asarray(startup["fields"][key])[startup_indices]*scale
        steady = np.asarray(baseline["fields"][key])*scale
        low = min(float(values.min()), float(startup_values.min()), float(steady.min()))
        high = max(float(values.max()), float(startup_values.max()), float(steady.max()))
        if high <= low:
            high = low + max(abs(low)*1e-6, 1e-9)
        limits[key] = [low, high]

    def gif(record, indices, filename, title):
        fig, axes = plt.subplots(4, 1, figsize=(10.5, 7.5), layout="constrained")
        meshes = []
        arrays = {key: np.asarray(record["fields"][key]) for key, _, _ in panels}
        frame_times = np.asarray(record["times"])
        for ax, (key, label, scale) in zip(axes, panels):
            mesh = draw(ax, arrays[key][indices[0]]*scale, Normalize(*limits[key]), label)
            tick_format = "%.2f" if key in ("M", "M_m") else "%.1f" if key == "p" else "%.0f"
            fig.colorbar(mesh, ax=ax, pad=.015, shrink=.85, aspect=15,
                         ticks=np.linspace(*limits[key], 5), format=tick_format)
            meshes.append(mesh)
        heading = fig.suptitle(title, fontsize=12, fontweight="bold")
        heading.set_text(title + f"  |  t = {frame_times[indices[0]]*1e3:.4f} ms")
        fig.canvas.draw()
        fig.set_layout_engine("none")
        frames = []
        for index in indices:
            for mesh, (key, _, scale) in zip(meshes, panels):
                mesh.set_array(np.tile(arrays[key][index]*scale, (X.shape[0]-1, 1)).ravel())
            heading.set_text(title + f"  |  t = {frame_times[index]*1e3:.4f} ms")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        frames[0].save(output_dir/filename, save_all=True, append_images=frames[1:],
                       duration=50, loop=0, optimize=False, disposal=2)
        plt.close(fig)
        print(f"Rendered {filename}: {len(frames)} frames", flush=True)

    gif(startup, startup_indices, "startup.gif", "Startup from steady state · first three cycles")
    gif(selected, late_indices, late_file, late_title)
    return dict(results_sha256=digest, selected_id=data["selected_id"],
        startup_id=startup_id, periodic_verified=periodic["verified"], steady_file="steady_contours.png",
        steady_pdf_file="steady_contours.pdf", startup_file="startup.gif", late_file=late_file,
        startup_window_s=[float(startup_times[startup_indices[0]]), float(startup_times[startup_indices[-1]])],
        late_window_s=[float(times[late_indices[0]]), float(times[late_indices[-1]])],
        collection_window_s=collection_window,
        startup_frames=len(startup_indices), late_frames=len(late_indices), frames_per_cycle=frames_per_cycle,
        playback_fps=20, gif_frame_duration_ms=50, shared_plasma_limits=limits,
        maximum_face_to_dense_sample_distance_m=alignment,
        description="Four panels show M, meridional Mach, meridional velocity and pressure. Each quasi-1D cell value is uniform across the physical normal duct height. Both GIFs use identical plasma limits; physical time is shown and playback is slowed. Startup loops deliberately restart at the initial steady state.")


def process(results_path=OUT/"results.json", *, generate_report=True, render=True, audit=True):
    """Update saved analyses/media, then build the single offline report."""
    results_path = Path(results_path)
    output_dir = results_path.parent
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not data.get("selected_id"):
        raise ValueError("A selected_id is required before postprocessing")
    comparison, periodicity, temporal = comparisons(data)
    data["comparison"] = plain(comparison)
    data["periodicity"] = plain(periodicity)
    data["temporal_comparison"] = plain(temporal)
    data["reflections"] = plain(reflection_analysis(data))
    data["postprocessing"] = dict(module="diffuser_postprocess", no_numerical_solves=True,
        averaging=comparison.get("averaging_kind", "unavailable"),
        averaging_window_s=[comparison.get("window_start_s"), comparison.get("window_end_s")],
        diagnostic_fields_averaged_directly=list(FIELD_NAMES))
    audited = independent_audit(data) if audit else None
    save(results_path, data)
    digest = sha256(results_path.read_bytes()).hexdigest()
    if audited is not None:
        audited["results_sha256"] = digest
        save(output_dir/"audit.json", audited)
    if render:
        geometry = json.loads((output_dir/"geometry.json").read_text(encoding="utf-8"))
        manifest = render_media(data, geometry, output_dir, digest)
        save(output_dir/"visuals.json", manifest)
    if generate_report:
        from .diffuser_report import build
        build(output_dir)
    return dict(results_sha256=digest, selected_id=data["selected_id"],
                periodic_verified=periodicity[data["selected_id"]]["verified"],
                audit_selected_passed=audited["selected_passed"] if audited else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT/"results.json")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()
    print(process(args.results, generate_report=not args.no_report, render=not args.no_media,
                  audit=not args.no_audit), flush=True)


if __name__ == "__main__":
    main()
