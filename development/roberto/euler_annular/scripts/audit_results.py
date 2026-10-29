"""Re-evaluate saved annular fields without advancing a Newton/timestep solve.

Run from the repository root:
  poetry run python -m development.roberto.euler_annular.scripts.audit_results

Audit integrity, original root convergence and analytical accuracy are separate
outcomes. The command fails for an incomplete/inconsistent audit or a recorded
successful root that no longer satisfies the original equations. Analytical
failures remain explicit in audits.json and verification_acceptance; this audit
does not waive them or replace the verification driver's acceptance policy.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import inspect
import json
from pathlib import Path
import re
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import nurbspy

from nozzle_hyperbolic.models.euler_annular import make_annular_solver
from nozzle_hyperbolic.models.euler_annular.state import primitive_to_conservative
from .verify_annular import (
    CASES, RECONSTRUCTIONS, REPOSITORY, STUDY, fingerprints, make_case,
    reference_fields, write_json,
)


COMPONENTS = ("mass", "meridional_momentum", "angular_momentum", "total_energy")
COMPONENT_UNITS = ("kg/s", "N", "N m", "W")
REPRODUCTION_TOLERANCE = 1e-10
CASE_FILE = re.compile(r"^(.+?)_(?:N)?([1-9][0-9]*)\.json$")
AUDIT_DEFINITION = "annular_saved_state_cp_mean_equation_monitoring_v2"


def _comparison(actual, saved, scale=1.0):
    """Compare arrays using explicit fixed scales, preserving all differences."""
    actual, saved = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
    if actual.shape != saved.shape:
        return dict(matches=False, current_shape=list(actual.shape), saved_shape=list(saved.shape))
    finite = bool(np.isfinite(actual).all() and np.isfinite(saved).all())
    difference = np.abs(actual-saved)
    normalized = difference/np.asarray(scale)
    return dict(
        matches=finite and bool(np.all(normalized <= REPRODUCTION_TOLERANCE)),
        bitwise_equal=bool(np.array_equal(actual, saved)), finite=finite,
        maximum_absolute_difference=float(np.max(difference)) if difference.size else 0.0,
        maximum_scaled_difference=float(np.max(normalized)) if normalized.size else 0.0,
        tolerance=REPRODUCTION_TOLERANCE,
    )


def _source_status(saved, current):
    common = set(saved) & set(current)
    changed = sorted(name for name in common if saved[name] != current[name])
    missing = sorted(set(saved)-set(current))
    return dict(
        matches_saved_snapshot=bool(saved) and not changed and not missing,
        saved_fingerprint_available=bool(saved),
        changed=changed, missing_from_current_fingerprint=missing,
        newly_fingerprinted=sorted(set(current)-set(saved)),
        saved=saved, current=current,
    )


def _backend_identity(fluid):
    cls = type(fluid)
    return dict(
        class_name=f"{cls.__module__}.{cls.__name__}",
        class_source=inspect.getfile(cls),
        fluid_name=getattr(fluid, "fluid_name", getattr(fluid, "name", None)),
        jax_backend=jax.default_backend(), jax_enable_x64=bool(jax.config.jax_enable_x64),
        jaxprop_import_path=str(Path(jxp.__file__).resolve()),
        nurbspy_import_path=str(Path(nurbspy.__file__).resolve()),
    )


def _named(values):
    return dict(zip(COMPONENTS, np.asarray(values).tolist()))


def _current_hashes():
    current = fingerprints()
    adapter = REPOSITORY/"src/nozzle_hyperbolic/thermodynamics/adapters.py"
    if adapter.is_file():
        current[str(adapter.relative_to(REPOSITORY))] = sha256(adapter.read_bytes()).hexdigest()
    return current


def audit_equation_monitoring(solver, run, physical, scaled, final_metrics):
    """Independent NumPy norms, initial reductions and engineering monitors.

    This does not call the production reducer, policy assessment, monitor
    tracker or annular adapter. Initial rows are independently reconstructed
    from actual face flux minus source, just like the final spatial audit.
    """
    metadata = run.get("monitoring")
    if not metadata:
        return None
    if metadata.get("definition") != "steady_spatial_fixed_scale_equation_v1":
        raise ValueError("Unsupported saved residual definition")
    policy = metadata["policy"]
    for key in ("relative_tolerance", "absolute_tolerance", "boundary_tolerance", "global_tolerance"):
        if not np.isfinite(policy[key]) or policy[key] <= 0:
            raise ValueError("Invalid saved residual tolerance")
    if policy["norm"] not in ("l1", "l2") or policy["weighting"] not in ("volume", "cell"):
        raise ValueError("Unsupported saved residual norm/weighting")
    volumes = np.asarray(solver.mesh.volumes)
    weights = (volumes/volumes.sum() if policy["weighting"] == "volume"
               else np.full(len(volumes), 1/len(volumes)))

    def norms(rows):
        return dict(l1=np.sum(weights[:, None]*abs(rows), axis=0),
                    l2=np.sqrt(np.sum(weights[:, None]*rows**2, axis=0)),
                    linf=np.max(abs(rows), axis=0), worst_cell=np.argmax(abs(rows), axis=0))

    initial = jnp.asarray(run["initial_state"])
    if initial.shape != solver.shape or not bool(solver.is_admissible(initial)):
        raise ValueError("Saved monitored initial state is invalid")
    initial_flux = np.asarray(solver.face_flux(initial))
    initial_source = np.asarray(solver.source(initial))
    initial_rows = initial_flux[1:]-initial_flux[:-1]-initial_source
    # Norm the public signed spatial rows, independently in NumPy. Keep its
    # canonical execution boundary: separately evaluating flux/source terms can
    # rearrange cancellation at roundoff near a nine-order reduction target.
    # The direct assembly remains a separate, fixed-scale identity audit.
    canonical_initial = np.asarray(solver.spatial_residual(initial))
    canonical_final = np.asarray(solver.spatial_residual(jnp.asarray(run["state"])))
    initial_norms = norms(canonical_initial/solver.spatial_flux_scales)
    current_norms = norms(canonical_final/solver.spatial_flux_scales)
    initial_selected = initial_norms[policy["norm"]]
    current_selected = current_norms[policy["norm"]]
    active = initial_selected > policy["initial_floor"]
    relative = np.divide(current_selected, initial_selected, out=np.zeros_like(current_selected),
                         where=initial_selected > 0)
    equation_ratios = np.maximum(current_selected/policy["absolute_tolerance"],
                                np.where(active, relative/policy["relative_tolerance"], 0.0))
    finite = bool(np.isfinite(scaled).all() and np.isfinite(initial_rows).all()
                  and np.isfinite(current_selected).all() and np.isfinite(initial_selected).all())
    equation_passed = (equation_ratios <= 1) & finite
    expected = {"equation_"+key: value for key, value in current_norms.items()}
    expected.update(equation_norm=current_selected, equation_relative=relative,
                    relative_active=active, equation_passed=equation_passed,
                    equation_max_ratio=float(np.max(equation_ratios)),
                    all_equations_passed=bool(equation_passed.all()),
                    boundary_passed=final_metrics["boundary_max"] <= policy["boundary_tolerance"],
                    global_passed=final_metrics["conservation_max"] <= policy["global_tolerance"],
                    boundary_ratio=final_metrics["boundary_max"]/policy["boundary_tolerance"],
                    global_ratio=final_metrics["conservation_max"]/policy["global_tolerance"])
    expected["overall_max_ratio"] = max(expected[key] for key in
                                        ("equation_max_ratio", "boundary_ratio", "global_ratio"))
    checks = {key: _comparison(actual, run["metrics"].get(key, [])) for key, actual in expected.items()}
    checks["initial_flux_source_identity"] = _comparison(canonical_initial, initial_rows, solver.spatial_flux_scales)
    checks["final_flux_source_identity"] = _comparison(canonical_final, physical, solver.spatial_flux_scales)
    # Comparing an already tolerance-divided number at a fixed absolute 1e-10
    # would amplify harmless raw arithmetic differences by 1/tolerance. Preserve
    # the original fixed-scale reproduction threshold for these two ratios.
    checks["boundary_ratio"] = _comparison(expected["boundary_ratio"], run["metrics"]["boundary_ratio"],
                                            1/policy["boundary_tolerance"])
    checks["global_ratio"] = _comparison(expected["global_ratio"], run["metrics"]["global_ratio"],
                                          1/policy["global_tolerance"])
    checks["initial_equation_norms"] = _comparison(initial_selected, metadata["initial_equation_norms"])
    checks["conservative_scales"] = _comparison(solver.conservative_scales, metadata["conservative_scales"])
    checks["spatial_flux_scales"] = _comparison(solver.spatial_flux_scales, metadata["spatial_flux_scales"])
    checks["global_flux_scales"] = _comparison(solver.global_flux_scales, metadata["global_flux_scales"])
    checks["volume_weights"] = _comparison(volumes, metadata["volumes"], volumes)
    history = run["history"]
    accepted = [row for row in history if row.get("accepted", True)]
    if not accepted or accepted[0].get("phase") != "initial" or accepted[0]["step"] != 0:
        raise ValueError("Monitored history needs its actual initial sample at update zero")
    checks["initial_history_norm"] = _comparison(initial_selected, accepted[0]["equation_norm"])
    checks["final_history_norm"] = _comparison(current_selected, accepted[-1]["equation_norm"])
    updates = [row for row in accepted if row.get("phase") != "initial"]
    bookkeeping = bool(len(updates) == metadata["accepted_updates"]
                       and len(history)-1 == metadata["attempts"]
                       and len(history)-len(accepted) == metadata["rejected_attempts"]
                       and [row["step"] for row in accepted] == list(range(len(accepted))))
    # A rejected candidate without an outer evaluation must not claim copied
    # accepted-state norms or engineering observations.
    bookkeeping = bookkeeping and all(
        "equation_norm" not in row and "monitors" not in row
        for row in history if not row.get("accepted", True))
    monitor_checks, monitor_expected = {}, {}
    state = np.asarray(run["state"])
    final_flux = np.asarray(solver.face_flux(jnp.asarray(state)))
    widths = np.diff(solver.mesh.m_faces)
    for spec in metadata["monitors"]:
        name, quantity, reduction = spec["name"], spec["resolved_quantity"], spec["reduction"]
        if quantity == "pressure_coefficient":
            denominator = spec["resolved_pressure_total"]-state[0, 0]
            if denominator <= 0:
                raise ValueError("Invalid saved Cp monitor normalization")
            cp = (state[1:-1, 0]-state[0, 0])/denominator
            if reduction == "inlet":
                value = 0.0
            elif reduction == "outlet":
                value = (state[-1, 0]-state[0, 0])/denominator
            else:
                value = float(np.sum(widths*cp))
                if reduction == "length_mean":
                    value /= widths.sum()
        elif quantity == "mass_flow":
            value = (final_flux[0, 0] if reduction == "inlet" else final_flux[-1, 0]
                     if reduction == "outlet" else
                     np.sum(widths*(final_flux[:-1, 0]+final_flux[1:, 0])/2)/widths.sum())
        else:
            raise ValueError("Unknown saved monitor quantity")
        window = [row["monitors"][name]["value"] for row in accepted[-spec["window"]:]]
        span = max(window)-min(window)
        threshold = spec["absolute_tolerance"]+spec["relative_tolerance"]*spec["reference"]
        stable = bool(len(window) == spec["window"] and span <= threshold)
        expected_monitor = dict(value=value, range=span, threshold=threshold, stable=stable,
                                samples=len(window), required=spec["convergence"])
        monitor_expected[name] = expected_monitor
        monitor_checks[name] = {key: _comparison(actual, run["metrics"]["monitors"][name][key])
                                for key, actual in expected_monitor.items()}
        monitor_checks[name]["final_history_value"] = _comparison(value, window[-1])
    monitors_passed = all(not row["required"] or row["stable"] for row in monitor_expected.values())
    checks["monitors_passed"] = _comparison(monitors_passed, run["metrics"]["monitors_passed"])
    root = bool(finite and equation_passed.all() and expected["boundary_passed"]
                and expected["global_passed"] and monitors_passed)
    reproduced = bool(bookkeeping and all(check["matches"] for check in checks.values())
                      and all(check["matches"] for group in monitor_checks.values() for check in group.values()))
    return dict(root_target_met=root, reproduced=reproduced,
                method="independent NumPy; actual numerical face flux minus integrated source",
                initial_equation_norms=initial_selected, final_equation_norms=current_selected,
                relative_active=active, equation_relative=relative, equation_passed=equation_passed,
                policy=policy, checks=checks, monitor_checks=monitor_checks,
                expected_monitors=monitor_expected, history_bookkeeping=bookkeeping,
                accepted_updates=len(updates), attempts=len(history)-1)


def audit_case(path, tolerance, current_hashes, current_versions):
    """Audit one saved case with current operators and the original saved state."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    name, cells = saved["case"], int(saved["cells"])
    if name not in CASES or cells < 2:
        raise ValueError("Unsupported case name or physical cell count")
    if set(saved["methods"]) != set(RECONSTRUCTIONS):
        raise ValueError("A saved comparison must retain both requested reconstructions")
    # verify_annular's declared verification backend; never infer a real-fluid
    # EOS from gamma or silently replace a file declaring another backend.
    declaration = saved.get("fluid", "jaxprop.FluidPerfectGas(Air)")
    if declaration != "jaxprop.FluidPerfectGas(Air)":
        raise ValueError(f"This saved-case schema does not support backend {declaration!r}")
    fluid = jxp.FluidPerfectGas(name="Air")
    case = make_case(name, fluid)
    mesh = case.geometry.mesh(m_faces=np.asarray(saved["m_faces"], dtype=float))
    if mesh.n_cells != cells:
        raise ValueError("Saved face coordinates and physical cell count disagree")
    shock_m = saved.get("shock_m", case.shock_m)
    inlet = case.reference.profile(0.0)
    outlet = case.reference.profile(case.geometry.length, shock_m=shock_m)
    pressure0, reference_pressure_in = float(inlet["p0"]), float(inlet["p"])
    inlet_regime = saved.get("inlet_regime", case.inlet_regime)
    outlet_regime = saved.get("outlet_regime", case.outlet_regime)
    angle = float(saved.get("alpha_in", case.alpha))
    backpressure = float(saved["backpressure"])
    inlet_data = (dict(state=np.asarray(inlet["W"])) if inlet_regime == "supersonic"
                  else dict(p0=pressure0, T0=case.reference.T0, alpha_in=angle))
    projected = case.reference.cell_average(mesh, shock_m=shock_m, return_conserved=True)
    analytical = reference_fields(case, mesh, projected["W"])
    analytical["Cp"] = ((analytical["p"]-reference_pressure_in)
                        /(pressure0-reference_pressure_in))
    mesh_checks = {
        key: _comparison(actual, saved[key], scale)
        for key, actual, scale in (
            ("m", mesh.m_centers, case.geometry.length),
            ("m_faces", mesh.m_faces, case.geometry.length),
            ("r", mesh.r_center, mesh.r_face[0]),
            ("volumes", mesh.volumes, np.asarray(mesh.volumes)),
        )
    }
    metadata_checks = dict(
        backpressure=_comparison(float(outlet["p"]), backpressure, pressure0),
        alpha_in=_comparison(case.alpha, angle),
        inlet_regime=inlet_regime == case.inlet_regime,
        outlet_regime=outlet_regime == case.outlet_regime,
        shock_m=(shock_m == case.shock_m),
    )
    result = dict(
        audit_definition=AUDIT_DEFINITION,
        input_file=path.name, input_sha256=sha256(path.read_bytes()).hexdigest(),
        case=name, cells=cells, root_tolerance=tolerance,
        reproduction_tolerance=REPRODUCTION_TOLERANCE,
        no_newton_or_timestep_updates=True,
        saved_backend_declaration=declaration,
        backend_declaration_origin=("saved metadata" if "fluid" in saved
                                    else "verify_annular schema: FluidPerfectGas(Air)"),
        requested_backend=_backend_identity(fluid),
        versions=dict(saved=saved.get("versions", {}), current=current_versions,
                      matching={key: current_versions.get(key) == value
                                for key, value in saved.get("versions", {}).items()}),
        sources=_source_status(saved.get("source_hashes", {}), current_hashes),
        mesh_checks=mesh_checks, case_metadata_checks=metadata_checks,
        methods={},
    )
    for reconstruction in RECONSTRUCTIONS:
        run = saved["methods"][reconstruction]
        options = dict(reconstruction=reconstruction)
        # These are the spatial choices actually recorded by the writer. Other
        # missing settings remain visibly current defaults in effective_options.
        for key in ("delta_factor", "flux_method"):
            if key in saved.get("settings", {}):
                options[key] = saved["settings"][key]
        options.update(run.get("options", {}))
        solver = make_annular_solver(
            fluid, mesh, inlet_data=inlet_data, outlet_pressure=backpressure,
            inlet_regime=inlet_regime, outlet_regime=outlet_regime,
            tau_w=saved.get("case_config", {}).get("walls", {}).get("shear_stress", 0.0),
            q_w=saved.get("case_config", {}).get("walls", {}).get("heat_flux", 0.0),
            **options,
        )
        state = np.asarray(run["state"], dtype=float)
        if state.shape != (cells+2, 4) or not np.isfinite(state).all():
            raise ValueError(f"Invalid full saved state for {reconstruction}")
        full = jnp.asarray(state)
        if not bool(solver.is_admissible(full)):
            result["methods"][reconstruction] = dict(
                recorded_success=bool(run["success"]), admissible=False,
                root_target_met=False, accuracy_target_met=False,
                saved_arrays_reproduced=False,
                error="Saved state or reconstructed face states are inadmissible; no unsafe EOS audit performed",
            )
            continue
        fields = {key: np.asarray(value) for key, value in solver.fields(full).items()}
        pressure_in = float(state[0, 0])
        cp_denominator = pressure0-pressure_in
        if not np.isfinite(cp_denominator) or cp_denominator <= 0:
            raise ValueError("Saved state has no positive stagnation/static Cp normalization")
        fields["Cp"] = (fields["p"]-pressure_in)/cp_denominator
        face_flux = np.asarray(solver.face_flux(full))
        source = np.asarray(solver.source(full))
        # Independently assemble the original physical rows from the actual
        # final numerical face flux and source, without evaluating a timestep.
        physical = face_flux[1:]-face_flux[:-1]-source
        factory_physical = np.asarray(solver.spatial_residual(full))
        bc = np.asarray(solver.boundary_residual(full))
        rho_ref, a_ref = solver.reference_density, solver.reference_sound_speed
        q_scale = rho_ref*np.asarray([1., a_ref, mesh.r_face[0]*a_ref, a_ref**2])
        flux_scale = np.asarray(mesh.volumes)[:, None]*q_scale*a_ref/case.geometry.length
        balance_scale = float(mesh.A_face[0])*a_ref*q_scale
        scaled = physical/flux_scale
        balance = face_flux[-1]-face_flux[0]-source.sum(axis=0)
        metrics = dict(
            spatial_rms=float(np.sqrt(np.mean(scaled**2))),
            spatial_max=float(np.max(np.abs(scaled))),
            spatial_component_rms=_named(np.sqrt(np.mean(scaled**2, axis=0))),
            spatial_component_max=_named(np.max(np.abs(scaled), axis=0)),
            boundary_rms=float(np.sqrt(np.mean(bc**2))),
            boundary_max=float(np.max(np.abs(bc))),
            conservation_max=float(np.max(np.abs(balance/balance_scale))),
            admissible=True,
        )
        boundary_mach = np.asarray(solver.metrics(full)["boundary_mach"])
        metrics["boundary_mach"] = boundary_mach
        regime_tolerance = 2e-3  # The factory's declared boundary-regime audit.
        regime_checks = {
            key: bool(mach >= 0 and (mach <= 1+regime_tolerance if regime == "subsonic"
                                    else mach >= 1-regime_tolerance))
            for key, mach, regime in zip(("inlet", "outlet"), boundary_mach,
                                        (inlet_regime, outlet_regime))
        }
        root_pass = (max(metrics[key] for key in
                         ("spatial_rms", "boundary_max", "conservation_max")) < tolerance
                     and all(regime_checks.values()))
        monitoring_audit = audit_equation_monitoring(solver, run, physical, scaled, metrics)
        if monitoring_audit is not None:
            root_pass = monitoring_audit["root_target_met"] and all(regime_checks.values())
        angular_norm = (abs(case.reference.mdot*case.reference.ell)
                        if case.reference.ell != 0
                        else abs(case.reference.mdot*solver.scales["ell"]))
        energy_norm = max(abs(case.reference.mdot*case.reference.h0),
                          abs(case.reference.mdot*a_ref**2))
        transport_norm = np.asarray([abs(case.reference.mdot), angular_norm, energy_norm])
        if not np.isfinite(transport_norm).all() or np.any(transport_norm <= 0):
            raise ValueError("Transport spreads require explicit positive reference-flux scales")
        conservative = np.asarray(primitive_to_conservative(
            full[1:-1], mesh.r_center, solver.fluid))
        errors = dict(
            Cp_max=float(np.max(np.abs(fields["Cp"]-analytical["Cp"]))),
            Cp_mean_absolute=float(np.sum(np.diff(mesh.m_faces)*np.abs(fields["Cp"]-analytical["Cp"]))/np.sum(np.diff(mesh.m_faces))),
            conservative_L1=(np.sum(mesh.volumes[:, None]*np.abs(conservative-projected["q"]), axis=0)
                              /(np.sum(mesh.volumes)*q_scale)).tolist(),
            p_L1=float(np.sum(mesh.volumes*np.abs(fields["p"]-analytical["p"]))
                       /np.sum(mesh.volumes)/pressure0),
            mass_flow_relative=float(abs(np.mean(face_flux[:, 0])-case.reference.mdot)
                                     /abs(case.reference.mdot)),
        )
        if shock_m is None:
            accuracy_pass = errors["Cp_mean_absolute"] <= 0.01 and max(errors["conservative_L1"]) < 1e-3
        else:
            index = int(np.argmin(np.diff(fields["v_m"])/np.diff(mesh.m_centers)))
            estimate = float(np.mean(mesh.m_centers[index:index+2]))
            errors.update(shock_m=estimate,
                          shock_error_cells=abs(estimate-shock_m)/float(np.max(mesh.widths)))
            accuracy_pass = errors["shock_error_cells"] <= 2
        field_scales = dict(p=pressure0, rho=rho_ref, a=a_ref, v_m=a_ref, v_theta=a_ref,
                            ell=solver.scales["ell"], h=a_ref**2, e=a_ref**2,
                            E=a_ref**2, h0=a_ref**2, T=case.reference.T0,
                            s=solver.scales["s"], G=1., M=1., M_m=1., M_theta=1., Cp=1.)
        comparisons = {
            key: _comparison(actual, run[key], scale)
            for key, actual, scale in (
                ("face_flux", face_flux, balance_scale),
                ("integrated_source", source, flux_scale),
                ("boundary_constraints", bc, 1.),
                ("physical_spatial_residual", physical, flux_scale),
            ) if key in run
        }
        comparisons["factory_spatial_from_face_flux"] = _comparison(
            factory_physical, physical, flux_scale)
        field_checks = {key: _comparison(fields[key], value, field_scales.get(key, 1.))
                        for key, value in run.get("fields", {}).items() if key in fields}
        reference_checks = dict(
            W=_comparison(projected["W"], saved["reference_W"],
                          [pressure0, a_ref, solver.scales["ell"], a_ref**2]),
            q=_comparison(projected["q"], saved["reference_q"], q_scale),
        )
        reference_checks.update({key: _comparison(analytical[key], value, field_scales.get(key, 1.))
                                 for key, value in saved.get("reference", {}).items()
                                 if key in analytical})
        metric_checks = {key: _comparison(metrics[key], value)
                         for key, value in run.get("metrics", {}).items() if key in metrics}
        reproduced = all(check["matches"] for group in
                         (comparisons, field_checks, reference_checks, metric_checks)
                         for check in group.values())
        if monitoring_audit is not None:
            reproduced = reproduced and monitoring_audit["reproduced"]
        result["methods"][reconstruction] = dict(
            recorded_success=bool(run["success"]), recorded_message=run.get("message"),
            equation_monitoring_audit=monitoring_audit,
            root_target_met=bool(root_pass), accuracy_target_met=bool(accuracy_pass),
            recorded_accuracy_target_met=run.get("analytical_target_met"),
            admissible=True, saved_arrays_reproduced=reproduced,
            effective_options=asdict(solver.options), actual_backend=_backend_identity(solver.fluid),
            native_fluid_identity_preserved=solver.fluid is fluid,
            original_saved_state_unchanged=bool(np.array_equal(state, np.asarray(run["state"]))),
            recomputed_metrics=metrics, boundary_mach=boundary_mach,
            boundary_regime_checks=regime_checks, boundary_regime_tolerance=regime_tolerance,
            comparisons=comparisons, saved_field_comparisons=field_checks,
            reference_comparisons=reference_checks, saved_metric_comparisons=metric_checks,
            errors=errors,
            analytical_target_definition=(dict(Cp_mean_absolute=0.01, Cp_comparison="<=",
                                               Cp_weighting="physical meridional cell width / domain length",
                                               conservative_L1_max=1e-3)
                                          if shock_m is None else dict(shock_error_cells=2.)),
            fixed_scales=dict(conservative=q_scale, spatial_flux=flux_scale,
                              global_balance=balance_scale),
            integrated_balance=dict(components=COMPONENTS, units=COMPONENT_UNITS,
                                    raw=balance, scales=balance_scale, scaled=balance/balance_scale),
            transported_flux_spread=dict(
                components=("mass", "angular_momentum", "total_energy"),
                raw=np.ptp(face_flux[:, [0, 2, 3]], axis=0), scales=transport_norm,
                scaled=np.ptp(face_flux[:, [0, 2, 3]], axis=0)/transport_norm,
                mass_scale_definition="abs(reference mass flow)",
                angular_scale_definition="abs(reference mdot*ell), or abs(reference mdot*positive ell_ref) at zero swirl",
                energy_scale_definition="max(abs(reference mdot*h0), abs(reference mdot*a_ref**2))",
            ),
            Cp_normalization=dict(reservoir_pressure=pressure0,
                                  numerical_inlet_pressure=pressure_in,
                                  denominator=cp_denominator,
                                  reference_inlet_pressure=reference_pressure_in),
            arrays=dict(state=state, fields=fields, conservative=conservative,
                        face_flux=face_flux, integrated_source=source,
                        physical_spatial_residual=physical, scaled_spatial_residual=scaled,
                        boundary_constraints=bc),
            boundary_row_names=dict(
                inlet=(("p", "v_m", "ell", "h") if inlet_regime == "supersonic"
                       else ("total_enthalpy", "entropy", "fixed_angle", "outgoing_acoustic_minus")),
                outlet=(("p", "v_m", "ell", "h") if outlet_regime == "supersonic"
                        else ("pressure", "outgoing_acoustic_plus", "entropy", "ell")),
            ),
        )
        print(f"AUDIT {name} N={cells} {reconstruction}: root={root_pass} "
              f"accuracy={accuracy_pass} reproduced={reproduced} Cp_mean_error={errors['Cp_mean_absolute']:.6g}", flush=True)
    geometry_ok = all(check["matches"] for check in mesh_checks.values())
    metadata_ok = all(value["matches"] if isinstance(value, dict) else value
                      for value in metadata_checks.values())
    result["audit_pass"] = bool(
        geometry_ok and metadata_ok
        and all(run["saved_arrays_reproduced"] for run in result["methods"].values())
        and all(not run["recorded_success"] or run["root_target_met"]
                for run in result["methods"].values())
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs"/"verification")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="Original spatial RMS, BC maximum and balance threshold")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse only audits with identical input/source hashes and tolerances")
    args = parser.parse_args()
    if not args.output.is_dir() or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("Use an existing output directory and a finite positive tolerance")
    files = [path for path in sorted(args.output.glob("*.json"))
             if CASE_FILE.fullmatch(path.name)]
    if not files:
        parser.error("No case_<cells>.json or case_N<cells>.json files found")
    current_hashes = _current_hashes()
    versions = {name: metadata.version(name) for name in ("jax", "jaxprop", "nurbspy", "numpy", "scipy")}
    summaries, failed = [], []
    for path in files:
        try:
            destination = path.with_suffix(".audit.json")
            cached = json.loads(destination.read_text(encoding="utf-8")) if args.resume and destination.is_file() else {}
            matches = (cached.get("input_sha256") == sha256(path.read_bytes()).hexdigest()
                       and cached.get("audit_definition") == AUDIT_DEFINITION
                       and cached.get("sources", {}).get("current") == current_hashes
                       and cached.get("root_tolerance") == args.tolerance
                       and cached.get("reproduction_tolerance") == REPRODUCTION_TOLERANCE
                       and cached.get("versions", {}).get("current") == versions
                       and cached.get("audit_pass") is True)
            result = cached if matches else audit_case(path, args.tolerance, current_hashes, versions)
            write_json(destination, result)
            summaries.append(dict(
                file=path.name, audit_file=destination.name, case=result["case"], cells=result["cells"],
                audit_pass=result["audit_pass"], source_matches=result["sources"]["matches_saved_snapshot"],
                methods={key: {name: run[name] for name in
                               ("recorded_success", "root_target_met", "accuracy_target_met", "saved_arrays_reproduced")}
                         for key, run in result["methods"].items()},
            ))
        except Exception as error:
            failure = dict(file=path.name, exception_type=type(error).__name__, message=str(error))
            failed.append(failure)
            write_json(path.with_suffix(".audit.json"), dict(audit_pass=False, error=failure))
            print(f"AUDIT FAILED {path.name}: {type(error).__name__}: {error}", flush=True)
        jax.clear_caches()
    finest = {}
    for result in summaries:
        if result["case"] not in finest or result["cells"] > finest[result["case"]]["cells"]:
            finest[result["case"]] = result
    final_hashes = _current_hashes()
    source_unchanged = final_hashes == current_hashes
    audit_pass = source_unchanged and not failed and all(result["audit_pass"] for result in summaries)
    all_roots = bool(summaries) and all(run["root_target_met"] for result in summaries for run in result["methods"].values())
    accuracy = bool(finest) and all(run["accuracy_target_met"] for result in finest.values() for run in result["methods"].values())
    report = dict(
        audit_pass=audit_pass, all_current_roots_met=all_roots,
        finest_grid_accuracy_met=accuracy, verification_acceptance=audit_pass and all_roots and accuracy,
        audited_files=len(summaries), failed_files=failed, cases=summaries,
        root_tolerance=args.tolerance, reproduction_tolerance=REPRODUCTION_TOLERANCE,
        current_versions=versions, current_source_hashes=current_hashes,
        source_unchanged_during_audit=source_unchanged,
        source_hashes_after=final_hashes,
        audit_driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        verification_driver_sha256=sha256(Path(__file__).with_name("verify_annular.py").read_bytes()).hexdigest(),
        python_executable=sys.executable,
        note="No Newton/timestep advances. Source drift is reported, not hidden; current equations are re-evaluated on saved fields. CLI status checks audit integrity; analytical failures remain verification failures.",
    )
    write_json(args.output/"audits.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("cases", "current_source_hashes", "source_hashes_after")}, indent=2), flush=True)
    return 0 if audit_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
