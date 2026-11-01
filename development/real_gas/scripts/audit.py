"""Independent steady residual, conservation and pointwise-reference assessment."""

import numpy as np


def _reductions(raw, volumes, scales, time, weighting):
    scaled = np.asarray(raw, float) * time / (volumes[:, None] * scales)
    weights = volumes / volumes.sum() if weighting == "volume" else np.full(len(volumes), 1 / len(volumes))
    largest = np.max(np.abs(scaled), axis=0)
    safe = np.where(largest > 0, largest, 1.0)
    return dict(
        l1=np.sum(weights[:, None] * np.abs(scaled), axis=0),
        l2=largest * np.sqrt(np.sum(weights[:, None] * (scaled / safe)**2, axis=0)),
        linf=largest, worst_cell=np.argmax(np.abs(scaled), axis=0),
    )


def _assessment(current, initial, policy):
    active = initial > policy.initial_floor
    relative = current / np.where(initial > 0, initial, 1.0)
    absolute_ok = current <= policy.absolute_tolerance
    relative_ok = active & (relative <= policy.relative_tolerance)
    if policy.criterion == "either":
        passed = absolute_ok | relative_ok
    elif policy.criterion == "both":
        passed = absolute_ok & (~active | relative_ok)
    elif policy.criterion == "relative":
        passed = np.where(active, relative_ok, absolute_ok)
    else:
        passed = absolute_ok
    finite = np.isfinite(current).all() and np.isfinite(initial).all()
    return dict(
        equation_norm=current, equation_initial_norm=initial,
        equation_relative=relative, relative_active=active,
        equation_passed=passed & finite, all_equations_passed=bool(np.all(passed) and finite),
        absolute_satisfied=absolute_ok, relative_satisfied=relative_ok,
    )


def evaluate_solution(context, solution, policy):
    """Recompute the original steady operator and reduce on the NumPy host.

    Accuracy is diagnostic: a converged coarse discretization need not be an
    accurate approximation. No analytic-field error tolerance is smuggled into
    the nonlinear stopping or initialization. Failures remain explicit.
    """
    operator = solution.operator
    model, mesh = operator.model, operator.mesh
    full = np.asarray(solution.state, float)
    volumes = np.asarray(mesh.volumes)
    raw = np.asarray(operator.spatial_residual(full))
    boundary = np.asarray(operator.boundary_residual(full, 0.0))
    face_flux = np.asarray(operator.face_flux(full))
    norms = _reductions(raw, volumes, np.asarray(model.conservative_scales),
                        model.reference_time, policy.weighting)
    initial = next((np.asarray(row["equation_norm"], float) for row in solution.history
                    if "equation_norm" in row), np.full(3, np.nan))
    assessment = _assessment(norms[policy.norm], initial, policy)
    boundary_max = float(np.max(np.abs(boundary)))
    global_defect = np.abs(np.sum(raw, axis=0) / np.asarray(model.global_flux_scales))
    finite = bool(np.isfinite(raw).all() and np.isfinite(boundary).all()
                  and np.isfinite(face_flux).all() and np.isfinite(full).all())
    admissible = bool(operator.is_admissible(full))
    domain = bool(
        np.all(full[:, 0] >= context.eos.settings["p_min"])
        and np.all(full[:, 0] <= context.eos.settings["p_max"])
        and np.all(full[:, 2] >= context.eos.settings["h_min"])
        and np.all(full[:, 2] <= context.eos.settings["h_max"])
    )
    left, right = (np.asarray(v) for v in operator.reconstruct(full))
    all_states = np.concatenate((full, left, right), axis=0)
    thermo_bounds = {}
    for key, column in (("p", 0), ("h", 2)):
        lower, upper = (context.eos.settings[f"{key}_{suffix}"] for suffix in ("min", "max"))
        values = all_states[:, column]
        span = upper - lower
        thermo_bounds[key] = dict(
            table_min=lower, table_max=upper,
            minimum=float(np.min(values)), maximum=float(np.max(values)),
            minimum_margin_to_lower=float(np.min(values) - lower),
            minimum_margin_to_upper=float(upper - np.max(values)),
            states_within_1e8_relative_of_bound=int(np.count_nonzero(
                (values - lower <= 1e-8 * span) | (upper - values <= 1e-8 * span))),
            all_cell_boundary_and_reconstructed_states_inside=bool(np.all((values >= lower) & (values <= upper))),
        )
    domain = domain and all(v["all_cell_boundary_and_reconstructed_states_inside"] for v in thermo_bounds.values())
    regime_passed = bool(model.regime_audit(full))
    checks = dict(
        native_solver_success=bool(solution.success), finite=finite,
        admissible_including_reconstructed_faces=admissible, within_bicubic_domain=domain,
        original_spatial_equations=assessment["all_equations_passed"],
        boundary_equations=bool(boundary_max <= policy.boundary_tolerance),
        global_conservation=bool(np.max(global_defect) <= policy.global_tolerance),
        selected_boundary_regime=regime_passed,
    )
    states = full[1:-1]
    fields = {}
    field_error = None
    try:
        thermo = context.eos.state(states[:, 2], states[:, 0])
        fields = {key: thermo[key] for key in
                  ("p", "h", "rho", "T", "s", "Q", "a_table", "a_euler", "phase_flag", "Z_table")}
        fields.update(
            x=np.asarray(mesh.centers), A=np.asarray(mesh.area_centers), u=states[:, 1],
            M_table=states[:, 1] / thermo["a_table"], M_euler=states[:, 1] / thermo["a_euler"],
            h0=states[:, 2] + states[:, 1]**2 / 2,
            mass_flow=thermo["rho"] * states[:, 1] * np.asarray(mesh.area_centers),
        )
    except Exception as error:
        field_error = repr(error)
        checks["finite_hyperbolic_diagnostic_fields"] = False

    reference = context.reference_fields(mesh.centers)
    scales = dict(
        p=model.p0, u=model.a0, h=model.a0**2, rho=model.rho0,
        T=model.T0, s=model.entropy_scale, Q=1.0, a_table=model.a0, a_euler=model.a0,
        M_table=1.0, M_euler=1.0, h0=model.a0**2, mass_flow=context.mdot,
    )
    weights = volumes / volumes.sum()
    errors = {}
    for key, scale in scales.items():
        if key not in fields:
            continue
        error = (np.asarray(fields[key]) - np.asarray(reference[key])) / scale
        errors[key] = dict(
            normalized_l1=float(np.sum(weights * np.abs(error))),
            normalized_l2=float(np.sqrt(np.sum(weights * error**2))),
            normalized_linf=float(np.max(np.abs(error))), scale=float(scale),
        )

    face_mass = face_flux[:, 0]
    mean_mass = float(np.mean(face_mass))
    mass_scale = max(abs(context.mdot), np.finfo(float).tiny)
    energy_flux = face_flux[:, 2]
    conservation = dict(
        numerical_face_mass_flow_mean=mean_mass,
        numerical_face_mass_flow_relative_spread=float(np.ptp(face_mass) / mass_scale),
        numerical_face_mass_flow_relative_reference_error=float(abs(mean_mass / context.mdot - 1)),
        numerical_face_energy_flux_relative_spread=float(np.ptp(energy_flux) /
                                                       max(abs(float(np.mean(energy_flux))), 1.0)),
        outlet_pressure_relative_backpressure_error=float(abs(full[-1, 0] / context.back_pressure - 1)),
    )
    if fields:
        conservation.update(
            physical_cell_mass_flow_relative_spread=float(np.ptp(fields["mass_flow"]) / mass_scale),
            physical_cell_total_enthalpy_relative_max=float(
                np.max(np.abs(fields["h0"] - context.upstream.h0)) / model.a0**2),
        )
    shock_diagnostic = None
    if context.shock_position is not None and fields:
        divergent = np.flatnonzero(np.asarray(mesh.centers) >= 0)
        pair = divergent[:-1]
        if len(pair):
            jump = np.diff(fields["p"])[pair]
            index = pair[int(np.argmax(jump))]
            estimate = float((mesh.centers[index] + mesh.centers[index + 1]) / 2)
            shock_diagnostic = dict(
                estimator="largest positive adjacent-cell pressure jump in divergent section",
                position_m=estimate, reference_position_m=context.shock_position,
                position_error_over_divergent_length=float(
                    (estimate - context.shock_position) / context.nozzle.x_out),
            )

    roe = None
    if operator.scheme.flux == "roe":
        try:
            data = operator.roe_data(full)
            roe = {key: dict(
                finite=bool(np.isfinite(np.asarray(value)).all()),
                minimum=float(np.min(np.asarray(value))), maximum=float(np.max(np.asarray(value))))
                for key, value in data.items() if key != "projectors"}
            roe["invalid_faces"] = int(np.count_nonzero(~np.asarray(data["valid"], bool)))
            checks["roe_faces_valid"] = roe["invalid_faces"] == 0
        except Exception as error:
            roe = dict(error=repr(error))
            checks["roe_faces_valid"] = False

    audit = dict(
        passed=bool(all(checks.values())), checks=checks,
        failed_checks=[key for key, passed in checks.items() if not passed],
        residual=dict(
            definition="original signed integrated spatial residual; independent NumPy fixed-scale reduction",
            norm=policy.norm, weighting=policy.weighting,
            **assessment, equation_l1=norms["l1"], equation_l2=norms["l2"],
            equation_linf=norms["linf"], worst_cell=norms["worst_cell"],
            boundary_max=boundary_max, global_defect=global_defect,
        ),
        conservation=conservation, shock=shock_diagnostic, roe=roe,
        thermo_bounds=thermo_bounds,
        field_evaluation_error=field_error,
        accuracy_definition="Volume-weighted numerical primitive cell-state versus continuous reference at physical cell centers; not exact cell-average primitive inversion",
        error_scales=scales,
        boundary_sound_speed="Production model uses separately interpolated table a; reference also reports density-derived Euler a",
    )
    return dict(fields=fields, errors=errors, audit=audit)
