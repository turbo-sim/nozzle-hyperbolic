"""Fresh original-equation checks, separate from analytical accuracy."""
import jax
import jax.numpy as jnp
import numpy as np

from development.real_gas.scripts.audit import _reductions, _assessment


def _position_criterion(error_cells, tolerance_cells, coordinate_scale, cell_width):
    """Account only for coordinate-subtraction roundoff at a declared limit."""
    allowance = 32*np.finfo(float).eps*coordinate_scale/cell_width
    return bool(abs(error_cells) <= tolerance_cells+allowance), float(allowance)


def evaluate_solution(context, solution, policy):
    operator = solution.operator
    model, mesh = operator.model, operator.mesh
    full = np.asarray(solution.state, dtype=float)
    raw = np.asarray(operator.spatial_residual(full))
    boundary = np.asarray(operator.boundary_residual(full, 0.0))
    flux = np.asarray(operator.face_flux(full))
    volumes = np.asarray(mesh.volumes)
    norms = _reductions(raw, volumes, np.asarray(model.conservative_scales), model.reference_time, policy.weighting)
    initial = next((np.asarray(row["equation_norm"], float) for row in solution.history
                    if "equation_norm" in row), np.full(model.nvar, np.nan))
    assessment = _assessment(norms[policy.norm], initial, policy)
    boundary_max = float(np.max(np.abs(boundary)))
    global_defect = np.abs(raw.sum(axis=0) / np.asarray(model.global_flux_scales))
    reconstructed = [np.asarray(value) for value in operator.reconstruct(full)]
    states = np.concatenate((full, *reconstructed), axis=0)
    domain = {}
    for key, index in (("p", 0), ("h", model.nvar-1)):
        values = states[:, index]
        lower, upper = getattr(model.fluid, key+"_min", None), getattr(model.fluid, key+"_max", None)
        passed = np.isfinite(values).all()
        if lower is not None:
            passed = passed and np.all(values >= lower)
        if upper is not None:
            passed = passed and np.all(values <= upper)
        domain[key] = dict(minimum=float(values.min()), maximum=float(values.max()),
                           table_min=lower, table_max=upper, passed=bool(passed))
    regime_audit = getattr(model, "solution_audit", None)
    checks = dict(
        native_solver_success=bool(solution.success),
        finite=bool(np.isfinite(raw).all() and np.isfinite(boundary).all() and np.isfinite(flux).all()),
        admissible_cells_traces_reconstructed_faces=bool(operator.is_admissible(full)),
        properties_inside_declared_domain=all(row["passed"] for row in domain.values()),
        original_spatial_equations=bool(assessment["all_equations_passed"]),
        boundary_equations=boundary_max <= policy.boundary_tolerance,
        global_conservation=bool(np.max(global_defect) <= policy.global_tolerance),
        selected_boundary_regimes=bool(regime_audit(full)) if regime_audit is not None else True,
    )
    fields = context.numerical_fields(model, full)
    reference = context.reference_fields(mesh.centers)
    scales = context.scales(model)
    weights = volumes/volumes.sum()
    errors = {}
    for key, scale in scales.items():
        if key not in fields or key not in reference:
            continue
        defect = (fields[key]-reference[key])/scale
        errors[key] = dict(normalized_l1=float(weights@np.abs(defect)),
                           normalized_l2=float(np.sqrt(weights@(defect**2))),
                           normalized_linf=float(np.max(np.abs(defect))), scale=scale)
    face_mass = flux[:, 0]
    conservation = dict(
        face_mass_flow_mean=float(face_mass.mean()),
        face_mass_flow_relative_spread=float(np.ptp(face_mass)/max(abs(float(face_mass.mean())), 1e-300)),
        outlet_pressure_relative_backpressure_error=float(abs(full[-1, 0]/context.back_pressure-1)),
        total_enthalpy_relative_spread=float(np.ptp(fields["h0"])/max(float(model.variable_scales[-1]), 1e-300)),
    )
    shock = None
    target = None
    if context.shock_position is not None:
        x, pressure = fields["x"], fields["p"]
        throat = 0.0 if context.real_gas else (float(context.physical.geometry.length)/2
                                             if model.name == "euler_nozzle" else float(mesh.faces[0]))
        indices = np.flatnonzero(x[:-1] >= throat)
        index = int(indices[np.argmax(np.diff(pressure)[indices])])
        position = float((x[index]+x[index+1])/2)
        shock = dict(estimator="largest positive adjacent-cell pressure jump downstream of throat/inlet",
                     position_m=position, reference_position_m=context.shock_position,
                     error_over_length=(position-context.shock_position)/(mesh.faces[-1]-throat),
                     position_error_cells=float((position-context.shock_position)/np.mean(np.diff(mesh.faces))))
        upstream = (x >= throat) & (x < position)
        downstream = x > position
        branch_checks = dict(
            internal_compression=bool(pressure[index+1] > pressure[index] and throat < position < mesh.faces[-1]),
            resolved_supersonic_upstream=bool(np.any(fields["M"][upstream] > 1.0)),
            downstream_subsonic=bool(np.any(downstream) and fields["M"][-1] < 1.0),
        )
        declared = None if context.real_gas else context.physical.definition.data["accuracy"]
        if declared is not None and "shock_position_cells" in declared:
            threshold = float(declared["shock_position_cells"])
            cell_width = float(np.mean(np.diff(mesh.faces)))
            coordinate_scale = max(abs(position), abs(context.shock_position),
                                   abs(float(mesh.faces[0])), abs(float(mesh.faces[-1])), cell_width)
            passed, allowance = _position_criterion(shock["position_error_cells"], threshold, coordinate_scale, cell_width)
            branch_checks["declared_shock_position_accuracy"] = passed
            shock.update(declared_position_tolerance_cells=threshold,
                         comparison_roundoff_allowance_cells=allowance)
        target = dict(passed=bool(all(branch_checks.values())), checks=branch_checks,
                      maximum_meridional_mach=float(np.max(fields["M"])),
                      accuracy_targets=declared,
                      accuracy_scope="CO2 has no declared shock-position tolerance; position/field errors remain diagnostics"
                      if declared is None else "Canonical case shock-position tolerance is checked; field errors are reported separately",
                      resolution_limited=not branch_checks["resolved_supersonic_upstream"])
    diagnostics = getattr(model, "boundary_diagnostics", None)
    full_jax = jnp.asarray(full)
    wave = None
    if model.outlet_formulation == "finite_wave":
        sampled_wave = model.outlet.wave(full_jax[1:-1])
        checks["finite_wave_curve_valid"] = bool(np.isfinite(np.asarray(sampled_wave.state)).all()
            and np.isfinite(np.asarray(sampled_wave.head_speed)).all()
            and np.isfinite(np.asarray(sampled_wave.tail_speed)).all())
        wave = dict(mode=sampled_wave.mode, head_speed=sampled_wave.head_speed,
                    tail_speed=sampled_wave.tail_speed, direct_flux=sampled_wave.direct_flux,
                    sampled_state=sampled_wave.state)
    elif diagnostics is not None:
        wave = diagnostics(full_jax)
    if wave is not None:
        wave = jax.tree.map(np.asarray, wave)
    audit = dict(passed=bool(all(checks.values())), checks=checks,
                 failed_checks=[key for key, value in checks.items() if not value],
                 residual=dict(**assessment, equation_l1=norms["l1"], equation_l2=norms["l2"],
                               equation_linf=norms["linf"], worst_cell=norms["worst_cell"],
                               boundary_max=boundary_max, global_defect=global_defect,
                               definition="original integrated signed residual; independent fixed-scale NumPy reduction"),
                 conservation=conservation, shock=shock, target=target, domain=domain, outlet_wave=wave,
                 accuracy_definition="Volume-weighted cell primitive diagnostics versus continuous reference point values; not cell-average inversion")
    return dict(fields=fields, reference_fields=reference, errors=errors, audit=audit,
                target_correct=bool(target["passed"]) if target is not None else None)
