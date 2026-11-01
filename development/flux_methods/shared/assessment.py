"""Canonical-case assessment and independent NumPy residual reductions.

Physical reference objects come from nozzle_hyperbolic.verification. The
reference-branch diagnostics were migrated as local postprocessing, with
provenance under shared/evidence; no old study import is required.
"""
import jax.numpy as jnp
import numpy as np
from nozzle_hyperbolic.verification import build_case
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from .branch_metrics import local_shock_metrics

def shifted_problem(case, cells, alignment):
    """Keep physical case fixed; smoothly shift interior faces by fractions of dx."""
    problem = build_case(case, cells)
    if alignment == 0:
        return problem
    from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
    from nozzle_hyperbolic.models.euler_nozzle.model import NozzleModel
    from nozzle_hyperbolic.models.euler_annular.model import AnnularModel
    old = problem.mesh.faces
    length = old[-1] - old[0]
    faces = old + alignment * length / cells * np.sin(np.pi * (old-old[0])/length)
    faces[[0, -1]] = old[[0, -1]]
    data = problem.definition.data
    boundary = data["boundaries"]
    common = dict(p0=data["inlet"]["p0"], T0=data["inlet"]["T0"],
                  inlet_regime=boundary["inlet"], outlet_regime=boundary["outlet"],
                  outlet_formulation=boundary["outlet_formulation"])
    if problem.definition.model_name == "euler_nozzle":
        mesh = problem.geometry.build_mesh(faces=faces)
        problem.model = NozzleModel(mesh, problem.fluid, back_pressure=problem.back_pressure, **common)
    else:
        mesh, native = build_annular_mesh(problem.geometry, cells, faces=faces)
        problem.native_mesh = native
        inlet = problem.inlet_data
        problem.model = AnnularModel(mesh, native, problem.fluid, outlet_pressure=problem.back_pressure,
                                     inlet_data=inlet, inlet_regime=boundary["inlet"],
                                     outlet_regime=boundary["outlet"],outlet_formulation=boundary["outlet_formulation"],
                                     tau_w=data["walls"]["shear_stress"],q_w=data["walls"]["heat_flux"])
    return problem

def assess(problem, state, operator):
    fields = problem.numerical_fields(state)
    point = problem.reference_fields()
    projected = problem.reference_cell_fields()
    projection = problem.cell_reference()
    if "s" not in fields:
        fields["s"] = np.asarray(problem.model.thermodynamic_state(jnp.asarray(state[1:-1]))["s"])
        projected["s"] = np.asarray(problem.model.thermodynamic_state(jnp.asarray(projection["W"]))["s"])
    x = np.asarray(problem.mesh.centers)
    volumes = np.asarray(problem.mesh.volumes)
    dx = np.diff(problem.mesh.faces)
    length = problem.mesh.faces[-1] - problem.mesh.faces[0]
    shock = problem.shock_m
    errors = {}
    for representation, ref in (("point", point), ("cell", projected)):
        errors[representation] = {}
        for name in ("p", "velocity", "T", "h0", "rho", "s"):
            error = np.asarray(fields[name])-np.asarray(ref[name])
            scale = problem.field_scales[name]
            errors[representation][name] = dict(l1=np.sum(abs(error)*volumes)/sum(volumes),
                                                linf=np.max(abs(error)),
                                                normalized_l1=np.sum(abs(error)*volumes)/sum(volumes)/scale,
                                                normalized_linf=np.max(abs(error))/scale)
    content = np.asarray(operator.storage(jnp.asarray(state[1:-1])))
    q = content / volumes[:, None]
    error = q - projection["q"]
    conservative_errors = dict(l1=(abs(error)*volumes[:,None]).sum(axis=0)/sum(volumes),
        normalized_l1=(abs(error)*volumes[:,None]).sum(axis=0)/sum(volumes)/problem.model.conservative_scales,
        linf=abs(error).max(axis=0))
    h0ref = float(problem.reference.h0) if problem.definition.model_name == "euler_annular" else float(problem.model.h0)
    enthalpy_scale = problem.field_scales["h0"]
    metrics = dict(h0_relative_max=np.max(abs(fields["h0"]-h0ref))/abs(h0ref),
                   h0_scale_max=np.max(abs(fields["h0"]-h0ref))/enthalpy_scale,
                   h0_min=float(np.min(fields["h0"])),h0_max=float(np.max(fields["h0"])),
                   minimum_pressure=np.min(fields["p"]), minimum_temperature=np.min(fields["T"]),
                   minimum_density=np.min(fields["rho"]))
    if "ell" in fields:
        ell = float(problem.reference_fields(np.array([0.]))["ell"][0])
        metrics["angular_momentum_relative_max"] = float(np.max(abs(fields["ell"]-ell))/max(abs(ell),1e-30))
    far = np.ones(len(x),dtype=bool)
    if shock is not None:
        window = abs(x-shock) <= max(0.05*length, 6*np.max(dx))
        # Original conservative window-wide envelope, retained for provenance.
        # The independent postprocessing audit additionally continues both
        # branches at each cell coordinate; reports prefer that local metric.
        near = problem.reference_fields(np.array([shock-1e-8*length,shock+1e-8*length]))
        # The envelope uses extrema of the exact point branches in the fixed
        # window. It is conservative about peak claims near varying geometry.
        p_lo, p_hi = np.min(point["p"][window]), np.max(point["p"][window])
        u_lo, u_hi = np.min(point["velocity"][window]), np.max(point["velocity"][window])
        t_lo, t_hi = np.min(point["T"][window]), np.max(point["T"][window])
        metrics.update(pressure_overshoot=max(0.,float(np.max(fields["p"][window])-p_hi))/p_hi,
                       velocity_undershoot=max(0.,float(u_lo-np.min(fields["velocity"][window])))/abs(u_lo),
                       temperature_overshoot=max(0.,float(np.max(fields["T"][window])-t_hi))/t_hi,
                       temperature_undershoot=max(0.,float(t_lo-np.min(fields["T"][window])))/t_lo,
                       shock_window=[float(x[window][0]),float(x[window][-1])],
                       branch_envelope=dict(p=[p_lo,p_hi],velocity=[u_lo,u_hi],T=[t_lo,t_hi]))
        # Pressure-jump gradient identifies location; width counts normalized
        # intermediate cells, with local analytical branch levels at the shock.
        gradient = np.diff(fields["p"])/np.diff(x)
        candidates = np.flatnonzero(abs((x[:-1]+x[1:])/2-shock)<max(.15*length,8*np.max(dx)))
        index = candidates[np.argmax(gradient[candidates])]
        location = .5*(x[index]+x[index+1])
        jump = near["p"][1]-near["p"][0]
        fraction = (fields["p"]-near["p"][0])/jump
        transition = window & (fraction>.1)&(fraction<.9)
        far = abs(x-shock)>max(.05*length,6*np.max(dx))
        metrics.update(shock_expected=shock,shock_position=location,
                       shock_position_error_cells=abs(location-shock)/np.mean(dx),
                       shock_thickness_cells=int(transition.sum()),
                       shock_thickness_length=float(dx[transition].sum()),
                       grid_shock_fraction=float((shock-problem.mesh.faces[np.searchsorted(problem.mesh.faces,shock)-1])/dx[np.searchsorted(problem.mesh.faces,shock)-1]),
                       entropy_step_min=float(np.min(np.diff(fields["s"])[window[:-1]&window[1:]])))
    metrics["far_field_normalized_linf"] = {name:float(np.max(abs(fields[name][far]-point[name][far]))/problem.field_scales[name]) if far.any() else None
                                              for name in ("p","velocity","T","h0","s")}
    face = np.asarray(operator.face_flux(jnp.asarray(state)))
    raw = np.asarray(operator.spatial_residual(jnp.asarray(state)))
    sources = np.asarray(operator.source(jnp.asarray(state)))
    metrics["telescoping_identity_max"] = float(np.max(abs(raw.sum(axis=0) - (face[-1]-face[0]-sources.sum(axis=0)))))
    local = local_shock_metrics(problem, state)
    metrics["local_branch_audit"] = local
    metrics["branch_excess_pct"] = 100*local.get("p_above_branch_relative", float("nan"))
    indices = np.asarray(local.get("window_indices", []), dtype=int)
    valid = (indices > 0) & (indices < len(x)-1)
    neighbors = indices[valid]
    if len(neighbors):
        down = local["downstream_branch"]
        p = np.asarray(fields["p"])
        u = np.asarray(fields["velocity"])
        p_peak = np.maximum(p[neighbors]-np.maximum(p[neighbors-1],p[neighbors+1]),0)
        u_dip = np.maximum(np.minimum(u[neighbors-1],u[neighbors+1])-u[neighbors],0)
        metrics["pressure_local_peak_pct"] = 100*float(np.max(p_peak/abs(np.asarray(down["p"])[valid])))
        metrics["velocity_local_dip_pct"] = 100*float(np.max(u_dip/abs(np.asarray(down["velocity"])[valid])))
    conservative_errors["equation_names"] = problem.model.equation_names
    return dict(fields=fields,reference_point=point,reference_cell=projected,
                errors=errors,conservative_errors=conservative_errors,profile_metrics=metrics,
                shock_position=shock,
                face_flux=face,raw_spatial_residual=raw,integrated_source=sources,
                boundary_residual=np.asarray(operator.boundary_residual(jnp.asarray(state),0.0)))


def independent_check(operator, state, initial, policy):
    """NumPy reduction of fresh original rows, independent of native monitors.

    This recomputes fluxes through the production Operator. Independence refers
    to the stopping reductions and saved-state rebuild, not a second flux code.
    """
    model = operator.model
    state, initial = jnp.asarray(state), jnp.asarray(initial)
    raw = np.asarray(operator.spatial_residual(state))
    raw_initial = np.asarray(operator.spatial_residual(initial))
    bc = np.asarray(operator.boundary_residual(state, 0.0))
    volume = np.asarray(operator.mesh.volumes)
    scales = np.asarray(model.conservative_scales)
    weights = volume/volume.sum() if policy["weighting"] == "volume" else np.full(len(volume),1/len(volume))

    def reduce(values):
        scaled = values*model.reference_time/(volume[:,None]*scales)
        absolute = abs(scaled)
        largest = absolute.max(axis=0)
        safe = np.where(largest>0,largest,1.)
        return dict(l1=(weights[:,None]*absolute).sum(axis=0),
                    l2=largest*np.sqrt((weights[:,None]*(scaled/safe)**2).sum(axis=0)),
                    linf=largest, worst_cell=absolute.argmax(axis=0))

    norms = reduce(raw)
    first = reduce(raw_initial)[policy["norm"]]
    norm = norms[policy["norm"]]
    relative = norm/np.where(first>0,first,1.)
    active = first > policy["initial_floor"]
    absolute_ok = norm <= policy["absolute_tolerance"]
    relative_ok = active & (relative <= policy["relative_tolerance"])
    criterion = policy["criterion"]
    if criterion == "either":
        equations = absolute_ok | relative_ok
    elif criterion == "both":
        equations = absolute_ok & (~active | relative_ok)
    elif criterion == "relative":
        equations = np.where(active,relative_ok,absolute_ok)
    else:
        equations = absolute_ok
    global_signed = raw.sum(axis=0)/np.asarray(model.global_flux_scales)
    boundary_max = float(abs(bc).max())
    finite = bool(np.isfinite(raw).all() and np.isfinite(bc).all() and np.isfinite(first).all())
    admissible = bool(operator.is_admissible(state))
    branch = getattr(model,"solution_audit",None)
    branch_ok = True if branch is None else bool(branch(state))
    boundary_ok = boundary_max <= policy["boundary_tolerance"]
    global_ok = bool(np.max(abs(global_signed)) <= policy["global_tolerance"])
    original_passed = bool(finite and admissible and equations.all() and boundary_ok and global_ok)
    face = np.asarray(operator.face_flux(state))
    source = np.asarray(operator.source(state))
    return dict(independent_original_residual_checks_passed=original_passed,
                all_checks_passed=original_passed and branch_ok,
                finite=finite,admissible=admissible,regime_branch_audit=branch_ok,
                equation_names=model.equation_names, fixed_scale_reductions=norms,
                initial_equation_norm=first,final_equation_relative=relative,
                relative_active=active,equation_passed=equations,
                boundary_max=boundary_max,boundary_passed=boundary_ok,
                global_signed_fixed_scale=global_signed,global_passed=global_ok,
                telescoping_defect=raw.sum(axis=0)-(face[-1]-face[0]-source.sum(axis=0)))
