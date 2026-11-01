"""Independent shock metrics using canonical analytical reference routines.

Migrated from the prior investigation; no import or runtime dependency on it.
These model-specific diagnostics are study postprocessing, not numerical fluxes.
"""
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

def shock_branches(problem, coordinates):
    """Continue canonical perfect-gas reference branches near the actual shock.

    The annular reference directly supplies smooth branch objects. The nozzle
    reference supplies its canonical area-Mach and normal-shock routines; its
    downstream branch is assembled from those values and the model EOS at its
    exact downstream entropy. This is assessment only, restricted to packaged
    calorically-perfect-gas verification cases.
    """
    if problem.shock_m is None:
        raise ValueError("Shock-branch metrics require a shocked reference")
    x = np.atleast_1d(np.asarray(coordinates, dtype=float))
    reference = problem.reference
    if problem.definition.model_name == "euler_annular":
        upstream = reference.profile(x, branch="sup")
        downstream_reference = reference.normal_shock(problem.shock_m)["downstream_reference"]
        downstream = downstream_reference.profile(x, branch="sub")
        for fields in (upstream, downstream):
            fields["velocity"] = fields["v_m"]
        return upstream, downstream
    if np.any(x < reference.length/2):
        raise ValueError("Nozzle shock continuation is local to the divergent section")
    upstream = reference.profile(x, reference.adapted)
    _, astar2, total_ratio = reference.shock_data(problem.shock_m)
    areas = problem.geometry.sample(x)["area"]
    mach = np.array([reference.mach(float(a)/astar2, "sub") for a in areas])
    pressure = reference.p0*total_ratio*reference.pressure_ratio(mach)
    length = problem.mesh.faces[-1]-problem.mesh.faces[0]
    entropy2 = float(problem.reference_fields(np.array([problem.shock_m+1e-8*length]))["s"][0])
    downstream_eos = problem.fluid.get_state(jxp.PSmass_INPUTS, jnp.asarray(pressure), entropy2)
    downstream = {key: np.broadcast_to(np.asarray(downstream_eos[key]),pressure.shape)
                  for key in ("p", "T", "rho", "a", "s", "h")}
    downstream["velocity"] = mach*downstream["a"]
    upstream["velocity"] = upstream["u"]
    upstream["s"] = float(problem.model.s0)+upstream["ds"]
    return upstream, downstream

def local_shock_metrics(problem, state):
    """Return branch-envelope, entropy and gradient-thickness measurements."""
    fields = problem.numerical_fields(state)
    if "s" not in fields:
        fields["s"] = np.asarray(problem.model.thermodynamic_state(jnp.asarray(state[1:-1]))["s"])
    x = np.asarray(problem.mesh.centers)
    widths = np.asarray(problem.mesh.widths)
    length = problem.mesh.faces[-1]-problem.mesh.faces[0]
    shock = problem.shock_m
    if shock is None:
        return dict(description="Smooth control; no shock window")
    half_window = max(.05*length, 6*np.max(widths))
    window = abs(x-shock) <= half_window
    requested_indices = np.flatnonzero(window)
    if problem.definition.model_name == "euler_nozzle":
        # The higher-entropy downstream isentrope cannot be continued through
        # areas smaller than its sonic area. Retain only the actual two-branch
        # domain; do not invent a branch by clipping A/A* or a root argument.
        _, astar2, _ = problem.reference.shock_data(shock)
        area = np.asarray(problem.geometry.sample(x)["area"])
        window &= (x>problem.reference.length/2)&(area>=astar2)
    indices = np.flatnonzero(window)
    if len(indices) < 2:
        return dict(assessment_unavailable=True, description="Fewer than two valid continued-branch cells", window_indices=indices)
    up, down = shock_branches(problem, x[window])
    envelopes = {}
    result = dict(
        description="Per-cell point branch envelope; values inside envelope count as normal smearing",
        shock_window=[float(x[indices[0]]), float(x[indices[-1]])],
        window_indices=indices,
        requested_window_indices=requested_indices,
        excluded_unavailable_branch_indices=np.setdiff1d(requested_indices,indices),
        coordinate=x[window], upstream_branch={k: up[k] for k in ("p", "velocity", "T", "s")},
        downstream_branch={k: down[k] for k in ("p", "velocity", "T", "s")},
    )
    for key in ("p", "velocity", "T"):
        low = np.minimum(up[key], down[key])
        high = np.maximum(up[key], down[key])
        value = np.asarray(fields[key])[window]
        above = np.maximum(0., value-high)/np.maximum(np.abs(high), 1e-300)
        below = np.maximum(0., low-value)/np.maximum(np.abs(low), 1e-300)
        envelopes[key] = dict(minimum=low, maximum=high,
                              above_relative=above, below_relative=below)
        result[key+"_above_branch_relative"] = float(above.max())
        result[key+"_below_branch_relative"] = float(below.max())
    result["branch_envelopes"] = envelopes
    projected = problem.reference_cell_fields()
    projection_departures = {}
    for key in ("p", "velocity", "T"):
        value = np.asarray(projected[key])[window]
        low, high = envelopes[key]["minimum"], envelopes[key]["maximum"]
        projection_departures[key+"_above_branch_relative"] = float(
            np.max(np.maximum(0.,value-high)/np.maximum(abs(high),1e-300)))
        projection_departures[key+"_below_branch_relative"] = float(
            np.max(np.maximum(0.,low-value)/np.maximum(abs(low),1e-300)))
    invariant = float(problem.reference.h0) if problem.definition.model_name == "euler_annular" else float(problem.model.h0)
    projection_departures.update(
        h0_min_relative=float(np.min(np.asarray(projected["h0"])[window]/invariant-1)),
        h0_max_relative=float(np.max(np.asarray(projected["h0"])[window]/invariant-1)),
        description="Exact conservative cell projection can itself depart from point branches/invariants in a shock-straddling cell; compare separately from point samples",
    )
    result["exact_conservative_projection_departures"] = projection_departures
    result["numerical_h0_min_relative"] = float(np.min(np.asarray(fields["h0"])[window]/invariant-1))
    result["numerical_h0_max_relative"] = float(np.max(np.asarray(fields["h0"])[window]/invariant-1))
    peak_index = indices[np.argmax(envelopes["p"]["above_relative"])]
    result["pressure_peak_cell"] = int(peak_index)
    result["pressure_peak_value"] = float(fields["p"][peak_index])
    # Local normalized progress removes smooth geometric variation of both
    # branches. Its maximum positive gradient gives a continuous thickness
    # even when zero stored cells fall between 10% and 90%.
    progress = (np.asarray(fields["p"])[window]-up["p"])/(down["p"]-up["p"])
    gradient_progress = np.diff(progress)/np.diff(x[window])
    steepest = int(np.argmax(gradient_progress))
    location = .5*(x[window][steepest]+x[window][steepest+1])
    positive_transition = float(gradient_progress[steepest]) > 0
    inverse = 1/float(gradient_progress[steepest]) if positive_transition else None
    local_dx = .5*(widths[indices[steepest]]+widths[indices[steepest+1]])
    gradient_pressure = np.diff(np.asarray(fields["p"])[window])/np.diff(x[window])
    near_up, near_down = shock_branches(problem, np.array([shock]))
    pressure_jump = float(near_down["p"][0]-near_up["p"][0])
    raw_inverse = pressure_jump/float(gradient_pressure.max()) if float(gradient_pressure.max()) > 0 else None
    transition = (progress>.1)&(progress<.9)
    result.update(
        pressure_progress=progress,
        shock_position_from_progress_gradient=float(location) if positive_transition else None,
        shock_position_error_cells=float(abs(location-shock)/np.mean(widths)) if positive_transition else None,
        shock_thickness_inverse_progress_gradient_length=float(inverse) if inverse is not None else None,
        shock_thickness_inverse_progress_gradient_cells=float(inverse/local_dx) if inverse is not None else None,
        shock_thickness_pressure_jump_over_gradient_length=float(raw_inverse) if raw_inverse is not None else None,
        shock_thickness_pressure_jump_over_gradient_cells=float(raw_inverse/local_dx) if raw_inverse is not None else None,
        shock_thickness_10_90_count=int(transition.sum()),
        shock_thickness_10_90_length=float(widths[window][transition].sum()),
    )
    entropy = np.asarray(fields["s"])[window]
    entropy1 = float(near_up["s"][0])
    entropy2 = float(near_down["s"][0])
    entropy_jump = entropy2-entropy1
    entropy_step = np.diff(entropy)
    scale = max(abs(entropy_jump), 1e-300)
    result["entropy"] = dict(
        description="Point-state diagnostics, not a discrete entropy-flux inequality",
        exact_upstream=entropy1, exact_downstream=entropy2, exact_jump=entropy_jump,
        minimum=float(entropy.min()), maximum=float(entropy.max()),
        overshoot_over_exact_jump=max(0., float(entropy.max())-entropy2)/scale,
        undershoot_over_exact_jump=max(0., entropy1-float(entropy.min()))/scale,
        negative_total_variation_over_exact_jump=float(np.maximum(-entropy_step,0).sum())/scale,
        total_variation_over_exact_jump=float(abs(entropy_step).sum())/scale,
        window_net_rise_over_exact_jump=float(entropy[-1]-entropy[0])/scale,
    )
    return result
