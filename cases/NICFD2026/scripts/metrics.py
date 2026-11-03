"""Independent convergence, published-profile errors and case-level verdicts."""

import jax
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from nozzle_hyperbolic.models.two_fluid import primitives

from .case import reference


def profile_kernel(model):
    """Bind one reusable field extractor; output includes two endpoint traces."""
    area = jnp.asarray(
        np.r_[
            model.mesh.area_faces[0], model.mesh.area_centers, model.mesh.area_faces[-1]
        ]
    )

    @jax.jit
    def extract(state):
        u = primitives(state)
        gas = model.fluid_g.get_state(jxp.HmassP_INPUTS, u[:, 4], u[:, 0])
        liquid = model.fluid_l.get_state(jxp.HmassP_INPUTS, u[:, 5], u[:, 0])
        eigenvalues = jax.vmap(lambda value: model._wave_data(value)[0])(u)
        sound = 0.5 * (eigenvalues[:, 2] - eigenvalues[:, 3])
        velocity = 0.5 * (eigenvalues[:, 2] + eigenvalues[:, 3])
        ug, ul, alpha = u[:, 1], u[:, 2], u[:, 3]
        mg, ml = alpha * gas["rho"], (1 - alpha) * liquid["rho"]
        return dict(
            x=model.mesh.points,
            A=area,
            r=jnp.sqrt(area / jnp.pi),
            p=u[:, 0],
            alpha_g=alpha,
            u_g=ug,
            u_l=ul,
            T_g=gas["T"],
            T_l=liquid["T"],
            h_g=u[:, 4],
            h_l=u[:, 5],
            rho_g=gas["rho"],
            rho_l=liquid["rho"],
            a_mix=sound,
            u_mix=velocity,
            Ma_mix=jnp.abs(velocity) / sound,
            quality_static=mg / (mg + ml),
            quality_flow=mg * ug / (mg * ug + ml * ul),
            slip=ug / ul,
            mdot=area * (mg * ug + ml * ul),
        )

    return extract


def regime_audit(fields, regime):
    mach = np.asarray(fields["Ma_mix"])[1:-1]
    finite = all(np.isfinite(np.asarray(value)).all() for value in fields.values())
    forward = bool(
        np.all(np.asarray(fields["u_g"])[1:-1] > 0)
        and np.all(np.asarray(fields["u_l"])[1:-1] > 0)
    )
    maximum, outlet = float(np.max(mach)), float(mach[-1])
    branch = (
        maximum < 1
        if regime == "subsonic"
        else maximum > 1 and outlet < 1 if regime == "supersonic_shock" else outlet > 1
    )
    return dict(
        passed=bool(finite and forward and branch),
        finite=bool(finite),
        forward=forward,
        maximum_mach=maximum,
        outlet_mach=outlet,
    )


def reference_errors(fields, model, family, regime, config):
    """Compare interior point profiles with arclength-weighted error norms.

    Published and current meshes differ. Interpolation excludes the two
    reference endpoint ghosts; no old runtime or old residual is inferred.
    """
    old = reference(family, regime)
    x, weights = np.asarray(fields["x"])[1:-1], np.asarray(model.mesh.widths)
    scales = {
        "p": config["bc"]["p0"],
        "alpha_g": 1.0,
        "Ma_mix": 1.0,
        "quality_static": 1.0,
        "quality_flow": 1.0,
        "T_g": config["bc"]["T_g0"],
        "T_l": config["bc"]["T_l0"],
    }
    result = {}
    for key in old.dtype.names:
        if key in ("x", "A", "r") or key not in fields:
            continue
        expected = np.interp(x, old["x"][1:-1], old[key][1:-1])
        delta = np.asarray(fields[key])[1:-1] - expected
        scale = float(scales.get(key, max(float(np.max(abs(old[key][1:-1]))), 1e-12)))
        rmse = float(np.sqrt(np.average(delta**2, weights=weights)))
        result[key] = dict(
            rmse=rmse,
            max_absolute=float(np.max(abs(delta))),
            mean_absolute=float(np.average(abs(delta), weights=weights)),
            scale=scale,
            normalized_rmse=rmse / scale,
        )
    new_mass = float(np.average(np.asarray(fields["mdot"])[1:-1], weights=weights))
    old_mass = float(np.mean(old["mdot"][1:-1]))
    result["mass_flow"] = dict(
        current=new_mass,
        published=old_mass,
        relative_difference=abs(new_mass - old_mass) / abs(old_mass),
    )
    return result


def independent_audit(workspace, state, initial_norm, recorded):
    """Reassemble original spatial, algebraic and invariant defects separately."""
    op, model, policy = workspace.operator, workspace.model, workspace.residual
    parts = {
        key: np.asarray(value)
        for key, value in op.residual_contributions(state).items()
    }
    physical = (
        parts["conservative"]
        + parts["stabilization"]
        + parts["nonconservative"]
        - parts["source"]
    )
    selected = physical + parts["equation_policy"]
    scaled = (
        selected
        * model.reference_time
        / (np.asarray(model.mesh.volumes)[:, None] * model.conservative_scales)
    )
    weights = (
        np.asarray(model.mesh.volumes)
        if policy.weighting == "volume"
        else np.ones(model.mesh.n_cells)
    )
    norms = {
        "l1": np.average(abs(scaled), axis=0, weights=weights),
        "l2": np.sqrt(np.average(scaled**2, axis=0, weights=weights)),
        "linf": np.max(abs(scaled), axis=0),
    }
    norm = norms[policy.norm]
    if "equation_norm" in recorded:
        np.testing.assert_allclose(
            norm, recorded["equation_norm"], rtol=2e-8, atol=1e-12
        )
    constraints = (
        np.asarray(model.cell_constraints(state[1:-1])) / model.constraint_scales
    )
    boundary = np.asarray(op.boundary_residual(state, 0.0))
    global_defect = np.sum(selected, axis=0) / model.global_flux_scales
    invariants = (
        model.invariant_projections
        @ np.sum(physical, axis=0)
        / model.invariant_flux_scales
    )
    absolute = norm <= policy.absolute_tolerance
    active = np.asarray(initial_norm) > policy.initial_floor
    relative = active & (norm <= np.asarray(initial_norm) * policy.relative_tolerance)
    if policy.criterion == "absolute":
        equation_pass = absolute
    elif policy.criterion == "relative":
        equation_pass = np.where(active, relative, absolute)
    elif policy.criterion == "both":
        equation_pass = absolute & np.where(active, relative, True)
    else:
        equation_pass = absolute | relative
    constraint_max = float(np.max(abs(constraints), initial=0.0))
    boundary_max = float(np.max(abs(boundary)))
    global_max = float(np.max(abs(global_defect)))
    invariant_max = float(np.max(abs(invariants)))
    admissible = bool(op.is_admissible(state))
    passed = bool(
        np.all(equation_pass)
        and np.isfinite(selected).all()
        and admissible
        and constraint_max <= policy.constraint_tolerance
        and boundary_max <= policy.boundary_tolerance
        and global_max <= policy.global_tolerance
        and invariant_max <= policy.global_tolerance
    )
    return dict(
        passed=passed,
        equation_norm=norm,
        boundary_max=boundary_max,
        constraint_max=constraint_max,
        global_max=global_max,
        invariant_max=invariant_max,
        admissible=admissible,
        unmodified_equation_l2=np.sqrt(
            np.average(
                (
                    physical
                    * model.reference_time
                    / (
                        np.asarray(model.mesh.volumes)[:, None]
                        * model.conservative_scales
                    )
                )
                ** 2,
                axis=0,
                weights=weights,
            )
        ),
        unmodified_global=np.sum(physical, axis=0) / model.global_flux_scales,
    )


def agreement(stage, thresholds):
    """State the scientific outcome before any report plots or timing tables."""
    if not stage or "errors" not in stage:
        return dict(
            status="not_reproduced",
            title="This case has not been reproduced.",
            detail=(stage or {}).get(
                "reason", "No valid current-solver state was returned."
            ),
            checks={},
        )
    if not stage.get("success") or not stage.get("audit", {}).get("passed"):
        return dict(
            status="not_converged",
            title="This case is not reproduced: the current solve did not converge.",
            detail="Any visible profile similarity is provisional. The original selected equations and independent convergence gates did not all pass. "
            + stage["reason"],
            checks={},
        )
    if not stage["regime_audit"]["passed"]:
        return dict(
            status="wrong_regime",
            title="This case does not match: the converged flow is on a different regime.",
            detail="Numerical root convergence alone does not establish the published physical branch.",
            checks={},
        )
    errors = stage["errors"]
    checks = dict(
        pressure=errors["p"]["normalized_rmse"]
        <= thresholds["pressure_normalized_rmse"],
        gas_velocity=errors["u_g"]["normalized_rmse"]
        <= thresholds["phase_velocity_normalized_rmse"],
        liquid_velocity=errors["u_l"]["normalized_rmse"]
        <= thresholds["phase_velocity_normalized_rmse"],
        void_fraction=errors["alpha_g"]["rmse"] <= thresholds["void_absolute_rmse"],
        gas_temperature=errors["T_g"]["normalized_rmse"]
        <= thresholds["temperature_normalized_rmse"],
        liquid_temperature=errors["T_l"]["normalized_rmse"]
        <= thresholds["temperature_normalized_rmse"],
        mass_flow=errors["mass_flow"]["relative_difference"]
        <= thresholds["mass_flow_relative_difference"],
    )
    good = all(checks.values())
    return dict(
        status="matches" if good else "disagrees",
        title=(
            "This case matches the published solution well."
            if good
            else "This case converges, but does not match the published solution well."
        ),
        detail=(
            "The current solve passes the original convergence and regime checks and all declared profile criteria."
            if good
            else "Failed agreement criteria: "
            + ", ".join(k.replace("_", " ") for k, v in checks.items() if not v)
        ),
        checks=checks,
    )
