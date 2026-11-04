"""Independent saved-state and algebra audits for the development FAS study.

No solver evolution occurs in a campaign audit. Fresh production spatial rows
are reduced with NumPy; analytical fields are independent assessment data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]


def clean(value):
    """Convert arrays and nonfinite diagnostics to portable JSON values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_hashes():
    """Hash all package Python and packaged physical-case YAML bytes."""
    files = {
        p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/nozzle_hyperbolic").rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yaml")
    }
    digest = sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "sha256": digest}


def _policy_dict(policy):
    from nozzle_hyperbolic.solvers.residuals import ResidualPolicy

    if policy is None:
        return asdict(ResidualPolicy(criterion="absolute"))
    return asdict(policy) if is_dataclass(policy) else dict(policy)


def _norms(integrated, volumes, scales, reference_time, weighting):
    """Independent physical-volume fixed-scale residual reductions."""
    scaled = np.asarray(integrated) * float(reference_time) / (
        volumes[:, None] * scales[None, :]
    )
    weights = (
        volumes / volumes.sum()
        if weighting == "volume"
        else np.full(len(volumes), 1.0 / len(volumes))
    )
    absolute = np.abs(scaled)
    largest = absolute.max(axis=0)
    safe = np.where(largest > 0, largest, 1.0)
    return {
        "l1": (weights[:, None] * absolute).sum(axis=0),
        "l2": largest
        * np.sqrt((weights[:, None] * (scaled / safe) ** 2).sum(axis=0)),
        "linf": largest,
    }


def _error_norms(value, reference, volumes, scale):
    error = np.asarray(value) - np.asarray(reference)
    l1 = float(np.sum(np.abs(error) * volumes) / np.sum(volumes))
    return {
        "l1": l1,
        "linf": float(np.max(np.abs(error))),
        "normalized_l1": l1 / float(scale),
        "normalized_linf": float(np.max(np.abs(error))) / float(scale),
    }


def audit_state(problem, state, operator, policy=None, initial_state=None):
    """Freshly audit one state after, and outside, its timed solve.

    ``original_checks_passed`` concerns the discrete equations only;
    ``analytical_passed`` and ``branch_passed`` are separate judgements. The
    catalogue's original target is applied without relaxation. At fewer than
    its recommended cell count, the result remains labelled ``accuracy_assessed``
    false even though the same target comparison is reported.
    """
    import jax.numpy as jnp

    policy = _policy_dict(policy)
    state = np.asarray(state, dtype=float)
    first = problem.initial_state if initial_state is None else initial_state
    first = np.asarray(first, dtype=float)
    volumes = np.asarray(problem.mesh.volumes)
    scales = np.asarray(problem.model.conservative_scales)
    raw = np.asarray(operator.spatial_residual(jnp.asarray(state)))
    raw_first = np.asarray(operator.spatial_residual(jnp.asarray(first)))
    bc = np.asarray(operator.boundary_residual(jnp.asarray(state), 0.0))
    constraints = np.asarray(operator.cell_constraints(jnp.asarray(state[1:-1])))
    norms = _norms(
        raw, volumes, scales, problem.model.reference_time, policy["weighting"]
    )
    initial_norm = _norms(
        raw_first, volumes, scales, problem.model.reference_time, policy["weighting"]
    )[policy["norm"]]
    equation = norms[policy["norm"]]
    active = initial_norm > policy["initial_floor"]
    relative = equation / np.where(initial_norm > 0, initial_norm, 1.0)
    absolute_ok = equation <= policy["absolute_tolerance"]
    relative_ok = active & (relative <= policy["relative_tolerance"])
    criterion = policy["criterion"]
    if criterion == "absolute":
        equation_ok = absolute_ok
    elif criterion == "relative":
        equation_ok = np.where(active, relative_ok, absolute_ok)
    elif criterion == "both":
        equation_ok = absolute_ok & (~active | relative_ok)
    elif criterion == "either":
        equation_ok = absolute_ok | relative_ok
    else:
        raise ValueError(f"Unknown stopping criterion {criterion!r}")
    boundary_max = float(np.max(np.abs(bc), initial=0.0))
    constraint_max = float(np.max(np.abs(constraints), initial=0.0))
    global_defect = np.abs(raw.sum(axis=0)) / np.asarray(
        problem.model.global_flux_scales
    )
    finite = bool(
        np.isfinite(state).all()
        and np.isfinite(raw).all()
        and np.isfinite(bc).all()
        and np.isfinite(constraints).all()
    )
    admissible = bool(operator.is_admissible(jnp.asarray(state)))
    checks = {
        "finite": finite,
        "admissible": admissible,
        "equations": bool(np.all(equation_ok)),
        "boundary": boundary_max <= policy["boundary_tolerance"],
        "constraints": constraint_max <= policy["constraint_tolerance"],
        "global": bool(np.all(global_defect <= policy["global_tolerance"])),
    }
    if hasattr(problem.model, "invariant_projections"):
        invariant = np.abs(
            np.asarray(problem.model.invariant_projections) @ raw.sum(axis=0)
        ) / np.asarray(problem.model.invariant_flux_scales)
        checks["invariants"] = bool(
            np.all(invariant <= policy["global_tolerance"])
        )
    fields = {key: np.asarray(v) for key, v in problem.numerical_fields(state).items()}
    reference = problem.reference_cell_fields()
    exact_q = np.asarray(problem.cell_reference()["q"])
    q = np.asarray(operator.storage(jnp.asarray(state[1:-1]))) / volumes[:, None]
    errors = {}
    for name, scale in problem.field_scales.items():
        if (
            name in fields
            and name in reference
            and np.shape(fields[name]) == np.shape(reference[name]) == volumes.shape
        ):
            errors[name] = _error_norms(fields[name], reference[name], volumes, scale)
    conservative_error = np.sum(np.abs(q - exact_q) * volumes[:, None], axis=0)
    conservative_error /= volumes.sum() * scales
    targets = problem.definition.data["accuracy"]
    accuracy_checks = {}
    if "normalized_field_l1" in targets:
        selected = [
            errors[name]["normalized_l1"]
            for name in ("p", "rho", "velocity", "M", "T")
            if name in errors
        ]
        accuracy_checks["normalized_field_l1"] = bool(
            max(selected, default=float("inf")) <= targets["normalized_field_l1"]
        )
    if "conservative_normalized_l1" in targets:
        accuracy_checks["conservative_normalized_l1"] = bool(
            np.max(conservative_error) < targets["conservative_normalized_l1"]
        )
    mach = np.asarray(fields["M"])
    x = np.asarray(problem.mesh.centers)
    shock = {
        "expected": problem.shock_position,
        "computed": None,
        "error_cells": None,
        "definition": "first interior supersonic-to-subsonic Mach crossing midpoint",
    }
    if problem.shock_position is not None:
        crossings = np.flatnonzero((mach[:-1] > 1.0) & (mach[1:] <= 1.0))
        if crossings.size:
            index = int(crossings[0])
            shock["computed"] = float(0.5 * (x[index] + x[index + 1]))
            shock["error_cells"] = abs(shock["computed"] - problem.shock_position) / float(
                np.max(np.diff(problem.mesh.faces))
            )
        accuracy_checks["shock_position"] = bool(
            shock["error_cells"] is not None
            and shock["error_cells"] <= targets.get("shock_position_cells", 2.0)
        )
    branch_checks = {
        "forward_velocity": bool(np.all(fields["velocity"] > 0)),
        "model_solution_audit": bool(problem.model.solution_audit(state)),
    }
    if problem.shock_position is not None:
        branch_checks["shock_present"] = shock["computed"] is not None
    if problem.definition.data["reference"]["condition"] == "throat_mach":
        branch_checks["subsonic_branch"] = bool(np.max(mach) <= 1.0 + 1e-8)
    face_flux = np.asarray(operator.face_flux(jnp.asarray(state)))
    source = np.asarray(operator.source(jnp.asarray(state)))
    telescoping = None
    if getattr(problem.model, "nonconservative_mode", "none") == "none":
        telescoping = float(
            np.max(np.abs(raw.sum(axis=0) - (face_flux[-1] - face_flux[0] - source.sum(axis=0))))
        )
    return clean(
        {
            "original_checks_passed": all(checks.values()),
            "original_checks": checks,
            "equation_norm": equation,
            "equation_relative_common_cold": relative,
            "equation_l1": norms["l1"],
            "equation_l2": norms["l2"],
            "equation_linf": norms["linf"],
            "boundary_max": boundary_max,
            "constraint_max": constraint_max,
            "global_defect": global_defect,
            "branch_passed": all(branch_checks.values()),
            "branch_checks": branch_checks,
            "analytical_passed": all(accuracy_checks.values()),
            "accuracy_assessed": len(volumes) >= targets["recommended_finest_cells"],
            "accuracy_checks": accuracy_checks,
            "accuracy_targets": targets,
            "errors": errors,
            "conservative_normalized_l1": conservative_error,
            "shock": shock,
            "mass_flow_face_mean": float(face_flux[:, 0].mean()),
            "mass_flow_face_relative_range": float(np.ptp(face_flux[:, 0]))
            / max(float(np.max(np.abs(face_flux[:, 0]))), 1e-300),
            "telescoping_identity_max": telescoping,
            "reference_representation": "volume-projected conserved cell averages",
            "policy": policy,
        }
    )


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_state(record):
    state = record.get("state")
    if isinstance(state, dict):
        state = state.get("full", state.get("state"))
    if state is None:
        state = record.get("solution", {}).get("state")
    return None if state is None else np.asarray(state, dtype=float)


def run_core_audit():
    """Check transfer identities, forcing parity and an exact FAS fixed point.

    These small algebra audits do not integrate a physical nozzle campaign.
    The fixed point is an arbitrary admissible state with its exact forcing,
    so its validity does not depend on an analytical nozzle solution.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from nozzle_hyperbolic.solvers.fvm import Scheme
    from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
    from .core import FASWorkspace, prolong_density, restrict_contents, restrict_integrated
    from .nozzle_case import build_hierarchy

    checks = {}
    # Unequal child volumes and five opaque transported fields exercise the
    # transfer contract without assuming Euler's equation count or identities.
    volumes = np.array([0.3, 0.7, 0.8, 0.4, 0.2, 0.5, 0.6, 0.9])
    coarse_volumes = volumes[::2] + volumes[1::2]
    q = np.arange(40, dtype=float).reshape(8, 5) / 7.0 - 2.0
    contents = volumes[:, None] * q
    restricted = np.asarray(restrict_contents(jnp.asarray(contents)))
    expected = contents[::2] + contents[1::2]
    np.testing.assert_allclose(restricted, expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(restricted.sum(0), contents.sum(0), rtol=2e-15, atol=3e-15)
    checks["nonuniform_content_restriction"] = {
        "passed": True,
        "max_global_content_error": float(np.max(np.abs(restricted.sum(0) - contents.sum(0)))),
        "transported_fields": 5,
    }
    constant = np.array([1.0, -2.0, 4.0, 0.25, 9.0])
    constant_contents = volumes[:, None] * constant[None, :]
    constant_coarse = np.asarray(restrict_contents(jnp.asarray(constant_contents))) / coarse_volumes[:, None]
    np.testing.assert_allclose(constant_coarse, np.broadcast_to(constant, (4, 5)), rtol=2e-15)
    e_coarse = np.arange(20, dtype=float).reshape(4, 5) / 9.0 - 1.0
    e_fine = np.asarray(prolong_density(jnp.asarray(e_coarse)))
    round_trip = np.asarray(restrict_contents(jnp.asarray(e_fine * volumes[:, None]))) / coarse_volumes[:, None]
    np.testing.assert_allclose(round_trip, e_coarse, rtol=2e-15, atol=2e-15)
    signed = np.vstack((np.arange(5), -np.arange(5), q[2:]))
    residual_coarse = np.asarray(restrict_integrated(jnp.asarray(signed)))
    np.testing.assert_allclose(residual_coarse, signed[::2] + signed[1::2], rtol=0, atol=0)
    checks["constant_preservation_and_correction_round_trip"] = {"passed": True}
    checks["signed_integrated_residual_sum"] = {"passed": True}

    problems = build_hierarchy("euler_nozzle/subsonic", (4, 8))
    scheme = Scheme(flux="hll", reconstruction="muscl_van_leer",
                    newton_iterations=20, newton_backtracking=True)
    workspace = FASWorkspace(
        [p.model for p in problems], scheme, ResidualPolicy(criterion="absolute"),
        settings={"pre_sweeps": 0, "post_sweeps": 0, "coarse_steps": 1,
                  "max_wall_s": 300.0}, mode="implicit",
    )
    level = workspace.levels[-1]
    op = level.operator
    old = jnp.asarray(problems[-1].initial_state)
    new = old.at[1:-1, 0].multiply(1.0001).at[1:-1, 1].add(0.01)
    dt, time = jnp.float64(0.02 * level.reference_time), jnp.float64(0.0)
    zero = jnp.zeros((op.mesh.n_cells, op.ntransport), dtype=jnp.float64)
    native = np.asarray(op.residual(new.ravel(), old.ravel(), dt, time))
    forced_zero = np.asarray(level.forced_residual(new.ravel(), old.ravel(), dt, time, zero))
    np.testing.assert_allclose(forced_zero, native, rtol=0, atol=0)
    forcing = jnp.asarray(np.arange(zero.size).reshape(zero.shape) + 0.5) * level.content_scale / level.reference_time
    forced = np.asarray(level.forced_residual(new.ravel(), old.ravel(), dt, time, forcing)).reshape(op.shape)
    expected_change = np.zeros(op.shape)
    factor = float(dt / jnp.maximum(1.0, dt / level.reference_time))
    expected_change[1:-1, :op.ntransport] = -factor * np.asarray(forcing / level.content_scale)
    actual_change = forced - forced_zero.reshape(op.shape)
    np.testing.assert_allclose(actual_change, expected_change, rtol=2e-13, atol=2e-15)
    checks["zero_forcing_native_residual_parity"] = {"passed": True, "max_error": float(np.max(np.abs(native-forced_zero)))}
    checks["nonzero_forcing_sign_scale_and_homogeneous_boundaries"] = {
        "passed": True, "max_error": float(np.max(np.abs(actual_change-expected_change)))
    }

    arguments = (new.ravel(), old.ravel(), dt, time, forcing)
    dense = np.asarray(jax.jit(level.forced_blocks["dense_jacobian"])(*arguments))
    colored = np.asarray(jax.jit(level.forced_blocks["colored_jacobian"])(*arguments))
    native_dense = np.asarray(op.dense_jacobian(new.ravel(), old.ravel(), dt, time))
    variable_scales = np.tile(np.asarray(level.model.variable_scales), op.shape[0])
    jac_scale = max(float(np.max(np.abs(dense * variable_scales[None, :]))), 1.0)
    colored_error = float(np.max(np.abs((colored-dense)*variable_scales[None, :])) / jac_scale)
    native_error = float(np.max(np.abs((native_dense-dense)*variable_scales[None, :])) / jac_scale)
    assert colored_error < 1e-11, colored_error
    assert native_error < 1e-11, native_error
    checks["forced_dense_colored_and_native_jacobian_parity"] = {
        "passed": True, "colored_scaled_relative_error": colored_error,
        "forcing_independence_scaled_relative_error": native_error,
    }

    completed, info = op.complete_boundaries(old, time)
    assert bool(info.converged and info.admissible), info
    exact_forcing = op.spatial_residual(completed)
    workspace._reset()
    corrected, okay = workspace._cycle(len(workspace.levels)-1, completed, exact_forcing)
    assert okay, workspace.work
    state_error = float(np.max(np.abs(np.asarray(corrected-completed)) / np.asarray(level.model.variable_scales)))
    assert state_error < 2e-11, state_error
    assert workspace.work["levels"][0]["accepted_steps"] == 0, workspace.work
    boundary_error = float(np.max(np.abs(np.asarray(op.boundary_residual(corrected, time)))))
    assert boundary_error < 2e-11, boundary_error
    checks["exact_forced_state_fas_fixed_point"] = {
        "passed": True, "scaled_state_change": state_error,
        "boundary_defect": boundary_error,
        "coarse_evolution_steps": workspace.work["levels"][0]["accepted_steps"],
        "work": workspace.work,
    }
    return clean({"passed": True, "checks": checks, "source": source_hashes()})


def audit_campaign(campaign_dir, *, output=None):
    """Rebuild saved fine states and compare methods using the original operator."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
    from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
    from .nozzle_case import build_problem

    directory = Path(campaign_dir)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = summary.get("configuration", {})
    default_policy = config.get("residual", config.get("residual_policy", {"criterion": "absolute"}))
    allowed = Scheme.__dataclass_fields__
    scheme_config = {k: v for k, v in config.get("scheme", {}).items() if k in allowed}
    snapshot = source_hashes()
    metadata = summary.get("metadata", {})
    expected_source = metadata.get("src_before", metadata.get("source_before", metadata.get("source", {})))
    expected_files = expected_source.get("files", {})
    changed = [name for name in sorted(set(expected_files) | set(snapshot["files"]))
               if expected_files.get(name) != snapshot["files"].get(name)]
    groups = []
    sample_failures = []
    operators = {}
    for group in summary.get("groups", []):
        case_id = group["case_id"]
        reconstruction = group["reconstruction"]
        records = []
        unique = {}
        representatives = {}
        repeat_consistency = {}
        for method in group.get("methods", []):
            method_id = method.get("method_id", method.get("id", method.get("method")))
            collection = [("warmup", method.get("warmup"))]
            collection.extend((f"sample_{i}", row) for i, row in enumerate(method.get("samples", [])))
            if not any(isinstance(row, dict) for _, row in collection):
                collection.append(("representative", method.get("representative")))
            for name, record in collection:
                if not isinstance(record, dict):
                    continue
                raw_file = record.get("raw_file")
                if raw_file:
                    path = directory / raw_file
                    if not path.exists():
                        sample_failures.append(f"Missing raw record {raw_file}")
                        continue
                    raw_bytes = path.read_bytes()
                    if record.get("raw_sha256") and sha256(raw_bytes).hexdigest() != record["raw_sha256"]:
                        sample_failures.append(f"Raw checksum mismatch {raw_file}")
                    record = json.loads(raw_bytes)
                state = _load_state(record)
                if state is None:
                    sample_failures.append(f"{case_id}/{reconstruction}/{method_id}/{name}: missing saved state")
                    continue
                cells = int(record.get("actual_cells", state.shape[0] - 2))
                options = dict(scheme_config, **{k: v for k, v in record.get("scheme", {}).items() if k in allowed})
                options["reconstruction"] = reconstruction
                key = (case_id, cells, json.dumps(options, sort_keys=True))
                if key not in operators:
                    problem = build_problem(case_id, cells)
                    operators[key] = (problem, Operator(problem.model, Scheme(**options)))
                problem, operator = operators[key]
                policy = ResidualPolicy(**record.get("residual_policy", default_policy))
                digest = sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()
                assessment_key = (digest, key, json.dumps(asdict(policy), sort_keys=True))
                if assessment_key not in unique:
                    unique[assessment_key] = audit_state(problem, state, operator, policy)
                assessment = unique[assessment_key]
                original_success = bool(record.get("success", False))
                fresh_success = bool(
                    assessment["original_checks_passed"] and assessment["branch_passed"]
                )
                if original_success and not fresh_success:
                    sample_failures.append(f"{case_id}/{reconstruction}/{method_id}/{name}: claimed success fails fresh checks")
                for field in ("original_checks_passed", "branch_passed", "analytical_passed"):
                    saved = record.get("assessment", {}).get(field)
                    if saved is not None and bool(saved) != bool(assessment[field]):
                        sample_failures.append(f"{case_id}/{reconstruction}/{method_id}/{name}: saved {field} disagrees with fresh audit")
                records.append({"method_id": method_id, "record": name, "state_sha256": digest,
                                "saved_success": original_success, "fresh_success": fresh_success,
                                "assessment": assessment})
                if original_success and fresh_success and name != "warmup":
                    scales = np.asarray(problem.model.variable_scales)
                    if method_id in representatives:
                        previous, _ = representatives[method_id]
                        difference = float(np.max(np.abs(state - previous) / scales)) if state.shape == previous.shape else float("inf")
                        consistency = repeat_consistency[method_id]
                        consistency["bitwise_identical"] &= bool(np.array_equal(state, previous))
                        consistency["max_scaled_difference"] = max(consistency["max_scaled_difference"], difference)
                        consistency["same_discrete_root"] &= difference <= 1e-6
                        consistency["samples_checked"] += 1
                    else:
                        representatives[method_id] = (state, scales)
                        repeat_consistency[method_id] = {"bitwise_identical": True,
                            "max_scaled_difference": 0.0, "same_discrete_root": True,
                            "threshold": 1e-6, "samples_checked": 1}
        comparisons = []
        baseline_name = next((name for name in representatives if "sequence" in str(name)), None)
        if baseline_name:
            baseline, _ = representatives[baseline_name]
            for method_id, (state, variable_scales) in representatives.items():
                if method_id == baseline_name:
                    continue
                if state.shape != baseline.shape:
                    comparisons.append({"baseline": baseline_name, "method_id": method_id,
                                        "same_discrete_root": False, "reason": "different_reached_grids"})
                    continue
                error = np.max(np.abs(state - baseline) / variable_scales)
                comparisons.append({"baseline": baseline_name, "method_id": method_id,
                                    "max_scaled_state_difference": float(error),
                                    "threshold": 1e-6, "same_discrete_root": bool(error <= 1e-6)})
        groups.append({"case_id": case_id, "reconstruction": reconstruction,
                       "records": records, "method_comparisons": comparisons,
                       "repeat_consistency": repeat_consistency,
                       "unique_states_recomputed": len(unique)})
    report = {"source_files_checked": len(expected_files), "source_changed": changed,
              "source_manifest_present": bool(expected_files), "source_after": snapshot,
              "sample_mismatches": sample_failures, "groups": groups,
              "passed": bool(expected_files and not changed and not sample_failures)}
    _save(output or directory / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--algebra", action="store_true")
    args = parser.parse_args()
    if args.algebra:
        report = run_core_audit()
        _save(args.output or STUDY / "outputs/algebra_audit.json", report)
    elif args.campaign:
        report = audit_campaign(args.campaign, output=args.output)
    else:
        parser.error("supply a campaign directory or --algebra")
    brief = (
        {"passed": report["passed"], "checks": {name: row.get("passed", False)
         for name, row in report["checks"].items()}}
        if args.algebra
        else {"passed": report["passed"], "source_changed": report["source_changed"],
              "sample_mismatches": report["sample_mismatches"], "groups": len(report["groups"])}
    )
    print(json.dumps(clean(brief), indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
