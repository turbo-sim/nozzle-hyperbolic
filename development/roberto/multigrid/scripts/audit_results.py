"""Independently re-evaluate the saved EulerNozzle grid-sequencing experiment.

No solve is repeated here. Saved full states, including their boundary values,
are evaluated with the unchanged physical source residual before any boundary
completion. Failed runs are retained and reported explicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np

from development.roberto.BC_residual.scripts.boundary_system import boundary_constraints
from development.roberto.BC_residual.scripts.source_adapter import independent_ghost_operators
from development.roberto.euler_nozzle import verify_euler_nozzle_steady as baseline
from development.roberto.newton_solve.scripts.problem import SteadyNozzleProblem


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
CASE_INDEX = {"shock35": 1, "shock70": 2, "subsonic": 4}
ROOT_TOL = 1e-8
MATCH_TOL = 1e-4


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def scaled_difference(first, second, scale):
    return float(np.max(np.abs(np.asarray(first)-np.asarray(second))/scale))


def shock_proxy(x, velocity):
    """Strongest downstream compression; not unique transient shock tracking."""
    x, velocity = np.asarray(x), np.asarray(velocity)
    centers = (x[1:]+x[:-1])/2
    gradient = np.diff(velocity)/np.diff(x)
    valid = (centers > x[-1]/2) & (centers < x[-2])
    if not np.any(valid):
        return None
    selected = np.flatnonzero(valid)[np.argmin(gradient[valid])]
    return float(centers[selected])


def source_audit(summary):
    before = summary["source_hashes_before"]
    after = summary["source_hashes_after"]
    current = {}
    for relative in before:
        path = REPO/relative
        current[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"pass": bool(before and before == after == current),
            "files": len(before), "before_equals_after": before == after,
            "current_equals_recorded": current == before}


def audit(output):
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    setup_lookup = {(s["case_id"], s["reconstruction"]): s for s in summary["setups"]}
    cache = {}

    def problem_for(case_id, reconstruction, points):
        key = case_id, reconstruction, int(points)
        if key not in cache:
            setup = setup_lookup[case_id, reconstruction]
            problem = SteadyNozzleProblem(
                baseline.CASES[CASE_INDEX[case_id]], n_points=int(points),
                solver_settings=setup["settings"],
            )
            constraints = boundary_constraints(problem, independent_ghost_operators(problem))
            cache[key] = problem, constraints
        return cache[key]

    records, saved = [], []
    for row in summary["runs"]:
        result = {"file": row["file"], "case_id": row["case_id"],
                  "reconstruction": row["reconstruction"], "method": row["method"],
                  "repeat": row["repeat"], "excluded": row["excluded"]}
        try:
            run = json.loads((output/row["file"]).read_text(encoding="utf-8"))
            stages = run["stages"]
            problem, constraints = problem_for(row["case_id"], row["reconstruction"], stages[-1]["n_points"])
            full = np.asarray(stages[-1]["final_state"], dtype=float)
            state = jnp.asarray(full)
            physical = np.asarray(problem.operators["physical_residual"](state, state, jnp.float64(1))).reshape(-1, 3)
            scale = problem.residual_scale.reshape(-1, 3)
            rms = float(np.sqrt(np.mean((physical/scale)**2)))
            complete = np.asarray(problem.operators["state_from_interior"](state[1:-1].ravel()))
            primitive_scale = np.array([problem.p0, problem.a0, problem.h0])
            bc_map = scaled_difference(full[[0, -1]], complete[[0, -1]], primitive_scale)
            bc_equations = float(np.max(np.abs(np.asarray(constraints(state)))))
            props = problem.fluid.get_state(jxp.HmassP_INPUTS, state[:, 2], state[:, 0])
            admissible = bool(np.isfinite(full).all() and np.all(full > 0))
            for key in ("rho", "a", "T"):
                values = np.asarray(props[key])
                admissible = admissible and bool(np.isfinite(values).all() and np.all(values > 0))
            mass_flux = np.asarray(problem.face_mass_flux(state))
            mass_imbalance = float(abs(mass_flux[-1]-mass_flux[0])/max(abs(mass_flux[0]), 1e-300))
            flux_scale = problem.face_area[0]*problem.rho0*problem.a0*np.array([1, problem.a0, problem.h0])
            balance = np.sum(physical, axis=0)/flux_scale
            max_balance = float(np.max(np.abs(balance)))
            fine_points = setup_lookup[row["case_id"], row["reconstruction"]]["n_points"]
            reached_fine = stages[-1]["n_points"] == fine_points
            root = bool(reached_fine and stages[-1]["native_success"]
                        and rms <= ROOT_TOL and bc_map <= ROOT_TOL and bc_equations <= ROOT_TOL
                        and admissible and mass_imbalance <= 1e-4 and max_balance <= 1e-5)
            count_checks, transfers, stage_checks = [], [], []
            for index, stage in enumerate(stages):
                history, trials = stage["history"], stage["trial_history"]
                stage_problem, stage_constraints = problem_for(row["case_id"], row["reconstruction"], stage["n_points"])
                stage_state = jnp.asarray(stage["final_state"], dtype=jnp.float64)
                stage_physical = np.asarray(stage_problem.operators["physical_residual"](stage_state, stage_state, jnp.float64(1))).ravel()
                stage_rms = float(np.sqrt(np.mean((stage_physical/stage_problem.residual_scale)**2)))
                declared = float(stage["tolerance"])
                stage_bc = float(np.max(np.abs(np.asarray(stage_constraints(stage_state)))))
                stage_consistent = bool(np.isclose(stage_rms, stage["history"][-1]["residual_rms"], rtol=1e-7, atol=1e-12)
                                        and bool(stage["native_success"]) == bool(stage_rms <= declared))
                stage_checks.append(dict(points=stage["n_points"], declared_tolerance=declared, original_spatial_rms=stage_rms,
                                         boundary_equation_max=stage_bc, native_success=stage["native_success"],
                                         residual_target_met=bool(stage_rms<=declared), status_consistent=stage_consistent))
                counted_steps = len(history)-1
                counted_trials = len(trials)
                counted_rejects = sum(not t["accepted"] for t in trials)
                count_checks.append(bool(
                    counted_steps == stage["iterations"]
                    and counted_trials == stage["trials"]
                    and counted_rejects == stage["rejected_trials"]
                    and counted_trials-counted_rejects == counted_steps
                ))
                if index:
                    previous = stages[index-1]
                    coarse, _ = problem_for(row["case_id"], row["reconstruction"], previous["n_points"])
                    target, _ = problem_for(row["case_id"], row["reconstruction"], stage["n_points"])
                    old = np.asarray(previous["final_state"])
                    interpolated = np.column_stack([np.interp(target.x, coarse.x, old[:, k]) for k in range(3)])
                    expected = np.asarray(target.operators["state_from_interior"](jnp.asarray(interpolated[1:-1].ravel())))
                    difference = scaled_difference(expected, stage["initial_state"], primitive_scale)
                    transfers.append({"points": stage["n_points"], "scaled_difference": difference,
                                      "pass": difference <= 1e-12})
            aggregate_counts = bool(
                sum(s["iterations"] for s in stages) == row["iterations"]
                and (stages[-1]["iterations"] if reached_fine else 0) == row["fine_iterations"]
                and sum(s["trials"] for s in stages) == row["trials"]
                and sum(s["rejected_trials"] for s in stages) == row["rejected_trials"]
                and np.isclose(sum(s["trials"]*(s["n_points"]-2)/(fine_points-2) for s in stages),
                               row["equivalent_fine_updates"], rtol=1e-12, atol=1e-12)
            )
            component_wall = sum(s["wall_s"]+s["transfer_s"] for s in stages)
            timing_sum = bool(row["wall_s"]+1e-6 >= component_wall)
            recorded_root = bool(row["root_success"])
            integrity = bool(root == recorded_root and all(count_checks) and aggregate_counts
                             and timing_sum and all(t["pass"] for t in transfers)
                             and all(s["status_consistent"] for s in stage_checks))
            result.update({"integrity_pass": integrity, "root": root,
                "recorded_root": recorded_root, "spatial_rms": rms,
                "boundary_map_max": bc_map, "boundary_equation_max": bc_equations,
                "admissible": admissible, "mass_imbalance": mass_imbalance,
                "integrated_balance": balance.tolist(), "conservation_max": max_balance,
                "stage_counts_pass": all(count_checks), "aggregate_counts_pass": aggregate_counts,
                "component_wall_s": component_wall, "timing_sum_pass": timing_sum,
                "transfers": transfers, "stage_residual_checks": stage_checks,
                "reached_fine": reached_fine,
                "shock_proxy_m": shock_proxy(problem.x[1:-1], full[1:-1, 1]) if row["case_id"].startswith("shock") else None})
            comparison_fields = {"p": full[:, 0]/problem.p0,
                "rho": np.asarray(props["rho"])/problem.rho0,
                "M": full[:, 1]/np.asarray(props["a"]),
                "v_m": full[:, 1]/problem.a0, "h": full[:, 2]/problem.h0}
            mach = comparison_fields["M"]
            boundary_branch = bool(0 < mach[0] < 1 and 0 < mach[-1] < 1)
            interior_branch = bool(np.max(mach[1:-1]) > 1 if row["case_id"].startswith("shock")
                                   else np.max(mach[1:-1]) < 1+1e-3)
            result["branch_check"] = {"pass": bool(boundary_branch and interior_branch),
                "inlet_M": float(mach[0]), "outlet_M": float(mach[-1]),
                "interior_max_M": float(np.max(mach[1:-1])),
                "boundary_subsonic": boundary_branch, "interior_branch": interior_branch}
            saved.append((result, full, primitive_scale, problem.dx, comparison_fields))
        except Exception as error:
            result.update(integrity_pass=False, root=False, error=f"{type(error).__name__}: {error}")
        records.append(result)

    comparisons = []
    for case_id, reconstruction in setup_lookup:
        measured = [s for s in saved if s[0]["case_id"] == case_id
                    and s[0]["reconstruction"] == reconstruction and not s[0]["excluded"]]
        single = [s for s in measured if s[0]["method"] == "single_grid"]
        if not single:
            comparisons.append({"case_id": case_id, "reconstruction": reconstruction,
                                "pass": False, "error": "No measured single-grid baseline"})
            continue
        anchor = single[0]
        for result, state, scale, dx, comparison_fields in measured:
            difference = scaled_difference(state, anchor[1], scale)
            shock_difference = (abs(result["shock_proxy_m"]-anchor[0]["shock_proxy_m"])/dx
                                if result["shock_proxy_m"] is not None else None)
            matches = bool(difference <= MATCH_TOL and (shock_difference is None or shock_difference <= 2))
            result.update(baseline_scaled_state_difference=difference,
                          baseline_shock_difference_cells=shock_difference, matched_solution=matches,
                          baseline_normalized_field_difference={key: float(np.max(np.abs(value-anchor[4][key])))
                                                                for key, value in comparison_fields.items()})
        for method in ("single_grid", "grid_sequence"):
            subset = [s for s in measured if s[0]["method"] == method]
            deterministic_difference = max((scaled_difference(s[1], subset[0][1], s[2]) for s in subset), default=None)
            comparisons.append({"case_id": case_id, "reconstruction": reconstruction, "method": method,
                "samples": len(subset), "repeat_scaled_state_difference": deterministic_difference,
                "pass": bool(subset and all(s[0]["root"] and s[0]["matched_solution"] for s in subset)
                             and deterministic_difference <= 1e-10)})

    sources = source_audit(summary["metadata"])
    measured_records = [r for r in records if not r["excluded"]]
    shared_before = summary["metadata"].get("shared_import_hashes_before")
    shared_after = summary["metadata"].get("shared_import_hashes_after")
    shared_current = ({name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in shared_before}
                      if shared_before else None)
    shared_pass = shared_before is None or shared_before == shared_after == shared_current
    audit_data = {"pass": bool(sources["pass"] and shared_pass and all(r["integrity_pass"] for r in records)
                               and measured_records and all(r["root"] for r in measured_records)
                               and all(r.get("branch_check", {}).get("pass", False) for r in measured_records)
                               and all(c["pass"] for c in comparisons)),
        "source": sources, "shared_imports_pass": shared_pass, "runs": len(records), "measured_runs": len(measured_records),
        "thresholds": {"spatial_rms": ROOT_TOL, "boundary_max": ROOT_TOL,
            "mass_imbalance": 1e-4, "integrated_balance": 1e-5,
            "scaled_primitive_agreement": MATCH_TOL, "shock_agreement_cells": 2},
        "results": records, "comparisons": comparisons}
    (output/"audit.json").write_text(json.dumps(json_safe(audit_data), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return audit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY/"outputs")
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({key: result[key] for key in ("pass", "runs", "measured_runs", "source")}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
