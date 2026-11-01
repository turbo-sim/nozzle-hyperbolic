"""Saved-data review only: no solver imports, state advances or timed work.

This evidence reproducer is outside the frozen five-file numerical client set.
The independent residual decisions below use saved reductions, not a second PDE
evaluation. The separate fresh-operator audit remains the solver-level check.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "development/flux_methods/stadtke/outputs/verification"
VARIANTS = {
    "stadtke_original": (False, False), "stadtke_jump": (True, False),
    "stadtke_endpoint": (False, True), "stadtke_both": (True, True),
}
THRESHOLD_PERCENT = 1e-8
ACCURACY_RELATIVE_TOLERANCE = 1e-8


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def review(row, item):
    fields, metrics = row["fields"], row["profile_metrics"]
    local, check = metrics["local_branch_audit"], row["independent_check"]
    policy, history = row["settings"]["residual"], row["history"]
    state, raw = np.asarray(row["state"]), np.asarray(row["raw_spatial_residual"])
    bc = np.asarray(row["boundary_residual"])
    pressure, velocity, h0 = (np.asarray(fields[k]) for k in ("p", "velocity", "h0"))
    h_ref = float(np.asarray(row["reference_point"]["h0"])[0])
    index = np.asarray(local["window_indices"], dtype=int)
    valid = (index > 0) & (index < len(pressure) - 1)
    interior = index[valid]
    down = local["downstream_branch"]
    high = np.maximum(local["upstream_branch"]["p"], down["p"])
    branch = float(np.max(np.maximum(0., pressure[index] - high) / abs(high)))
    peak = float(np.max(np.maximum(0., pressure[interior] - np.maximum(
        pressure[interior - 1], pressure[interior + 1])) / abs(np.asarray(down["p"])[valid])))
    dip = float(np.max(np.maximum(0., np.minimum(velocity[interior - 1],
        velocity[interior + 1]) - velocity[interior]) / abs(np.asarray(down["velocity"])[valid])))
    h_error = float(np.max(abs(h0 - h_ref)) / abs(h_ref))
    work = dict(
        accepted_steps=sum(h.get("phase") == "accepted" for h in history),
        rejected_attempts=sum(h.get("phase") == "rejected" for h in history),
        newton_corrections=sum(h.get("inner_iterations", 0) for h in history),
        rejected_newton_trials=sum(h.get("rejected_newton_trials", 0) for h in history),
    )
    norm = np.asarray(check["fixed_scale_reductions"][policy["norm"]])
    initial = np.asarray(check["initial_equation_norm"])
    relative = norm / np.where(initial > 0, initial, 1.)
    active = initial > policy["initial_floor"]
    absolute_ok = norm <= policy["absolute_tolerance"]
    relative_ok = active & (relative <= policy["relative_tolerance"])
    criterion = policy["criterion"]
    equation = (absolute_ok | relative_ok if criterion == "either" else
                absolute_ok & (~active | relative_ok) if criterion == "both" else
                np.where(active, relative_ok, absolute_ok) if criterion == "relative" else absolute_ok)
    finite = all(np.isfinite(v).all() for v in (state, raw, bc, pressure, velocity, h0, norm, initial))
    boundary = float(np.max(abs(bc))) <= policy["boundary_tolerance"]
    global_ok = max(map(abs, check["global_signed_fixed_scale"])) <= policy["global_tolerance"]
    original = finite and bool(check["admissible"]) and bool(equation.all()) and boundary and global_ok
    all_checks = original and bool(check["regime_branch_audit"])
    flags = VARIANTS[row["variant"]]
    accepted = [h["step"] for h in history if h.get("phase") == "accepted"]
    decisions = dict(
        input_digest=digest(CAMPAIGN / item["file"]) == item["sha256"],
        flags=tuple(row["scheme"][k] for k in ("custom_jump_correction", "custom_endpoint_envelope")) == flags == tuple(row[k] for k in ("custom_jump_correction", "custom_endpoint_envelope")),
        public_flux=row["scheme"]["flux"] == row["flux"] == "stadtke",
        cold_state=row["initialization"] == "cold",
        state_shape=state.shape == (row["cells"] + 2, len(row["metrics"]["equation_norm"])),
        finite=finite,
        positive_p_T_rho=all(np.all(np.asarray(fields[k]) > 0) for k in ("p", "T", "rho")),
        history_work=work == row["work"],
        accepted_steps=accepted == list(range(1, len(accepted) + 1)),
        history_endpoints=history[0]["phase"] == "initial" and (not row["success"] or history[-1]["phase"] == "accepted"),
        equation_mask=np.array_equal(equation, check["equation_passed"]),
        original_decision=original == check["independent_original_residual_checks_passed"],
        all_checks_decision=all_checks == check["all_checks_passed"],
        native_success_agrees=bool(row["success"]) == all_checks,
        branch_metric=bool(np.isclose(100 * branch, metrics["branch_excess_pct"], rtol=2e-13, atol=2e-13)),
        pressure_local_metric=bool(np.isclose(100 * peak, metrics["pressure_local_peak_pct"], rtol=2e-13, atol=2e-13)),
        velocity_local_metric=bool(np.isclose(100 * dip, metrics["velocity_local_dip_pct"], rtol=2e-13, atol=2e-13)),
        h0_metric=bool(np.isclose(h_error, metrics["h0_relative_max"], rtol=2e-13, atol=2e-13)),
    )
    return dict(
        file=item["file"], file_sha256=item["sha256"], case=row["case"], cells=row["cells"],
        reconstruction=row["reconstruction"], variant=row["variant"], alignment=row["alignment"],
        flags=list(flags), native_success=row["success"], native_reason=row["reason"],
        saved_independent_all=bool(check["all_checks_passed"]), checks=decisions, work=work,
        pressure_branch_excess_pct=100 * branch, pressure_local_neighbor_peak_pct=100 * peak,
        velocity_local_neighbor_dip_pct=100 * dip, h0_relative_linf_pct=100 * h_error,
        h0_min_relative_pct=100 * float(np.min(h0 / h_ref - 1)),
        h0_max_relative_pct=100 * float(np.max(h0 / h_ref - 1)),
        pressure_point_l1=row["errors"]["point"]["p"]["l1"],
        pressure_cell_l1=row["errors"]["cell"]["p"]["l1"],
        pressure_point_normalized_l1=row["errors"]["point"]["p"]["normalized_l1"],
        pressure_cell_normalized_l1=row["errors"]["cell"]["p"]["normalized_l1"],
        velocity_point_l1=row["errors"]["point"]["velocity"]["l1"],
        h0_point_l1=row["errors"]["point"]["h0"]["l1"], exception=row.get("exception"),
        saved_data_consistent=all(decisions.values()),
    )


def accuracy_statistics(pairs, metric):
    old = np.asarray([original[metric] for original, corrected in pairs])
    new = np.asarray([corrected[metric] for original, corrected in pairs])
    if np.any(old <= 0):
        raise ValueError("Error-ratio comparison requires positive baseline errors")
    ratios, delta = new / old, new - old
    worse = ratios > 1. + ACCURACY_RELATIVE_TOLERANCE
    better = ratios < 1. - ACCURACY_RELATIVE_TOLERANCE

    def stats(values):
        return (dict(min=float(np.min(values)), median=float(np.median(values)),
                     max=float(np.max(values))) if len(values) else None)

    largest = int(np.argmax(ratios))
    original, corrected = pairs[largest]
    return dict(count=len(pairs), count_worse=int(worse.sum()), count_better=int(better.sum()),
                count_within_relative_band=int((~worse & ~better).sum()),
                ratio_all=stats(ratios), ratio_worse_only=stats(ratios[worse]),
                absolute_delta_all=stats(delta), absolute_delta_worse_only=stats(delta[worse]),
                baseline_error=stats(old), corrected_error=stats(new),
                largest_ratio_example=dict(case=original["case"], cells=original["cells"],
                    reconstruction=original["reconstruction"], original_file=original["file"],
                    corrected_file=corrected["file"], original_error=float(old[largest]),
                    corrected_error=float(new[largest]), ratio=float(ratios[largest])))


def fresh_snapshot(records):
    path = CAMPAIGN / "fresh_audit.json"
    if not path.exists():
        return None
    payload = path.read_bytes()
    audit = json.loads(payload)
    native = {r["file"]: r["file_sha256"] for r in records}
    covered = [r for r in audit["records"] if r["file"] in native and r["file_sha256"] == native[r["file"]]]
    return dict(
        file=path.relative_to(ROOT).as_posix(), sha256=sha256(payload).hexdigest(),
        record_count=len(audit["records"]), hash_matched_native_rows=len(covered),
        native_rows_remaining=128 - len(covered), audit_valid=bool(audit["audit_valid"]),
        snapshot_source_match=bool(audit["snapshot_source_match"]),
        snapshot_client_match=bool(audit["snapshot_client_match"]),
        snapshot_case_match=bool(audit["snapshot_case_match"]),
        all_covered_checks_passed=all(r.get("all_checks_passed", False) for r in covered),
        complete_flag_relative_to_its_own_input=bool(audit["complete"]),
        all_128_native_rows_covered=len(covered) == 128,
    )


def main():
    summary = json.loads((CAMPAIGN / "summary.json").read_text())
    if len(summary["records"]) != 128:
        print("PENDING_NATIVE_COVERAGE", len(summary["records"]), 128)
        return
    records, raw_rows, anomalies = [], [], []
    for item in summary["records"]:
        row = json.loads((CAMPAIGN / item["file"]).read_text())
        raw_rows.append(row)
        try:
            record = review(row, item)
            records.append(record)
            if not record["saved_data_consistent"]:
                anomalies.append(dict(file=item["file"], failed_checks=[k for k, v in record["checks"].items() if not v]))
        except Exception as error:
            anomalies.append(dict(file=item["file"], postprocessing_error=repr(error), native_success=row.get("success"), reason=row.get("reason")))
    source, client = raw_rows[0]["source_sha256"], raw_rows[0]["implementation_sha256"]
    source_match = all(digest(ROOT / name) == value for name, value in source.items())
    client_match = all(digest(ROOT / "development/flux_methods/shared" / name) == value for name, value in client.items())
    epoch_match = all(r["source_sha256"] == source and r["implementation_sha256"] == client for r in raw_rows)
    case_match = all(digest(ROOT / "src/nozzle_hyperbolic/models" / r["case"].split("/")[0] / "cases" / (r["case"].split("/")[1] + ".yaml")) == r["case_sha256"] for r in raw_rows)
    keys = Counter((r["case"], r["cells"], r["reconstruction"], r["variant"]) for r in raw_rows)
    requested = Counter((r["case"], r["cells"], r["reconstruction"], r["variant"]) for r in summary["requested"])
    coverage = keys == requested and len(keys) == 128 and all(n == 1 for n in keys.values())
    groups = defaultdict(list)
    for r in raw_rows:
        groups[(r["case"], r["cells"], r["reconstruction"], r["alignment"])].append(r)
    mismatches = []
    for key, group in groups.items():
        common = ("initial_state", "settings", "source_sha256", "implementation_sha256", "case_sha256", "case_definition")
        equal = len(group) == 4 and all(all(r[k] == group[0][k] for r in group) for k in common)
        schemes = [{k: v for k, v in r["scheme"].items() if k not in ("custom_jump_correction", "custom_endpoint_envelope")} for r in group]
        if not equal or not all(s == schemes[0] for s in schemes):
            mismatches.append(list(key))
    aggregate = []
    for variant in VARIANTS:
        for reconstruction in ("constant", "muscl_van_leer"):
            group = [r for r in records if r["variant"] == variant and r["reconstruction"] == reconstruction]
            aggregate.append(dict(variant=variant, reconstruction=reconstruction, count=len(group),
                native_successes=sum(r["native_success"] for r in group),
                local_pressure_peak_count=sum(r["pressure_local_neighbor_peak_pct"] > THRESHOLD_PERCENT for r in group),
                local_velocity_dip_count=sum(r["velocity_local_neighbor_dip_pct"] > THRESHOLD_PERCENT for r in group),
                maximum_pressure_local_peak_pct=max((r["pressure_local_neighbor_peak_pct"] for r in group), default=None),
                maximum_pressure_branch_excess_pct=max((r["pressure_branch_excess_pct"] for r in group), default=None),
                maximum_h0_relative_linf_pct=max((r["h0_relative_linf_pct"] for r in group), default=None),
                work_totals={k: sum(r["work"][k] for r in group) for k in ("accepted_steps", "rejected_attempts", "newton_corrections", "rejected_newton_trials")}))
    by_config = defaultdict(dict)
    for r in records:
        by_config[(r["case"], r["cells"], r["reconstruction"], r["alignment"])][r["variant"]] = r
    comparisons, accuracy = [], []
    for variant in VARIANTS:
        if variant == "stadtke_original":
            continue
        pairs = [(g["stadtke_original"], g[variant]) for g in by_config.values() if "stadtke_original" in g and variant in g]
        comparisons.append(dict(variant=variant, matched_configurations=len(pairs),
            accepted_steps_fewer=sum(c["work"]["accepted_steps"] < o["work"]["accepted_steps"] for o, c in pairs),
            accepted_steps_more=sum(c["work"]["accepted_steps"] > o["work"]["accepted_steps"] for o, c in pairs),
            newton_corrections_fewer=sum(c["work"]["newton_corrections"] < o["work"]["newton_corrections"] for o, c in pairs),
            newton_corrections_more=sum(c["work"]["newton_corrections"] > o["work"]["newton_corrections"] for o, c in pairs),
            original_pressure_peak_removed=sum(o["pressure_local_neighbor_peak_pct"] > THRESHOLD_PERCENT and c["pressure_local_neighbor_peak_pct"] <= THRESHOLD_PERCENT for o, c in pairs),
            new_pressure_peak=sum(o["pressure_local_neighbor_peak_pct"] <= THRESHOLD_PERCENT and c["pressure_local_neighbor_peak_pct"] > THRESHOLD_PERCENT for o, c in pairs),
            h0_linf_worse=sum(c["h0_relative_linf_pct"] > o["h0_relative_linf_pct"] * (1 + 1e-10) + 1e-12 for o, c in pairs),
            pressure_point_l1_worse=sum(c["pressure_point_l1"] > o["pressure_point_l1"] * (1 + 1e-10) + 1e-14 for o, c in pairs),
            pressure_cell_l1_worse=sum(c["pressure_cell_l1"] > o["pressure_cell_l1"] * (1 + 1e-10) + 1e-14 for o, c in pairs)))
        metrics = ("pressure_cell_l1", "pressure_cell_normalized_l1", "h0_relative_linf_pct")
        per_case = []
        for case in sorted({original["case"] for original, corrected in pairs}):
            for reconstruction in ("constant", "muscl_van_leer"):
                subset = [(o, c) for o, c in pairs if o["case"] == case and o["reconstruction"] == reconstruction]
                per_case.append(dict(case=case, reconstruction=reconstruction,
                    metrics={metric: accuracy_statistics(subset, metric) for metric in metrics}))
        accuracy.append(dict(variant=variant,
            metrics={metric: accuracy_statistics(pairs, metric) for metric in metrics},
            per_case_reconstruction=per_case))
    result = dict(role="Independent saved-data assessment; no solver imported or state advanced",
        native_records=len(raw_rows), unique_configurations=len(keys), coverage_complete=coverage,
        all_native_succeeded=all(r["success"] for r in raw_rows), native_reasons=dict(Counter(r["reason"] for r in raw_rows)),
        source_epoch_unchanged=source_match and client_match and epoch_match and case_match,
        source_sha256=source, implementation_sha256=client,
        controls_match_except_flags=not mismatches, control_mismatches=mismatches,
        saved_data_review_passed=not anomalies and coverage and source_match and client_match and epoch_match and case_match and not mismatches,
        anomalies=anomalies, local_extremum_count_threshold_percent=THRESHOLD_PERCENT,
        aggregate=aggregate, comparisons_to_original=comparisons, accuracy_tradeoffs=accuracy,
        accuracy_comparison_relative_tolerance=ACCURACY_RELATIVE_TOLERANCE,
        accuracy_units=dict(pressure_cell_l1="Pa; volume-weighted primitive pressure error versus exact conservative cell projection",
            pressure_cell_normalized_l1="Pressure cell-L1 divided by the model reference pressure scale; dimensionless",
            h0_relative_linf_pct="Maximum absolute point total-enthalpy deviation as percent of analytical total enthalpy; deltas are percentage points"),
        finest_grid=[r for r in records if r["cells"] == 400], records=records,
        fresh_audit_at_postprocessing=fresh_snapshot(records),
        review_implementation_sha256=digest(Path(__file__)),
        independence_limit="Residual decisions use saved reductions and BC/global data, not a new flux evaluation. Fresh-operator audit coverage is separately bound by digest; its complete flag refers to its own input snapshot.",
        interpretation="Local one-cell extrema, branch-envelope errors, point-reference accuracy and iteration counts are separate. No elapsed-time speedup or universal no-overshoot theorem is inferred.",
        new_flow_solves=0, new_time_steps=0)
    target = Path(__file__).with_name("stadtke_matrix_postprocess.json")
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("SAVED_DATA_REVIEW", result["saved_data_review_passed"], "NATIVE", len(raw_rows), "ANOMALIES", anomalies)
    for r in aggregate:
        print(r)
    print("FRESH", result["fresh_audit_at_postprocessing"])
    print("SUMMARY_SHA", digest(target))


if __name__ == "__main__":
    main()
