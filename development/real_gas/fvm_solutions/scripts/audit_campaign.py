"""Audit completeness and consistency of saved campaign evidence, without solves."""
import json
from itertools import product
import numpy as np

from .common import OUTPUT, identity, save
from .profiles import compatible


def main():
    campaign = json.loads((OUTPUT / "campaign.json").read_text(encoding="utf-8"))
    settings = campaign["settings"]
    rows = campaign["results"]
    expected = set(product(
        [f + "_" + r for f, r in product(settings["fluid_cases"], settings["regimes"])],
        settings["fluxes"], settings["reconstructions"], settings["grids"],
    ))
    actual = [(r["case_id"], r["flux"], r["reconstruction"], r["cells"]) for r in rows]
    checks = dict(
        expected_configuration_matrix=set(actual) == expected and len(actual) == len(expected),
        all_stages_completed=all(r["reason"] != "not_completed" for r in rows),
        unique_run_ids=len({r["id"] for r in rows}) == len(rows),
        source_unchanged=campaign["metadata"]["src_unchanged"],
        analytical_references_verified=all(c["baseline_audit"]["passed"] for c in campaign["cases"]),
    )
    details = []
    source_id = identity(campaign["metadata"]["source_before"])
    profile_id = identity(settings)
    for row in rows:
        if row["reason"] == "not_completed":
            continue
        state = np.asarray(row["state"], dtype=float)
        tests = dict(
            same_source=row["source_sha256"] == source_id,
            same_effective_case_settings=compatible(
                row["profile_sha256"], profile_id,
                row["case_id"][:-(len("adapted")+1)] if row["case_id"].endswith("_adapted")
                else row["case_id"][:-(len("shock")+1)],
                "adapted" if row["case_id"].endswith("_adapted") else "shock"),
            full_state_shape=state.shape == (row["cells"] + 2, 3),
            finite_full_state=bool(np.isfinite(state).all()),
            verified_label=bool(row["verified"]) == bool(row["success"] and row["audit"].get("passed", False)),
            failed_attempts_retained=all(not a["success"] and "state" in a and "history" in a for a in row["attempts"]),
            native_success_is_converged=not row["success"] or row["reason"] == "converged",
            native_flags_consistent=not row["success"] or all(row["metrics"].get(k, False) for k in
                ("finite", "admissible", "all_equations_passed", "boundary_passed", "global_passed")),
        )
        for name, values in row.get("fields", {}).items():
            tests[f"field_{name}_shape"] = len(values) == row["cells"]
        for name, error in row.get("errors", {}).items():
            tests[f"error_{name}_finite_nonnegative"] = all(
                v is not None and np.isfinite(v) and v >= 0
                for k, v in error.items() if k.startswith("normalized_")
            )
        details.append(dict(id=row["id"], passed=all(tests.values()),
                            failed_checks=[k for k, passed in tests.items() if not passed]))
    checks["completed_record_consistency"] = all(r["passed"] for r in details)
    by_key = {(r["case_id"], r["flux"], r["reconstruction"], r["cells"]): r for r in rows}
    sequencing = []
    for case_id, flux, reconstruction in product(
            [c["id"] for c in campaign["cases"]], settings["fluxes"], settings["reconstructions"]):
        previous = None
        for cells in settings["grids"]:
            row = by_key[(case_id, flux, reconstruction, cells)]
            initialization = row.get("initialization", {})
            prolonged = initialization.get("method") == "componentwise linear prolongation"
            passed = not prolonged or (previous is not None and previous["success"]
                                       and initialization.get("source_cells") == previous["cells"])
            sequencing.append(dict(id=row["id"], passed=bool(passed)))
            previous = row
    checks["prolongation_uses_previous_successful_numerical_grid"] = all(r["passed"] for r in sequencing)
    result = dict(
        passed=all(checks.values()), checks=checks,
        failed_checks=[k for k, passed in checks.items() if not passed],
        summary=campaign["summary"], records=details, grid_sequencing=sequencing,
        interpretation="A complete, internally consistent campaign may contain genuine nonlinear/accuracy failures; these remain report results.",
    )
    save(OUTPUT / "campaign_audit.json", result)
    print(json.dumps(dict(passed=result["passed"], failed_checks=result["failed_checks"], summary=campaign["summary"])))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
