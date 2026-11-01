"""Classify retained Sod results without changing any measurements/statuses.

Run after verify_sod, including when its strict aggregate exit status is1:
    poetry run python -B -m development.roe_method.scripts.summarize_sod

The original1e-8 analytic-transfer criterion and all original passed statuses
remain intact. Discrete flux-budget conservation is reported independently.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
        default=Path(__file__).resolve().parents[1]/"outputs/roe_transient.json")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not data.get("completed"):
        raise RuntimeError("Verification is not complete; keep the live output unchanged")
    numerical_passed, analytic_passed = 0, 0
    for row in data["records"]:
        if "conservative_errors" not in row:
            row["classification"] = dict(numerical_passed=False,
                                         analytic_boundary_transfer_passed=False)
            continue
        maximum = lambda values:max(abs(value) for value in values)
        numerical = bool(row["success"] and row["time"]==row["target_time"]
            and row["maximum_boundary_defect"]<=1e-9
            and row["maximum_reported_conservation_defect"]<=1e-9
            and maximum(row["independent_summed_transfer_budget"])<=1e-9
            and all(value>0 for value in row["minimum_over_11_output_frames"].values())
            and max(row["conservative_errors"]["L1"])<.08)
        analytic = maximum(row["exact_pre_boundary_interaction_budget"])<=1e-8
        row["classification"] = dict(numerical_passed=numerical,
            analytic_boundary_transfer_passed=analytic,
            actual_minus_exact_boundary_transfer=[a-b for a,b in zip(
                row["summed_numerical_transfer"],row["exact_transfer"])],
            normalized_actual_minus_exact_boundary_transfer=[a-b for a,b in zip(
                row["exact_pre_boundary_interaction_budget"],row["independent_summed_transfer_budget"])])
        numerical_passed += numerical
        analytic_passed += analytic
    data["assessment"] = dict(original_aggregate_passed=data["passed"],
        numerical_checks_passed=numerical_passed,
        analytic_boundary_transfer_checks_passed=analytic_passed,
        total_runs=len(data["records"]),
        all_errors_reduce_on_refinement=all(row["L1_reduced_in_all_equations"] for row in data["refinement"]),
        implicit_boundary_agreement_passed=all(row["passed"] for row in data["implicit_boundary_comparisons"]),
        interpretation="The finite-grid scheme can diffuse tiny tails to an endpoint before the exact waves arrive. "
            "Analytic pre-boundary-interaction transfer then differs from actual endpoint flux transfer. "
            "This difference is retained under the original1e-8 criterion and is distinct from discrete conservation, "
            "which compares total content change with actual accepted numerical transfers.",
        postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.input.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(data["assessment"],indent=2))


if __name__=="__main__":
    main()
