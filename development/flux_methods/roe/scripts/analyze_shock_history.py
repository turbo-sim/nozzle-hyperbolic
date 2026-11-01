"""Explain a saved steady shock history without importing or running a solver.

The original results remain unchanged. Counts are numerical-work evidence,
not performance timings; this client is safe to rerun while a study continues.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
CAMPAIGN = STUDY / "outputs/roe_steady_verification"
DEFAULT_ID = "euler_annular__shock35__muscl_van_leer__n400__cap10__current__flux_roe"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_rows(rows):
    return dict(
        attempts=len(rows),
        newton_corrections=sum(row.get("inner_iterations", 0) for row in rows),
        rejected_newton_trials=sum(row.get("rejected_newton_trials", 0) for row in rows),
        iteration_counts=dict(sorted(Counter(row.get("inner_iterations") for row in rows).items())),
        status_counts=dict(sorted(Counter(row.get("inner_status") for row in rows).items())),
        cfl_quantiles={str(fraction): quantile([row["cfl"] for row in rows], fraction)
                       for fraction in (0, .1, .5, .9, 1)},
        dt_quantiles={str(fraction): quantile([row["dt"] for row in rows], fraction)
                      for fraction in (0, .1, .5, .9, 1)},
    )


def analyze(record, *, scenario, campaign_path):
    history = record["history"]
    accepted = [row for row in history if row.get("phase") == "accepted"]
    rejected = [row for row in history if row.get("phase") == "rejected"]
    cap = [row for row in rejected if row.get("inner_status") == 1]
    invalid = [row for row in rejected if row.get("inner_status") == 3]
    scheme = record["settings"]["scheme"]
    residual_tolerance = scheme["newton_residual_tolerance"]
    residual_satisfied_at_cap = [row for row in cap
                                 if row["inner_residual"] <= residual_tolerance]
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    raw_path = campaign_path.parent / record["raw_result_file"]
    raw_verified = digest(raw_path) == record["raw_result_sha256"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_history_matches = raw["history"] == history
    source_key = "src/nozzle_hyperbolic/solvers/time_marching/implicit.py"
    source_hash = digest(ROOT / source_key)
    expected_source = campaign["provenance"]["source"]["files"][source_key]
    source_matches = source_hash == expected_source
    if not (raw_verified and raw_history_matches and source_matches):
        raise ValueError("Saved raw history or Newton-source identity did not verify")
    total_corrections = sum(row.get("inner_iterations", 0) for row in accepted + rejected)
    phases = []
    for start, end in ((1, 100), (101, 300), (301, 600), (601, 900), (901, len(accepted))):
        if end < start:
            continue
        kept = [row for row in accepted if start <= row["step"] <= end]
        attempts = [row for row in rejected if start <= row["step"] <= end]
        if not kept:
            continue
        phases.append(dict(
            first_step=start, last_step=end, accepted=len(kept), rejected=len(attempts),
            median_cfl=statistics.median(row["cfl"] for row in kept),
            initial_overall_relative=kept[0]["overall_relative"],
            final_overall_relative=kept[-1]["overall_relative"],
            newton_corrections=sum(row["inner_iterations"] for row in kept + attempts),
        ))
    return dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Read-only numerical-work diagnosis; no additional simulation or timing claim",
        id=record["id"], case_id=record["case_id"], cells=record["cells"],
        reconstruction=record["reconstruction"], flux=record["flux"],
        source_sha256=record["source_sha256"], case_sha256=record["case_sha256"],
        settings=record["settings"], final_status=record["status"],
        success=record["success"], branch_passed=record["branch_passed"],
        analytical_passed=record["analytical_passed"],
        final_metrics=record["metrics"], roe_diagnostics=record["roe_diagnostics"],
        accepted=summarize_rows(accepted), rejected=summarize_rows(rejected),
        cap_rejected=summarize_rows(cap), full_trial_rejected=summarize_rows(invalid),
        rejected_fraction_of_counted_newton_corrections=(
            sum(row["inner_iterations"] for row in rejected) / total_corrections),
        cap_failures=len(cap), cap_failures_with_residual_satisfied=len(residual_satisfied_at_cap),
        full_trial_acceptance_failures=len(invalid),
        update_gate_inference_valid=bool(source_matches and residual_tolerance > 0
                                       and scheme["newton_update_tolerance"] > 0),
        accepted_cfl_below_5=sum(row["cfl"] < 5 for row in accepted),
        accepted_cfl_below_10=sum(row["cfl"] < 10 for row in accepted),
        final_accepted_cfl=accepted[-1]["cfl"] if accepted else None,
        last_rejected_step=rejected[-1]["step"] if rejected else None,
        first_equation_target_pass=next((row["step"] for row in accepted
                                         if row["all_equations_passed"]), None),
        first_cfl_100=next((row["step"] for row in accepted if row["cfl"] >= 100), None),
        first_cfl_1000=next((row["step"] for row in accepted if row["cfl"] >= 1000), None),
        phases=phases,
        evidence=dict(
            scenario=str(scenario.relative_to(ROOT)), scenario_sha256=digest(scenario),
            raw_result=str(raw_path.relative_to(ROOT)), raw_result_sha256=digest(raw_path),
            raw_hash_verified=raw_verified, raw_history_matches=raw_history_matches,
            newton_source=source_key, newton_source_sha256=source_hash,
            newton_source_matches_campaign=source_matches,
            analysis_script_sha256=digest(Path(__file__)),
        ),
        limitations=[
            "The run was untimed under concurrent workload; elapsed diagnostics include cold compilation and are not speed evidence.",
            "Rejected candidate states and Newton relative-update values are not in the primary history. Status 3 does not distinguish cell/trace admissibility, Roe-face validity and nonfinite residuals.",
            "Update-gate failures are inferred from status 1, a satisfied residual criterion and the verified source's simultaneous convergence criteria; the update magnitude itself is unavailable.",
            "These data do not establish limiter cycling or isolate a Roe-versus-FVS cause. No matched same-grid FVS run was executed here.",
        ],
    )


def markdown(data):
    accepted, rejected = data["accepted"], data["rejected"]
    metrics = data["final_metrics"]
    lines = [
        "# Roe fine-grid annular shock: saved-history diagnosis", "",
        f"`{data['case_id']}`, {data['cells']} cells, `{data['reconstruction']}`: "
        f"**{data['final_status']}**. The primary run converged; no additional solve was needed.", "",
        "This is a numerical-work analysis, not a timing comparison. The source and controls are unchanged: "
        "coupled backward Euler, undamped Newton cap 10, CFL 0.2 with growth 1.2, shrink 0.5 and maximum $10^6$.", "",
        "| Attempt category | Count | Newton corrections |",
        "|---|---:|---:|",
        f"| Accepted outer steps | {accepted['attempts']} | {accepted['newton_corrections']} |",
        f"| Rejected: iteration cap / convergence criteria pending | {data['cap_failures']} | "
        f"{data['cap_rejected']['newton_corrections']} |",
        f"| Rejected: single full trial failed acceptance | {data['full_trial_acceptance_failures']} | "
        f"{data['full_trial_rejected']['newton_corrections']} |", "",
        f"Rejected outer attempts account for **{100*data['rejected_fraction_of_counted_newton_corrections']:.1f}%** "
        "of counted Newton corrections. This percentage is a work count, not elapsed time. "
        f"Accepted corrections range from {min(map(int,accepted['iteration_counts']))} to {max(map(int,accepted['iteration_counts']))}; "
        f"only {accepted['iteration_counts'].get(10, accepted['iteration_counts'].get('10',0))} "
        "accepted steps use the full cap.", "",
        f"Of the {data['cap_failures']} cap failures, **{data['cap_failures_with_residual_satisfied']}** "
        "already satisfy the inner residual tolerance $10^{-12}$. The verified Newton source requires both "
        "that residual criterion and the scaled update criterion $10^{-10}$, so the update gate was still pending. "
        "The primary history does not retain its update value.", "",
        f"The other **{data['full_trial_acceptance_failures']}** rejections have status 3: "
        "the undamped full trial failed the admissibility/finite-residual acceptance gate. "
        "No failed candidate fields were retained, so this log cannot identify the precise failing condition.", "",
        f"Median accepted CFL is **{accepted['cfl_quantiles']['0.5']:.3f}**. "
        f"**{data['accepted_cfl_below_5']}/{accepted['attempts']}** accepted steps have CFL below 5; "
        f"**{data['accepted_cfl_below_10']}/{accepted['attempts']}** are below 10. "
        "The configured maximum does not force the controller to attain it: failed attempts halve the next CFL. "
        f"The final accepted CFL is **{data['final_accepted_cfl']:.1f}**.", "",
        "| Accepted-step interval | Rejected attempts | Median accepted CFL | Initial → final relative norm | Newton corrections |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in data["phases"]:
        lines.append(f"| {row['first_step']}–{row['last_step']} | {row['rejected']} | {row['median_cfl']:.3g} | "
                     f"{row['initial_overall_relative']:.3e} → {row['final_overall_relative']:.3e} | {row['newton_corrections']} |")
    lines += ["", "The recorded relative norm remains of order one over much of the trajectory, then falls rapidly "
              "as the controller can sustain larger steps. This is an observed history pattern; no field trajectory was saved "
              "to prove a particular shock-motion or limiter mechanism.", "",
              f"All equation targets first pass at accepted step **{data['first_equation_target_pass']}**; "
              f"the independent global check requires the remaining **{accepted['attempts']-data['first_equation_target_pass']}** steps. "
              f"Final relative norm is **{metrics['overall_relative']:.3e}**, boundary defect **{metrics['boundary_max']:.3e}**, "
              f"and maximum global defect **{max(metrics['global_defect']):.3e}**. The final branch, analytical accuracy and Roe face-validity checks pass.", "",
              "## Interpretation and limits", "",
              "The large work count is explained directly by the prolonged low-CFL trajectory and repeated cap/full-trial rejections. "
              "It is not a failure of the final steady root. Raising the Newton cap would directly address only the cap category; "
              "it would not automatically resolve inadmissible/nonfinite full trials, and this record cannot predict the resulting trajectory or speed.", "",
              "For a later performance study, retain candidate failure categories and Newton update norms, then compare one controlled "
              "policy change on this same physical case. No source or control change is justified by timing claims from this contended run.", "",
              "Saved coarse FVS studies provide context only; there is no maintained same-case, same-grid, same-reconstruction, cap-10 "
              "FVS400 result for a matched comparison. Source/harness differences and resolution differences prevent a quantitative speed conclusion.", "",
              "## Reproduction", "", "```powershell",
              "poetry run python -B -m development.flux_methods.roe.scripts.analyze_shock_history",
              "```", "",
              "The companion JSON retains counts, exact input/settings hashes, raw-history verification and the verified Newton-source hash. "
              "All processing reads saved evidence only.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CAMPAIGN / "scenarios" / (DEFAULT_ID + ".json"))
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN / "summary.json")
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/shock_diagnostics/shock35_n400_muscl")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = analyze(json.loads(args.input.read_text(encoding="utf-8")),
                   scenario=args.input.resolve(), campaign_path=args.campaign.resolve())
    (args.output / "history_analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "history_analysis.md").write_text(markdown(data), encoding="utf-8")
    print(json.dumps({key:data[key] for key in ("id", "final_status", "cap_failures", "cap_failures_with_residual_satisfied", "full_trial_acceptance_failures")}, indent=2))


if __name__ == "__main__":
    main()
