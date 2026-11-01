"""Present saved actual-CO2 attempts; no solver imports or simulation reruns.

``load_attempts`` returns compact records. ``render_markdown`` returns a report
fragment with links relative to the study root by default. The command-line
entry point writes presentation files under outputs/ and changes no run data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


STUDY = Path(__file__).resolve().parents[1]


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _maximum(values):
    finite = [
        float(v) for v in (values or []) if v is not None and math.isfinite(float(v))
    ]
    return max(finite) if finite else None


def _hashes(study, paths):
    return {
        str(path.relative_to(study))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in paths
        if path.is_file()
    }


def load_attempts(study=STUDY):
    """Read every saved 25/5-cell attempt, preserving incomplete budgets."""
    study = Path(study)
    records = []
    for directory in sorted((study / "outputs").glob("real_gas_steady*")):
        if not directory.is_dir():
            continue
        setup = _read(directory / "setup.json", {})
        invocation = _read(directory / "invocation.json", {})
        for result_path in sorted(directory.glob("*/result.json")):
            result = _read(result_path, {})
            history = _read(result_path.parent / "history.json", [])
            sampled = [row for row in history if "equation_norm" in row]
            latest = sampled[-1] if sampled else {}
            accepted = [row for row in history if row.get("phase") == "accepted"]
            audit = result.get("original_residual_audit", {})
            metrics = result.get("driver_metrics", {})
            work = result.get("work", {})
            calls = work.get("runtime_eos_calls", {})
            numerical_pairs = {
                name: count
                for name, count in calls.items()
                if name.startswith("numerical:") and count
            }
            ph_only = bool(numerical_pairs) and set(numerical_pairs) == {"numerical:PH"}
            budget = "TimeoutError" in result.get("exception", "")
            outcome = (
                "PASS"
                if result.get("success")
                else ("BUDGET — incomplete" if budget else "FAIL")
            )
            inner_tolerance = result.get("scheme", {}).get("newton_residual_tolerance")
            inner_audit = (
                bool(accepted)
                and inner_tolerance is not None
                and all(
                    row.get("inner_status") == 0
                    and row.get("inner_residual") is not None
                    and row["inner_residual"] <= inner_tolerance
                    for row in accepted
                )
            )
            global_values = audit.get(
                "global_defect",
                metrics.get("global_defect", latest.get("global_defect", [])),
            )
            records.append(
                dict(
                    attempt=directory.name,
                    case=result.get("case", result_path.parent.name),
                    cells=result.get("cells"),
                    outcome=outcome,
                    reason=result.get("reason"),
                    exception=result.get("exception"),
                    budget_incomplete=budget,
                    budget_seconds_per_case=invocation.get("max_wall_seconds_per_case"),
                    steady_verified=bool(result.get("success")),
                    final_state_saved=bool(audit)
                    and (result_path.parent / "final.npz").is_file(),
                    accepted_steps=work.get("accepted_steps", len(accepted)),
                    rejected_steps=work.get("rejected_steps"),
                    newton_corrections=work.get("newton_corrections_all_trials"),
                    accepted_newton_status_residual_audit=inner_audit,
                    last_inner_residual=(
                        accepted[-1].get("inner_residual") if accepted else None
                    ),
                    numerical_ph_calls=calls.get("numerical:PH", 0),
                    runtime_eos_calls=calls,
                    numerical_pairs=numerical_pairs,
                    numerical_ph_only=ph_only,
                    equation_l2=audit.get(
                        "equation_l2",
                        metrics.get(
                            "equation_l2",
                            latest.get("equation_l2", latest.get("equation_norm")),
                        ),
                    ),
                    overall_relative=metrics.get(
                        "overall_relative", latest.get("overall_relative")
                    ),
                    boundary_max=audit.get(
                        "boundary_max",
                        metrics.get("boundary_max", latest.get("boundary_max")),
                    ),
                    global_defect=global_values,
                    global_max=_maximum(global_values),
                    admissible=audit.get(
                        "operator_admissible",
                        metrics.get("admissible", latest.get("admissible")),
                    ),
                    positive_properties=audit.get(
                        "positive_pressure_density_temperature_sound_speed"
                    ),
                    model_audit=audit.get("model_audit_passed"),
                    all_equations_passed=audit.get(
                        "all_equations_passed",
                        metrics.get(
                            "all_equations_passed", latest.get("all_equations_passed")
                        ),
                    ),
                    boundary_passed=audit.get(
                        "boundary_passed",
                        metrics.get("boundary_passed", latest.get("boundary_passed")),
                    ),
                    global_passed=audit.get(
                        "global_passed",
                        metrics.get("global_passed", latest.get("global_passed")),
                    ),
                    source_unchanged=result.get("source_unchanged"),
                    source_aggregate_sha256=setup.get("source_aggregate_sha256"),
                    elapsed_s_diagnostic=result.get(
                        "elapsed_s_including_compile_audit"
                    ),
                    physical=result.get("physical"),
                    settings=result.get("settings"),
                    scheme=result.get("scheme"),
                    result_link=str(result_path.relative_to(study / "outputs")).replace(
                        "\\", "/"
                    ),
                    raw_json_sha256=_hashes(
                        study,
                        (
                            directory / "setup.json",
                            directory / "invocation.json",
                            directory / "source_manifest.json",
                            result_path,
                            result_path.parent / "history.json",
                            result_path.parent / "case_setup.json",
                        ),
                    ),
                    saved_state_sha256=_hashes(
                        study,
                        (
                            result_path.parent / "initial.npz",
                            result_path.parent / "final.npz",
                        ),
                    ),
                )
            )
    return sorted(
        records, key=lambda row: (-(row["cells"] or 0), row["case"], row["attempt"])
    )


def _number(value):
    return (
        "—"
        if value is None or not math.isfinite(float(value))
        else f"{float(value):.2e}"
    )


def _flag(value):
    return "—" if value is None else ("yes" if value else "no")


def _expected_policy(row):
    scheme, settings = row.get("scheme") or {}, row.get("settings") or {}
    residual = settings.get("residual", {})
    return (
        all(
            scheme.get(key) == value
            for key, value in {
                "newton_iterations": 10,
                "newton_backtracking": False,
                "newton_update_tolerance": 1e-10,
                "newton_residual_tolerance": 1e-12,
                "flux": "roe",
                "reconstruction": "constant",
                "time_integrator": "euler-implicit",
                "boundary_treatment": "coupled",
            }.items()
        )
        and all(
            settings.get(key) == value
            for key, value in {
                "cfl": 0.2,
                "cfl_growth": 1.2,
                "cfl_max": 1e6,
            }.items()
        )
        and all(
            residual.get(key) == value
            for key, value in {
                "criterion": "either",
                "relative_tolerance": 1e-6,
                "absolute_tolerance": 1e-9,
                "boundary_tolerance": 1e-9,
                "global_tolerance": 1e-9,
            }.items()
        )
    )


def render_markdown(attempts=None, *, study=STUDY, link_prefix="outputs/"):
    """Return a concise report fragment; never treat budget stops as passes."""
    attempts = load_attempts(study) if attempts is None else attempts
    if not attempts:
        return "No saved actual-CO₂ steady attempts are available.\n"
    passed = sum(row["steady_verified"] for row in attempts)
    budgets = sum(row["budget_incomplete"] for row in attempts)
    failures = len(attempts) - passed - budgets
    lines = [
        "## Actual-CO₂ steady verification",
        "",
        f"Saved attempts: **{passed} steady passes, {budgets} incomplete budget stops, {failures} other failures**.",
        "The 25-cell attempts remain in the record. A coarse-grid pass does not replace a finer-grid verification.",
        "",
        "Both models use CO₂/HEOS at reservoir 8 MPa / 320 K and backpressure 7.98 MPa, acoustic outlets, "
        "Roe flux with piecewise-constant reconstruction, coupled backward Euler, and their model-default cold initialization. "
        "The only follow-up change is cell count. No analytical flow-accuracy or runtime-ranking claim is made.",
        "",
        "| Case | Cells | Outcome | Accepted / rejected steps | Newton corrections | Numerical PH calls |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in attempts:
        label = f'[{row["case"]}]({link_prefix}{row["result_link"]})'
        lines.append(
            f'| {label} | {row["cells"]} | **{row["outcome"]}** | '
            f'{row["accepted_steps"]} / {row["rejected_steps"]} | '
            f'{row["newton_corrections"]} | {row["numerical_ph_calls"]:,} |'
        )
    lines += [
        "",
        "| Case / cells | Last equation L2 norms | BC max | Global max | Admissible | PH only | Source unchanged |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in attempts:
        norms = ", ".join(_number(v) for v in (row["equation_l2"] or []))
        lines.append(
            f'| {row["case"]} / {row["cells"]} | {norms} | '
            f'{_number(row["boundary_max"])} | {_number(row["global_max"])} | '
            f'{_flag(row["admissible"])} | {_flag(row["numerical_ph_only"])} | '
            f'{_flag(row["source_unchanged"])} |'
        )
    lines += [
        "",
        "For budget stops, the last sampled spatial residual/BC/global checks are shown; no final steady state "
        "or final model audit is claimed. Full histories and the timeout exception are retained. "
        "The nominal 600-second budget is checked between driver trials, so a trial can finish after that limit.",
        "",
    ]
    for row in attempts:
        lines.append(
            f'- {row["case"]}, {row["cells"]} cells: active relative norm '
            f'{_number(row["overall_relative"])}; accepted Newton status/residual audit '
            f'{_flag(row["accepted_newton_status_residual_audit"])}; '
            f'last inner residual {_number(row["last_inner_residual"])}; '
            f'positive-property final audit {_flag(row["positive_properties"])}; '
            f'final model audit {_flag(row["model_audit"])}.'
        )
    hashes = {
        row["source_aggregate_sha256"]
        for row in attempts
        if row["source_aggregate_sha256"]
    }
    if not all(_expected_policy(row) for row in attempts):
        lines += [
            "",
            "**Some recorded settings differ from the study policy below. Inspect their linked JSON records before comparing results.**",
        ]
    lines += [
        "",
        "Study policy: up to 10 undamped Newton corrections; update tolerance 1e-10 and inner residual "
        "tolerance 1e-12; equation stopping is relative 1e-6 or absolute 1e-9, with independent "
        "BC/global limits 1e-9. CFL starts at 0.2 and grows by 1.2 to at most 1e6. "
        "Accepted-step Newton convergence is distinct from convergence of the original steady spatial equations.",
    ]
    if len(hashes) == 1:
        lines += [
            "",
            f"All saved attempts record production source aggregate `{next(iter(hashes))}`.",
        ]
    elif hashes:
        lines += [
            "",
            "**Source aggregates differ across attempts; inspect the individual manifests.**",
        ]
    lines += [
        "",
        "The host bridge performs scalar PH flashes and evaluates base/perturbed properties for its existing "
        "finite-difference JVP. This explains substantial recorded EOS work, but external CPU workloads were present "
        "and no matched flux/backend comparison was performed. These data do not attribute cost to Roe alone.",
        "",
    ]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    attempts = load_attempts(args.study)
    outputs = args.study / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "real_gas_summary.json").write_text(
        json.dumps(attempts, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(attempts, study=args.study, link_prefix="")
    (outputs / "real_gas_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
