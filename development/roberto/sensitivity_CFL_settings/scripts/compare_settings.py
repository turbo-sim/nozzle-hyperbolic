"""Newton-cap and CFL sensitivity using the current shared FVM driver.

Run from the repository root with
``poetry run python -B -m development.roberto.sensitivity_CFL_settings.scripts.compare_settings``.
New results use a separate schema/output directory from the old nodal campaign.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import jaxprop as jxp

from development.roberto.sensitivity_backpressure.scripts.problem import (
    CASE_IDS,
    ImplicitNozzle,
    SCHEMA,
    scheme,
)
from development.roberto.sensitivity_backpressure.scripts.run_sweep import hashes
from nozzle_hyperbolic.verification import build_case

STUDY = Path(__file__).resolve().parents[1]
OUTPUT = STUDY / "outputs" / "shared_fvm"
CAPS = (1, 3, 5, 10, 100)
# The production driver grows accepted CFLs and shrinks rejected attempts.
# It has no SER exponent; no obsolete SER option is silently approximated.
CFL_VARIANTS = {
    "baseline": dict(cfl=0.2, growth=1.2, maximum=1e6),
    "start1": dict(cfl=1.0, growth=1.2, maximum=1e6),
    "start100": dict(cfl=100.0, growth=1.2, maximum=1e6),
    "max100": dict(cfl=0.2, growth=1.2, maximum=100.0),
    "max100000": dict(cfl=0.2, growth=1.2, maximum=1e5),
    "fixed_cfl": dict(cfl=0.2, growth=1.0, maximum=1e6),
    "growth14": dict(cfl=0.2, growth=1.4, maximum=1e6),
    "growth105": dict(cfl=0.2, growth=1.05, maximum=1e6),
    "growth2": dict(cfl=0.2, growth=2.0, maximum=1e6),
}


def safe(value):
    """Convert host diagnostics to strict, portable JSON."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def write_index(report, output):
    dump(output / "summary.json", report)
    rows = report["runs"]
    if rows:
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(safe(rows))


def measure(problem, pressure, case, cap, variant, output, max_updates):
    """Replay a cold field with warm bindings and retain every rejected attempt."""
    initial = problem.cold_initial(pressure)
    history = []
    start = perf_counter()

    def record(row):
        history.append(dict(row, elapsed_s=perf_counter() - start))

    result = problem.solve(
        initial, pressure, max_updates, callback=record, **CFL_VARIANTS[variant]
    )
    run_id = f"{case}_n{cap}_{variant}"
    exact = problem.reference.profile(problem.x, pressure / problem.reference.p0)
    properties = problem.fluid.get_state(
        jxp.HmassP_INPUTS, result.state[:, 2], result.state[:, 0]
    )
    fields = dict(
        p=result.state[:, 0],
        rho=np.asarray(properties["rho"]),
        M=result.state[:, 1] / np.asarray(properties["a"]),
    )
    errors = {
        key: float(
            np.mean(np.abs(value[1:-1] - exact[key][1:-1]))
            / (
                problem.reference.p0
                if key == "p"
                else problem.rho0 if key == "rho" else 1.0
            )
        )
        for key, value in fields.items()
    }
    trials = [row for row in history if row["phase"] != "initial"]
    brief = dict(
        run_id=run_id,
        case=case,
        cap=cap,
        cfl_variant=variant,
        success=result.success,
        reason=result.reason,
        accepted_steps=sum(row["accepted"] for row in trials),
        rejected_steps=sum(not row["accepted"] for row in trials),
        inner_iterations=sum(row["inner_iterations"] for row in trials),
        spatial_max=float(np.max(result.metrics["equation_norm"])),
        boundary_max=result.metrics["boundary_max"],
        global_max=float(np.max(result.metrics["global_defect"])),
        analytical_pass=all(value <= 0.03 for value in errors.values()),
        solve_s=result.elapsed_s,
        initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        file=f"runs/{run_id}.json",
    )
    # Relative updates are not exposed by the production driver. Do not infer
    # them or call the accepted outer count the total Newton work.
    dump(
        output / brief["file"],
        dict(
            schema=SCHEMA,
            **brief,
            state=result.state,
            history=history,
            metrics=result.metrics,
            analytical_errors=errors,
            fields=fields,
            controller=CFL_VARIANTS[variant],
            scheme=asdict(problem.scheme),
        ),
    )
    return brief


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--points",
        type=int,
        default=201,
        help="Total states: physical cells plus two endpoint traces",
    )
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument(
        "--cases", nargs="+", choices=list(CASE_IDS), default=list(CASE_IDS)
    )
    parser.add_argument("--caps", nargs="+", type=int, default=list(CAPS))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One subsonic case, caps 1/10, baseline CFL",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Accepted for CLI compatibility; rendering is offline",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        from .current_report import render

        render(output)
        return
    if args.points < 4 or args.max_iterations < 1 or any(cap < 1 for cap in args.caps):
        parser.error(
            "Use at least four points, positive Newton caps and a positive step budget"
        )
    if (output / "summary.json").exists():
        parser.error("Select a fresh --output directory for a new numerical campaign")
    report = dict(
        schema=SCHEMA,
        settings=vars(args),
        versions={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxprop", "numpy")
        },
        source_before=hashes(),
        cases={},
        runs=[],
        compilation=[],
        controller="accepted-step CFL growth and rejected-step shrink; no SER",
        cells=args.points - 2,
    )
    case_ids = ["subsonic"] if args.smoke else args.cases
    caps = [1, 10] if args.smoke else args.caps
    for case in case_ids:
        physical = build_case("euler_nozzle/" + CASE_IDS[case], cells=args.points - 2)
        pressure = physical.back_pressure
        for cap in caps:
            problem = ImplicitNozzle(args.points, numerical_scheme=scheme(cap))
            initial = problem.cold_initial(pressure)
            compile_s = problem.warm_up(initial, pressure)
            report["compilation"].append(
                dict(case=case, cap=cap, first_call_s=compile_s)
            )
            report["cases"][case] = dict(
                x_m=problem.x,
                reference_pressure=problem.reference.profile(
                    problem.x, pressure / problem.reference.p0
                )["p"],
                physical_case=physical.definition.to_dict(),
            )
            variants = (
                list(CFL_VARIANTS)
                if cap in (1, 100) and not args.smoke
                else ["baseline"]
            )
            for variant in variants:
                print(f"{case}, Newton cap {cap}, {variant}", flush=True)
                row = measure(
                    problem, pressure, case, cap, variant, output, args.max_iterations
                )
                report["runs"].append(row)
                write_index(report, output)
        seeds = {r["initial_sha256"] for r in report["runs"] if r["case"] == case}
        assert (
            len(seeds) == 1
        ), "Sensitivity runs must use identical physical cold seeds"
    report["source_after"] = hashes()
    report["source_unchanged"] = report["source_before"] == report["source_after"]
    write_index(report, output)
    from .current_report import render

    render(output)
    if not report["source_unchanged"]:
        raise RuntimeError(
            "Production source changed during the campaign; results retain the mismatch"
        )
    if not all(row["success"] for row in report["runs"]):
        raise SystemExit(
            "Some solves failed; all states and rejected work were retained in "
            + str(output)
        )
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
