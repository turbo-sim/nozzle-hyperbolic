"""Compare cold custom Newton and single-loop physical-storage Newton/PTC.

Run from the repository root::

    poetry run python -m development.roberto.newton_solve.scripts.investigate_ptc --no-show

All compared methods receive identical copied seeds. The implicit reference
is computed only after all direct/PTC runs for that case; it never initializes
them. First-call compilation and discarded warm-up updates are reported apart
from serial solve times. Only the original simple cold initialization is used.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[3]
CASE_IDS = ("supersonic", "shock35", "shock70", "sonic", "subsonic")
DISPLAY_CASES = ("subsonic", "sonic", "shock35", "shock70", "supersonic")
RESIDUAL_RMS_TOL = 1.0e-8
SAME_SOLUTION_RMS_TOL = 1.0e-4


def json_value(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON report value: {type(value)}")


def point_hash(point):
    import numpy as np

    return hashlib.sha256(np.asarray(point, dtype=np.float64).tobytes()).hexdigest()


def core_fingerprints():
    paths = (
        "src/nozzle_hyperbolic/models/euler_nozzle/fvm.py",
        "src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py",
        "src/nozzle_hyperbolic/numerics/reconstruction.py",
        "development/roberto/newton_solve/scripts/problem.py",
        "development/roberto/newton_solve/scripts/pseudo_time.py",
        "development/roberto/newton_solve/scripts/direct_solvers.py",
        "development/roberto/newton_solve/scripts/ptc.py",
        "development/roberto/newton_solve/scripts/ptc_storage.py",
        "development/roberto/newton_solve/scripts/investigate_ptc.py",
    )
    return {
        path: hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()
        for path in paths
    }


def save_csv(path, records):
    """Preserve scalar history fields and encode nested backend metadata as JSON."""
    flat = []
    for record in records:
        row = {}
        for name, value in record.items():
            row[name] = (
                json.dumps(value, default=json_value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value
            )
        flat.append(row)
    fields = list(dict.fromkeys(name for row in flat for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(flat)


def write_report(report, output):
    """Persist incomplete and completed runs without converting failures to roots."""
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, default=json_value), encoding="utf-8"
    )
    rows = []
    for case in report["cases"]:
        for run in comparison_runs(case):
            folder = output / case["id"] / run["phase"]
            folder.mkdir(parents=True, exist_ok=True)
            save_csv(
                folder / f"pressure_{run['method']}.csv",
                [
                    dict(x_m=float(x), p_Pa=float(p))
                    for x, p in zip(
                        case["x_m"][1:-1], run["diagnostics"]["fields"]["p"][1:-1]
                    )
                ],
            )
            rows.append(
                dict(
                    case=case["id"],
                    phase=run["phase"],
                    method=run["method"],
                    initial_cfl=run.get("initial_cfl"),
                    wall_s=run["wall_s"],
                    residual_rms=run["diagnostics"]["residual_rms"],
                    root_success=run["root_success"],
                    same_root=run.get("same_root"),
                    speedup_vs_implicit=run.get("speedup_vs_implicit"),
                    accepted_updates=len(run["history"]) - 1,
                    total_trials=(
                        len(run["trial_history"])
                        if "trial_history" in run
                        else run["backend"].get(
                            "ntrials", run["backend"].get("implicit_step_trials")
                        )
                    ),
                    nfev=run.get("nfev"),
                    njev=run.get("njev"),
                    analytical_pass=run["diagnostics"]["analytical_pass"],
                    message=run["message"],
                )
            )
    if rows:
        save_csv(output / "results.csv", rows)


def comparison_runs(case):
    """Return the three independent methods, including the implicit baseline."""
    runs = [dict(run) for run in case["runs"] if run["phase"] == "simple"]
    reference = case.get("implicit_references", {}).get("simple")
    if reference is not None:
        runs.append(dict(reference, method="implicit", phase="simple"))
    return sorted(
        runs, key=lambda run: ("implicit", "ptc", "newton").index(run["method"])
    )


def label(run):
    return {
        "implicit": "Implicit integration",
        "ptc": "Newton/PTC",
        "newton": "Controlled Newton",
    }[run["method"]]


def plot_report(report, output, no_show):
    """Two overview figures: one panel per case, three methods per panel."""
    import matplotlib

    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    cases = report["cases"]
    report["plot_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    rows = (len(cases) + 2) // 3
    figures = {}
    colors = dict(
        zip(
            ("implicit", "ptc", "newton"),
            plt.colormaps["plasma"](np.linspace(0.2, 0.8, 3)),
        )
    )
    styles = {"implicit": "-", "ptc": "--", "newton": ":"}
    titles = {
        "subsonic": "Subsonic throughout",
        "sonic": "Sonic throat / subsonic exit",
        "shock35": "Shock at 35% of divergence",
        "shock70": "Shock at 70% of divergence",
        "supersonic": "Adapted supersonic exit",
    }
    for kind in ("convergence_histories", "pressure_profiles"):
        fig, axes = plt.subplots(
            rows, 3, figsize=(14, 4 * rows), layout="constrained", squeeze=False
        )
        for axis, case in zip(axes.flat, cases):
            if "x_m" not in case:
                raise ValueError(
                    "Saved report lacks profile coordinates; rerun the comparison."
                )
            for run in comparison_runs(case):
                method = run["method"]
                text = label(run) + (
                    " (not converged)" if not run["root_success"] else ""
                )
                if kind == "convergence_histories":
                    history = run["history"]
                    axis.semilogy(
                        [row["accepted_index"] for row in history],
                        [max(row["residual_rms"], 1e-16) for row in history],
                        color=colors[method],
                        linestyle=styles[method],
                        label=text,
                        lw=1.5,
                    )
                    axis.plot(
                        history[-1]["accepted_index"],
                        max(history[-1]["residual_rms"], 1e-16),
                        marker="o" if run["root_success"] else "x",
                        color=colors[method],
                        ms=4,
                    )
                else:
                    axis.plot(
                        np.asarray(case["x_m"])[1:-1],
                        np.asarray(run["diagnostics"]["fields"]["p"])[1:-1] / 1000,
                        color=colors[method],
                        linestyle=styles[method],
                        label=text,
                        lw=1.5,
                    )
            if kind == "convergence_histories":
                axis.set_xscale("symlog", linthresh=10)
                axis.axhline(report["residual_rms_tol"], color="0.4", ls="-.", lw=0.8)
                axis.set(
                    xlabel="Accepted outer update (symlog)",
                    ylabel="Original steady residual RMS",
                )
            else:
                axis.plot(
                    case["exact_x_m"],
                    np.asarray(case["exact_pressure_Pa"]) / 1000,
                    color="black",
                    ls="-.",
                    lw=1,
                    label="Analytic reference",
                )
                axis.set(xlabel="x [m]", ylabel="Pressure [kPa]")
            axis.set_title(
                titles[case["id"]]
                + f"\n$p_b/p_0={case['back_pressure']/case['reservoir_pressure']:.6f}$"
                + (
                    "\nControlled Newton: not converged"
                    if any(
                        run["method"] == "newton" and not run["root_success"]
                        for run in comparison_runs(case)
                    )
                    else ""
                )
            )
            axis.grid(alpha=0.2)
            if len(cases) == axes.size and case is cases[0]:
                axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(cases) :]:
            axis.axis("off")
        if len(cases) < axes.size:
            axis = axes.flat[-1]
            handles = [
                Line2D(
                    [], [], color=colors[m], ls=styles[m], label=label(dict(method=m))
                )
                for m in ("implicit", "ptc", "newton")
            ]
            if kind == "convergence_histories":
                handles.extend(
                    [
                        Line2D([], [], color="0.4", ls="-.", label="Root threshold"),
                        Line2D(
                            [],
                            [],
                            color="0.4",
                            marker="o",
                            ls="none",
                            label="Converged endpoint",
                        ),
                        Line2D(
                            [],
                            [],
                            color="0.4",
                            marker="x",
                            ls="none",
                            label="Unconverged endpoint",
                        ),
                    ]
                )
            else:
                handles.append(
                    Line2D([], [], color="black", ls="-.", label="Analytic reference")
                )
            axis.legend(handles=handles, loc="center", frameon=False)
        fig.suptitle(
            "Steady nozzle: "
            + (
                "convergence of three solve approaches"
                if kind == "convergence_histories"
                else "final returned pressure fields (failed runs labeled)"
            )
        )
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{kind}.{extension}", dpi=180)
        figures[kind] = fig
    write_report(report, output)
    write_overview(report, output)
    if no_show:
        for fig in figures.values():
            plt.close(fig)
    else:
        plt.show()


def write_overview(report, output):
    """Keep the two overview figures and failed/converged status easy to review."""
    lines = [
        "# Five-case steady-nozzle solver comparison",
        "",
        "All approaches start independently from the same original cold field.",
        'Controlled Newton uses custom PySolverView `library="pysolver"`,',
        '`method="newton_raphson"`, with initial fraction and growth controls plus backtracking.',
        "Newton/PTC uses the same custom backend for its shifted corrections.",
        "",
        "## Convergence histories",
        "",
        "![Five-case convergence histories](convergence_histories.png)",
        "",
        "[Vector PDF](convergence_histories.pdf). Each panel is one backpressure; colors and line styles identify the three approaches.",
        "The x-axis counts accepted outer updates, including the initial state at zero.",
        "Its symmetric-log scale is linear from zero to ten updates and logarithmic thereafter, so short Newton paths remain visible beside long PTC histories.",
        "Inner implicit Newton work and rejected trials are excluded from that count; equal counts do not mean equal cost.",
        "All methods use the same fixed-scale original-residual RMS target. Curves show actual residuals, not running minima.",
        "",
        "## Final returned pressure fields",
        "",
        "![Five-case pressure comparison](pressure_profiles.png)",
        "",
        "[Vector PDF](pressure_profiles.pdf). Black curves are the analytic references.",
        "Unconverged runs are explicitly labeled and show their last returned field, not a claimed steady solution.",
        "Ghost-cell values are excluded from numerical curves. Analytic agreement is audited separately from root convergence.",
        "",
        "## Results",
        "",
        "| Case | Method | Updates | Original RMS | Root converged | Analytic check | Same implicit root | Solve [s] |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for case in report["cases"]:
        for run in comparison_runs(case):
            d = run["diagnostics"]
            same = run.get("same_root")
            if run["method"] == "implicit":
                same = run["root_success"]
            lines.append(
                f"| {case['id']} | {label(run)} | {len(run['history'])-1} | "
                f"{d['residual_rms']:.3e} | {run['root_success']} | {d['analytical_pass']} | "
                f"{same} | {run['wall_s']:.3f} |"
            )
    counts = {
        method: sum(
            run["root_success"]
            for case in report["cases"]
            for run in comparison_runs(case)
            if run["method"] == method
        )
        for method in ("implicit", "ptc", "newton")
    }
    lines[2:2] = [
        f"**Original-residual targets reached:** implicit {counts['implicit']}/{len(report['cases'])}, "
        f"Newton/PTC {counts['ptc']}/{len(report['cases'])}, controlled Newton {counts['newton']}/{len(report['cases'])}.",
        "",
    ]
    lines += [
        "",
        "[Full JSON, including backend options and audits](summary.json); [measurements CSV](results.csv).",
        "Per-case subdirectories contain accepted histories, rejected PTC trials, and final pressure CSVs.",
        "JAX warmup is excluded from solve timings. These are single serial implementation measurements, not a repeated speed benchmark.",
        "Core source hashes identify the measured code; plot_source_sha256 identifies the rendering script after plot-only changes.",
        "The sonic-limit analytic discrepancy is a spatial-verification issue and can persist even when all original residual targets are met.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=DISPLAY_CASES,
        default=list(DISPLAY_CASES),
    )
    parser.add_argument("--initial-cfl", type=float, default=1.0)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--newton-max-iterations", type=int, default=300)
    parser.add_argument("--implicit-max-iterations", type=int, default=5000)
    parser.add_argument("--first-step-fraction", type=float, default=0.25)
    parser.add_argument("--step-growth-factor", type=float, default=1.5)
    parser.add_argument("--minimum-cfl", type=float, default=1e-6)
    parser.add_argument("--maximum-cfl", type=float, default=1e8)
    parser.add_argument("--ser-exponent", type=float, default=0.7)
    parser.add_argument("--cfl-growth-limit", type=float, default=1.5)
    parser.add_argument("--cfl-shrink-limit", type=float, default=0.5)
    parser.add_argument("--max-relative-step", type=float, default=0.15)
    parser.add_argument("--nonmonotone-window", type=int, default=10)
    parser.add_argument("--residual-growth-limit", type=float, default=2.0)
    parser.add_argument("--model-error-limit", type=float, default=1.0)
    parser.add_argument("--newton-switch-ratio", type=float, default=1e-3)
    parser.add_argument("--max-retries", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DIRECTORY.parent / "outputs")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.phases = ["simple"]
    if (
        args.points < 5
        or min(
            args.max_iterations,
            args.newton_max_iterations,
            args.implicit_max_iterations,
        )
        < 1
    ):
        parser.error("Use at least five points and positive iteration budgets")
    import math

    if not math.isfinite(args.initial_cfl) or args.initial_cfl <= 0:
        parser.error("Initial CFL must be finite and positive")
    args.output = args.output.resolve()
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        report = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        plot_report(report, args.output, args.no_show)
        return

    import numpy as np
    from development.roberto.euler_nozzle import verify_euler_nozzle_steady as baseline
    from development.roberto.newton_solve.scripts.direct_solvers import solve_newton
    from development.roberto.newton_solve.scripts.problem import SteadyNozzleProblem
    from development.roberto.newton_solve.scripts.pseudo_time import solve_implicit
    from development.roberto.newton_solve.scripts.ptc import solve_ptc
    from development.roberto.newton_solve.scripts.ptc_storage import PhysicalStorage

    class LinearWarmup:
        def residual(self, z):
            return np.asarray(z) - [1.0, 2.0]

        def gradient(self, z):
            return np.eye(2)

    began = perf_counter()
    warm = solve_newton(
        LinearWarmup(), [0.5, 0.5], first_step_fraction=1.0, max_iterations=3
    )
    if not warm["native_success"]:
        raise RuntimeError("Custom Newton backend warmup failed")
    backend_warmup_s = perf_counter() - began
    report = dict(
        settings=vars(args),
        residual_rms_tol=RESIDUAL_RMS_TOL,
        same_solution_rms_tol=SAME_SOLUTION_RMS_TOL,
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "jax", "jaxprop", "pysolver-view")
        },
        core_source_sha256=core_fingerprints(),
        backend_import_warmup_s=backend_warmup_s,
        timing_note="Single serial warm measurement per method/seed/CFL, one BLAS thread. Outer times include construction, monitoring and rejected trials. First-call warmup and its discarded implicit update are separate. Matched-seed implicit references run only after all direct/PTC solves for that case. Speed ratios require the same independently verified discrete root.",
        cases=[],
    )
    cases = dict(zip(CASE_IDS, baseline.CASES))
    for case_id in args.cases:
        problem = SteadyNozzleProblem(
            cases[case_id],
            n_points=args.points,
            solver_settings=dict(
                reconstruction="muscl_van_leer", flux_variant="regularized"
            ),
        )
        storage = PhysicalStorage(problem)
        case = dict(
            id=case_id,
            name=(
                problem.case["name"]
                if "name" in problem.case
                else problem.case.get("label", case_id)
            ),
            back_pressure=problem.p_back,
            reservoir_pressure=problem.p0,
            x_m=problem.x,
            exact_x_m=problem.x_fine,
            exact_pressure_Pa=np.asarray(problem.reference_fine["p"]),
            solver_settings=problem.settings,
            first_call_costs=problem.warm_up(),
            storage_first_call_costs=storage.warm_up(problem.x0),
            warmup_state_used_as_seed=False,
            seeds={},
            runs=[],
            implicit_references={},
        )
        report["cases"].append(case)
        seeds = {}
        for phase in args.phases:
            raw = np.asarray(problem.initial_state).copy()
            seed = problem.pack(raw).copy()
            full = np.asarray(problem.full_state(seed)).copy()
            metadata = dict(
                kind=phase,
                initial_sha256=point_hash(seed),
                full_initial_state_sha256=point_hash(full),
                boundaries_applied=True,
                residual_rms=float(
                    np.linalg.norm(problem.residual(seed)) / np.sqrt(seed.size)
                ),
            )
            case["seeds"][phase] = metadata
            seeds[phase] = seed, full
        write_report(report, args.output)

        def measure(callback, seed, method, phase, **options):
            problem.reset_counters()
            storage.reset_counters()
            started = perf_counter()
            result = callback(**options)
            total = perf_counter() - started
            result["solver_wall_s"] = result["wall_s"]
            offset = result.get("history_started_at_s", started) - started
            for key in ("history", "trial_history"):
                if key in result:
                    result[key] = [
                        dict(row, elapsed_s=row["elapsed_s"] + offset)
                        for row in result[key]
                    ]
            result.update(
                wall_s=total,
                history_started_at_s=started,
                method=method,
                phase=phase,
                initial_sha256=point_hash(seed),
                model_work=problem.model_work(),
                storage_work=storage.model_work(),
            )
            if (
                result["backend"]["library"] != "pysolver"
                or result["backend"]["method"] != "newton_raphson"
            ):
                raise RuntimeError("A noncustom Newton backend was selected")
            result["diagnostics"] = problem.diagnostics(result["x"])
            result["root_success"] = bool(
                result["diagnostics"]["residual_rms"] <= RESIDUAL_RMS_TOL
            )
            result["same_root"] = None
            return result

        for phase, (seed, full) in seeds.items():
            folder = args.output / case_id / phase
            folder.mkdir(parents=True, exist_ok=True)
            print(f"{case_id} {phase}: custom Newton baseline", flush=True)
            newton = measure(
                lambda: solve_newton(
                    problem,
                    seed.copy(),
                    RESIDUAL_RMS_TOL,
                    args.newton_max_iterations,
                    args.first_step_fraction,
                    args.step_growth_factor,
                ),
                seed,
                "newton",
                phase,
            )
            case["runs"].append(newton)
            save_csv(folder / "history_newton.csv", newton["history"])
            write_report(report, args.output)
            for initial_cfl in (args.initial_cfl,):
                print(
                    f"{case_id} {phase}: Newton/PTC initial CFL={initial_cfl:g}",
                    flush=True,
                )

                def progress(row):
                    if (
                        args.progress_every
                        and row["accepted_index"] % args.progress_every == 0
                    ):
                        print(
                            f"  accepted={row['accepted_index']} RMS={row['residual_rms']:.3e} CFL={row['cfl']:.3e} t={row['elapsed_s']:.2f}s",
                            flush=True,
                        )

                result = measure(
                    lambda: solve_ptc(
                        problem,
                        storage,
                        seed.copy(),
                        residual_rms_tol=RESIDUAL_RMS_TOL,
                        max_iterations=args.max_iterations,
                        initial_cfl=initial_cfl,
                        minimum_cfl=args.minimum_cfl,
                        maximum_cfl=args.maximum_cfl,
                        first_step_fraction=args.first_step_fraction,
                        step_growth_factor=args.step_growth_factor,
                        ser_exponent=args.ser_exponent,
                        cfl_growth_limit=args.cfl_growth_limit,
                        cfl_shrink_limit=args.cfl_shrink_limit,
                        max_relative_step=args.max_relative_step,
                        nonmonotone_window=args.nonmonotone_window,
                        residual_growth_limit=args.residual_growth_limit,
                        model_error_limit=args.model_error_limit,
                        newton_switch_ratio=args.newton_switch_ratio,
                        max_retries=args.max_retries,
                        progress=progress,
                    ),
                    seed,
                    "ptc",
                    phase,
                )
                result["initial_cfl"] = initial_cfl
                case["runs"].append(result)
                tag = f"{initial_cfl:g}".replace(".", "p")
                save_csv(folder / f"history_ptc_cfl_{tag}.csv", result["history"])
                save_csv(folder / f"trials_ptc_cfl_{tag}.csv", result["trial_history"])
                write_report(report, args.output)
                print(
                    f"  done: {result['wall_s']:.3f}s RMS={result['diagnostics']['residual_rms']:.3e}, {result['message']}",
                    flush=True,
                )

        # These references are deliberately unavailable during every PTC run above.
        for phase, (seed, full) in seeds.items():
            problem.reset_counters()
            started = perf_counter()
            reference = solve_implicit(
                problem,
                RESIDUAL_RMS_TOL,
                args.implicit_max_iterations,
                initial_state=full.copy(),
            )
            total = perf_counter() - started
            offset = reference["history_started_at_s"] - started
            reference["history"] = [
                dict(row, elapsed_s=row["elapsed_s"] + offset)
                for row in reference["history"]
            ]
            reference.update(
                wall_s=total,
                history_started_at_s=started,
                initial_sha256=point_hash(seed),
                full_initial_state_sha256=point_hash(full),
                model_work=problem.model_work(),
            )
            reference["diagnostics"] = problem.diagnostics(reference["x"])
            reference["root_success"] = bool(
                reference["diagnostics"]["residual_rms"] <= RESIDUAL_RMS_TOL
            )
            case["implicit_references"][phase] = reference
            save_csv(
                args.output / case_id / phase / "history_implicit.csv",
                reference["history"],
            )
            for run in [run for run in case["runs"] if run["phase"] == phase]:
                if run["initial_sha256"] != reference["initial_sha256"]:
                    raise AssertionError(
                        "Compared methods received different initial points"
                    )
                run["scaled_solution_difference_from_implicit"] = float(
                    np.linalg.norm(run["x"] - reference["x"]) / np.sqrt(seed.size)
                )
                run["same_root"] = bool(
                    run["root_success"]
                    and reference["root_success"]
                    and run["scaled_solution_difference_from_implicit"]
                    <= SAME_SOLUTION_RMS_TOL
                )
                run["speedup_vs_implicit"] = (
                    reference["wall_s"] / run["wall_s"] if run["same_root"] else None
                )
            write_report(report, args.output)
    plot_report(report, args.output, args.no_show)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
