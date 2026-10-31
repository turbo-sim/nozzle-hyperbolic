"""Matched source-style implicit solvers: internal versus coupled boundary equations.

Run 1 is excluded for each case/method; runs 2--11 provide ten warm timings.
No source files are edited. Both paths use source JAX Newton/colored/banded
machinery; the coupled system only changes boundary residual/unknown handling.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import jax.numpy as jnp
import numpy as np

from development.roberto.historical.BC_residual.scripts.coupled_step import CoupledStep
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from development.roberto.historical.newton_solve.scripts.investigate_ptc import json_value
from development.roberto.historical.newton_solve.scripts.problem import SteadyNozzleProblem

MODES = ("production", "coupled")
CASE_IDS = ("supersonic", "shock35", "shock70", "sonic", "subsonic")


def integrate(base, coupled, mode, initial, max_updates, progress=False):
    """One identical cold restart; no rebuilding or compiling per-repeat closures."""
    full = jnp.asarray(np.array(initial, copy=True))
    step = base.step if mode == "production" else coupled.step
    previous_dt = float(base.compute_dt(full)[0])
    first_ssr, previous_ssr = None, np.inf
    history, trials = [], []
    started = perf_counter()

    def record(index, dt=0.0):
        spatial, boundary, _ = (float(v) for v in coupled.metrics(full))
        history.append(
            dict(
                update=index,
                elapsed_s=perf_counter() - started,
                dt_s=dt,
                spatial_rms=spatial,
                boundary_rms=boundary,
                residual_rms=max(spatial, boundary),
            )
        )
        return max(spatial, boundary)

    rms = record(0)
    message = "Maximum accepted updates reached"
    for index in range(1, max_updates + 1):
        if rms <= 1e-8:
            break
        dt_cap, sound = base.compute_dt(full)
        dt_cap = float(dt_cap)
        if first_ssr is not None:
            dt_cap *= float(
                np.clip(
                    (first_ssr / max(previous_ssr, 1e-300))
                    ** base.settings["ser_beta"],
                    1.0,
                    base.settings["ser_cfl_max"] / base.settings["CFL"],
                )
            )
        dt = min(previous_dt * 1.2, dt_cap)
        old = np.asarray(full)
        accepted = False
        for retry in range(9):
            candidate, ssr = step(full, jnp.float64(dt))
            new = np.asarray(
                candidate
            )  # Synchronizes the complete JIT update before timing.
            increments = (
                np.max(np.abs(new[:, 0] - old[:, 0]) / (old[:, 0] + 1e-10)),
                np.max(np.abs(new[:, 2] - old[:, 2]) / (np.abs(old[:, 2]) + 1e-10)),
                np.max(
                    np.abs(new[:, 1] - old[:, 1])
                    / (np.abs(old[:, 1]) + np.asarray(sound) + 1e-10)
                ),
            )
            accepted = bool(
                np.isfinite(new).all()
                and np.all(new > 0)
                and all(v <= limit for v, limit in zip(increments, (0.2, 0.1, 0.2)))
            )
            trials.append(dict(update=index, retry=retry, dt_s=dt, accepted=accepted))
            if accepted:
                break
            dt *= 0.5
        if not accepted:
            message = "Implicit update failed after nine attempts"
            break
        full = candidate
        previous_dt, previous_ssr = dt, float(ssr)
        if first_ssr is None:
            first_ssr = previous_ssr
        rms = record(index, dt)
        if progress and index % 50 == 0:
            print(f"  {mode}: {index} accepted, RMS={rms:.3e}", flush=True)
    if rms <= 1e-8:
        message = "Spatial and boundary residual targets reached"
    elapsed = perf_counter() - started
    # Independent final audits and reference comparisons are outside solve timing.
    full = np.asarray(full)
    point = base.pack(full)
    diagnostics = base.diagnostics(point)
    return dict(
        mode=mode,
        point=point,
        final_state=full,
        x_m=base.x,
        pressure_exact_Pa=base.reference["p"],
        history=history,
        trials=trials,
        solve_s=elapsed,
        accepted_updates=len(history) - 1,
        rejected_trials=sum(not t["accepted"] for t in trials),
        success=bool(rms <= 1e-8 and diagnostics["residual_rms"] <= 1e-8),
        message=message,
        diagnostics=diagnostics,
    )


def source_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "src").rglob("*.py")
    }


def save(report, output):
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, default=json_value), encoding="utf-8"
    )
    rows = []
    for case in report["cases"]:
        for run in case["measured_runs"]:
            rows.append(
                dict(
                    case=case["id"],
                    mode=run["mode"],
                    measurement=run["measurement"],
                    execution_run=run["measurement"] + 1,
                    order=run["order"],
                    solve_s=run["solve_s"],
                    success=run["success"],
                    accepted_updates=run["accepted_updates"],
                    rejected_trials=run["rejected_trials"],
                )
            )
    if rows:
        with (output / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_IDS, default=list(CASE_IDS))
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Measured runs per case/method, after one excluded run",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "steady",
    )
    args = parser.parse_args()
    if args.points < 5 or args.max_updates < 1 or args.runs < 1:
        parser.error("Use at least five points and positive budgets/run count")
    args.output = args.output.resolve()
    if args.plot_only:
        from development.roberto.historical.BC_residual.scripts.report_implicit import (
            write_report,
        )

        write_report(args.output)
        return
    root = Path(__file__).resolve().parents[5]
    before = source_hashes(root)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        settings=vars(args),
        methods=list(MODES),
        source_sha256=before,
        cases=[],
        backend="Source-native JAX Newton, colored block Jacobian, identical block-Thomas solver code; no PySolverView or Optimistix timestep backend",
        timing_note="One full excluded cold solve per case/method, followed by ten measured cold solves by default. Kernels built once and reused; each solve resets fields and controller. Method execution order alternates. Solve timing includes monitoring/retries and synchronized device work; setup, final audits, output and plotting excluded.",
    )
    definitions = dict(zip(CASE_IDS, baseline.CASES))
    for case_id in args.cases:
        # Pin the measured comparison settings independently of example defaults.
        base = SteadyNozzleProblem(
            definitions[case_id],
            n_points=args.points,
            solver_settings=dict(
                CFL=20.0,
                newton_iter=8,
                newton_tol=1e-8,
                ser_beta=1.0,
                ser_cfl_max=500.0,
                dt_growth_factor=1.2,
            ),
        )
        coupled = CoupledStep(base)
        initial = np.asarray(base.full_state(base.x0)).copy()
        case = dict(
            id=case_id,
            name=definitions[case_id]["label"],
            back_pressure_ratio=base.p_back / base.p0,
            solver_settings=dict(base.settings),
            excluded_runs=[],
            measured_runs=[],
            runs=[],
            timings={},
        )
        report["cases"].append(case)
        for mode in MODES:
            print(f"{case_id} {mode}: excluded compilation run", flush=True)
            warm = integrate(
                base, coupled, mode, initial, args.max_updates, progress=True
            )
            case["excluded_runs"].append(warm)
            save(report, args.output)
        for repeat in range(1, args.runs + 1):
            order = MODES if repeat % 2 else MODES[::-1]
            for position, mode in enumerate(order):
                run = integrate(base, coupled, mode, initial, args.max_updates)
                run.update(measurement=repeat, order=position)
                case["measured_runs"].append(run)
                print(
                    f"{case_id} {mode} measurement {repeat}/{args.runs}: {run['solve_s']:.3f}s, {run['accepted_updates']} updates, success={run['success']}",
                    flush=True,
                )
            save(report, args.output)
        for mode in MODES:
            runs = [r for r in case["measured_runs"] if r["mode"] == mode]
            times = np.array([r["solve_s"] for r in runs])
            case["timings"][mode] = dict(
                count=len(runs),
                min_s=float(times.min()),
                max_s=float(times.max()),
                mean_s=float(times.mean()),
                all_success=all(r["success"] for r in runs),
            )
            # Plot one actual median-duration trajectory, never an averaged residual curve.
            representative = sorted(runs, key=lambda r: r["solve_s"])[len(runs) // 2]
            case["runs"].append(representative)
            fields = list(representative["history"][0].keys())
            with (args.output / f"{case_id}_{mode}_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(representative["history"])
            np.savetxt(
                args.output / f"{case_id}_{mode}_pressure.csv",
                np.column_stack(
                    [
                        representative["x_m"],
                        representative["final_state"][:, 0],
                        representative["pressure_exact_Pa"],
                    ]
                ),
                delimiter=",",
                header="x_m,pressure_Pa,pressure_exact_Pa",
                comments="",
            )
        source, result = case["runs"]
        case["scaled_state_difference_rms"] = float(
            np.linalg.norm(result["point"] - source["point"]) / np.sqrt(base.x0.size)
        )
        case["maximum_pressure_difference_Pa"] = float(
            np.max(
                np.abs(result["final_state"][1:-1, 0] - source["final_state"][1:-1, 0])
            )
        )
        save(report, args.output)
    assert before == source_hashes(root), "Source files changed during experiment"
    report["source_files_unchanged"] = True
    save(report, args.output)
    from development.roberto.historical.BC_residual.scripts.report_implicit import write_report

    write_report(args.output)


if __name__ == "__main__":
    main()
