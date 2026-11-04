"""Sequential, resumable FAS-versus-grid-sequencing nozzle benchmark.

The controller imports no flow solver. Each case/reconstruction uses a fresh
worker, binds its levels once, excludes one complete solve, and then rotates
method order across measured complete solves. Numerical audits and I/O are
outside the synchronized solve timer.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

from .common import (
    ROOT,
    STUDY,
    archive_snapshot,
    clean,
    identity,
    provenance,
    save,
    snapshot,
)


LABELS = {
    "sequence": "Grid sequencing",
    "fas_implicit": "FAS implicit V-cycle",
    "fas_lagged": "FAS implicit, frozen step Jacobian",
    "fas_aggregate": "FAS implicit, aggregated coarse Jacobian",
    "fas_fmg": "FAS with coarse-to-fine initialization",
    "fas_explicit": "FAS RK3 V-cycle",
}


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/fmg_20261104")
    parser.add_argument("--settings", type=Path, default=STUDY / "settings_fmg.json")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--reconstructions", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--grids", nargs="+", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--worker", nargs=2, metavar=("CASE", "RECONSTRUCTION"), help=argparse.SUPPRESS
    )
    return parser.parse_args()


def _configuration(args):
    config = json.loads(args.settings.read_text(encoding="utf-8"))
    for name in ("cases", "reconstructions", "methods", "grids", "repeats"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    for name in ("max_cycles", "max_wall_s"):
        if getattr(args, name) is not None:
            config["fas"][name] = getattr(args, name)
    if config["repeats"] < 0:
        raise ValueError("repeats cannot be negative")
    return config


def _group_path(output, case, reconstruction):
    return output / "groups" / f"{case.rsplit('/', 1)[-1]}__{reconstruction}.json"


class SequenceWorkspace:
    """Production driver replay with retained level model/workspace identities."""

    def __init__(self, problems, scheme, policy, steady, workspace_cache=None):
        from nozzle_hyperbolic.solvers.fvm import SteadySettings
        from nozzle_hyperbolic.solvers.fvm.drivers import SteadyWorkspace

        self.problems = {p.mesh.n_cells: p for p in problems}
        self.scheme = scheme
        self.settings = SteadySettings(residual=policy, **steady)
        cache = {} if workspace_cache is None else workspace_cache
        self.workspaces = {}
        for n, problem in self.problems.items():
            key = (id(problem.model), repr(scheme), repr(policy))
            if key not in cache:
                cache[key] = SteadyWorkspace(problem.model, scheme, policy)
            self.workspaces[n] = cache[key]

    def solve(self):
        from nozzle_hyperbolic.solvers.fvm.drivers import solve_grid_sequence

        stages_history, level_work, stage_progress = [], [], []
        started = perf_counter()

        def on_stage(model, solution, metadata):
            n = model.mesh.n_cells
            for row in solution.history:
                stages_history.append(dict(row, cells=n))
            level_work.append(
                {
                    "cells": n,
                    "accepted_steps": sum(
                        r.get("phase") == "accepted" for r in solution.history
                    ),
                    "rejected_steps": sum(
                        r.get("phase") == "rejected" for r in solution.history
                    ),
                    "newton_iterations": sum(
                        int(r.get("inner_iterations", 0)) for r in solution.history
                    ),
                    "elapsed_s": solution.elapsed_s,
                    "success": bool(solution.success),
                }
            )
            stage_progress.append(
                {
                    "cells": n,
                    "elapsed_s": perf_counter() - started,
                    "success": bool(solution.success),
                    "reason": solution.reason,
                    "metrics": solution.metrics,
                }
            )

        result, stages = solve_grid_sequence(
            lambda n: self.problems[n].model,
            tuple(self.problems),
            self.scheme,
            self.settings,
            workspace_factory=lambda model, scheme, settings: self.workspaces[
                model.mesh.n_cells
            ],
            on_stage=on_stage,
            carry_cfl=True,
            retry_cold=False,
            stop_on_failure=True,
        )
        result.work = {"levels": level_work, "stage_progress": stage_progress}
        result.stage_history = stages_history
        result.stages = [
            {k: v for k, v in stage.items() if k not in ("attempts", "preparation")}
            for stage in stages
        ]
        result.history = [
            dict(row, cells=result.operator.mesh.n_cells) for row in result.history
        ]
        result.success = bool(
            result.success and result.operator.mesh.n_cells == max(self.problems)
        )
        return result


def _worker(args):
    # Configure the backend before importing numerical clients or creating arrays.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import platform
    import jaxlib
    from nozzle_hyperbolic.solvers.compilation import (
        CompilationReport,
        configure_compilation_cache,
    )
    from nozzle_hyperbolic.solvers.fvm import Scheme
    from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
    from .audit import audit_state
    from .common import environment_probe
    from .core import FASWorkspace
    from .nozzle_case import build_hierarchy, numerical_fields, reference_fields

    output = args.output.resolve()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["configuration"]
    if snapshot() != metadata["source"]:
        raise RuntimeError("Numerical source changed after campaign creation")
    configure_compilation_cache(output / "cache")
    case, reconstruction = args.worker
    case = case if "/" in case else "euler_nozzle/" + case
    policy = ResidualPolicy(**config["residual"])
    scheme = Scheme(reconstruction=reconstruction, **config["scheme"])
    setup_start = perf_counter()
    with CompilationReport(stream=io.StringIO()) as shared_compile:
        problems = build_hierarchy(case, config["grids"])
        problem_by_n = {p.mesh.n_cells: p for p in problems}
        fine = problems[-1]
        # Populate independent assessment references outside all solve timers.
        dense_reference = reference_fields(fine)
        fine.cell_reference()
        fine.reference_cell_fields()
        jax.block_until_ready(fine.model.initial_state())
    shared_setup_s = perf_counter() - setup_start
    group = {
        "case_id": case,
        "label": fine.definition.label,
        "reconstruction": reconstruction,
        "grids": config["grids"],
        "case_definition": fine.definition.to_dict(),
        "case_source_sha256": fine.definition.source_sha256,
        "back_pressure": fine.back_pressure,
        "shock_position": fine.shock_position,
        "reference": clean(dense_reference),
        "shared_setup_s": shared_setup_s,
        "shared_compilation": shared_compile.stats,
        "environment_before": environment_probe(),
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in jax.devices()],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "methods": [],
    }
    workspaces = {}
    level_cache = {}
    sequence_cache = {}
    for method in config["methods"]:
        started = perf_counter()
        with CompilationReport(stream=io.StringIO()) as compile_setup:
            family = "sequence" if method.startswith("sequence") else "fas"
            if family == "sequence":
                steady_settings = dict(config["steady"])
                steady_settings.update(
                    config.get("method_settings", {}).get(method, {})
                )
                workspaces[method] = SequenceWorkspace(
                    problems,
                    scheme,
                    policy,
                    steady_settings,
                    workspace_cache=sequence_cache,
                )
            else:
                fas_settings = dict(config["fas"])
                fas_settings.update(config.get("method_settings", {}).get(method, {}))
                extra = {}
                if method == "fas_aggregate":
                    from .aggregation import AggregatedJacobian

                    extra["coarse_jacobian_factory"] = AggregatedJacobian
                workspaces[method] = FASWorkspace(
                    [p.model for p in problems],
                    scheme,
                    residual_policy=policy,
                    settings=fas_settings,
                    mode="explicit" if method == "fas_explicit" else "implicit",
                    level_cache=level_cache,
                    **extra,
                )
        group["methods"].append(
            {
                "method_id": method,
                "label": config.get("method_labels", {}).get(
                    method, LABELS.get(method, method)
                ),
                "family": family,
                "requested_repeats": config["repeats"],
                "samples": [],
                "setup_s": perf_counter() - started,
                "setup_compilation": compile_setup.stats,
            }
        )
    group_file = _group_path(output, case, reconstruction)
    save(group_file, group)
    full_samples = {key: [] for key in config["methods"]}

    def run_one(method, index=None):
        label = "warmup" if index is None else f"sample_{index:03d}"
        print(f"START {case} {reconstruction} {method} {label}", flush=True)
        with CompilationReport(stream=io.StringIO()) as observed:
            started = perf_counter()
            result = workspaces[method].solve()
            jax.block_until_ready(result.state)
            elapsed = perf_counter() - started
        actual_n = result.operator.mesh.n_cells
        problem = problem_by_n[actual_n]
        state = np.asarray(result.state)
        assessment = audit_state(problem, state, result.operator, policy)
        fields = numerical_fields(problem, state)
        raw_file = (
            Path("runs")
            / f"{case.rsplit('/', 1)[-1]}__{reconstruction}"
            / method
            / f"{label}.json"
        )
        record = {
            "case_id": case,
            "reconstruction": reconstruction,
            "method_id": method,
            "sample_index": index,
            "excluded": index is None,
            "success": bool(result.success),
            "reason": result.reason,
            "actual_cells": actual_n,
            "grids": config["grids"],
            "elapsed_s": elapsed,
            "driver_elapsed_s": result.elapsed_s,
            "scheme": asdict(result.operator.scheme),
            "fas_settings": (
                getattr(workspaces[method], "settings", None)
                if not method.startswith("sequence")
                else None
            ),
            "steady_settings": (
                asdict(workspaces[method].settings)
                if method.startswith("sequence")
                else None
            ),
            "residual_policy": asdict(policy),
            "state": state,
            "state_sha256": sha256(state.tobytes()).hexdigest(),
            "fields": fields,
            "field_scales": problem.field_scales,
            "history": result.history,
            "metrics": result.metrics,
            "work": getattr(result, "work", {}),
            "stages": getattr(result, "stages", None),
            "stage_history": getattr(result, "stage_history", None),
            "initialization_history": getattr(result, "initialization_history", None),
            "assessment": assessment,
            "compilation": observed.stats,
            "raw_file": raw_file.as_posix(),
        }
        save(output / raw_file, record)
        record["raw_sha256"] = sha256((output / raw_file).read_bytes()).hexdigest()
        # Use the portable form in both the live group and final report.
        record = clean(record)
        print(
            f"END {case} {reconstruction} {method} {label}: {record['success']} "
            f"{record['reason']} {elapsed:.3f}s cycles={record['work'].get('cycles', '-')}",
            flush=True,
        )
        return record

    active = []
    for item in group["methods"]:
        method = item["method_id"]
        try:
            item["warmup"] = run_one(method)
        except Exception as error:
            item["warmup"] = {
                "success": False,
                "reason": "exception",
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "excluded": True,
            }
            print(item["warmup"]["traceback"], flush=True)
        if item["warmup"]["success"]:
            active.append(method)
        else:
            item["measurement_status"] = (
                "excluded complete solve failed; no warm timing population"
            )
        save(group_file, group)

    for index in range(config["repeats"]):
        order = (
            active[index % len(active) :] + active[: index % len(active)]
            if active
            else []
        )
        for method in order:
            item = next(m for m in group["methods"] if m["method_id"] == method)
            try:
                record = run_one(method, index)
            except Exception as error:
                record = {
                    "success": False,
                    "reason": "exception",
                    "exception": repr(error),
                    "traceback": traceback.format_exc(),
                    "sample_index": index,
                    "elapsed_s": None,
                }
            full_samples[method].append(record)
            item["samples"].append(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    not in (
                        "state",
                        "fields",
                        "history",
                        "stage_history",
                        "stages",
                        "initialization_history",
                    )
                }
            )
            save(group_file, group)

    for item in group["methods"]:
        records = full_samples[item["method_id"]]
        timed = [r for r in records if r.get("elapsed_s") is not None]
        if timed:
            durations = np.asarray([r["elapsed_s"] for r in timed])
            median_index = int(np.argmin(np.abs(durations - np.median(durations))))
            item["representative"] = timed[median_index]
            item["representative_note"] = (
                "Actual measured complete solve nearest median elapsed time"
            )
            item["timing"] = {
                "count": len(timed),
                "mean_s": float(durations.mean()),
                "min_s": float(durations.min()),
                "max_s": float(durations.max()),
                "std_s": float(durations.std(ddof=1)) if len(timed) > 1 else 0.0,
            }
            first_state = next((r["state"] for r in records if "state" in r), None)
            item["repeat_consistency"] = {
                "all_success": all(r["success"] for r in records),
                "identical_states": bool(
                    first_state is not None
                    and all(r.get("state") == first_state for r in records)
                ),
            }
        else:
            item["representative"] = item["warmup"]
            item["timing"] = {"count": 0, "mean_s": None, "min_s": None, "max_s": None}
    baseline = next((m for m in group["methods"] if m["method_id"] == "sequence"), None)
    for item in group["methods"]:
        rep = item["representative"]
        passed = False
        deviation = None
        if (
            baseline is not None
            and rep.get("state")
            and baseline["representative"].get("state")
        ):
            a, b = np.asarray(rep["state"]), np.asarray(
                baseline["representative"]["state"]
            )
            if a.shape == b.shape:
                deviation = float(
                    np.max(
                        np.abs(a - b) / np.asarray(fine.model.variable_scales)[None, :]
                    )
                )
                passed = deviation <= config["comparison_field_tolerance"]
        item["field_agreement"] = {
            "passed": passed,
            "max_scaled_state_difference": deviation,
            "tolerance": config["comparison_field_tolerance"],
            "baseline": "sequence",
        }
        complete = len(item["samples"]) == config["repeats"] and config["repeats"] > 0
        checks = all(
            r.get("success")
            and r.get("assessment", {}).get("original_checks_passed")
            and r.get("assessment", {}).get("branch_passed")
            and r.get("assessment", {}).get("analytical_passed")
            for r in item["samples"]
        )
        item["timing_eligible"] = bool(complete and checks and passed)
        item["warm_compile_reuse_passed"] = bool(
            item["samples"]
            and all(
                r.get("compilation", {}).get("backend_requests", -1) == 0
                for r in item["samples"]
            )
        )
    group["environment_after"] = environment_probe()
    group["timing_environment_verified"] = all(
        group[key].get("status") == "clean"
        for key in ("environment_before", "environment_after")
    )
    for item in group["methods"]:
        item["numerical_timing_eligible"] = item["timing_eligible"]
        item["timing_eligible"] = bool(
            item["timing_eligible"] and group["timing_environment_verified"]
        )
    group["source_after"] = snapshot()
    if group["source_after"] != metadata["source"]:
        raise RuntimeError("Numerical source changed during benchmark group")
    group["complete"] = True
    save(group_file, group)
    return 0


def assemble(output, report=True):
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    groups = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((output / "groups").glob("*.json"))
    ]
    result = {
        "metadata": metadata,
        "configuration": metadata["configuration"],
        "groups": groups,
    }
    result["audits"] = {
        "package_source_unchanged": all(
            g.get("source_after", {}).get("src") == metadata["source"]["src"]
            for g in groups
        ),
        "requested_groups": len(metadata["configuration"]["cases"])
        * len(metadata["configuration"]["reconstructions"]),
        "completed_groups": sum(bool(g.get("complete")) for g in groups),
    }
    audit_path = output / "audit.json"
    if audit_path.exists():
        result["audits"]["fresh_campaign_audit"] = json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    save(output / "summary.json", result)
    if report:
        from .report import build

        build(output, STUDY / "REPORT.html")
    return result


def main():
    args = _arguments()
    if args.worker:
        return _worker(args)
    output = args.output.resolve()
    if args.assemble_only:
        assemble(output, not args.no_report)
        return 0
    config = _configuration(args)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError("Use a fresh output directory or --resume")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["configuration"] != config or metadata["source"] != snapshot():
            raise ValueError(
                "Resume configuration/source differs from the recorded campaign"
            )
    else:
        output.mkdir(parents=True, exist_ok=True)
        metadata = provenance()
        metadata.update(
            configuration=config,
            src_before=snapshot()["src"],
            timing_protocol=f"One excluded complete solve, then {config['repeats']} requested warm replays; rotate method order; all-level solve/transfer/monitor/rejection cost included; audits and I/O excluded.",
            first_use_caveat="Warmup compile/load events share kernels and depend on method order; they are not independent cold-method compilation measurements.",
            failed_warmup_policy="Retain unsuccessful excluded solve; do not repeatedly benchmark a known failed configuration.",
            shock_fraction_definition="Fraction of diverging section from throat to exit; x_shock/L=(1+fraction)/2.",
        )
        save(metadata_path, metadata)
        archive_snapshot(output / "source_snapshot.zip", metadata["source"])
    child_env = os.environ.copy()
    child_env["NOZZLE_MULTIGRID_CONTROLLER_PID"] = str(os.getpid())
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_env[name] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    for case in config["cases"]:
        for recon in config["reconstructions"]:
            path = _group_path(output, case, recon)
            if (
                args.resume
                and path.exists()
                and json.loads(path.read_text(encoding="utf-8")).get("complete")
            ):
                continue
            command = [
                sys.executable,
                "-B",
                "-u",
                "-m",
                "development.multigrid.scripts.run",
                "--output",
                str(output),
                "--worker",
                case,
                recon,
            ]
            print(f"GROUP {case} / {recon}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, env=child_env, check=False)
            if completed.returncode:
                print(
                    f"Worker failed with status {completed.returncode}; saved groups retained.",
                    flush=True,
                )
                assemble(output, False)
                return completed.returncode
            assemble(output, False)
    assemble(output, not args.no_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
