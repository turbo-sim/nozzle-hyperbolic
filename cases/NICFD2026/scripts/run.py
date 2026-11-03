"""Execute the six published NICFD cases and identical cold/warm replays."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import (
    Solution,
    SteadyWorkspace,
    prepare_continuation_state,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.convergence_output import strict_json

from .case import (
    ROOT,
    FAMILIES,
    REGIMES,
    build_case,
    environment,
    prepare_fluids,
    specification,
)
from .metrics import (
    agreement,
    independent_audit,
    profile_kernel,
    reference_errors,
    regime_audit,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(strict_json(value), encoding="utf-8")
    temporary.replace(path)


def identity(state):
    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def numerical_sources():
    repo = ROOT.parents[1]
    files = list((repo / "src/nozzle_hyperbolic").rglob("*.py"))
    files += [ROOT / "scripts" / name for name in ("case.py", "metrics.py", "run.py")]
    files += [ROOT / "protocol.json"]
    return {
        str(path.relative_to(repo))
        .replace("\\", "/"): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(files)
    }


class CompilationLog(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = 0

    def emit(self, record):
        message = record.getMessage()
        if record.name.startswith("jax") and (
            message.startswith("Compiling ") or "Finished XLA compilation" in message
        ):
            self.events += 1


class TimeLimit(Exception):
    """Resource-limited failure; it never establishes numerical convergence."""


def execute(family, regime):
    case_dir = ROOT / family / regime
    out = case_dir / "outputs"
    out.mkdir(exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        raise ValueError(
            f"Existing campaign at {result_path}; choose a clean campaign before rerunning"
        )
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    config = specification(family, regime)
    jax.config.update("jax_enable_compilation_cache", False)
    record = dict(
        family=family,
        regime=regime,
        created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_revision=json.loads((ROOT / "provenance.json").read_text())[
            "revision"
        ],
        environment=environment(),
        source_before=numerical_sources(),
        specification=config,
        protocol=protocol,
        runs=[],
        preparation_failure=None,
    )
    write_json(result_path, record)
    started = time.perf_counter()
    try:
        # Only the shocked two-component YAML omits N_h. Reuse the explicitly
        # published within-family property definition, recording that omission.
        property_config = specification(family, "subsonic")
        record["property_resolution"] = dict(
            source="subsonic/settings.yaml within this family",
            omitted_gas_N_h="N_h" not in config["eos"]["gas"],
            note="No explicit table setting is changed; identical within-family tables are reused.",
        )
        fluids, properties = prepare_fluids(property_config, family)
        record["property_preparation_s"] = time.perf_counter() - started
        record["properties"] = properties
        write_json(result_path, record)
    except Exception as error:
        record.update(
            preparation_failure=dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
            property_preparation_s=time.perf_counter() - started,
        )
        record["source_after"] = numerical_sources()
        record["verdict"] = agreement(None, protocol["agreement"])
        record["verdict"]["detail"] = "Property preparation failed: " + str(error)
        write_json(result_path, record)
        print(f"{family}/{regime}: property preparation failed: {error}", flush=True)
        return record

    cache, cold_states = {}, {}
    root_logger, jax_logger = logging.getLogger(), logging.getLogger("jax")
    original_root, original_jax = list(root_logger.handlers), list(jax_logger.handlers)
    compilation = CompilationLog()
    root_logger.handlers, jax_logger.handlers = [compilation], []
    try:
        with jax.log_compiles(True):
            for mode in protocol["timing_modes"]:
                previous, hierarchy_failed = None, False
                run = dict(mode=mode, stages=[], complete=False)
                start_events = compilation.events
                for cells in protocol["meshes"]:
                    if hierarchy_failed and cells != protocol["meshes"][-1]:
                        run["stages"].append(
                            dict(
                                cells=cells,
                                success=False,
                                reason="not_attempted_after_coarse_failure",
                                solve_s=0.0,
                                preparation_s=0.0,
                                total_s=0.0,
                            )
                        )
                        continue
                    stage = dict(
                        cells=cells,
                        mode=mode,
                        success=False,
                        history=[],
                        reason="preparing",
                        compilation_events=0,
                    )
                    event_start, preparation_started = (
                        compilation.events,
                        time.perf_counter(),
                    )
                    try:
                        if cells not in cache:
                            model, scheme, settings, effective = build_case(
                                config,
                                fluids,
                                cells,
                                family=family,
                                initialization="reference",
                                reconstruction=protocol["reconstruction"],
                                newton_iterations=protocol["newton_iterations"],
                                max_steps=protocol["max_steps"],
                            )
                            workspace = SteadyWorkspace(
                                model, scheme, settings.residual
                            )
                            cache[cells] = (
                                model,
                                scheme,
                                settings,
                                effective,
                                workspace,
                                profile_kernel(model),
                            )
                        model, scheme, settings, effective, workspace, extractor = (
                            cache[cells]
                        )
                        stage["effective"] = effective
                        if previous is not None and previous.success:
                            initial, transfer = prepare_continuation_state(
                                model, previous, scheme, operator=workspace.operator
                            )
                            stage["initialization"] = (
                                "successful transported-density transfer with constraint recovery"
                            )
                            stage["transfer"] = transfer
                            if not transfer["success"]:
                                raise ValueError(
                                    "Constraint-consistent transfer failed: "
                                    + str(transfer)
                                )
                        else:
                            initial = model.initial_state()
                            stage["initialization"] = (
                                "independent published-profile finest check after coarse failure"
                                if hierarchy_failed
                                else "published-profile first-grid seed"
                            )
                            completed, info = workspace.operator.complete_boundaries(
                                initial, 0.0
                            )
                            completed, info = jax.device_get((completed, info))
                            stage["boundary_preparation"] = info._asdict()
                            if bool(info.converged and info.admissible):
                                initial = jnp.asarray(completed)
                        jax.block_until_ready(initial)
                        stage["initial_state_sha256"] = identity(initial)
                        stage["preparation_s"] = (
                            time.perf_counter() - preparation_started
                        )
                        last = dict(state=initial, candidate=initial)
                        raw_step = workspace.operator.step

                        def retained_step(*args):
                            outcome = raw_step(*args)
                            last["candidate"] = outcome[0]
                            return outcome

                        # Retain the actual accepted state if a resource limit
                        # fires. Numerical operations and acceptance stay in src.
                        workspace.operator.step = retained_step
                        solve_started, last_print = time.perf_counter(), [0.0]

                        def progress(row):
                            elapsed = time.perf_counter() - solve_started
                            stage["history"].append(dict(row, elapsed_s=elapsed))
                            if row["phase"] != "initial" and row["accepted"]:
                                last["state"] = last["candidate"]
                            if (
                                row["phase"] == "initial"
                                or elapsed - last_print[0] >= 20
                            ):
                                last_print[0] = elapsed
                                print(
                                    f"{family}/{regime} {mode} N={cells}: {row['phase']} "
                                    f"step {row['step']}, {elapsed:.1f}s",
                                    flush=True,
                                )
                            if elapsed > protocol["stage_time_limit_s"]:
                                raise TimeLimit("declared per-stage wall-time limit")

                        try:
                            result = solve_single_grid(
                                model,
                                scheme,
                                settings,
                                initial_state=initial,
                                workspace=workspace,
                                callback=progress,
                            )
                        except TimeLimit:
                            state = np.asarray(last["state"])
                            result = Solution(
                                state,
                                False,
                                "resource_time_limit",
                                stage["history"],
                                workspace.sample(state),
                                elapsed_s=time.perf_counter() - solve_started,
                                operator=workspace.operator,
                            )
                        finally:
                            workspace.operator.step = raw_step
                        jax.block_until_ready(result.state)
                        stage.update(
                            success=result.success,
                            reason=result.reason,
                            metrics=result.metrics,
                            solve_s=time.perf_counter() - solve_started,
                        )
                        stage["total_s"] = stage["preparation_s"] + stage["solve_s"]
                        stage["compilation_events"] = compilation.events - event_start
                        # Field extraction and independent audit are outside timers.
                        data = {
                            k: np.asarray(v)
                            for k, v in extractor(jnp.asarray(result.state)).items()
                        }
                        stem = f"{mode}_N{cells}"
                        np.savez_compressed(
                            out / (stem + ".npz"), state=result.state, **data
                        )
                        stage["fields_file"] = stem + ".npz"
                        stage["regime_audit"] = regime_audit(data, regime)
                        stage["errors"] = reference_errors(
                            data, model, family, regime, config
                        )
                        baseline = stage["history"][0]["equation_norm"]
                        stage["audit"] = independent_audit(
                            workspace, result.state, baseline, result.metrics
                        )
                        stage["case_success"] = bool(
                            result.success
                            and stage["audit"]["passed"]
                            and stage["regime_audit"]["passed"]
                        )
                        stage["final_state_sha256"] = identity(result.state)
                        if mode == "cold":
                            cold_states[cells] = np.asarray(result.state).copy()
                        elif cells in cold_states:
                            stage["cold_warm_agree"] = bool(
                                np.allclose(
                                    result.state,
                                    cold_states[cells],
                                    rtol=1e-11,
                                    atol=1e-10,
                                )
                            )
                            stage["cold_warm_scaled_max_difference"] = float(
                                np.max(
                                    abs(result.state - cold_states[cells])
                                    / model.variable_scales
                                )
                            )
                        previous = replace(result, success=stage["case_success"])
                        if not stage["case_success"]:
                            hierarchy_failed, previous = True, None
                        print(
                            f"{family}/{regime} {mode} N={cells}: {stage['reason']}; "
                            f"regime={stage['regime_audit']['passed']}; {stage['total_s']:.2f}s",
                            flush=True,
                        )
                    except Exception as error:
                        stage.update(
                            success=False,
                            case_success=False,
                            reason="preparation_or_audit_failed",
                            error=dict(
                                type=type(error).__name__,
                                message=str(error),
                                traceback=traceback.format_exc(),
                            ),
                        )
                        stage.setdefault(
                            "preparation_s", time.perf_counter() - preparation_started
                        )
                        stage.setdefault("solve_s", 0.0)
                        stage.setdefault(
                            "total_s", stage["preparation_s"] + stage["solve_s"]
                        )
                        hierarchy_failed, previous = True, None
                        print(
                            f"{family}/{regime} {mode} N={cells}: {error}", flush=True
                        )
                    run["stages"].append(stage)
                    write_json(result_path, dict(record, current_run=run))
                run.update(
                    complete=all(s.get("case_success", False) for s in run["stages"]),
                    total_s=sum(s["total_s"] for s in run["stages"]),
                    preparation_s=sum(s["preparation_s"] for s in run["stages"]),
                    solve_s=sum(s["solve_s"] for s in run["stages"]),
                    compilation_events=compilation.events - start_events,
                )
                run["warm_replay_valid"] = bool(
                    mode == "warm"
                    and not run["compilation_events"]
                    and any("fields_file" in s for s in run["stages"])
                    and all(
                        s.get("cold_warm_agree", False)
                        for s in run["stages"]
                        if s["reason"] != "not_attempted_after_coarse_failure"
                    )
                )
                record["runs"].append(run)
                write_json(result_path, record)
    finally:
        root_logger.handlers, jax_logger.handlers = original_root, original_jax
    record["source_after"] = numerical_sources()
    record["source_unchanged"] = record["source_before"] == record["source_after"]
    reported = record["runs"][-1]["stages"][-1] if record["runs"] else None
    record["verdict"] = agreement(reported, protocol["agreement"])
    record["verdict"]["reported_cells"] = protocol["meshes"][-1]
    write_json(result_path, record)
    print(f"{family}/{regime}: {record['verdict']['title']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--regime", choices=(*REGIMES, "all"), default="all")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.family == "all" or args.regime == "all":
            parser.error("A worker executes one physical case")
        execute(args.family, args.regime)
        return
    families = FAMILIES if args.family == "all" else (args.family,)
    regimes = REGIMES if args.regime == "all" else (args.regime,)
    existing = [
        ROOT / family / regime / "outputs/result.json"
        for family in families
        for regime in regimes
        if (ROOT / family / regime / "outputs/result.json").exists()
    ]
    if existing:
        raise SystemExit(
            "Saved campaigns already exist; no logs or results were overwritten:\n"
            + "\n".join(str(path) for path in existing)
        )
    failures = []
    for family in families:
        for regime in regimes:
            log = ROOT / family / regime / "outputs/execution.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "cases.NICFD2026.scripts.run",
                        "--family",
                        family,
                        "--regime",
                        regime,
                        "--worker",
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                )
            if result.returncode:
                failures.append(f"{family}/{regime}")
            print(
                f"Finished {family}/{regime}: worker exit {result.returncode}; {log}",
                flush=True,
            )
    if failures:
        raise SystemExit("Case execution errors: " + ", ".join(failures))


if __name__ == "__main__":
    main()
