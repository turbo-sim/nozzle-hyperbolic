"""Single-grid 2x2 ablation: parallel excluded compilation, serial warm samples."""

import argparse
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep
import traceback
from zoneinfo import ZoneInfo

from .common import ROOT, STUDY, VARIANTS, CASES, GRIDS, RECONSTRUCTIONS, archive, clean, group_name, identity, read, save, sha, sources


def setup(cache):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    from nozzle_hyperbolic.solvers.compilation import configure_compilation_cache
    from cases.NICFD2026.scripts.case import prepare_fluids, specification
    configure_compilation_cache(cache)
    return prepare_fluids(specification("two_component", "subsonic"), "two_component")


def build(fluids, regime, reconstruction, cells, variant):
    import numpy as np
    from cases.NICFD2026.scripts.case import build_case, specification
    from nozzle_hyperbolic.solvers.fvm import SteadyWorkspace, SteadySettings
    from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
    from .numerics import adapt
    base, scheme, _, effective = build_case(specification("two_component", regime), fluids, cells,
        initialization="reference", reconstruction=reconstruction, newton_iterations=20)
    model = adapt(base, scheme, variant)
    assert np.array_equal(np.asarray(model._volumes), np.asarray(model.mesh.volumes))
    policy = ResidualPolicy(criterion="absolute", absolute_tolerance=1e-9, boundary_tolerance=1e-9,
                            constraint_tolerance=1e-9, global_tolerance=1e-9)
    settings = SteadySettings(residual=policy, cfl=2., cfl_growth=1.2, cfl_max=1e6,
                             max_steps=2000, max_retries=32, stagnation_steps=64)
    return model, scheme, settings, SteadyWorkspace(model, scheme, policy), effective


def python_activity():
    """Read-only Python CPU snapshot; environment evidence, not an idle guarantee."""
    if os.name != "nt":
        return dict(verified=False, reason="Windows-only supplementary CPU snapshot")
    try:
        command = "@(Get-Process python,pythonw -ErrorAction SilentlyContinue | Select-Object Id,CPU) | ConvertTo-Json -Compress"
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=15)
        rows = json.loads(result.stdout.strip() or "[]")
        if isinstance(rows, dict):
            rows = [rows]
        return dict(verified=result.returncode == 0,
                    cpu_s={str(r["Id"]): r["CPU"] for r in rows if r["Id"] not in (os.getpid(), os.getppid())})
    except Exception as error:
        return dict(verified=False, exception=repr(error))


def solve_once(binding, initial, variant, phase, budget):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from nozzle_hyperbolic.solvers.compilation import CompilationReport
    from nozzle_hyperbolic.solvers.fvm import Solution, solve_single_grid
    model, scheme, settings, ws, effective = binding
    initial = jnp.asarray(initial)
    initial_norm = np.asarray(ws.sample(initial)["equation_norm"])
    retained = dict(state=initial, candidate=initial, old=initial, best_state=initial,
                    best_norm=float(np.max(initial_norm)), last_rejected=None)
    history, raw_step = [], ws.operator.step

    class Budget(Exception):
        pass

    def step(old, dt, time):
        candidate, info = raw_step(old, dt, time)
        retained.update(old=old, candidate=candidate)
        return candidate, info

    def callback(row):
        history.append(row)
        if row["phase"] == "accepted":
            retained["state"] = retained["candidate"]
        elif row["phase"] == "rejected":
            retained["last_rejected"] = dict(row=row, state=retained["candidate"])
        if row.get("equation_norm") is not None:
            norm = float(np.max(row["equation_norm"]))
            if np.isfinite(norm) and norm < retained["best_norm"]:
                retained.update(best_norm=norm, best_state=retained["state"])
        if row["phase"] == "accepted" and row["step"] % 40 == 0:
            print(f"PROGRESS {phase} {variant} step={row['step']} R={max(row['equation_norm']):.4g}", flush=True)
        if perf_counter() - started > budget:
            raise Budget()

    exception = None
    ws.operator.step = step
    started = perf_counter()
    with CompilationReport(stream=io.StringIO()) as compilation:
        try:
            result = solve_single_grid(model, scheme, settings, initial_state=initial, workspace=ws, callback=callback)
        except Exception as error:
            exception = traceback.format_exc() if not isinstance(error, Budget) else None
            result = Solution(np.asarray(retained["state"]), False,
                "wall_budget" if isinstance(error, Budget) else "exception", history,
                ws.sample(retained["state"], initial_norm), elapsed_s=perf_counter() - started, operator=ws.operator)
        finally:
            ws.operator.step = raw_step
        jax.block_until_ready(result.state)
    elapsed = perf_counter() - started
    # Everything below is excluded from the solve timer.
    from cases.NICFD2026.scripts.case import specification
    from cases.NICFD2026.scripts.metrics import independent_audit, profile_kernel, regime_audit, reference_errors, agreement
    metrics = result.metrics or ws.sample(result.state, initial_norm)
    try:
        audit = independent_audit(ws, result.state, initial_norm, metrics)
    except Exception:
        audit = dict(passed=False, exception=traceback.format_exc())
    fields = clean(jax.device_get(profile_kernel(model)(jnp.asarray(result.state))))
    branch = regime_audit(fields, model._study_regime)
    errors = reference_errors(fields, model, "two_component", model._study_regime, specification("two_component", model._study_regime))
    thresholds = read(ROOT / "cases/NICFD2026/protocol.json")["agreement"]
    profile = agreement(dict(success=result.success, audit=audit, regime_audit=branch,
                             errors=errors, reason=result.reason), thresholds)
    state = np.asarray(result.state)
    counts = dict(accepted_steps=sum(r["phase"] == "accepted" for r in history),
        rejected_steps=sum(r["phase"] == "rejected" for r in history),
        newton_iterations=sum(r.get("inner_iterations", 0) for r in history),
        rejected_newton_trials=sum(r.get("rejected_newton_trials", 0) for r in history))
    return clean(dict(variant=variant, phase=phase, success=bool(result.success), reason=result.reason,
        root_pass=bool(result.success and audit["passed"] and branch["passed"]),
        elapsed_s=elapsed, compilation=compilation.stats, state=state, state_sha256=sha256(state.tobytes()).hexdigest(),
        initial_state_sha256=sha256(np.asarray(initial).tobytes()).hexdigest(), initial_norm=initial_norm,
        audit=audit, branch=branch, errors=errors, profile=profile, fields=fields, history=history, work=counts,
        best_state=np.asarray(retained["best_state"]), best_norm=retained["best_norm"],
        last_rejected=retained["last_rejected"], exception=exception))


def worker(args):
    import numpy as np
    regime, reconstruction, n = args.worker
    cells = int(n)
    name = group_name(regime, reconstruction, cells)
    folder = args.output / "groups" / name
    source = sources()
    metadata = read(args.output / "metadata.json")
    if source != metadata["source"]:
        raise ValueError("Campaign numerical source changed")
    fluids, fluid_metadata = setup(args.cache)
    import jax
    import jax.numpy as jnp
    from cases.NICFD2026.scripts.case import environment
    save(folder / ("environment_" + args.phase + ".json"), dict(**environment(), backend=jax.default_backend(),
        x64=jax.config.jax_enable_x64, fluids=fluid_metadata, activity=python_activity()))
    seed_path = folder / "seed.json"
    seed_binding = build(fluids, regime, reconstruction, cells, "baseline")
    if seed_path.exists():
        seed_record = read(seed_path)
        initial = jnp.asarray(seed_record["state"])
    else:
        started = perf_counter()
        original = seed_binding[0].initial_state()
        completed, info = jax.device_get(seed_binding[3].operator.complete_boundaries(original, 0.))
        okay = bool(info.converged and info.admissible)
        initial = jnp.asarray(completed if okay else original)
        seed_record = dict(kind="published profile; same prepared field for all four variants",
            preparation_success=okay, preparation=clean(info), preparation_s=perf_counter() - started,
            preparation_failure_policy="retain failure; use original coupled seed for every variant",
            state=np.asarray(initial), state_sha256=sha256(np.asarray(initial).tobytes()).hexdigest(),
            points=np.asarray(seed_binding[0].mesh.points), volumes=np.asarray(seed_binding[0].mesh.volumes),
            variable_scales=np.asarray(seed_binding[0].variable_scales))
        save(seed_path, seed_record)
    assert seed_record["state_sha256"] == sha256(np.asarray(initial).tobytes()).hexdigest()
    order = list(VARIANTS)
    rotation = metadata["groups"].index([regime, reconstruction, cells]) % 4
    order = order[rotation:] + order[:rotation]
    for variant in order:
        path = folder / (variant + ("__compile.json" if args.phase == "compile" else ".json"))
        if path.exists() and args.resume:
            continue
        if path.exists():
            raise ValueError(f"Refusing to replace {path}")
        binding = seed_binding if variant == "baseline" else build(fluids, regime, reconstruction, cells, variant)
        model, scheme, settings, ws, effective = binding
        model._study_regime = regime
        print("START", args.phase, name, variant, flush=True)
        preflight = None
        if args.phase == "warm":
            # Load already compiled executables before the timed full solve.
            started = perf_counter()
            ws.sample(initial)
            limits = ws.operator.timestep(initial, 2.)
            candidate, info = ws.operator.step(initial, jnp.float64(limits.dt), jnp.float64(0.))
            jax.block_until_ready(candidate)
            model.solution_audit(initial)
            preflight = dict(elapsed_s=perf_counter() - started, discarded_step=clean(info))
        activity_before = python_activity() if args.phase == "warm" else None
        result = solve_once(binding, initial, variant, args.phase,
                            args.compile_budget if args.phase == "compile" else args.warm_budget)
        attempts = []
        if args.phase == "warm" and result["compilation"]["backend_requests"]:
            attempts.append(result)
            result = solve_once(binding, initial, variant, "warm_cache_retry", args.warm_budget)
        clean_warm = args.phase == "warm" and result["compilation"]["backend_requests"] == 0
        result.update(regime=regime, reconstruction=reconstruction, cells=cells,
            inverse_free=variant in ("inverse_old_sonic", "both"), new_sonic=variant in ("sonic_only", "both"),
            scheme=asdict(scheme), steady=asdict(settings), effective=effective,
            preflight=preflight, discarded_cache_event_attempts=attempts,
            warm_s=result["elapsed_s"] if clean_warm else None, compilation_free=clean_warm,
            seed_file=str(seed_path.relative_to(args.output)), seed_sha256=sha(seed_path),
            activity_before=activity_before, activity_after=python_activity() if args.phase == "warm" else None,
            source_before=source, source_after=sources())
        result["source_unchanged"] = result["source_before"] == result["source_after"]
        save(path, result)
        print("END", args.phase, name, variant, "root", result["root_pass"], "reason", result["reason"],
              "seconds", result["elapsed_s"], "backend_requests", result["compilation"]["backend_requests"], flush=True)
        if not result["source_unchanged"]:
            raise RuntimeError("Source changed during solve")


def supervise(args, groups, phase):
    pending, running = list(groups), []
    failures = []
    count = args.jobs if phase == "compile" else 1
    while pending or running:
        while pending and len(running) < count:
            group = pending.pop(0)
            name = group_name(*group)
            expected = args.output / "groups" / name
            suffix = "__compile.json" if phase == "compile" else ".json"
            if args.resume and all((expected / (v + suffix)).exists() for v in VARIANTS):
                continue
            log_path = args.output / "logs" / (name + "__" + phase + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("ab")
            command = [sys.executable, "-B", "-u", "-m", "development.two_fluid_test_v2.scripts.run",
                "--output", str(args.output), "--cache", str(args.cache), "--phase", phase,
                "--worker", *map(str, group), "--compile-budget", str(args.compile_budget),
                "--warm-budget", str(args.warm_budget)]
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            running.append((process, stream, name))
            print("LAUNCH", phase, name, "pid", process.pid, flush=True)
        for item in list(running):
            process, stream, name = item
            if process.poll() is not None:
                stream.close()
                running.remove(item)
                print("FINISH", phase, name, "exit", process.returncode, flush=True)
                if process.returncode:
                    failures.append(dict(group=name, exit_code=process.returncode))
        if running:
            sleep(1.)
    save(args.output / ("workers_" + phase + ".json"), dict(complete=not failures, failures=failures))
    if failures:
        raise RuntimeError(f"{len(failures)} workers failed; inspect retained logs and resume after resolving the cause")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/ablation_20261104")
    parser.add_argument("--cache", type=Path, default=ROOT / ".jax_cache")
    parser.add_argument("--phase", choices=("all", "compile", "warm"), default="all")
    parser.add_argument("--worker", nargs=3, metavar=("CASE", "RECONSTRUCTION", "CELLS"))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--grids", nargs="+", type=int, default=list(GRIDS))
    parser.add_argument("--reconstructions", nargs="+", choices=RECONSTRUCTIONS, default=list(RECONSTRUCTIONS))
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent excluded compilation workers; warm samples are always serial")
    parser.add_argument("--compile-budget", type=float, default=1800.)
    parser.add_argument("--warm-budget", type=float, default=300.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or any(n < 2 for n in args.grids):
        raise ValueError("Use at least one worker and at least two cells")
    if args.worker:
        worker(args)
        return
    groups = [[case, rec, n] for n in args.grids for rec in args.reconstructions for case in args.cases]
    metadata = dict(source=sources(), groups=groups, variants=VARIANTS,
        configuration=dict(grids=args.grids, cases=args.cases, reconstructions=args.reconstructions,
                           compile_budget=args.compile_budget, warm_budget=args.warm_budget),
        protocol="One excluded complete compile/load run, discarded one-step preflight loading, then one serial compilation-free timed full solve from the identical seed; no sequencing or multigrid")
    path = args.output / "metadata.json"
    if path.exists():
        if not args.resume or read(path) != clean(metadata):
            raise ValueError("Existing or incompatible campaign; use a fresh directory")
    else:
        save(path, metadata)
        archive(args.output / "source.zip", metadata["source"])
        save(args.output / "started.json", dict(time=datetime.now(ZoneInfo("Europe/Paris")).isoformat(), cache=args.cache))
    if args.phase in ("all", "compile"):
        supervise(args, groups, "compile")
    if args.phase in ("all", "warm"):
        supervise(args, groups, "warm")
    save(args.output / ("completion_" + args.phase + ".json"), dict(completed=True, groups=len(groups), source_unchanged=sources() == metadata["source"]))


if __name__ == "__main__":
    main()
