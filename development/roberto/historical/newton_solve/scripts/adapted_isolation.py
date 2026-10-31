"""Bound custom direct-solver work in a task-owned subprocess.

Retain actual returned states and streamed accepted observations; timeouts
remain diagnostic candidates rather than claimed optimizer returns.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[5]


def _clean(value):
    """Reuse the study serializer lazily, avoiding its driver import cycle."""
    from development.roberto.historical.newton_solve.scripts.experiment_support import clean

    return clean(value)


def _read_observations(path):
    """Read flushed native observations, ignoring an interrupted final line."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    observations = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            observations.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return observations


def solve_isolated(
    problem,
    initial,
    method,
    *,
    output,
    run_id,
    process_timeout_s=120,
    max_iterations=300,
    native_tolerance=1e-10,
    max_wall_s=180,
    custom_options=None,
):
    """Run one controlled custom call with an external time limit.

    The timeout includes interpreter startup, model reconstruction and JIT.
    Normal returns retain the native solve timing separately. A timeout has
    no authoritative solution: only the last observed candidate or initial
    point can be offered for a clearly labeled diagnostic evaluation.
    """
    import numpy as np
    from development.roberto.historical.newton_solve.scripts.investigate_ptc import point_hash

    if method not in ("newton", "hybr", "lm", "slsqp") or custom_options is None:
        raise ValueError("Expected a custom direct method and explicit custom options")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)):
        raise ValueError(
            "run_id must contain only letters, digits, underscores and hyphens"
        )
    if not np.isfinite(process_timeout_s) or process_timeout_s <= 0:
        raise ValueError("process_timeout_s must be finite and positive")
    initial = np.array(initial, dtype=np.float64, copy=True)
    directory = Path(output).resolve() / "workers" / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    result_path = directory / "result.json"
    observation_path = directory / "observations.jsonl"
    setup_path = directory / "setup.json"
    # These named artifacts belong to this specific experimental run.
    result_path.unlink(missing_ok=True)
    setup_path.unlink(missing_ok=True)
    observation_path.write_text("", encoding="utf-8")
    options = dict(
        max_iterations=max_iterations,
        native_tolerance=native_tolerance,
        max_wall_s=max_wall_s,
    )
    request = dict(
        case=problem.case,
        n_points=len(problem.x),
        settings=problem.settings,
        initial=initial,
        initial_sha256=point_hash(initial),
        method=method,
        options=options,
        custom_options=custom_options,
    )
    request_path.write_text(
        json.dumps(_clean(request), indent=2, allow_nan=False), encoding="utf-8"
    )
    environment = dict(
        os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg"
    )
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        environment[name] = "1"
    started = perf_counter()
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "development.roberto.historical.newton_solve.scripts.adapted_isolation",
            "--worker",
            str(request_path),
        ],
        cwd=str(ROOT),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
    )
    timed_out = False
    try:
        code = child.wait(timeout=process_timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Kill only the Popen child created above, never unrelated solver work.
        child.kill()
        code = child.wait()
    elapsed = perf_counter() - started
    setup = (
        json.loads(setup_path.read_text(encoding="utf-8"))
        if setup_path.is_file()
        else {}
    )
    worker = dict(
        run_id=str(run_id),
        pid=child.pid,
        exit_code=code,
        process_timeout_s=float(process_timeout_s),
        elapsed_s=elapsed,
        timed_out=timed_out,
        request_path=str(request_path),
        observation_path=str(observation_path),
        worker_setup_s=setup.get("worker_setup_s"),
        initial_sha256=request["initial_sha256"],
    )
    if not timed_out and code == 0 and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("worker_initial_sha256") != request["initial_sha256"]:
            raise RuntimeError(
                "Worker seed fingerprint differs from the requested seed"
            )
        result.update(
            worker=worker,
            worker_elapsed_s=elapsed,
            wall_s_scope="native solve and callbacks; worker setup recorded separately",
        )
        return result

    history = _read_observations(observation_path)
    last = next(
        (row["x"] for row in reversed(history) if row.get("x") is not None), None
    )
    point = np.asarray(last, dtype=float) if last is not None else initial.copy()
    error_name = "WorkerTimeout" if timed_out else "WorkerError"
    message = (
        f"Isolated native worker exceeded {process_timeout_s:g} s including setup; "
        "no authoritative returned optimizer point is available."
        if timed_out
        else f"Isolated native worker exited with code {code} without a returned result."
    )
    print(f"    {method}: {message}", flush=True)
    effective = dict(custom_options, max_iterations=max_iterations,
                     native_tolerance=native_tolerance, max_wall_s=max_wall_s)
    first = next((row["x"] for row in history if row.get("x") is not None), None)
    return dict(
        method=method,
        x=None,
        last_observed_candidate=last,
        audited_point=point,
        diagnostic_point_source="last_observation" if last is not None else "initial",
        authoritative_returned_point=False,
        history=history,
        wall_s=elapsed,
        wall_s_scope="whole isolated worker incl setup (native return unavailable)",
        exception=dict(type=error_name, message=message),
        cancellation_reason=(
            "absolute isolated-process wall-time budget"
            if timed_out
            else "isolated worker error"
        ),
        native_success=False,
        native_status=None,
        native_message=message,
        iterations=None,
        nfev=None,
        njev=None,
        model_work=dict(
            residual_calls=None, jacobian_calls=None, residual_s=None, jacobian_s=None
        ),
        first_observed_seed_max_difference=(
            None
            if first is None
            else float(np.max(np.abs(np.asarray(first) - initial)))
        ),
        worker=worker,
        worker_elapsed_s=elapsed,
        backend=dict(
            library="pysolver",
            method=method,
            info=None,
            effective_options=effective,
            effective_options_source="requested configuration; native return unavailable",
            hessian_model=None,
            history_note="Flushed initial/accepted custom-driver states; no returned point.",
            initial_step_controls_available=True,
        ),
    )


def run_worker(request_path):
    """Rebuild one model, stream observations and save an actual native return."""
    started = perf_counter()
    import numpy as np
    from development.roberto.historical.newton_solve.scripts.investigate_ptc import point_hash
    from development.roberto.historical.newton_solve.scripts.experiment_support import RestAdmissibleNozzleProblem

    request_path = Path(request_path).resolve()
    directory = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    initial = np.asarray(request["initial"], dtype=np.float64)
    assert (
        point_hash(initial) == request["initial_sha256"]
    ), "JSON seed roundtrip changed the initial state"
    problem = RestAdmissibleNozzleProblem(
        request["case"],
        n_points=request["n_points"],
        solver_settings=request["settings"],
    )
    # Only the two operators needed by direct optimizers are precompiled.
    problem.residual(initial)
    problem.gradient(initial)
    setup = dict(
        worker_setup_s=perf_counter() - started, initial_sha256=point_hash(initial)
    )
    (directory / "setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    with (directory / "observations.jsonl").open(
        "a", encoding="utf-8", buffering=1
    ) as stream:

        def observe(record):
            """Persist the supplied detached observation, without model evaluations."""
            stream.write(
                json.dumps(_clean(record), separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            stream.flush()

        from development.roberto.historical.newton_solve.scripts.adapted_solvers import solve_custom
        options = request["options"]
        result = solve_custom(
            problem, initial, request["method"], observation_sink=observe,
            max_iterations=options["max_iterations"],
            native_tolerance=options["native_tolerance"], max_wall_s=options["max_wall_s"],
            **request["custom_options"],
        )
    result.update(worker_initial_sha256=point_hash(initial), **setup)
    temporary = directory / "result.json.tmp"
    temporary.write_text(
        json.dumps(_clean(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(directory / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    run_worker(args.worker)


if __name__ == "__main__":
    main()
