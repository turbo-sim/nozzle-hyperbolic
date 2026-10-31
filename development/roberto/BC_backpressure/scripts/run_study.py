"""Compare current and pressure-wave outlets with implicit time integration.

Both methods start from identical uniform or numerical adapted fields. Run through launch
to select the recorded source snapshot; analytical fields are references only.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from .time_integration import solve_implicit
from .problem import BoundaryProblem

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
CASES = {
    "adapted": baseline.CASES[0],
    "shock35": baseline.CASES[1],
    "shock70": baseline.CASES[2],
}
VARIANTS = ("source", "pressure_wave")
INITIALIZATIONS = ("uniform", "adapted")
SETTINGS = dict(
    baseline.SOLVER,
    CFL=0.2,
    ser_beta=0.0,
    ser_cfl_max=1e6,
    dt_growth_factor=1.2,
    newton_iter=8,
    newton_tol=1e-10,
)


def point_hash(point):
    """Fingerprint the copied dimensionless numerical coordinates."""
    return hashlib.sha256(np.asarray(point, dtype=np.float64).tobytes()).hexdigest()


def controller_settings(max_iterations=5000):
    """Expose the shared implicit helper's effective control policies."""
    settings = SETTINGS
    return dict(
        policy="Fixed CFL growth after acceptance; halving after failure; no SER",
        shrink_factor=0.5,
        implicit_residual_tolerance=1e-10,
        recommendation="historical/euler_annular/cfl_investigation/REPORT.md",
        nominal_cfl=settings["CFL"],
        ser_beta=settings["ser_beta"],
        ser_cfl_max=settings["ser_cfl_max"],
        dt_growth_factor=settings["dt_growth_factor"],
        max_halvings=12,
        max_trials_per_update=13,
        relative_increment_limits=dict(p=0.2, h=0.1, u_scaled=0.2),
        increment_check_scope="Full state including algebraic ghosts",
        newton_max_iterations=settings["newton_iter"],
        newton_relative_update_tolerance=settings["newton_tol"],
        jacobian=settings["jacobian"],
        linear_solver=settings["linear_solver"],
        spatial_rms_tolerance=1e-8,
        max_accepted_updates=max_iterations,
    )


def clean(value):
    """Encode actual records with unavailable values represented by null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def source_hashes():
    import nozzle_hyperbolic

    package = Path(nozzle_hyperbolic.__file__).resolve().parent
    return {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }


def shared_hashes():
    names = [
        "development/roberto/historical/euler_nozzle/verify_euler_nozzle_steady.py",
        "development/roberto/historical/newton_solve/scripts/problem.py",
        "development/roberto/historical/newton_solve/scripts/pseudo_time.py",
    ]
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }


def save(report, output):
    (output / "summary.json").write_text(
        json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8"
    )


def compile_problem(problem, point):
    """Record first-call costs without using the discarded update as a seed."""
    costs = {}
    callbacks = [
        ("residual", lambda: problem.residual(point)),
        ("jacobian", lambda: problem.gradient(point)),
        ("state", lambda: problem.full_state(point)),
        (
            "step",
            lambda: problem.step(
                problem.full_state(point),
                problem.compute_dt(problem.full_state(point))[0] / 64,
            ),
        ),
        ("mass_flux", lambda: problem.face_mass_flux(problem.full_state(point))),
    ]
    for name, callback in callbacks:
        started = perf_counter()
        jax.block_until_ready(callback())
        costs[name + "_first_call_s"] = perf_counter() - started
    return costs


def implicit(problem, point, tolerance=1e-8, max_iterations=5000):
    """Use the existing controller and record each accepted boundary state."""
    problem.boundary_trace = []
    problem.record_boundary_trace = True
    try:
        result = solve_implicit(
            problem,
            residual_rms_tol=tolerance,
            max_iterations=max_iterations,
            initial_state=problem.full_state(np.asarray(point)),
            max_halvings=12,
            record_controls=True,
            state_is_admissible=problem.state_is_admissible,
        )
    finally:
        problem.record_boundary_trace = False
    assert len(result["history"]) == len(problem.boundary_trace)
    for row, info in zip(result["history"], problem.boundary_trace):
        row["boundary"] = info
    return result


def uniform(problem):
    return problem.pack(np.tile([problem.p0, 0.0, problem.h0], (len(problem.x), 1)))


def physical_success(diagnostics):
    """Require the intended shock and pressure-active forward outflow."""
    return bool(
        diagnostics["residual_rms"] <= 1e-8
        and 0 < diagnostics["exit_M"] < 1
        and diagnostics["analytical_pass"]
        and diagnostics["mass_imbalance"] < 1e-4
        and diagnostics["boundary_pressure_relative_error"] < 1e-6
        and diagnostics["shock_location_m"] is not None
        and diagnostics["boundary"]["mode"] != 7
    )


def attach_audit(problem, result, initial, reference=None):
    diagnostics = problem.diagnostics(np.asarray(result["x"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        initial_diagnostics = problem.diagnostics(np.asarray(initial))
    result.update(
        variant=problem.variant,
        solver="implicit",
        diagnostics=diagnostics,
        root_success=bool(diagnostics["residual_rms"] <= 1e-8),
        physical_success=physical_success(diagnostics),
        initial_point_sha256=point_hash(initial),
        initial_diagnostics=initial_diagnostics,
    )
    if reference is not None:
        result["distance_to_source_reference"] = float(
            np.linalg.norm(np.asarray(result["x"]) - np.asarray(reference))
            / np.sqrt(len(initial))
        )
    return result


def verify_saved(report, output):
    """Independently recheck saved fields, common seeds and accepted CFLs."""
    assert source_hashes() == report["source_before"] == report["source_after"]
    assert shared_hashes() == report["shared_before"] == report["shared_after"]
    findings = []
    for case in report["cases"]:
        assert {(r["initialization"], r["variant"]) for r in case["runs"]} == {
            (i, v) for i in INITIALIZATIONS for v in VARIANTS
        }
        assert len(case["runs"]) == 4
        for run in case["runs"]:
            assert run["solver"] == "implicit"
            assert run["initialization"] in INITIALIZATIONS
            problem = BoundaryProblem(
                CASES[case["id"]],
                variant=run["variant"],
                n_points=report["metadata"]["mesh"]["n_points"],
                solver_settings=run["solver_settings"],
            )
            initial = (
                uniform(problem)
                if run["initialization"] == "uniform"
                else np.asarray(report["adapted"]["x"])
            )
            assert run["initial_point_sha256"] == point_hash(initial)
            d = problem.diagnostics(np.asarray(run["x"]))
            initial_rms = float(
                np.linalg.norm(problem.residual(initial)) / np.sqrt(len(run["x"]))
            )
            np.testing.assert_allclose(
                run["history"][0]["residual_rms"], initial_rms, atol=1e-12, rtol=1e-9
            )
            np.testing.assert_allclose(
                d["residual_rms"],
                run["diagnostics"]["residual_rms"],
                atol=1e-12,
                rtol=1e-9,
            )
            assert bool(d["residual_rms"] <= 1e-8) == run["root_success"]
            assert physical_success(d) == run["physical_success"]
            expected_cfl = SETTINGS["CFL"]
            for trial in run["trial_history"]:
                np.testing.assert_allclose(
                    trial["effective_cfl"], expected_cfl, rtol=1e-12
                )
                expected_cfl = (
                    min(expected_cfl * 1.2, 1e6)
                    if trial["accepted"]
                    else expected_cfl * 0.5
                )
                assert trial["inner_iterations"] <= 8
                if trial["accepted"]:
                    assert trial["inner_relative_update"] <= 1e-10
                    assert trial["implicit_residual_rms"] <= 1e-10
            history = run["history"]
            accepted = [trial for trial in run["trial_history"] if trial["accepted"]]
            assert len(history) == run["iterations"] + 1 == len(accepted) + 1
            assert history[0].get("effective_cfl") is None
            for index, (row, trial) in enumerate(zip(history[1:], accepted), start=1):
                assert (
                    row["accepted_index"] == index
                    and trial["anchor_index"] == index - 1
                )
                assert (
                    0
                    < row["effective_cfl"]
                    <= report["metadata"]["controller"]["ser_cfl_max"]
                )
                np.testing.assert_allclose(
                    row["dt_s"], trial["dt_s"], rtol=1e-13, atol=0
                )
                np.testing.assert_allclose(
                    row["effective_cfl"], trial["effective_cfl"], rtol=1e-13, atol=0
                )
            findings.append(
                dict(
                    case=case["id"],
                    initialization=run["initialization"],
                    variant=run["variant"],
                    residual_rms=d["residual_rms"],
                    physical_success=physical_success(d),
                    accepted_cfl_samples=len(accepted),
                    cfl_min=min(
                        (h["effective_cfl"] for h in history[1:]), default=None
                    ),
                    cfl_max=max(
                        (h["effective_cfl"] for h in history[1:]), default=None
                    ),
                )
            )
    live = {
        p.relative_to(ROOT / "src" / "nozzle_hyperbolic")
        .as_posix(): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in (ROOT / "src" / "nozzle_hyperbolic").rglob("*.py")
    }
    audit = dict(
        passed=True,
        source_snapshot_unchanged=True,
        shared_imports_unchanged=True,
        live_source_unchanged=live == report["source_before"],
        all_solver_initial_histories_match_selected_seed=True,
        accepted_cfl_matches_recorded_trials=True,
        summary_sha256=hashlib.sha256(
            (output / "summary.json").read_bytes()
        ).hexdigest(),
        runs=findings,
    )
    (output / "audit.json").write_text(
        json.dumps(clean(audit), indent=2), encoding="utf-8"
    )
    print(
        "Independent field / initialization / accepted-CFL / source audit passed.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs" / "recommended_cfl"
    )
    parser.add_argument("--n-points", type=int, default=201)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(STUDY):
        raise ValueError("Output outside study")
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    if args.plot_only or args.audit_only:
        report = json.loads(summary.read_text(encoding="utf-8"))
        if args.audit_only:
            verify_saved(report, output)
        from .report import render

        render(clean(report), output, STUDY / "REPORT.html")
        return
    if summary.exists():
        if not args.resume:
            raise FileExistsError("Use --resume or a new study output directory")
        report = json.loads(summary.read_text(encoding="utf-8"))
        assert (
            report["source_before"] == source_hashes()
            and report["shared_before"] == shared_hashes()
        )
        assert report["metadata"]["mesh"]["n_points"] == args.n_points
        assert report["metadata"]["controller"] == controller_settings(
            args.max_iterations
        )
        assert report["metadata"]["settings"] == SETTINGS
    else:
        report = dict(
            metadata=dict(
                versions={
                    name: version(name)
                    for name in ["jax", "jaxlib", "jaxprop", "numpy", "scipy"]
                },
                mesh=dict(n_points=args.n_points, n_unknowns=3 * (args.n_points - 2)),
                settings=SETTINGS,
                controller=controller_settings(args.max_iterations),
                precision="float64",
                initialization="Matched uniform stagnation and numerical adapted starts",
                source_baseline="Working-source snapshot at study creation",
                pressure_wave_flux="Physical sampled trace on incoming/sonic/wall final face; FVS on outgoing branch",
                timing_note="Single serial observations including checked-step first-call compilation; not repeated performance benchmarks",
            ),
            source_before=source_hashes(),
            shared_before=shared_hashes(),
            cases=[],
        )
        source = BoundaryProblem(
            CASES["adapted"],
            variant="source",
            n_points=args.n_points,
            solver_settings=SETTINGS,
        )
        initial = uniform(source)
        compilation = compile_problem(source, initial)
        adapted = implicit(source, initial, 1e-11, args.max_iterations)
        if not adapted["native_success"]:
            raise RuntimeError("Adapted solve failed: " + adapted["message"])
        adapted.update(
            diagnostics=source.diagnostics(adapted["x"]),
            initial_point_sha256=point_hash(adapted["x"]),
            compilation=compilation,
        )
        report["adapted"] = adapted
        save(report, output)
    for case_id in ("shock35", "shock70"):
        print(f"=== {case_id} ===", flush=True)
        case = next((c for c in report["cases"] if c["id"] == case_id), None)
        if case is None:
            source = BoundaryProblem(
                CASES[case_id],
                variant="source",
                n_points=args.n_points,
                solver_settings=SETTINGS,
            )
            compilation = compile_problem(source, uniform(source))
            reference = implicit(source, uniform(source), 1e-11, args.max_iterations)
            reference["diagnostics"] = source.diagnostics(reference["x"])
            if (
                not reference["native_success"]
                or not reference["diagnostics"]["analytical_pass"]
            ):
                raise RuntimeError("Independent source reference failed")
            case = dict(
                id=case_id,
                label=CASES[case_id]["label"],
                x_m=source.x,
                p_back_ratio=source.p_back / source.p0,
                reference=source.reference,
                source_reference=reference,
                runs=[],
                reference_compilation=compilation,
            )
            report["cases"].append(case)
        for initialization, variant in [
            (i, v) for i in INITIALIZATIONS for v in VARIANTS
        ]:
            if any(
                r["variant"] == variant and r["initialization"] == initialization
                for r in case["runs"]
            ):
                continue
            print(
                f"  {initialization} / {variant}: compile, audit and integrate...",
                flush=True,
            )
            problem = BoundaryProblem(
                CASES[case_id],
                variant=variant,
                n_points=args.n_points,
                solver_settings=SETTINGS,
            )
            initial = (
                uniform(problem)
                if initialization == "uniform"
                else np.asarray(report["adapted"]["x"])
            )
            compilation = compile_problem(problem, initial)
            assembly = problem.assembly_audit(initial)
            assert (
                assembly["finite_jacobian"]
                and assembly["colored_dense_relative_max"] < 1e-10
            )
            assert assembly["mass_flux_divergence_max"] < 1e-10
            if variant == "source":
                assert assembly["source_residual_scaled_max"] < 1e-10
                assert assembly["source_jacobian_relative_max"] < 1e-10
                assert assembly["source_step_scaled_max"] < 1e-10
            result = implicit(problem, initial, max_iterations=args.max_iterations)
            attach_audit(problem, result, initial, case["source_reference"]["x"])
            result.update(
                initialization=initialization,
                compilation=compilation,
                assembly_audit=assembly,
                solver_settings=problem.settings,
            )
            case["runs"].append(result)
            save(report, output)
            print(
                f'    RMS={result["diagnostics"]["residual_rms"]:.3e}; physical={result["physical_success"]}',
                flush=True,
            )
    report["source_after"] = source_hashes()
    report["shared_after"] = shared_hashes()
    report["source_unchanged"] = report["source_before"] == report["source_after"]
    save(report, output)
    verify_saved(report, output)
    from .report import render

    render(clean(report), output, STUDY / "REPORT.html")


if __name__ == "__main__":
    main()
