"""Can an implicitly computed adapted expansion seed direct shocked solves?

Use run_adapted_initialization for the frozen source comparison. No production
equations, boundary conditions or fluxes are changed. The optional outlet-layer
seed is an initialization change, not a modification of the solved equations.
"""

import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import jaxprop as jxp
import numpy as np

from development.roberto.historical.newton_solve.scripts.experiment_support import (
    CASES,
    STUDY,
    clean,
    point_hash,
    source_hashes,
)
from development.roberto.historical.newton_solve.scripts.experiment_support import (
    RestAdmissibleNozzleProblem,
)
from development.roberto.historical.newton_solve.scripts.adapted_isolation import solve_isolated
from development.roberto.historical.newton_solve.scripts.pseudo_time import solve_implicit

METHODS = ("newton", "hybr", "lm", "slsqp")


def save(report, output):
    """Checkpoint all outcomes, including unfinished and wrong-branch solves."""
    (output / "summary.json").write_text(
        json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8"
    )


def field_audit(problem, point):
    """Distinguish the original PDE root from the requested physical branch."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = problem.diagnostics(np.asarray(point))
    result.update(
        exit_M=float(result["fields"]["M"][-2]),
        outlet_ghost_pressure_Pa=float(result["fields"]["p"][-1]),
        requested_backpressure_Pa=problem.p_back,
    )
    result["outlet_pressure_weight"] = float(
        0.5 * (1 + np.tanh((1 - abs(result["exit_M"])) / 0.004))
    )
    return result


def implicit_from_rest(problem):
    """Compute a numerical reference without an analytical initial guess."""
    raw = np.tile([problem.p0, 0.0, problem.h0], (len(problem.x), 1))
    result = solve_implicit(
        problem,
        residual_rms_tol=1e-11,
        max_iterations=5000,
        initial_state=problem.full_state(problem.pack(raw)),
        record_controls=True,
        state_is_admissible=problem.state_is_admissible,
    )
    if not result["native_success"]:
        raise RuntimeError("Implicit reference failed: " + result["message"])
    result["diagnostics"] = field_audit(problem, result["x"])
    return result


def outlet_adjusted_seed(problem, adapted_point, fraction):
    """Keep the adapted upstream field; initialize a subsonic outlet layer.

    Over the final fraction of the interior domain, blend pressure smoothly
    toward the requested backpressure. Obtain h(p,s0) and the spouting velocity
    from h0 = h + u²/2. No shock relation or shock-position information is used.
    These fields are initial guesses only; all six methods solve original R=0.
    """
    full = np.array(problem.full_state(adapted_point), copy=True)
    interior = full[1:-1].copy()
    x = problem.x[1:-1]
    start = x[-1] - fraction * (x[-1] - x[0])
    mask = x > start
    t = (x[mask] - start) / (x[-1] - start)
    weight = t * t * (3 - 2 * t)
    pressure = (1 - weight) * interior[mask, 0] + weight * problem.p_back
    props = problem.fluid.get_state(
        jxp.PSmass_INPUTS, pressure, np.full(pressure.shape, problem.s0)
    )
    enthalpy = np.asarray(props["h"])
    if np.any(enthalpy >= problem.h0):
        raise ValueError(
            "Outlet-layer pressure does not admit a positive spouting velocity"
        )
    interior[mask, 0] = pressure
    interior[mask, 2] = enthalpy
    interior[mask, 1] = np.sqrt(2 * (problem.h0 - enthalpy))
    full[1:-1] = interior
    return problem.pack(full)


def audit_saved(report, output):
    """Independently re-evaluate seeds, returned roots and branch decisions."""
    assert source_hashes() == report["source_before"] == report["source_after"]
    tolerance = report["metadata"]["residual_rms_tol"]
    evidence = dict(
        cases=[],
        source_unchanged=True,
        summary_sha256=hashlib.sha256(
            (output / "summary.json").read_bytes()
        ).hexdigest(),
    )
    for case in report["cases"]:
        problem = RestAdmissibleNozzleProblem(
            CASES[case["id"]], n_points=report["metadata"]["n_points"]
        )
        reference = np.asarray(case["implicit_reference"]["x"])
        reference_diagnostics = field_audit(problem, reference)
        assert reference_diagnostics["residual_rms"] <= 1e-11
        assert reference_diagnostics["analytical_pass"]
        assert reference_diagnostics["exit_M"] < 1
        np.testing.assert_array_equal(
            case["seeds"]["adapted"]["x"], report["adapted"]["x"]
        )
        records = []
        for seed in case["seeds"].values():
            assert point_hash(seed["x"]) == seed["initial_point_sha256"]
        for run in case["runs"]:
            assert (
                run["initial_point_sha256"]
                == case["seeds"][run["initialization"]]["initial_point_sha256"]
            )
            expected_library = "pysolver"
            assert run["backend"]["library"] == expected_library
            if expected_library == "pysolver":
                assert all(h["accepted"] is True for h in run["history"])
                options = run["backend"]["effective_options"]
                if run["method"] in ("newton", "slsqp"):
                    assert (
                        options["first_step_fraction"]
                        == report["metadata"]["first_step_fraction"]
                    )
                    assert (
                        options["step_growth_factor"]
                        == report["metadata"]["step_growth_factor"]
                    )
                else:
                    assert (
                        options["initial_radius"]
                        == report["metadata"]["initial_radius"]
                    )
                    assert (
                        options["growth_factor"]
                        == report["metadata"]["step_growth_factor"]
                    )
            assert run["first_observed_seed_max_difference"] in (None, 0.0)
            if run["authoritative_returned_point"]:
                np.testing.assert_array_equal(run["x"], run["audited_point"])
            diagnostics = field_audit(problem, run["audited_point"])
            np.testing.assert_allclose(
                diagnostics["residual_rms"],
                run["diagnostics"]["residual_rms"],
                rtol=1e-9,
                atol=1e-12,
            )
            root = bool(
                run["authoritative_returned_point"]
                and diagnostics["residual_rms"] <= tolerance
            )
            distance = float(
                np.linalg.norm(np.asarray(run["audited_point"]) - reference)
                / np.sqrt(reference.size)
            )
            same = bool(root and distance <= 1e-4)
            assert root == run["root_success"] and same == run["same_shocked_root"]
            assert diagnostics["analytical_pass"] == run["analytical_pass"]
            assert (
                bool(diagnostics["mass_imbalance"] < 1e-4)
                == run["mass_conservation_pass"]
            )
            records.append(
                dict(
                    method=run["method"],
                    initialization=run["initialization"],
                    residual_rms=diagnostics["residual_rms"],
                    root_success=root,
                    same_shocked_root=same,
                    exit_M=diagnostics["exit_M"],
                )
            )
        evidence["cases"].append(dict(id=case["id"], runs=records))
    (output / "audit.json").write_text(
        json.dumps(clean(evidence), indent=2), encoding="utf-8"
    )
    print(
        "Independent root, branch, seed, backend and source audit passed.", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=STUDY / "outputs" / "adapted_initialization"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["shock35", "shock70"],
        default=["shock35", "shock70"],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--seeds",
        nargs="+",
        choices=["adapted", "outlet_adjusted"],
        default=["adapted", "outlet_adjusted"],
    )
    parser.add_argument("--n-points", type=int, default=201)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--process-timeout-s", type=float, default=120)
    parser.add_argument("--max-wall-s", type=float, default=90)
    parser.add_argument("--first-step-fraction", type=float, default=0.25)
    parser.add_argument("--initial-radius", type=float, default=0.25)
    parser.add_argument("--outlet-fraction", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.outlet_fraction <= 1:
        parser.error("outlet-fraction must lie in (0,1]")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "summary.json"
    if args.audit_only or args.plot_only:
        report = json.loads(summary.read_text(encoding="utf-8"))
        if args.audit_only:
            audit_saved(report, args.output)
        from development.roberto.historical.newton_solve.scripts.adapted_reporting import render

        render(report, args.output)
        return
    metadata = dict(
        versions={
            name: version(name)
            for name in (
                "pysolver-view",
                "jaxprop",
                "numpy",
                "jax",
                "scipy",
            )
        },
        n_points=args.n_points,
        max_iterations=args.max_iterations,
        process_timeout_s=args.process_timeout_s,
        max_wall_s=args.max_wall_s,
        first_step_fraction=args.first_step_fraction,
        initial_radius=args.initial_radius,
        step_growth_factor=1.5,
        outlet_fraction=args.outlet_fraction,
        residual_rms_tol=1e-8,
        frozen_baseline_commit=os.environ.get("NEWTON_BASELINE_COMMIT"),
        formulation="Original scaled R=0; optimizers minimize zero subject to R=0",
        timing_note="Single serial observations, not repeated speed benchmarks",
    )
    if args.resume and summary.exists():
        report = json.loads(summary.read_text(encoding="utf-8"))
        assert report["metadata"] == metadata, "Run settings or versions changed"
        assert report["source_before"] == source_hashes(), "Imported source changed"
    else:
        if summary.exists():
            raise FileExistsError("Use --resume or a separate output directory")
        report = dict(
            metadata=metadata,
            source_before=source_hashes(),
            source_unchanged=None,
            cases=[],
        )
        print(
            "Generating adapted expansion by implicit integration from uniform rest...",
            flush=True,
        )
        adapted_problem = RestAdmissibleNozzleProblem(
            CASES["supersonic"], n_points=args.n_points
        )
        compile_costs = adapted_problem.warm_up()
        adapted = implicit_from_rest(adapted_problem)
        report["adapted"] = dict(
            implicit=adapted,
            x=adapted["x"],
            initial_point_sha256=point_hash(adapted["x"]),
            compile_costs=compile_costs,
            backpressure_ratio=adapted_problem.p_back / adapted_problem.p0,
        )
        save(report, args.output)
    adapted_point = np.asarray(report["adapted"]["x"])
    adapted_residual = np.asarray(
        RestAdmissibleNozzleProblem(
            CASES["supersonic"], n_points=args.n_points
        ).residual(adapted_point)
    )
    for case_id in args.cases:
        print(f"=== {case_id} ===", flush=True)
        problem = RestAdmissibleNozzleProblem(CASES[case_id], n_points=args.n_points)
        problem.warm_up()
        case = next((c for c in report["cases"] if c["id"] == case_id), None)
        if case is None:
            reference = implicit_from_rest(problem)
            seeds = {}
            for name, point in [
                ("adapted", adapted_point),
                (
                    "outlet_adjusted",
                    outlet_adjusted_seed(problem, adapted_point, args.outlet_fraction),
                ),
            ]:
                diagnostics = field_audit(problem, point)
                seeds[name] = dict(
                    x=point.copy(),
                    initial_point_sha256=point_hash(point),
                    diagnostics=diagnostics,
                )
            # An exact adapted root also stops an RMS-driven implicit solve immediately.
            stopped = solve_implicit(
                problem,
                initial_state=problem.full_state(adapted_point),
                state_is_admissible=problem.state_is_admissible,
            )
            case = dict(
                id=case_id,
                label=CASES[case_id]["label"],
                x_m=problem.x,
                p_back_ratio=problem.p_back / problem.p0,
                settings=problem.settings,
                reference=problem.reference,
                implicit_reference=reference,
                implicit_from_adapted=stopped,
                seeds=seeds,
                runs=[],
                adapted_residual_change=float(
                    np.max(np.abs(problem.residual(adapted_point) - adapted_residual))
                ),
            )
            report["cases"].append(case)
            save(report, args.output)
        for initialization in args.seeds:
            seed = case["seeds"][initialization]
            print(
                f'  {initialization}: initial RMS {seed["diagnostics"]["residual_rms"]:.3e}; exit M {seed["diagnostics"]["exit_M"]:.4f}',
                flush=True,
            )
            for method in args.methods:
                if any(
                    r["method"] == method and r["initialization"] == initialization
                    for r in case["runs"]
                ):
                    continue
                print(f"  -> {method}", flush=True)
                custom = dict(
                        residual_rms_tol=1e-8,
                        first_step_fraction=args.first_step_fraction,
                        initial_radius=args.initial_radius,
                        step_growth_factor=1.5,
                )
                result = solve_isolated(
                    problem,
                    seed["x"],
                    method,
                    output=args.output,
                    run_id=f"{case_id}_{initialization}_{method}",
                    process_timeout_s=args.process_timeout_s,
                    max_iterations=args.max_iterations,
                    max_wall_s=args.max_wall_s,
                    custom_options=custom,
                )
                diagnostics = field_audit(problem, result["audited_point"])
                distance = float(
                    np.linalg.norm(
                        np.asarray(result["audited_point"])
                        - np.asarray(case["implicit_reference"]["x"])
                    )
                    / np.sqrt(adapted_point.size)
                )
                root = bool(
                    result["authoritative_returned_point"]
                    and diagnostics["residual_rms"] <= 1e-8
                )
                result.update(
                    initialization=initialization,
                    initial_point_sha256=seed["initial_point_sha256"],
                    diagnostics=diagnostics,
                    root_success=root,
                    scaled_distance_to_shocked_root=distance,
                    same_shocked_root=bool(root and distance <= 1e-4),
                    mass_conservation_pass=bool(diagnostics["mass_imbalance"] < 1e-4),
                    analytical_pass=diagnostics["analytical_pass"],
                )
                case["runs"].append(result)
                save(report, args.output)
                print(
                    f'    root={root}, shocked branch={result["same_shocked_root"]}, RMS={diagnostics["residual_rms"]:.3e}, M_exit={diagnostics["exit_M"]:.3f}',
                    flush=True,
                )
    report["source_after"] = source_hashes()
    report["source_unchanged"] = report["source_before"] == report["source_after"]
    assert report["source_unchanged"]
    save(report, args.output)
    audit_saved(report, args.output)
    from development.roberto.historical.newton_solve.scripts.adapted_reporting import render

    render(report, args.output)
    print(f"Saved results and report under {args.output}", flush=True)


if __name__ == "__main__":
    main()
