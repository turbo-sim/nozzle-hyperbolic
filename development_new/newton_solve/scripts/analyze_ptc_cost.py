"""Diagnose PTC cost and progress without changing the production solver.

Run: poetry run python -m development_new.newton_solve.scripts.analyze_ptc_cost
The counter-enabled implicit kernel is an in-memory copy with one extra return.
All PTC corrections still use the custom PySolverView backend.
"""

import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import jax
import numpy as np
from development_new.euler_nozzle import verify_euler_nozzle_steady as baseline
from development_new.newton_solve.scripts import problem as problem_module
from development_new.newton_solve.scripts import ptc
from development_new.newton_solve.scripts.ptc_storage import PhysicalStorage
from development_new.newton_solve.scripts.pseudo_time import solve_implicit
from development_new.newton_solve.scripts.investigate_ptc import CASE_IDS, json_value
from nozzle_hyperbolic.models.euler_nozzle import fvm

MEASURED_SETTINGS = dict(
    CFL=20.0,
    newton_iter=8,
    newton_tol=1e-8,
    ser_beta=1.0,
    ser_cfl_max=500.0,
    dt_growth_factor=1.2,
)


def counted_factory(return_inner_status=False):
    """Return an otherwise unchanged factory with the inner count exposed."""
    source = inspect.getsource(fvm.make_jax_stadtke_step)
    target = "_, U_k, _ = jax.lax.while_loop("
    assert source.count(target) == 1
    source = source.replace(
        target, "inner_count, U_k, inner_relative_update = jax.lax.while_loop("
    )
    target = "U_k = jax.lax.fori_loop(0, newton_iter, body, U_k)"
    assert source.count(target) == 1
    source = source.replace(
        target,
        target
        + "\n            inner_count = jnp.int32(newton_iter)\n            inner_relative_update = jnp.float64(jnp.nan)",
    )
    target = "return U_new, ssr"
    assert source.count(target) == 1
    returned = "return U_new, ssr, inner_count"
    if return_inner_status:
        returned += ", inner_relative_update"
    source = source.replace(target, returned)
    namespace = dict(fvm.__dict__)
    exec(compile(source, "<counted implicit factory>", "exec"), namespace)
    return namespace["make_jax_stadtke_step"]


def implicit_replay(case, settings):
    with patch.object(problem_module, "make_jax_stadtke_step", counted_factory()):
        problem = problem_module.SteadyNozzleProblem(
            case, solver_settings=dict(MEASURED_SETTINGS, **settings)
        )
    problem.warm_up()
    raw_step = problem.step
    records = []

    def step(state, dt):
        new, ssr, count = raw_step(state, dt)
        jax.block_until_ready(new)
        tau = float(problem.compute_dt(state)[0]) / problem.settings["CFL"]
        records.append(
            dict(
                dt=float(dt),
                effective_cfl=float(dt) / tau,
                inner_solves=int(count),
                ssr=float(ssr),
            )
        )
        return new, ssr

    problem.step = step
    result = solve_implicit(problem, initial_state=problem.full_state(problem.x0))
    accepted = []
    previous_trials = 0
    for row in result["history"][1:]:
        trials = row["implicit_step_trials"]
        accepted.append(
            dict(
                records[trials - 1],
                accepted_index=row["accepted_index"],
                residual_rms=row["residual_rms"],
                trials=trials - previous_trials,
            )
        )
        previous_trials = trials
    return problem, dict(
        success=result["native_success"],
        updates=len(result["history"]) - 1,
        residual_rms=result["history"][-1]["residual_rms"],
        x=result["x"],
        total_linear_solves=sum(x["inner_solves"] for x in records),
        trials=len(records),
        accepted_history=accepted,
        all_trials=records,
        note="Counter instrumentation adds host monitoring; replay timings are not benchmarked.",
    )


def profile_custom_ptc(problem, iterations):
    """Time the actual public custom backend and its nested QR/condition work."""
    import pysolver_view.roots.solvers.linear_model as lm

    storage = PhysicalStorage(problem)
    storage.warm_up()
    problem.reset_counters()
    timers = dict(affine_s=0.0, qr_s=0.0, condition_s=0.0, affine_calls=0, qr_calls=0)
    original_affine, original_qr = ptc._one_custom_correction, lm.qr
    original_condition = lm.LinearModel._check_condition

    def affine(*args, **kwargs):
        start = perf_counter()
        try:
            return original_affine(*args, **kwargs)
        finally:
            timers["affine_s"] += perf_counter() - start
            timers["affine_calls"] += 1

    def qr(*args, **kwargs):
        start = perf_counter()
        try:
            return original_qr(*args, **kwargs)
        finally:
            timers["qr_s"] += perf_counter() - start
            timers["qr_calls"] += 1

    def condition(*args, **kwargs):
        start = perf_counter()
        try:
            return original_condition(*args, **kwargs)
        finally:
            timers["condition_s"] += perf_counter() - start

    with patch.object(ptc, "_one_custom_correction", affine), patch.object(
        lm, "qr", qr
    ), patch.object(lm.LinearModel, "_check_condition", condition):
        result = ptc.solve_ptc(problem, storage, problem.x0, max_iterations=iterations)
    return dict(
        timers,
        total_s=result["wall_s"],
        accepted_updates=result["iterations"],
        model_work=problem.model_work(),
        storage_work=storage.model_work(),
        backend=result["backend"],
        root_success=result["root_success"],
        message=result["message"],
        note="Short prefix cost profile, not a new converged solution; nested timers overlap.",
    )


def matched_first_correction(case):
    """Compare one implicit Newton correction to full-fraction custom PTC."""
    with patch.object(problem_module, "make_jax_stadtke_step", counted_factory()):
        problem = problem_module.SteadyNozzleProblem(
            case, solver_settings=dict(MEASURED_SETTINGS, newton_iter=1, newton_tol=0)
        )
    problem.warm_up()
    storage = PhysicalStorage(problem)
    storage.warm_up()
    z = problem.x0.copy()
    state = problem.full_state(z)
    dt = 20 * storage.characteristic_dt(z)
    new, _, count = problem.step(state, dt)
    jax.block_until_ready(new)
    r = problem.residual(z)
    matrix = problem.gradient(z) + storage.matrix(z) / dt
    candidate, backend = ptc._one_custom_correction(z, r, matrix, 1.0)
    return dict(
        cfl=20,
        inner_count=int(count),
        backend=backend,
        scaled_max_difference=float(np.max(np.abs(problem.pack(new) - candidate))),
        scaled_rms_difference=float(
            np.linalg.norm(problem.pack(new) - candidate) / np.sqrt(z.size)
        ),
    )


def plot_analysis(report, comparison, output):
    """Separate controller-induced slow progress from measured per-trial cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.colormaps["plasma"](np.linspace(0.2, 0.8, 3))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for axis, name in zip(axes[:2], ("shock35", "shock70")):
        case = next(c for c in comparison["cases"] if c["id"] == name)
        history = next(r for r in case["runs"] if r["method"] == "ptc")["history"][1:]
        for hist, text, color, style in (
            (
                report["replay"][name]["default"]["accepted_history"],
                "Implicit (up to 8 inner solves)",
                colors[0],
                "-",
            ),
            (history, "Current Newton/PTC", colors[1], "--"),
            (
                report["replay"][name]["one_inner"]["accepted_history"],
                "Implicit controller, 1 inner solve",
                colors[2],
                ":",
            ),
        ):
            axis.semilogy(
                [v["accepted_index"] for v in hist],
                [v.get("effective_cfl", v.get("cfl")) for v in hist],
                color=color,
                ls=style,
                label=text,
            )
        axis.set(title=name, xlabel="Accepted outer update", ylabel="Accepted CFL")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    prof = report["profile"]
    jac = prof["model_work"]["jacobian_s"]
    model = (
        prof["model_work"]["residual_s"]
        + prof["storage_work"]["matrix_s"]
        + prof["storage_work"]["timescale_s"]
    )
    values = [
        prof["qr_s"],
        prof["condition_s"],
        prof["affine_s"] - prof["qr_s"] - prof["condition_s"],
        jac,
        model,
        prof["total_s"] - prof["affine_s"] - jac - model,
    ]
    labels = [
        "Dense pivoted QR",
        "Condition checks",
        "Other affine/backend",
        "Spatial Jacobian",
        "Residual / storage / CFL",
        "Remaining controller",
    ]
    axes[2].barh(
        labels[::-1],
        100 * np.array(values[::-1]) / prof["total_s"],
        color=plt.colormaps["plasma"](np.linspace(0.2, 0.8, 6)),
    )
    axes[2].set(
        xlabel="Fraction of profiled prefix time [%]",
        title=f"{prof['accepted_updates']}-update custom PTC prefix",
    )
    axes[2].grid(axis="x", alpha=0.2)
    fig.suptitle("PTC performance: timestep recovery and per-correction cost")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"cost_and_cfl.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/performance_analysis",
    )
    parser.add_argument("--profile-iterations", type=int, default=200)
    parser.add_argument("--ptc-ablations", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/summary.json",
    )
    args = parser.parse_args()
    if args.profile_iterations < 1:
        parser.error("profile-iterations must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).resolve().parents[3]
        / "src/nozzle_hyperbolic/models/euler_nozzle/fvm.py"
    )
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = dict(production_source_sha256=before, replay={}, controller_ablations={})
    if args.plot_only:
        report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        plot_analysis(
            report, json.loads(args.comparison.read_text(encoding="utf-8")), output
        )
        return

    def save():
        (output / "analysis.json").write_text(
            json.dumps(report, indent=2, default=json_value), encoding="utf-8"
        )

    cases = dict(zip(CASE_IDS, baseline.CASES))
    original = json.loads(args.comparison.read_text(encoding="utf-8"))
    report["metadata"] = dict(
        comparison_summary_sha256=hashlib.sha256(
            args.comparison.read_bytes()
        ).hexdigest(),
        diagnostic_script_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "jax", "jaxprop", "pysolver-view")
        },
    )
    for name in ("subsonic", "sonic", "shock35", "shock70", "supersonic"):
        print("Replay", name, flush=True)
        problem, full = implicit_replay(cases[name], {})
        _, single = implicit_replay(cases[name], dict(newton_iter=1, newton_tol=0))
        saved = next(c for c in original["cases"] if c["id"] == name)[
            "implicit_references"
        ]["simple"]
        full["max_state_difference_from_saved"] = float(
            np.max(np.abs(full["x"] - np.asarray(saved["x"])))
        )
        report["replay"][name] = dict(default=full, one_inner=single)
        print(
            name,
            "default",
            full["updates"],
            full["total_linear_solves"],
            "single",
            single["updates"],
            single["success"],
            flush=True,
        )
        save()
    print("Matched first correction", flush=True)
    report["first_correction"] = matched_first_correction(cases["shock35"])
    print("Profile custom PTC", flush=True)
    problem = problem_module.SteadyNozzleProblem(
        cases["shock35"], solver_settings=MEASURED_SETTINGS
    )
    problem.warm_up()
    report["profile"] = profile_custom_ptc(problem, args.profile_iterations)
    save()
    if args.ptc_ablations:
        storage = PhysicalStorage(problem)
        storage.warm_up()
        for name, options in (
            ("initial_cfl20", dict(initial_cfl=20)),
            ("cfl20_full_fraction", dict(initial_cfl=20, first_step_fraction=1)),
            (
                "cfl20_relaxed_residual_guard",
                dict(initial_cfl=20, first_step_fraction=1, residual_growth_limit=1e6),
            ),
        ):
            print("PTC ablation", name, flush=True)
            result = ptc.solve_ptc(
                problem, storage, problem.x0.copy(), max_iterations=3000, **options
            )
            result["options"] = options
            result["diagnostics"] = problem.diagnostics(result["x"])
            reference = next(c for c in original["cases"] if c["id"] == "shock35")[
                "implicit_references"
            ]["simple"]
            distance = float(
                np.linalg.norm(result["x"] - np.asarray(reference["x"]))
                / np.sqrt(result["x"].size)
            )
            result["scaled_distance_from_implicit"] = distance
            result["same_implicit_root"] = bool(
                result["root_success"] and distance <= 1e-4
            )
            report["controller_ablations"][name] = result
            print(
                name,
                result["iterations"],
                result["root_success"],
                result["residual_rms"],
                flush=True,
            )
            save()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    plot_analysis(report, original, output)
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
