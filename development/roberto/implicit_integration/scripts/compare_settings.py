"""Three five-case implicit studies: Newton caps, CFL at cap 100, CFL at cap 1.

poetry run python -m development.roberto.implicit_integration.scripts.compare_settings --no-show
Numerical setup and the implicit reference controller are reused; no FVM
formula is copied. Counters expose the actual early-terminated inner work.
"""

import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import jax
import numpy as np
from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
from development.roberto.historical.newton_solve.scripts import problem as problem_module
from development.roberto.historical.newton_solve.scripts.analyze_ptc_cost import counted_factory
from development.roberto.historical.newton_solve.scripts.pseudo_time import solve_implicit
from development.roberto.historical.newton_solve.scripts.investigate_ptc import CASE_IDS, json_value

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
CASE_ORDER = ("subsonic", "sonic", "shock35", "shock70", "supersonic")
CAPS = (1, 3, 5, 10, 100)
CFL_BASE = dict(CFL=20.0, ser_cfl_max=500.0, ser_beta=1.0, dt_growth_factor=1.2)
CFL_VARIANTS = {
    "baseline": dict(label="Baseline", **CFL_BASE),
    "start1": dict(label="Start CFL 1", **dict(CFL_BASE, CFL=1.0)),
    "start100": dict(label="Start CFL 100", **dict(CFL_BASE, CFL=100.0)),
    "max100": dict(label="Max CFL 100", **dict(CFL_BASE, ser_cfl_max=100.0)),
    "max100000": dict(label="Max CFL 100000", **dict(CFL_BASE, ser_cfl_max=100000.0)),
    "no_ser": dict(label="No SER ramp", **dict(CFL_BASE, ser_beta=0.0)),
    "beta2": dict(label="SER exponent 2", **dict(CFL_BASE, ser_beta=2.0)),
    "growth105": dict(
        label="Timestep growth 1.05", **dict(CFL_BASE, dt_growth_factor=1.05)
    ),
    "growth2": dict(label="Timestep growth 2", **dict(CFL_BASE, dt_growth_factor=2.0)),
}
STUDIES = {
    "newton_iterations": dict(
        title="Newton iteration caps",
        variants=[(n, "baseline", f"Cap {n}") for n in CAPS],
    ),
    "cfl_newton100": dict(
        title="CFL sensitivity: Newton cap 100",
        variants=[(100, k, v["label"]) for k, v in CFL_VARIANTS.items()],
    ),
    "cfl_newton1": dict(
        title="CFL sensitivity: Newton cap 1",
        variants=[(1, k, v["label"]) for k, v in CFL_VARIANTS.items()],
    ),
}
TITLES = {
    "subsonic": "Fully subsonic",
    "sonic": "Sonic throat / subsonic exit",
    "shock35": "Shock at 35%",
    "shock70": "Shock at 70%",
    "supersonic": "Adapted supersonic exit",
}


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    path.write_text(
        json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(safe(rows))


def run_id(case, cap, variant):
    return f"{case}_n{cap}_{variant}"


def write_index(report, output):
    dump(output / "summary.json", report)
    csv_write(output / "results.csv", report["runs"])


def measure(problem, raw_step, raw_dt, case, cap, key, output, max_iterations):
    config = CFL_VARIANTS[key]
    problem.settings.update({k: config[k] for k in ("CFL", "ser_cfl_max", "ser_beta")})
    problem.settings["dt_growth_factor"] = config["dt_growth_factor"]
    original_cfl = 20.0

    def compute_dt(state):
        dt, sound = raw_dt(state)
        return dt * (config["CFL"] / original_cfl), sound

    inner = []

    def step(state, dt):
        new, ssr, count, relative = raw_step(state, dt)
        inner.append(
            dict(
                inner_solves=int(count),
                inner_relative_update=float(relative),
                inner_converged=bool(
                    np.isfinite(float(relative))
                    and float(relative) <= problem.settings["newton_tol"]
                ),
            )
        )
        return new, ssr

    problem.compute_dt = compute_dt
    problem.step = step
    initial = problem.full_state(problem.x0)
    seed_hash = hashlib.sha256(np.asarray(problem.pack(initial)).tobytes()).hexdigest()
    problem.reset_counters()
    result = solve_implicit(
        problem,
        max_iterations=max_iterations,
        initial_state=initial,
        dt_growth_factor=config["dt_growth_factor"],
        record_controls=True,
    )
    # These costs include controller/counter monitoring, not compilation or final audits.
    work = problem.model_work()
    assert len(inner) == len(result["trial_history"])
    for trial, counts in zip(result["trial_history"], inner):
        trial.update(counts)
    cumulative = 0
    for row in result["history"]:
        trials = row["implicit_step_trials"]
        if trials:
            row.update(inner[trials - 1])
        cumulative = sum(v["inner_solves"] for v in inner[:trials])
        row["cumulative_linear_solves"] = cumulative
    diagnostics = problem.diagnostics(result["x"])
    root = bool(diagnostics["residual_rms"] <= 1e-8)
    rid = run_id(case, cap, key)
    folder = output / "runs" / rid
    folder.mkdir(parents=True, exist_ok=True)
    accepted = [t for t in result["trial_history"] if t["accepted"]]
    brief = dict(
        run_id=rid,
        case=case,
        cap=cap,
        cfl_variant=key,
        label=config["label"],
        start_cfl=config["CFL"],
        max_cfl=config["ser_cfl_max"],
        ser_exponent=config["ser_beta"],
        dt_growth=config["dt_growth_factor"],
        updates=result["iterations"],
        linear_solves=sum(v["inner_solves"] for v in inner),
        trial_steps=len(inner),
        rejected_steps=result["backend"]["rejected_trials"],
        inner_converged_accepted=sum(v["inner_converged"] for v in accepted),
        root_success=root,
        analytic_pass=diagnostics["analytical_pass"],
        residual_rms=diagnostics["residual_rms"],
        mass_imbalance=diagnostics["mass_imbalance"],
        solve_s=result["wall_s"],
        initial_sha256=seed_hash,
        file=str((folder / "run.json").relative_to(output)),
        message=result["message"],
    )
    result.update(
        diagnostics=diagnostics,
        settings=dict(problem.settings),
        controller=dict(config),
        model_work=work,
        **{k: v for k, v in brief.items() if k not in result},
    )
    dump(folder / "run.json", result)
    csv_write(folder / "history.csv", result["history"])
    csv_write(folder / "trials.csv", result["trial_history"])
    csv_write(
        folder / "pressure.csv",
        [
            dict(x_m=float(x), pressure_Pa=float(p))
            for x, p in zip(problem.x[1:-1], diagnostics["fields"]["p"][1:-1])
        ],
    )
    return brief


def study_findings(study, runs, loaded):
    """Build findings from the saved measurements, including nondefault runs."""
    get = lambda case, cap, key="baseline": runs[run_id(case, cap, key)]
    selected = [
        get(case, cap, key)
        for case in CASE_ORDER
        for cap, key, _ in STUDIES[study]["variants"]
    ]
    lines = [
        "",
        "## Observed findings",
        "",
        f"Original residual targets reached: {sum(r['root_success'] for r in selected)}/{len(selected)}; agreement with the case's baseline root: {sum(r['same_baseline_root'] for r in selected)}/{len(selected)}.",
        "",
    ]
    if study == "newton_iterations":
        least = sum(
            get(case, 1)["root_success"]
            and get(case, 1)["linear_solves"]
            == min(
                get(case, c)["linear_solves"]
                for c in CAPS
                if get(case, c)["root_success"]
            )
            for case in CASE_ORDER
            if any(get(case, c)["root_success"] for c in CAPS)
        )
        lines += [
            f"Cap 1 uses the least total linear work among converged baseline-CFL runs in {least}/{len(CASE_ORDER)} cases. Extra inner accuracy changes the trajectory, so outer count need not improve monotonically.",
            "Each entry is **accepted outer updates / actual linear solves**, including rejected-trial work.",
            "",
            "| Case | Cap 1 | Cap 3 | Cap 5 | Cap 10 | Cap 100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for case in CASE_ORDER:
            lines.append(
                "| "
                + TITLES[case]
                + " | "
                + " | ".join(
                    f"{get(case,c)['updates']} / {get(case,c)['linear_solves']}"
                    for c in CAPS
                )
                + " |"
            )
        x, y = get("shock35", 1), get("shock35", 100)
        lines += [
            "",
            f"For shock35, cap 1 gives {x['updates']} outer updates and {x['linear_solves']} linear solves; cap 100 gives {y['updates']} updates and {y['linear_solves']} solves.",
        ]
        x, y = get("shock70", 3), get("shock70", 100)
        lines += [
            f"For shock70, cap 3 gives {x['updates']} outer updates, compared with {y['updates']} at cap 100. These measurements distinguish outer progress from total work.",
            "The smooth-case curves show how much extra inner convergence matters once the controller is fixed. This is a measured comparison, not a theorem for arbitrary meshes or flows.",
            "",
            "| Case | Largest observed inner count at cap-100 baseline (all trials) | Accepted steps satisfying the inner tolerance |",
            "|---|---:|---:|---:|",
        ]
        for case in CASE_ORDER:
            row = get(case, 100)
            data = loaded[row["run_id"]]
            lines.append(
                f"| {TITLES[case]} | {max(t['inner_solves'] for t in data['trial_history'])} | {row['inner_converged_accepted']}/{row['updates']} |"
            )
        lines += [
            "",
            "Cap 100 does not force 100 solves per timestep. A low cap can reach the steady spatial target without satisfying the tight inner timestep tolerance at every accepted step; that is different from fully converging each transient equation.",
        ]
    else:
        cap = 100 if study == "cfl_newton100" else 1
        lines += [
            "The settings perturb one controller parameter at a time. Each entry is **accepted outer updates / total linear solves**.",
            "",
            "| Variant | Subsonic | Sonic limit | Shock 35% | Shock 70% | Supersonic |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key, config in CFL_VARIANTS.items():
            lines.append(
                "| "
                + config["label"]
                + " | "
                + " | ".join(
                    f"{get(case,cap,key)['updates']} / {get(case,cap,key)['linear_solves']}"
                    for case in CASE_ORDER
                )
                + " |"
            )
        x, y = get("sonic", cap), get("sonic", cap, "max100000")
        lines += [
            "",
            f"Raising maximum CFL from 500 to 100000 changes sonic-limit updates from {x['updates']} to {y['updates']}.",
        ]
        x, y = get("shock35", cap), get("shock35", cap, "max100000")
        u, v = get("shock70", cap), get("shock70", cap, "max100000")
        lines += [
            f"The same change gives {x['updates']} to {y['updates']} for shock35 and {u['updates']} to {v['updates']} for shock70. Its importance therefore depends on the trajectory, not just the nominal ceiling."
        ]
        x, y = get("shock70", cap), get("shock70", cap, "growth2")
        lines += [
            f"With timestep growth 2 instead of 1.2, shock70 outer updates change from {x['updates']} to {y['updates']}, while linear solves change from {x['linear_solves']} to {y['linear_solves']}. Compare both measures before judging the ramp."
        ]
        for case in ("shock35", "shock70"):
            x, y = get(case, cap), get(case, cap, "start100")
            lines += [
                f"For {case}, starting at CFL 100 instead of 20 changes outer updates {x['updates']} to {y['updates']} and linear solves {x['linear_solves']} to {y['linear_solves']}."
            ]
        slow = sum(
            get(case, cap, "no_ser")["updates"] > get(case, cap)["updates"]
            for case in CASE_ORDER
        )
        lines += [
            f"Disabling SER increases outer updates in {slow}/{len(CASE_ORDER)} measured cases. A larger SER exponent or timestep growth bound is not automatically a reduction in total work.",
            "The nine settings are individual perturbations, not a factorial optimization; their combinations and robustness outside these cases are not established.",
        ]
    return lines + [""]


def render(report, output, no_show=True):
    report["render_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    write_index(report, output)
    import matplotlib

    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = {r["run_id"]: r for r in report["runs"]}
    loaded = {
        k: json.loads((output / v["file"]).read_text(encoding="utf-8"))
        for k, v in runs.items()
    }
    overview = [
        "# Implicit-integration sensitivity studies",
        "",
        "Three studies use the same five physical cases, mesh, cold seeds, corrected FVS and safe Van Leer.",
        "Newton counts are caps, with inner relative-update tolerance $10^{-8}$. The independent original spatial residual RMS target is also $10^{-8}$.",
        "Actual linear-solve counts include rejected timestep trials. Compilation and final audits are excluded from solve times; all variants include the same monitoring.",
        "Single serial timings are descriptive, not repeated speed benchmarks. A converged discrete root can still miss the sonic-limit analytic accuracy check.",
        "",
    ]
    overview[2:2] = [
        f"**Measured outcome:** {sum(r['root_success'] for r in report['runs'])}/{len(report['runs'])} unique runs reached the original residual target; {sum(r['same_baseline_root'] for r in report['runs'])}/{len(report['runs'])} match their case's baseline root.",
        "The detailed reports compare outer updates, total linear work and timing separately. More aggressive timestep growth can reduce outer iterations while increasing total work.",
        f"Analytic accuracy failures among the unique runs: {sum(not r['analytic_pass'] for r in report['runs'])}. Root convergence is audited separately from analytic agreement.",
        "",
    ]
    for study, definition in STUDIES.items():
        folder = output / study
        folder.mkdir(parents=True, exist_ok=True)
        variants = definition["variants"]
        colors = plt.colormaps["plasma"](np.linspace(0.2, 0.8, len(variants)))
        styles = ["-", "--", ":", "-."]
        for kind in (
            "convergence_histories",
            "pressure_profiles",
            "convergence_time",
            "cfl_histories",
        ):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
            for axis, case in zip(axes.flat, CASE_ORDER):
                meta = report["cases"][case]
                failures = []
                for n, (cap, key, label) in enumerate(variants):
                    rid = run_id(case, cap, key)
                    if rid not in runs:
                        continue
                    run = loaded[rid]
                    brief = runs[rid]
                    h = run["history"]
                    if not brief["root_success"]:
                        failures.append(label)
                    if kind == "pressure_profiles":
                        axis.plot(
                            np.asarray(meta["x_m"])[1:-1],
                            np.asarray(run["diagnostics"]["fields"]["p"])[1:-1] / 1000,
                            color=colors[n],
                            ls=styles[n % 4],
                            lw=1.4,
                            label=label,
                        )
                    else:
                        hist = h[1:] if kind == "cfl_histories" else h
                        x = [
                            (
                                r["elapsed_s"]
                                if kind == "convergence_time"
                                else r["accepted_index"]
                            )
                            for r in hist
                        ]
                        y = [
                            (
                                r["effective_cfl"]
                                if kind == "cfl_histories"
                                else max(r["residual_rms"], 1e-16)
                            )
                            for r in hist
                        ]
                        axis.semilogy(
                            x, y, color=colors[n], ls=styles[n % 4], lw=1.3, label=label
                        )
                        if x:
                            axis.plot(
                                x[-1],
                                y[-1],
                                marker="o" if brief["root_success"] else "x",
                                color=colors[n],
                                ms=3,
                            )
                if kind == "pressure_profiles":
                    axis.plot(
                        meta["exact_x_m"],
                        np.asarray(meta["exact_pressure_Pa"]) / 1000,
                        "k-.",
                        lw=1,
                        label="Analytic",
                    )
                    axis.set(xlabel="x [m]", ylabel="Pressure [kPa]")
                else:
                    axis.set_xscale(
                        "symlog", linthresh=0.01 if kind == "convergence_time" else 10
                    )
                    axis.set(
                        xlabel=(
                            "Solve time [s] (symlog)"
                            if kind == "convergence_time"
                            else "Accepted outer update (symlog)"
                        ),
                        ylabel=(
                            "Actual accepted CFL"
                            if kind == "cfl_histories"
                            else "Original steady residual RMS"
                        ),
                    )
                    if kind != "cfl_histories":
                        axis.axhline(1e-8, color=".4", ls="-.", lw=0.8)
                title = TITLES[case] + f"\n$p_b/p_0={meta['p_back']/meta['p0']:.6f}$"
                if failures:
                    title += "\nTarget missed: " + ", ".join(failures)
                axis.set_title(title, fontsize=10)
                axis.grid(alpha=0.2)
            axes.flat[-1].axis("off")
            handles = [
                Line2D([], [], color=colors[n], ls=styles[n % 4], label=v[2])
                for n, v in enumerate(variants)
            ]
            if kind == "pressure_profiles":
                handles.append(
                    Line2D([], [], color="k", ls="-.", label="Analytic reference")
                )
            else:
                handles.extend(
                    [
                        Line2D(
                            [],
                            [],
                            color=".4",
                            marker="o",
                            ls="none",
                            label="Converged endpoint",
                        ),
                        Line2D(
                            [],
                            [],
                            color=".4",
                            marker="x",
                            ls="none",
                            label="Unconverged endpoint",
                        ),
                    ]
                )
            axes.flat[-1].legend(
                handles=handles, loc="center", frameon=False, fontsize=9
            )
            fig.suptitle(definition["title"] + ": " + kind.replace("_", " "))
            for ext in ("png", "pdf"):
                fig.savefig(folder / f"{kind}.{ext}", dpi=180, bbox_inches="tight")
            if no_show:
                plt.close(fig)
        lines = [
            f"# {definition['title']}",
            "",
            "All numerical curves start from the same original cold state. Colors use plasma from 0.2 to 0.8; each panel fixes one physical case/backpressure.",
            "Failed runs are labeled in panel titles and endpoint markers; their pressure curves are final returned fields, not claimed steady solutions.",
            "",
            "## Settings",
            "",
            "| Variant | Newton cap | Starting CFL | Maximum CFL | SER exponent | Timestep growth bound |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cap, key, label in variants:
            c = CFL_VARIANTS[key]
            lines.append(
                f"| {label} | {cap} | {c['CFL']:g} | {c['ser_cfl_max']:g} | {c['ser_beta']:g} | {c['dt_growth_factor']:g} |"
            )
        lines += [
            "",
            "The inner stopping tolerance is unchanged. Maximum CFL is a controller cap, not the necessarily accepted CFL. Fixed increment/domain checks can force smaller steps.",
            "SER exponent zero disables residual-based acceleration while keeping the base CFL.",
            "",
            "## Summary",
            "",
            "| Case | Roots reached | Baseline updates / linear solves | Lowest linear work among roots | Lowest warm time among roots | Analytic failures among roots |",
            "|---|---:|---:|---|---|---:|",
        ]
        for case in CASE_ORDER:
            records = [
                runs[run_id(case, cap, key)]
                for cap, key, _ in variants
                if run_id(case, cap, key) in runs
            ]
            good = [r for r in records if r["root_success"]]
            base = next(
                (
                    r
                    for r in records
                    if r["cfl_variant"] == "baseline"
                    and r["cap"]
                    == (100 if study == "newton_iterations" else variants[0][0])
                ),
                records[0],
            )
            work = min(good, key=lambda r: r["linear_solves"]) if good else None
            time = min(good, key=lambda r: r["solve_s"]) if good else None
            name = lambda r: (
                f"Cap {r['cap']}" if study == "newton_iterations" else r["label"]
            )
            lines.append(
                f"| {TITLES[case]} | {len(good)}/{len(records)} | {base['updates']} / {base['linear_solves']} | "
                + (f"{name(work)}: {work['linear_solves']}" if work else "None")
                + " | "
                + (f"{name(time)}: {time['solve_s']:.3f} s" if time else "None")
                + f" | {sum(not r['analytic_pass'] for r in good)} |"
            )
        lines += study_findings(study, runs, loaded)
        for kind, title, caption in (
            (
                "convergence_histories",
                "Convergence versus accepted updates",
                "Outer counts exclude rejected trials and inner Newton work. A cap of 100 does not mean 100 solves per accepted step.",
            ),
            (
                "convergence_time",
                "Convergence versus warm solve time",
                "Includes monitoring and rejected trials. JAX compilation is recorded separately.",
            ),
            (
                "pressure_profiles",
                "Final pressure versus position",
                "Black is the dense analytic reference. Numerical ghost cells are omitted; converged curves can overlap. The sonic-limit discrepancy is retained.",
            ),
            (
                "cfl_histories",
                "Actual accepted CFL histories",
                "Shows how the ramp and increment safeguards interact. A configured maximum need never be reached.",
            ),
        ):
            lines += [
                "",
                f"## {title}",
                "",
                f"![{title}]({kind}.png)",
                "",
                caption + f" [Vector PDF]({kind}.pdf).",
            ]
        rows = [
            runs[run_id(case, cap, key)]
            for case in CASE_ORDER
            for cap, key, _ in variants
            if run_id(case, cap, key) in runs
        ]
        csv_write(folder / "results.csv", rows)
        lines += [
            "",
            "[All measured runs](results.csv). Individual run JSON, accepted histories, all-trial histories and pressures are under [runs/](../runs/).",
            "The same baseline runs are reused between studies, rather than presenting duplicate executions as independent repetitions.",
            "",
        ]
        (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        overview += [
            f"## {definition['title']}",
            "",
            f"[Detailed report]({study}/REPORT.md) and [table]({study}/results.csv).",
            "",
            "### Convergence versus accepted updates",
            "",
            f"![Convergence versus accepted updates]({study}/convergence_histories.png)",
            "",
            "Each panel fixes one physical case. Outer counts exclude inner corrections and rejected trials.",
            "",
            "### Convergence versus warm solve time",
            "",
            f"![Convergence versus warm solve time]({study}/convergence_time.png)",
            "",
            "Solve times include monitoring and rejected trials; compilation is excluded. These are single measurements.",
            "",
            "### Final pressure profiles",
            "",
            f"![Final pressure profiles]({study}/pressure_profiles.png)",
            "",
            "Black curves show the analytic reference. Numerical curves overlap closely; the sonic-limit discrepancy remains.",
            "",
            "### Actual accepted CFL histories",
            "",
            f"![Actual accepted CFL histories]({study}/cfl_histories.png)",
            "",
            "These are actual accepted values after the ramp and safety checks, rather than configured maximum CFL values.",
            "",
        ]
    overview += [
        "## Scope",
        "",
        "This is a controlled one-factor-at-a-time sensitivity, not a factorial optimization or a universal stability test.",
        "Review both root and analytic checks. Timings include counter monitoring and remain single samples.",
        "[Full run index](summary.json), [full CSV](results.csv). Root agreement and seed consistency are independently audited.",
        "source_sha256 records the code used for the solves; render_source_sha256 records later reporting-only revisions.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(overview), encoding="utf-8")
    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        render(
            json.loads((output / "summary.json").read_text(encoding="utf-8")),
            output,
            args.no_show,
        )
        return
    if args.points < 5 or args.max_iterations < 1:
        parser.error("Use at least 5 points and a positive budget")
    sources = [
        ROOT / p
        for p in (
            "src/nozzle_hyperbolic/models/euler_nozzle/fvm.py",
            "src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py",
            "src/nozzle_hyperbolic/numerics/reconstruction.py",
            "development/roberto/historical/newton_solve/scripts/problem.py",
            "development/roberto/historical/newton_solve/scripts/pseudo_time.py",
            "development/roberto/historical/newton_solve/scripts/analyze_ptc_cost.py",
            "development/roberto/implicit_integration/scripts/compare_settings.py",
        )
    ]
    report = dict(
        settings=vars(args),
        versions={
            p: importlib.metadata.version(p)
            for p in ("jax", "numpy", "scipy", "jaxprop")
        },
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        cases={},
        runs=[],
        compilation=[],
    )
    cases = dict(zip(CASE_IDS, baseline.CASES))
    for case in CASE_ORDER:
        for cap in CAPS:
            print(f"Compile {case}, Newton cap {cap}", flush=True)
            with patch.object(
                problem_module,
                "make_jax_stadtke_step",
                counted_factory(return_inner_status=True),
            ):
                problem = problem_module.SteadyNozzleProblem(
                    cases[case],
                    n_points=args.points,
                    solver_settings=dict(
                        CFL=CFL_BASE["CFL"], newton_iter=cap, newton_tol=1e-8
                    ),
                )
            costs = problem.warm_up()
            report["compilation"].append(
                dict(case=case, cap=cap, first_call_costs=costs)
            )
            report["cases"][case] = dict(
                x_m=problem.x,
                exact_x_m=problem.x_fine,
                exact_pressure_Pa=np.asarray(problem.reference_fine["p"]),
                p_back=problem.p_back,
                p0=problem.p0,
            )
            raw_step, raw_dt = problem.step, problem.compute_dt
            keys = list(CFL_VARIANTS) if cap in (1, 100) else ["baseline"]
            for key in keys:
                print(f"  {case}, cap {cap}, {key}", flush=True)
                brief = measure(
                    problem,
                    raw_step,
                    raw_dt,
                    case,
                    cap,
                    key,
                    output,
                    args.max_iterations,
                )
                report["runs"].append(brief)
                write_index(report, output)
                print(
                    f"    {brief['updates']} updates, {brief['linear_solves']} linear solves, root={brief['root_success']}, {brief['solve_s']:.3f}s",
                    flush=True,
                )
            del problem
        jax.clear_caches()
    # Compare root fields against the converged cap-100 baseline for each case.
    for case in CASE_ORDER:
        rows = [r for r in report["runs"] if r["case"] == case]
        anchor = next(
            r for r in rows if r["cap"] == 100 and r["cfl_variant"] == "baseline"
        )
        ref = json.loads((output / anchor["file"]).read_text(encoding="utf-8"))
        for row in rows:
            result = json.loads((output / row["file"]).read_text(encoding="utf-8"))
            distance = float(
                np.linalg.norm(np.asarray(result["x"]) - np.asarray(ref["x"]))
                / np.sqrt(len(ref["x"]))
            )
            row["scaled_distance_from_baseline"] = distance
            row["same_baseline_root"] = bool(
                row["root_success"] and anchor["root_success"] and distance <= 1e-4
            )
        assert len({r["initial_sha256"] for r in rows}) == 1
    write_index(report, output)
    assert all(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
        for p, h in report["source_sha256"].items()
    )
    render(report, output, args.no_show)
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
