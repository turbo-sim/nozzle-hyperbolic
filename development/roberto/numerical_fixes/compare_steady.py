"""Run the shared steady-nozzle example with legacy and corrected numerics.

poetry run python -m development.roberto.numerical_fixes.compare_steady --no-show

No setup, reference solution, timestepper, or acceptance threshold is copied.
The shared example supplies all five physical cases. Local probes separately
exercise rest/sonic states and AD where converged flow plots may hide defects.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

VARIANTS = {
    "legacy": dict(flux_variant="legacy", reconstruction="muscl_van_leer"),
    "corrected": dict(flux_variant="regularized", reconstruction="muscl_van_leer"),
    "smooth": dict(
        flux_variant="regularized",
        reconstruction="muscl_van_leer_smooth",
        smooth_limiter_coefficient=1.0,
    ),
}


def json_safe(value):
    """Keep non-finite probe outcomes explicit without invalid JSON numbers."""
    import numpy as np

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(report, output):
    """Persist partial results as well as completed comparisons."""
    (output / "summary.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = []
    for run in report["runs"]:
        for index, case in enumerate(run["report"]["cases"]):
            history = case.get("convergence_history")
            if history:
                path = output / run["folder"] / f"case_{index+1}_convergence.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, history[0].keys())
                    writer.writeheader()
                    writer.writerows(history)
            rows.append(
                dict(
                    variant=run["variant"],
                    repeat=run["repeat"],
                    case_index=index,
                    case=case["case"],
                    back_pressure_ratio=case["back_pressure_ratio"],
                    steps=case["steps"],
                    solve_s=case["wall_s"],
                    compile_s=case["compile_s"],
                    ssr=case["ssr"],
                    converged=case["converged"],
                    passed=case["passed"],
                    max_normalized_L1=max(
                        case["normalized_L1"]["all_physical_points"].values()
                    ),
                    mass_imbalance=case["relative_boundary_mass_imbalance"],
                    throat_M=case["throat_M_numerical"],
                    shock_position_error_cells=case.get("shock_position_error_cells"),
                )
            )
    if rows:
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def compare_fields(report, output, scales):
    """Check matched physical references and measure changes from legacy."""
    import numpy as np

    comparisons = []
    for run in report["runs"]:
        if run["variant"] == "legacy":
            continue
        legacy = next(
            r
            for r in report["runs"]
            if r["variant"] == "legacy" and r["repeat"] == run["repeat"]
        )
        for index, case in enumerate(run["report"]["cases"]):
            left = np.genfromtxt(
                output / legacy["folder"] / f"case_{index+1}.csv",
                delimiter=",",
                names=True,
            )
            right = np.genfromtxt(
                output / run["folder"] / f"case_{index+1}.csv",
                delimiter=",",
                names=True,
            )
            for field in ("x_m", "area_m2", "p_exact_Pa", "rho_exact", "M_exact"):
                np.testing.assert_array_equal(left[field], right[field])
            assert (
                case["back_pressure_Pa"]
                == legacy["report"]["cases"][index]["back_pressure_Pa"]
            )
            difference = {}
            for key, field in (("p", "p_num_Pa"), ("rho", "rho_num"), ("M", "M_num")):
                delta = np.abs(right[field][1:-1] - left[field][1:-1]) / scales[key]
                difference[key] = dict(
                    mean=float(np.mean(delta)), maximum=float(np.max(delta))
                )
            comparisons.append(
                dict(
                    variant=run["variant"],
                    repeat=run["repeat"],
                    case=case["case"],
                    normalized_field_difference_from_legacy=difference,
                )
            )
    report["field_comparisons"] = comparisons


def local_probes(fluid, delta_factor):
    """Measure flux consistency/continuity and limiter AD independently of PDEs."""
    import jax
    import jax.numpy as jnp
    import jaxprop as jxp
    import numpy as np
    from nozzle_hyperbolic.models.euler_nozzle import flux_jax
    from nozzle_hyperbolic.numerics.reconstruction import (
        _van_leer_jax_safe,
        _van_leer_jax_regularized,
    )

    properties = fluid.get_state(jxp.PT_INPUTS, 1e5, 300.0)
    h, a, rho = (float(properties[name]) for name in ("h", "a", "rho"))
    scale = rho * a * np.array([1.0, a, h])

    def state(mach):
        return jnp.asarray([[1e5, mach * a, h]])

    fluxes = dict(
        legacy=flux_jax._jax_F_fvs_stadtke_legacy, corrected=flux_jax._jax_F_fvs_stadtke
    )
    probes = dict(equal_states=[], entropy_activation=[], limiters=[])
    for label, function in fluxes.items():
        evaluate = jax.jit(
            lambda left, right: function(left, right, fluid, delta_factor)
        )
        for mach in (0.0, 1e-8, 0.5, 1 - 1e-7, 1.0, 1 + 1e-7, 2.0):
            primitive = state(mach)
            u = mach * a
            physical = np.array(
                [rho * u, rho * u * u + 1e5, rho * u * (h + 0.5 * u * u)]
            )
            actual = np.asarray(evaluate(primitive, primitive))[0]
            variable_scale = jnp.asarray([1e5, a, h])
            z = primitive[0] / variable_scale
            single = lambda value: evaluate(
                (value * variable_scale)[None, :], primitive
            )[0] / jnp.asarray(scale)
            forward = np.asarray(jax.jacfwd(single)(z))
            reverse = np.asarray(jax.jacrev(single)(z))
            probes["equal_states"].append(
                dict(
                    variant=label,
                    mach=mach,
                    flux_finite=bool(np.isfinite(actual).all()),
                    scaled_flux_defect=float(
                        np.linalg.norm((actual - physical) / scale)
                    ),
                    forward_ad_finite=bool(np.isfinite(forward).all()),
                    reverse_ad_finite=bool(np.isfinite(reverse).all()),
                )
            )
        for epsilon in (1e-5, 1e-7, 1e-9):
            gap = (
                np.asarray(evaluate(state(1 + epsilon), state(1.1)))
                - np.asarray(evaluate(state(1 - epsilon), state(1.1)))
            )[0] / scale
            probes["entropy_activation"].append(
                dict(
                    variant=label,
                    epsilon=epsilon,
                    scaled_flux_gap=float(np.linalg.norm(gap)),
                )
            )

    # The historical expression is a mathematical control, not a second
    # reconstruction implementation used by any nozzle calculation.
    original = lambda p: jnp.where(
        p[0] * p[1] > 0, 2 * p[0] * p[1] / (p[0] + p[1] + 1e-300), 0.0
    )
    limiters = dict(
        original=original,
        safe=lambda p: _van_leer_jax_safe(p[0], p[1]),
        smooth=lambda p: _van_leer_jax_regularized(p[0], p[1], 2.5e-5),
    )
    for label, function in limiters.items():
        for point in ((0.0, 0.0), (1.0, -1.0), (1.0, 2.0), (1e-6, 1e-6)):
            argument = jnp.asarray(point)
            probes["limiters"].append(
                dict(
                    variant=label,
                    differences=list(point),
                    value=float(function(argument)),
                    forward_ad_finite=bool(
                        np.isfinite(np.asarray(jax.jacfwd(function)(argument))).all()
                    ),
                    reverse_ad_finite=bool(
                        np.isfinite(np.asarray(jax.jacrev(function)(argument))).all()
                    ),
                )
            )
    return json_safe(probes)


def plot(report, output, show=False):
    """Overlay numerical fields and matched analytic references per case."""
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [r for r in report["runs"] if r["repeat"] == 0]
    if not runs:
        return
    colors = dict(zip(VARIANTS, plt.colormaps["plasma"](np.linspace(0.2, 0.8, 3))))
    fig_history, history_axes = plt.subplots(
        2, 3, figsize=(13, 7), layout="constrained", squeeze=False
    )
    for index, case in enumerate(runs[0]["report"]["cases"]):
        axis = history_axes.flat[index]
        for run in runs:
            history = run["report"]["cases"][index].get("convergence_history")
            if not history:
                raise ValueError(
                    "Missing full history: rerun the comparison before plotting."
                )
            axis.semilogy(
                [point["iteration"] for point in history],
                [point["ssr"] for point in history],
                color=colors[run["variant"]],
                linestyle={"legacy": "-", "corrected": "--", "smooth": ":"}[
                    run["variant"]
                ],
                label=run["variant"],
            )
        axis.axhline(
            runs[0]["report"]["solver"]["residuals"],
            color="0.4",
            linestyle="-.",
            linewidth=0.8,
            label="Stopping threshold",
        )
        axis.set(
            title=f"{case['case']}\n$p_b/p_0={case['back_pressure_ratio']:.6f}$",
            xlabel="Accepted implicit update",
            ylabel="Native steady residual (SSR)",
        )
        axis.grid(alpha=0.2, which="both")
    history_axes.flat[-1].axis("off")
    handles, labels = history_axes.flat[0].get_legend_handles_labels()
    history_axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig_history.suptitle("Steady-nozzle convergence: one backpressure per panel")
    fig_history.savefig(output / "convergence_histories.png", dpi=180)
    fig_history.savefig(output / "convergence_histories.pdf")
    if not show:
        plt.close(fig_history)
    for index, case in enumerate(runs[0]["report"]["cases"]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for run in runs:
            data = np.genfromtxt(
                output / run["folder"] / f"case_{index+1}.csv",
                delimiter=",",
                names=True,
            )
            for axis, field, reference, label in zip(
                axes,
                ("p_num_Pa", "rho_num", "M_num"),
                ("p_exact_Pa", "rho_exact", "M_exact"),
                ("Pressure [Pa]", "Density [kg/m³]", "Mach"),
            ):
                if run is runs[0]:
                    axis.plot(
                        data["x_m"], data[reference], "k--", lw=1, label="Reference"
                    )
                axis.plot(
                    data["x_m"][1:-1],
                    data[field][1:-1],
                    color=colors[run["variant"]],
                    lw=1,
                    label=run["variant"],
                )
                axis.set(xlabel="x [m]", ylabel=label)
                axis.grid(alpha=0.2)
        axes[0].legend()
        fig.suptitle(case["case"])
        fig.savefig(output / f"case_{index+1}_overlay.png", dpi=180)
        fig.savefig(output / f"case_{index+1}_overlay.pdf")
        if not show:
            plt.close(fig)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results_steady"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import jaxprop as jxp
    import numpy as np
    from development.roberto.historical.euler_nozzle import verify_euler_nozzle_steady as baseline
    from nozzle_hyperbolic.models.euler_nozzle import flux_jax

    if not hasattr(flux_jax, "_jax_F_fvs_stadtke_legacy"):
        raise SystemExit(
            "This comparison requires the pending numerical-fix working tree; it is not available in the structural-only checkout."
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(
            json.loads((output / "summary.json").read_text(encoding="utf-8")),
            output,
            show=not args.no_show,
        )
        return
    root = Path(__file__).resolve().parents[3]
    sources = (
        Path(baseline.__file__).resolve(),
        root / "src/nozzle_hyperbolic/models/euler_nozzle/fvm.py",
        root / "src/nozzle_hyperbolic/models/euler_nozzle/flux_jax.py",
        root / "src/nozzle_hyperbolic/numerics/reconstruction.py",
        Path(__file__).resolve(),
    )
    report = dict(
        versions={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxprop", "numpy", "scipy")
        },
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        setup_module="development.roberto.historical.euler_nozzle.verify_euler_nozzle_steady",
        comparison_note="Same five cases, backend, mesh, initialization, production timestepper and thresholds. Legacy means legacy flux with current safe Van Leer, not a complete historical AD implementation. Solve/compile times separate; one repeat is not a speed benchmark.",
        runs=[],
    )
    fluid = jxp.FluidPerfectGas(name="Air")
    reservoir = fluid.get_state(
        jxp.PT_INPUTS, baseline.P_STAGNATION, baseline.T_STAGNATION
    )
    limiter_scales = [
        baseline.P_STAGNATION,
        float(reservoir["a"]),
        float(reservoir["h"]),
    ]
    scales = dict(p=baseline.P_STAGNATION, rho=float(reservoir["rho"]), M=1.0)
    unchanged_settings = dict(baseline.SOLVER)
    for repeat in range(args.repeats):
        for variant, settings in VARIANTS.items():
            folder = output / f"repeat_{repeat}" / variant
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Running {variant}, repeat {repeat+1}/{args.repeats}", flush=True)
            options = dict(settings)
            options["record_history"] = True
            if variant == "smooth":
                options["limiter_scales"] = limiter_scales
            with (folder / "run.log").open(
                "w", encoding="utf-8"
            ) as log, contextlib.redirect_stdout(log):
                result = baseline.main(
                    show=False,
                    solver_overrides=options,
                    output_dir=folder,
                    strict=False,
                )
            assert baseline.SOLVER == unchanged_settings
            report["runs"].append(
                dict(
                    variant=variant,
                    repeat=repeat,
                    folder=str(folder.relative_to(output)),
                    report=result,
                )
            )
            save(report, output)
            print(
                "  "
                + ", ".join(
                    f"{c['steps']} updates / {'pass' if c['passed'] else 'check failed'}"
                    for c in result["cases"]
                ),
                flush=True,
            )
    compare_fields(report, output, scales)
    report["local_probes"] = local_probes(fluid, baseline.SOLVER["delta_factor"])
    save(report, output)
    plot(report, output, show=not args.no_show)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
