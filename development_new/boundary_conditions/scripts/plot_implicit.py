"""Five-backpressure figures for the two implicit boundary formulations."""

import argparse
import json
from pathlib import Path

LABELS = {
    "production": "Source: internal BCs",
    "coupled": "Experimental: coupled BC equations",
}


def five_panels(title):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3)]
    axes += [fig.add_subplot(grid[1, 1:3]), fig.add_subplot(grid[1, 3:5])]
    fig.suptitle(title)
    return fig, axes


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    colors = dict(zip(LABELS, plt.colormaps["plasma"]([0.2, 0.8])))
    residual_fig, residual_axes = five_panels(
        "Implicit integration: convergence for five backpressures"
    )
    pressure_fig, pressure_axes = five_panels(
        "Implicit integration: final pressure for five backpressures"
    )
    for index, case in enumerate(report["cases"]):
        rax, pax = residual_axes[index], pressure_axes[index]
        for run in case["runs"]:
            mode = run["mode"]
            style = dict(
                color=colors[mode],
                ls="-" if mode == "production" else "--",
                label=LABELS[mode],
                lw=1.5,
            )
            if mode == "coupled":
                style.update(
                    marker="o",
                    mfc="none",
                    ms=3,
                    markevery=max(1, len(run["history"]) // 12),
                )
            h = run["history"]
            rax.semilogy(
                [p["update"] for p in h], [p["residual_rms"] for p in h], **style
            )
            pressure_style = {k: v for k, v in style.items() if k != "markevery"}
            if mode == "coupled":
                pressure_style["markevery"] = max(1, (len(run["x_m"]) - 2) // 15)
            pax.plot(
                np.asarray(run["x_m"])[1:-1],
                np.asarray(run["final_state"])[1:-1, 0] / 1000,
                **pressure_style,
            )
        first = case["runs"][0]
        pax.plot(
            np.asarray(first["x_m"])[1:-1],
            np.asarray(first["pressure_exact_Pa"])[1:-1] / 1000,
            color="0.15",
            ls=":",
            lw=1,
            label="Analytic reference",
        )
        title = f"{case['name']}\n$p_b/p_0={case['back_pressure_ratio']:.6f}$"
        rax.set(
            title=title, xlabel="Accepted implicit update", ylabel="Steady residual RMS"
        )
        rax.axhline(1e-8, color="0.5", ls=":", lw=0.8)
        pax.set(title=title, xlabel="Axial position [m]", ylabel="Pressure [kPa]")
        rax.grid(alpha=0.2, which="both")
        pax.grid(alpha=0.2)
    for index in range(len(report["cases"]), 5):
        residual_axes[index].set_visible(False)
        pressure_axes[index].set_visible(False)
    residual_axes[0].legend(fontsize=8)
    pressure_axes[0].legend(fontsize=8)
    for fig, name in (
        (residual_fig, "convergence_histories"),
        (pressure_fig, "pressure_distributions"),
    ):
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.pdf")
        plt.close(fig)

    timing_fig, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    positions = np.arange(len(report["cases"]))
    for mode, shift in (("production", -0.12), ("coupled", 0.12)):
        stats = [case["timings"][mode] for case in report["cases"]]
        means = np.array([s["mean_s"] for s in stats])
        low = means - np.array([s["min_s"] for s in stats])
        high = np.array([s["max_s"] for s in stats]) - means
        axis.errorbar(
            positions + shift,
            means,
            yerr=[low, high],
            fmt="o",
            capsize=5,
            color=colors[mode],
            label=LABELS[mode] + ": mean and min–max",
        )
        for index, case in enumerate(report["cases"]):
            values = [r["solve_s"] for r in case["measured_runs"] if r["mode"] == mode]
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(
                index + shift + jitter, values, s=12, alpha=0.35, color=colors[mode]
            )
    axis.set_xticks(positions, [c["name"] for c in report["cases"]], fontsize=9)
    axis.set(
        ylabel="Warm full-solve time [s]",
        ylim=(0, None),
        title=f"{report['settings']['runs']} measured runs per method/case; first full run excluded",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=9)
    timing_fig.savefig(output / "timings.png", dpi=180)
    timing_fig.savefig(output / "timings.pdf")
    plt.close(timing_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "steady",
    )
    plot(parser.parse_args().output)
