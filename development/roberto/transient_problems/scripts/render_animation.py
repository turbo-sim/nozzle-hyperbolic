"""Export actual saved transient fields as shareable GIFs and scientific keyframes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation, pyplot as plt
import numpy as np

from .common import STUDY, save

FIELDS = (
    ("p", "Pressure [kPa]", 1.0e-3),
    ("rho", "Density [kg/m³]", 1.0),
    ("u", "Velocity [m/s]", 1.0),
    ("M", "Mach number [–]", 1.0),
    ("T", "Temperature [K]", 1.0),
)
LABELS = dict(
    euler_explicit="Euler explicit",
    rk3="SSPRK3",
    implicit_coupled="Euler implicit, coupled",
    implicit_segregated="Euler implicit, segregated",
)


def render(case, output, fps, dpi):
    times = np.asarray(case["times"])
    x = np.asarray(case["x"])
    methods = case["methods"]
    exact = case.get("reference", {}).get("fields", {})
    fig, grid = plt.subplots(3, 2, figsize=(10.0, 8.0), dpi=dpi)
    axes = grid.ravel()
    colors = plt.colormaps["plasma"](np.linspace(0.2, 0.8, 4))
    styles = ("-", "-.", "--", ":")
    lines = []
    for axis, (field, label, factor) in zip(axes, FIELDS):
        axis.set_title(label, loc="left", fontsize=10)
        all_values = [
            np.asarray(method["fields"][field]) * factor for method in methods
        ]
        if field in exact:
            all_values.append(np.asarray(exact[field]) * factor)
        lower, upper = min(np.min(value) for value in all_values), max(
            np.max(value) for value in all_values
        )
        if field == "p" and case["id"] == "nozzle_step":
            config = case["physical_inputs"]
            back = config["reservoir_pressure"] * config["back_pressure_ratio"] * factor
            lower = min(lower, back)
            axis.axhline(back, color=".5", lw=0.7, ls=":", label="External pressure")
        pad = max(0.06 * (upper - lower), 0.01 * abs(upper), 1.0e-6)
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_xlim(case["faces"][0], case["faces"][-1])
        axis.set_xlabel("Coordinate [m]", fontsize=9)
        axis.grid(color=".9", linewidth=0.5)
        axis.tick_params(labelsize=8)
        for index, method in enumerate(methods):
            (line,) = axis.plot(
                x,
                np.asarray(method["fields"][field][0]) * factor,
                color=colors[index],
                ls=styles[index],
                lw=1.6,
                label=LABELS[method["id"]],
            )
            lines.append((line, np.asarray(method["fields"][field]) * factor))
        if field in exact:
            (line,) = axis.plot(
                x,
                np.asarray(exact[field][0]) * factor,
                color=".15",
                ls=(0, (4, 3)),
                lw=1.25,
                label="Exact Sod",
            )
            lines.append((line, np.asarray(exact[field]) * factor))
    geometry = axes[-1]
    area = np.asarray(case["area_centers"])
    geometry.plot(x, area / area[0], color=".35", lw=1.5)
    geometry.fill_between(x, area / area[0], color=".95")
    geometry.set(
        title="Fixed geometry and boundary state",
        xlabel="Coordinate [m]",
        ylabel="Area / inlet area [–]",
    )
    geometry.set_xlim(case["faces"][0], case["faces"][-1])
    geometry.set_ylim(0.0, 1.65)
    geometry.grid(color=".9", linewidth=0.5)
    geometry.tick_params(labelsize=8)
    status = geometry.text(
        0.03, 0.95, "", transform=geometry.transAxes, va="top", fontsize=8
    )
    title = fig.suptitle("", fontsize=13, y=0.985)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Requested CFL = 0.5 · All curves use the same physical model, mesh and numerical flux",
        ha="center",
        fontsize=8,
        color=".35",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.885), h_pad=1.4)

    def frame(index):
        for line, values in lines:
            line.set_ydata(values[index])
        title.set_text(
            f"{case['label']} · t = {times[index]*1.e3:.4f} ms · τ = {times[index]/case['reference_time']:.3f}"
        )
        modes = list(
            dict.fromkeys(method["outlet_regime"][index] for method in methods)
        )
        status.set_text(
            "Outlet: "
            + ", ".join(modes)
            + "\n"
            + "Maximum normalized budget: "
            + f"{max(np.max(np.abs(method['metrics']['conservation_budget'][index])) for method in methods):.2e}"
        )
        return [line for line, _ in lines] + [title, status]

    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{case['id']}.gif"
    movie = animation.FuncAnimation(
        fig, frame, frames=len(times), interval=1000.0 / fps, blit=False
    )
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    keyframes = []
    for label, index in (
        ("initial", 0),
        ("middle", len(times) // 2),
        ("final", len(times) - 1),
    ):
        frame(index)
        for suffix in ("png", "pdf"):
            figure = output / f"{case['id']}_{label}.{suffix}"
            fig.savefig(figure, dpi=160 if suffix == "png" else dpi)
            keyframes.append(figure.name)
    plt.close(fig)
    return dict(
        case=case["id"],
        file=path.name,
        bytes=path.stat().st_size,
        sha256=sha256(path.read_bytes()).hexdigest(),
        frames=len(times),
        fps=fps,
        dpi=dpi,
        keyframes=keyframes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=STUDY / "outputs/verification/summary.json"
    )
    parser.add_argument("--output", type=Path, default=STUDY / "outputs/animations")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not data.get("completed") or not data.get("passed"):
        raise ValueError("Animation export requires a complete successful campaign")
    records = []
    for case in data["cases"]:
        print("Rendering", case["id"], flush=True)
        records.append(render(case, args.output, args.fps, args.dpi))
    save(
        args.output / "manifest.json",
        dict(
            campaign=str(args.input),
            campaign_sha256=sha256(args.input.read_bytes()).hexdigest(),
            renderer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            records=records,
        ),
    )
    print("Saved", len(records), "computed-flow animations", flush=True)


if __name__ == "__main__":
    main()
