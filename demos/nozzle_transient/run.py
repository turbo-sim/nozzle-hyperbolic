"""Nozzle pressure release from rest, using SSPRK3 and a finite-wave outlet.

Run: poetry run python demos/nozzle_transient/run.py
Needs FFmpeg on PATH or FFMPEG_BINARY. All geometry, EOS and conditions are here.
"""

from dataclasses import asdict
from pathlib import Path

# Edit this section, then run this file directly. There are no CLI arguments.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
NUMBER_OF_CELLS = 100
CFL = 0.4
END_TIME = None  # Seconds; None selects the physical default.
VERBOSE_JAX = False  # One final compile/load status line.
FIGURE_DPI = 300  # Static PNGs; PDFs are vector graphics.

# More recorded states give smooth slow motion without interpolating flow data.
# 601 frames at 20 fps plus the holds below produce about 35 seconds of video.
NUMBER_OF_FRAMES = 601
VIDEO_FPS = 20
VIDEO_DPI = 160  # 1600 x 1000 H.264 MP4.
INITIAL_HOLD_SECONDS = 2.0
FINAL_HOLD_SECONDS = 3.0

# Nozzle geometry and reservoir conditions, all in SI units.
RESERVOIR_PRESSURE = 2e5
RESERVOIR_TEMPERATURE = 300.0
BACK_PRESSURE = 0.01 * RESERVOIR_PRESSURE
NOZZLE_LENGTH = 0.2
INLET_AREA = 3e-4
THROAT_AREA = 1.5e-4
EARLY_FRAME_FRACTION = 0.6

import os
import shutil

import jax

jax.config.update("jax_enable_x64", True)
from nozzle_hyperbolic.solvers.compilation import (
    CompilationReport,
    configure_compilation_cache,
)

# Keep this disk cache between Python runs. The bound Operator below also
# retains its executables in-process. An environment override is supported.
JAX_CACHE_DIRECTORY = configure_compilation_cache(OUTPUT_DIRECTORY / "jax_cache")
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from nozzle_hyperbolic.plot_settings import set_plot_options
from matplotlib.animation import FFMpegWriter

from nozzle_hyperbolic.geometry.nozzle import SymmetricNozzleGeometry
from nozzle_hyperbolic.models.euler_nozzle import NozzleModel
from nozzle_hyperbolic.models.euler_nozzle.analytical.shock_tube import sod_exact
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme, solve_transient
from nozzle_hyperbolic.solvers.convergence_output import (
    ConvergenceTable,
    solution_summary,
    strict_json,
)


# Physical setup and API usage. Supporting helpers follow below.
def main():
    """Release the external pressure and follow the nozzle start-up."""
    configure_figures()
    configure_video_encoder()
    fluid = jxp.FluidPerfectGas("Air")
    geometry = SymmetricNozzleGeometry(
        length=NOZZLE_LENGTH, inlet_area=INLET_AREA, throat_area=THROAT_AREA
    )
    mesh = geometry.build_mesh(NUMBER_OF_CELLS)
    reservoir = fluid.get_state(
        jxp.PT_INPUTS, RESERVOIR_PRESSURE, RESERVOIR_TEMPERATURE
    )
    initial = jnp.broadcast_to(
        jnp.array([RESERVOIR_PRESSURE, 0.0, reservoir["h"]]), (NUMBER_OF_CELLS + 2, 3)
    )

    # The cells start at rest at reservoir pressure. At t=0 the exterior drops
    # to BACK_PRESSURE. The finite-wave outlet law switches between pressure,
    # sonic and outward-only branches as the exit flow develops.
    model = NozzleModel(
        mesh,
        fluid,
        p0=RESERVOIR_PRESSURE,
        T0=RESERVOIR_TEMPERATURE,
        back_pressure=BACK_PRESSURE,
        outlet_regime="auto",
        outlet_formulation="finite_wave",
        initial_state=initial,
    )
    acoustic_time = NOZZLE_LENGTH / float(reservoir["a"])
    end_time = 8.0 * acoustic_time if END_TIME is None else END_TIME
    times = recording_times(end_time, min(2.0 * acoustic_time, end_time))
    print(
        f"Reservoir: {RESERVOIR_PRESSURE / 1e5:g} bar, "
        f"{RESERVOIR_TEMPERATURE:g} K; exterior: {BACK_PRESSURE / 1e5:g} bar; "
        f"duration: {end_time / acoustic_time:g} acoustic transit times."
    )
    notes = dict(
        acoustic_time_s=acoustic_time,
        back_pressure_pa=BACK_PRESSURE,
        outlet_formulation="finite_wave",
        output_time_spacing="piecewise linear; 60% of frames in first two acoustic transit times",
    )
    data, history, diagnostics, summary = solve_problem(model, times, notes)
    plot_history(data, history, diagnostics)
    render_movie(data, "Nozzle pressure release")
    check_completed(summary)


# Supporting solver, geometry and figure helpers.
def configure_figures():
    """Use the package's publication style with portable serif fonts."""
    set_plot_options(fontsize=11, grid=False)
    plt.rcParams.update(
        {
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "axes.linewidth": 0.8,
            "axes.labelpad": 5,
            "axes.titlepad": 7,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axis(axis):
    """Keep labels readable and use a light major grid."""
    axis.grid(True, which="major", color="0.88", linewidth=0.5)
    axis.tick_params(which="both", direction="in")
    if axis.get_yscale() == "linear":
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))


def export_figure(figure, name, output):
    """Save a sharp raster preview and a vector PDF with embedded fonts."""
    figure.savefig(output / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def configure_video_encoder():
    """Find FFmpeg before spending time on the numerical integration."""
    local = OUTPUT_DIRECTORY / "tools" / "ffmpeg.exe"
    binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if binary is None and local.is_file():
        binary = str(local)
    binary = shutil.which(binary) if binary is not None else None
    if binary is None:
        raise RuntimeError(
            "Set FFMPEG_BINARY to FFmpeg's executable, or add ffmpeg to PATH."
        )
    matplotlib.rcParams["animation.ffmpeg_path"] = binary
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def recording_times(end_time, focus_time):
    """Spend more movie time on the early wave motion, using real saved states."""
    if NUMBER_OF_FRAMES < 5 or VIDEO_FPS <= 0:
        raise ValueError("Use at least five output frames and positive VIDEO_FPS.")
    if not 0 < EARLY_FRAME_FRACTION < 1 or end_time <= 0:
        raise ValueError("Need a positive end time and 0 < EARLY_FRAME_FRACTION < 1.")
    if end_time <= focus_time:
        return np.linspace(0.0, end_time, NUMBER_OF_FRAMES)

    early_count = max(
        1, min(NUMBER_OF_FRAMES - 2, round(EARLY_FRAME_FRACTION * NUMBER_OF_FRAMES))
    )
    early = np.linspace(0.0, focus_time, early_count, endpoint=False)
    late = np.linspace(focus_time, end_time, NUMBER_OF_FRAMES - early_count)
    return np.concatenate((early, late))


def solve_problem(model, times, notes):
    """Advance conserved contents with RK3 and retain every actual outcome."""
    scheme = Scheme(
        reconstruction="muscl_van_leer", flux="hll", time_integrator="runge-kutta-3"
    )
    # Other reconstructions include constant, muscl_minmod and muscl_van_albada.
    # Other Euler fluxes include rusanov, stadtke, roe and osher. Integrators
    # include euler-explicit, euler-implicit and runge-kutta-4. This example uses
    # RK3 with segregated algebraic trace completion at every explicit stage.
    operator = Operator(model, scheme)
    original = jnp.asarray(model.initial_state(), dtype=jnp.float64)
    initial, info = operator.complete_boundaries(original, 0.0)
    if not bool(info.converged and info.admissible):
        raise RuntimeError(f"Initial boundary completion failed: {info}")
    np.testing.assert_array_equal(initial[1:-1], original[1:-1])

    # Reuse this binding: initial trace completion does not require rebuilding
    # the numerical kernels or changing the initial physical cell contents.
    table = ConvergenceTable(model.equation_names, mode="transient", every=100)
    snapshots = solve_transient(
        model,
        times[1:],
        scheme,
        cfl=CFL,
        initial_state=initial,
        operator=operator,
        callback=table,
    )
    result = snapshots[-1]
    if table.last_step != result.history[-1]["step"]:
        table(result.history[-1], force=True)

    actual_times = np.r_[0.0, [snapshot.time for snapshot in snapshots]]
    states = np.stack(
        [np.asarray(initial)] + [snapshot.state for snapshot in snapshots]
    )
    sampled = [model.fields(jnp.asarray(state)) for state in states]
    data = {
        "times": actual_times,
        "x": np.asarray(model.mesh.centers),
        "faces": np.asarray(model.mesh.faces),
        "area_faces": np.asarray(model.mesh.area_faces),
        "full_states": states,
        **{
            key: np.stack([np.asarray(fields[key]) for fields in sampled])
            for key in ("p", "rho", "u", "M", "T")
        },
    }
    summary = solution_summary(result)
    summary.update(notes)
    summary.update(
        requested_end_time_s=float(times[-1]),
        requested_frames=len(times),
        frames=len(actual_times),
        video_fps=VIDEO_FPS,
        video_playback_s=(
            len(actual_times) / VIDEO_FPS + INITIAL_HOLD_SECONDS + FINAL_HOLD_SECONDS
        ),
        final_pressure_span_over_mean=float(
            np.ptp(data["p"][-1]) / np.mean(data["p"][-1])
        ),
        final_velocity_span_over_sound_speed=float(np.ptp(data["u"][-1]) / model.a0),
    )
    diagnostics = dict(
        time_s=actual_times[1:],
        boundary_max=[float(s.metrics["boundary_max"]) for s in snapshots],
    )

    # Save numerical data before plotting, including a failed run's reached time.
    np.savez_compressed(OUTPUT_DIRECTORY / "solution.npz", **data)
    (OUTPUT_DIRECTORY / "summary.json").write_text(
        strict_json(summary), encoding="utf-8"
    )
    (OUTPUT_DIRECTORY / "history.json").write_text(
        strict_json(
            dict(
                summary=summary,
                scheme=asdict(scheme),
                requested_cfl=CFL,
                output_diagnostics=diagnostics,
                history=result.history,
            )
        ),
        encoding="utf-8",
    )
    print("\n" + strict_json(summary), flush=True)
    return data, result.history, diagnostics, summary


def plot_history(data, history, diagnostics):
    """Show conservation budgets separately from algebraic boundary defects."""
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), layout="constrained")
    rows = [row for row in history if row["accepted"]]
    budget = [np.max(np.abs(row["budget"])) for row in rows]
    axes[0].semilogy(
        1e3 * np.array([row["time_s"] for row in rows]),
        np.ma.masked_less_equal(budget, 0),
        color=plt.get_cmap("plasma")(0.3),
    )
    axes[0].set(ylabel="Normalized content-budget error", title="(a) Conservation")

    boundary = diagnostics["boundary_max"]
    if max(boundary, default=0) > 0:
        axes[1].semilogy(
            1e3 * np.asarray(diagnostics["time_s"]),
            np.ma.masked_less_equal(boundary, 0),
            color=plt.get_cmap("plasma")(0.65),
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "All sampled defects are zero",
            transform=axes[1].transAxes,
            ha="center",
        )
    axes[1].set(ylabel="Scaled boundary defect", title="(b) Boundary equations")
    for axis in axes:
        axis.set(xlabel=r"$t$ [ms]", xlim=(0, 1e3 * data["times"][-1]))
        finish_axis(axis)
    export_figure(figure, "history", OUTPUT_DIRECTORY)


def render_movie(data, title, reference=None, reference_valid_until=None):
    """Render slow playback of actual saved frames, with fixed axes and units."""
    figure, axes = plt.subplots(
        2, 2, figsize=(10, 6.25), sharex=True, layout="constrained"
    )
    figure.get_layout_engine().set(w_pad=0.15, h_pad=0.08, wspace=0.08, hspace=0.08)
    color = plt.get_cmap("plasma")(0.3)
    panels = (
        ("p", 1e-5, r"$p$ [bar]", "(a) Pressure"),
        ("rho", 1.0, r"$\rho$ [kg/m$^3$]", "(b) Density"),
        ("u", 1.0, r"$u$ [m/s]", "(c) Velocity"),
        ("M", 1.0, r"$M$ [-]", "(d) Mach number"),
    )
    numerical_lines = []
    reference_lines = []

    for axis, (key, scale, label, panel_title) in zip(axes.flat, panels, strict=True):
        values = scale * data[key]
        low, high = float(np.min(values)), float(np.max(values))
        if reference is not None:
            low = min(low, float(np.min(scale * reference[key])))
            high = max(high, float(np.max(scale * reference[key])))
            (line,) = axis.plot(
                reference["x"], scale * reference[key][0], "--", color="0.15", lw=1.2
            )
            reference_lines.append(line)

        (line,) = axis.plot(
            data["x"],
            values[0],
            color=color,
            lw=1.5,
            marker="o",
            ms=2.3,
            markevery=max(1, len(data["x"]) // 20),
        )
        numerical_lines.append(line)
        margin = 0.07 * max(high - low, abs(high), 1e-3)
        axis.set(
            xlim=(data["faces"][0], data["faces"][-1]),
            ylim=(low - margin, high + margin),
            ylabel=label,
        )
        axis.set_title(panel_title, loc="left")
        finish_axis(axis)

    for axis in axes[-1]:
        axis.set_xlabel(r"$x$ [m]")
    handles = [Line2D([], [], color=color, marker="o", ms=3, label="MUSCL–HLL / RK3")]
    if reference is not None:
        handles.append(
            Line2D([], [], color="0.15", ls="--", label="Analytical: unbounded domain")
        )
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    heading = figure.suptitle("", fontsize=12)

    writer = FFMpegWriter(
        fps=VIDEO_FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"]
    )
    snapshot_indices = {
        0: "initial",
        len(data["times"]) // 2: "middle",
        len(data["times"]) - 1: "final",
    }
    if reference_valid_until is not None:
        before_boundary = int(
            np.argmin(np.abs(data["times"] - 0.95 * reference_valid_until))
        )
        snapshot_indices[before_boundary] = "before_boundary"

    with writer.saving(figure, str(OUTPUT_DIRECTORY / "transient.mp4"), dpi=VIDEO_DPI):
        for index, time in enumerate(data["times"]):
            for line, (key, scale, _, _) in zip(numerical_lines, panels, strict=True):
                line.set_ydata(scale * data[key][index])
            if reference is not None:
                for line, (key, scale, _, _) in zip(
                    reference_lines, panels, strict=True
                ):
                    line.set_ydata(scale * reference[key][index])
                    line.set_color("0.15" if time < reference_valid_until else "0.5")
                status = (
                    "Before boundary interaction: exact Riemann reference"
                    if time < reference_valid_until
                    else "Boundary interaction: unbounded reference is a comparison"
                )
            else:
                status = "Pressure release from rest; approximate open outlet"
            heading.set_text(f"{title}  |  $t = {1e3 * time:.3f}$ ms\n{status}")

            # Holding the first/last states makes the initial data and final
            # outcome readable. Intermediate frames are never interpolated.
            repetitions = 1
            if index == 0:
                repetitions += round(INITIAL_HOLD_SECONDS * VIDEO_FPS)
            if index == len(data["times"]) - 1:
                repetitions += round(FINAL_HOLD_SECONDS * VIDEO_FPS)
            for _ in range(repetitions):
                writer.grab_frame()
            if index in snapshot_indices:
                stem = f"frame_{snapshot_indices[index]}"
                figure.savefig(
                    OUTPUT_DIRECTORY / f"{stem}.png",
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )
                figure.savefig(OUTPUT_DIRECTORY / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def check_completed(summary):
    """Keep failure visible after exporting the time and states actually reached."""
    if not summary["success"] or summary["frames"] != summary["requested_frames"]:
        raise RuntimeError("The integration failed; its actual outputs remain saved.")
    print(
        f"Saved figures and a {summary['video_playback_s']:.1f} s video to {OUTPUT_DIRECTORY.resolve()}"
    )


if __name__ == "__main__":
    with CompilationReport(OUTPUT_DIRECTORY / "compilation.json", verbose=VERBOSE_JAX):
        main()
