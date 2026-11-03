"""Subsonic diffusion, an internal shock and adapted annular expansion.

Run: poetry run python demos/euler_annular/run.py
This radial channel diverges: expansion/shock cases prescribe a supersonic
inlet, whereas the subsonic case uses a reservoir inlet. Angles are radians.
"""

from dataclasses import asdict
from pathlib import Path

# Edit these inputs; no command-line arguments or YAML files are used.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
NUMBER_OF_CELLS = 80
CASES_TO_RUN = ("subsonic", "shocked", "adapted")
MAX_PSEUDO_STEPS = 4000
VERBOSE_JAX = False
FIGURE_DPI = 300

# Radial channel: radii and the full wall-to-wall gap are in metres.
RESERVOIR_PRESSURE = 2e5
RESERVOIR_TEMPERATURE = 300.0
INLET_RADIUS = 0.1
OUTLET_RADIUS = 0.2
CHANNEL_GAP = 0.005
INLET_ANGLE_DEGREES = 15.0
SUBSONIC_INLET_MACH = 0.3
SUPERSONIC_INLET_MACH = 1.6
SHOCK_POSITION_FRACTION = 0.65


import jax

# Configure precision before constructing any model or JAX array.
jax.config.update("jax_enable_x64", True)
from nozzle_hyperbolic.solvers.compilation import (
    CompilationReport,
    configure_compilation_cache,
)

# Configure before creating JAX arrays. Keep this cache for later Python runs;
# JAX_COMPILATION_CACHE_DIR can select another persistent location.
JAX_CACHE_DIRECTORY = configure_compilation_cache(OUTPUT_DIRECTORY / "jax_cache")
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Save figures; no GUI is required.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from nozzle_hyperbolic.plot_settings import set_plot_options

from nozzle_hyperbolic.solvers.fvm import (
    Scheme,
    SteadySettings,
    SteadyWorkspace,
    solve_single_grid,
)
from nozzle_hyperbolic.solvers.convergence_output import (
    ConvergenceTable,
    solution_summary,
    strict_json,
)

from nozzle_hyperbolic.geometry.annular import radial_channel
from nozzle_hyperbolic.geometry.fvm import build_annular_mesh
from nozzle_hyperbolic.models.euler_annular import AnnularModel
from nozzle_hyperbolic.models.euler_annular.analytical import PerfectGasReference


# Physical setup and API usage. Supporting helpers follow below.
def main():
    """Build the physical model, solve each requested regime and save the results."""
    configure_figures()
    fluid = jxp.FluidPerfectGas("Air")
    p0, T0 = RESERVOIR_PRESSURE, RESERVOIR_TEMPERATURE
    alpha = np.deg2rad(INLET_ANGLE_DEGREES)
    # Meridional coordinate is radial distance here. Height is the full gap;
    # AnnularModel obtains face areas, volume, radius and swirl geometry from it.
    geometry = radial_channel(INLET_RADIUS, OUTLET_RADIUS, CHANNEL_GAP)
    scheme = Scheme(
        reconstruction="muscl_van_leer",
        flux="hll",
        time_integrator="euler-implicit",
        newton_backtracking=True,
    )
    # Other supported Euler fluxes: stadtke, rusanov, roe, osher.
    # Other reconstruction examples: constant or muscl_van_albada.
    # boundary_treatment="segregated" solves the same implicit endpoint laws
    # separately; "auto" uses coupled traces for backward Euler.
    settings = SteadySettings(max_steps=MAX_PSEUDO_STEPS, cfl=0.2, cfl_max=1e4)
    # At adapted operation all exit characteristics point outward: its pressure
    # is a design/reference target, not an extra imposed supersonic condition.
    records = []
    for name in CASES_TO_RUN:
        reference = PerfectGasReference.from_inlet(
            fluid,
            geometry,
            p0=p0,
            T0=T0,
            M_m=SUBSONIC_INLET_MACH if name == "subsonic" else SUPERSONIC_INLET_MACH,
            alpha=alpha,
        )
        shock = SHOCK_POSITION_FRACTION * geometry.length if name == "shocked" else None
        inlet = reference.profile(np.array([0.0]))["W"][0]
        pressure = float(
            reference.profile(np.array([geometry.length]), shock_m=shock)["p"][0]
        )
        mesh, native = build_annular_mesh(geometry, NUMBER_OF_CELLS)
        inlet_data = (
            dict(p0=p0, T0=T0, alpha_in=alpha)
            if name == "subsonic"
            else dict(state=inlet, p0=p0)
        )
        model = AnnularModel(
            mesh,
            native,
            fluid,
            inlet_data=inlet_data,
            outlet_pressure=pressure,
            inlet_regime="subsonic" if name == "subsonic" else "supersonic",
            outlet_regime="supersonic" if name == "adapted" else "subsonic",
        )
        # State order is (p, v_m, ell, h), with ell=r*v_theta. The default cold
        # field is uniform; no analytical shock or profile initializes it.
        print(f"\n=== {name}: exit pressure={pressure/1e5:.6f} bar ===", flush=True)
        table = ConvergenceTable(model.equation_names, every=20)
        workspace = SteadyWorkspace(model, scheme, settings.residual)
        # Reuse this exact workspace for another solve of the same fixed model.
        # Rebuilding/mutating the model can require new tracing/compilation.
        result = solve_single_grid(
            model, scheme, settings, callback=table, workspace=workspace
        )
        if table.last_step != result.history[-1]["step"]:
            table(result.history[-1], force=True)
        raw = model.fields(jnp.asarray(result.state))
        fields = {
            key: np.asarray(raw[source])
            for key, source in (("p", "p"), ("u", "v_m"), ("M", "M_m"), ("T", "T"))
        }
        reference_x = np.linspace(0.0, geometry.length, 801)
        exact = reference.profile(reference_x, shock_m=shock)
        exact = {
            key: exact[source]
            for key, source in (("p", "p"), ("u", "v_m"), ("M", "M_m"), ("T", "T"))
        }
        exact["x"] = reference_x
        records.append((name, model, result, fields, exact))
    save_results(records, OUTPUT_DIRECTORY, settings)


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


def plot_profiles(records, output):
    """Compare the operating regimes with independent reference profiles."""
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(records)))
    two_fluid = "alpha_g" in records[0]["fields"]

    if two_fluid:
        figure, axes = plt.subplots(
            2, 3, figsize=(10.4, 6.2), sharex=True, layout="constrained"
        )
        panels = (
            ("p", 1e-5, r"$p$ [bar]", "(a) Pressure"),
            ("u_g", 1.0, r"$u_g,\ u_l$ [m/s]", "(b) Phase velocities"),
            ("alpha_g", 1.0, r"$\alpha_g$ [-]", "(c) Gas volume fraction"),
            ("Ma_mix", 1.0, r"$M_\mathrm{mix}$ [-]", "(d) Mixture Mach number"),
            ("T_g", 1.0, r"$T_g,\ T_l$ [K]", "(e) Phase temperatures"),
            ("rho_g", 1.0, r"$\rho_g$ [kg/m$^3$]", "(f) Gas density"),
        )
    else:
        figure, axes = plt.subplots(
            2, 2, figsize=(7.6, 5.2), sharex=True, layout="constrained"
        )
        panels = (
            ("p", 1e-5, r"$p$ [bar]", "(a) Pressure"),
            ("M", 1.0, r"$M$ [-]", "(b) Mach number"),
            ("u", 1.0, r"$u$ [m/s]", "(c) Flow velocity"),
            ("T", 1.0, r"$T$ [K]", "(d) Temperature"),
        )

    case_handles = []
    for record, color in zip(records, colors, strict=True):
        fields = record["fields"]
        reference = record["reference"]
        x = 100 * record["x"]
        case_handles.append(
            Line2D([], [], color=color, label=record["name"].capitalize())
        )

        for axis, (key, scale, label, title) in zip(axes.flat, panels, strict=True):
            axis.plot(
                100 * reference["x"], scale * reference[key], "--", color=color, lw=1.2
            )
            axis.plot(
                x,
                scale * fields[key],
                color=color,
                marker="o",
                ms=2.5,
                markevery=max(1, len(x) // 16),
                lw=1.3,
            )

            # A distinct marker/style separates the liquid from the gas.
            if key in ("u_g", "T_g"):
                liquid_key = "u_l" if key == "u_g" else "T_l"
                axis.plot(
                    100 * reference["x"],
                    reference[liquid_key],
                    "-.",
                    color=color,
                    lw=1.0,
                )
                axis.plot(
                    x,
                    fields[liquid_key],
                    ":",
                    color=color,
                    marker="s",
                    ms=2.3,
                    markevery=max(1, len(x) // 16),
                    lw=1.4,
                )

            axis.set_ylabel(label)
            axis.set_title(title, loc="left")
            finish_axis(axis)

    for axis in axes[-1]:
        axis.set_xlabel(r"$x$ [cm]")
    method_handles = [
        Line2D([], [], color="0.2", marker="o", ms=3, label="Finite volume"),
        Line2D([], [], color="0.2", ls="--", label="Reference"),
    ]
    if two_fluid:
        method_handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color="0.2",
                    ls=":",
                    marker="s",
                    ms=3,
                    label="Liquid: finite volume",
                ),
                Line2D([], [], color="0.2", ls="-.", label="Liquid: reference"),
            ]
        )
    figure.legend(
        handles=case_handles + method_handles,
        loc="outside upper center",
        ncols=4 if two_fluid else 3,
    )
    export_figure(figure, "flow", output)


def plot_convergence(records, output):
    """Separate spatial convergence, global balance and algebraic defects."""
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.25), layout="constrained")
    colors = plt.get_cmap("plasma")(np.linspace(0.2, 0.8, len(records)))
    handles = []
    has_constraints = False

    for record, color in zip(records, colors, strict=True):
        rows = [row for row in record["history"] if row.get("accepted")]
        steps = [row["step"] for row in rows]
        handles.append(Line2D([], [], color=color, label=record["name"].capitalize()))
        quantities = (
            [np.max(row["equation_norm"]) for row in rows],
            [np.max(np.abs(row["global_defect"])) for row in rows],
            [row["boundary_max"] for row in rows],
        )
        for axis, values in zip(axes, quantities, strict=True):
            axis.semilogy(steps, np.ma.masked_less_equal(values, 0), color=color)

        constraints = [row["constraint_max"] for row in rows]
        if max(constraints, default=0) > 0:
            has_constraints = True
            axes[2].semilogy(
                steps, np.ma.masked_less_equal(constraints, 0), "--", color=color
            )

        rejected = [row["step"] for row in record["history"] if not row.get("accepted")]
        # Rejected attempts have no new spatial norm. Their marker height is
        # an axes coordinate, so no artificial residual value is introduced.
        axes[0].plot(
            rejected,
            [0.035] * len(rejected),
            "x",
            ms=3.5,
            color=color,
            transform=axes[0].get_xaxis_transform(),
        )

    tolerance = records[0]["settings"]["residual"]["global_tolerance"]
    axes[1].axhline(tolerance, color="0.35", ls=":", lw=1.0)
    axes[0].set(ylabel=r"Maximum scaled $L_2$ norm", title="(a) Spatial residual")
    axes[1].set(ylabel=r"Maximum scaled balance defect", title="(b) Global balance")
    axes[2].set(ylabel=r"Maximum scaled defect", title="(c) Algebraic equations")
    for axis in axes:
        axis.set_xlabel("Pseudo-time updates")
        finish_axis(axis)

    handles.append(Line2D([], [], color="0.35", ls=":", label="Global tolerance"))
    if has_constraints:
        handles.extend(
            [
                Line2D([], [], color="0.2", label="Boundary equations"),
                Line2D([], [], color="0.2", ls="--", label="Cell constraints"),
            ]
        )
    figure.legend(handles=handles, loc="outside upper center", ncols=3)
    figure.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    figure.text(
        0.5,
        0.01,
        r"$\times$: rejected attempt; exact zeros omitted on logarithmic axes.",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    export_figure(figure, "convergence", output)


def save_results(records, output, settings):
    """Save solver data first, then render the publication figures."""
    output.mkdir(parents=True, exist_ok=True)
    plot_records = []
    summaries = {}

    for name, model, result, fields, reference in records:
        summary = solution_summary(result)
        summaries[name] = summary
        print(f"\n{name}: {result.reason}; success={result.success}")
        print(
            f"Accepted/rejected updates: {summary['accepted_steps']}/{summary['rejected_steps']}"
        )
        print(
            f"Inner corrections, including rejected work: {summary['inner_iterations']}"
        )
        print("Final scaled equation norms:", result.metrics["equation_norm"])
        print(
            f"Boundary={summary['boundary_max']:.3e}; "
            f"constraints={summary['constraint_max']:.3e}; "
            f"global balance={summary['global_balance_max']:.3e}"
        )
        print(
            f"Mach range: {np.min(fields['M']):.4f} to {np.max(fields['M']):.4f}; "
            f"exit pressure: {fields['p'][-1] / 1e5:.5f} bar"
        )

        payload = dict(
            summary=summary,
            scheme=asdict(result.operator.scheme),
            settings=asdict(settings),
            history=result.history,
            equation_names=model.equation_names,
        )
        (output / f"{name}.json").write_text(strict_json(payload), encoding="utf-8")
        np.savez_compressed(
            output / f"{name}.npz",
            x=model.mesh.centers,
            full_state=result.state,
            **fields,
            **{f"reference_{key}": value for key, value in reference.items()},
        )
        plot_records.append(
            dict(
                name=name,
                x=model.mesh.centers,
                fields=fields,
                reference=reference,
                history=result.history,
                settings=asdict(settings),
            )
        )

    (output / "summary.json").write_text(strict_json(summaries), encoding="utf-8")
    plot_profiles(plot_records, output)
    plot_convergence(plot_records, output)
    print(f"\nSaved results to {output.resolve()}")
    if not all(record[2].success for record in records):
        raise RuntimeError(
            "A steady case failed; its actual outcome and fields were saved."
        )


if __name__ == "__main__":
    # One concise cache/compilation summary; details also go to compilation.json.
    with CompilationReport(OUTPUT_DIRECTORY / "compilation.json", verbose=VERBOSE_JAX):
        main()
