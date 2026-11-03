"""NICFD nitrogen-water nozzle: subsonic, shocked and adapted flow.

Run: poetry run python demos/two_fluid/run.py
All physical/numerical inputs are below: no YAML and no case-builder imports.
The default initial guess is openly interpolated from the published CSV profile;
INITIALIZATION = "cold" selects an independent 10 m/s guess instead.
"""

from dataclasses import asdict
from pathlib import Path

# Edit these inputs; the NICFD model is constructed entirely in this script.
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
NUMBER_OF_CELLS = 100
CASES_TO_RUN = ("subsonic", "shocked", "adapted")
MAX_PSEUDO_STEPS = 500
VERBOSE_JAX = False
FIGURE_DPI = 300

# "reference" uses the published CSV only as a visible initial guess and
# comparison. "cold" uses an independent 10 m/s guess, without a convergence
# guarantee. Neither choice loads model settings from another case or YAML.
INITIALIZATION = "reference"
RESERVOIR_PRESSURE = 2e6  # Pa
GAS_STAGNATION_TEMPERATURE = 295.15  # K
LIQUID_STAGNATION_TEMPERATURE = 295.15
GAS_MASS_FLOW_FRACTION = 0.5
INLET_SLIP = 1.0  # u_g / u_l


import jax

jax.config.update("jax_enable_x64", True)
from nozzle_hyperbolic.solvers.compilation import (
    CompilationReport,
    configure_compilation_cache,
)

# Configure before creating JAX arrays. This executable cache is separate from
# JAXProp's property tables. JAX_COMPILATION_CACHE_DIR overrides its location.
JAX_CACHE_DIRECTORY = configure_compilation_cache(OUTPUT_DIRECTORY / "jax_cache")
import jax.numpy as jnp
import jaxprop as jxp
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from nozzle_hyperbolic.plot_settings import set_plot_options

from nozzle_hyperbolic.geometry.duct import get_nozzle_elliot
from nozzle_hyperbolic.numerics.mesh import Mesh
from nozzle_hyperbolic.models.two_fluid import (
    TwoFluidModel,
    BoundaryConfig,
    ClosureConfig,
    CompatibilityConfig,
    StabilizationConfig,
    primitives,
)
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

# Published data are used only as a visible initial guess and comparison. The
# frozen source revision and attribution are recorded in cases/NICFD2026.
REFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "cases" / "NICFD2026" / "two_component"
)
REGIMES = {
    "subsonic": ("subsonic", 16e5, 2.0),
    "shocked": ("supersonic_shock", 12e5, 5.0),
    "adapted": ("supersonic_adapted", 98600.0, 2.0),
}


# Physical setup and API usage. Supporting helpers follow below.
def main():
    """Build the physical model, solve each requested regime and save the results."""
    configure_figures()
    if INITIALIZATION not in ("reference", "cold"):
        raise ValueError('INITIALIZATION must be "reference" or "cold".')
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # HEOS bicubic PH tables: JAXProp builds/caches them on first execution.
    # Later runs load these same tables. No perfect-gas liquid approximation.
    table_dir = str(OUTPUT_DIRECTORY / "fluid_tables")
    eos = dict(
        backend="HEOS",
        p_min=1e4,
        p_max=6e6,
        N_h=145,
        N_p=200,
        gradient_method="forward",
        table_dir=table_dir,
    )
    gas = jxp.FluidBicubic(fluid_name="nitrogen", h_min=5e4, h_max=4e5, **eos)
    liquid = jxp.FluidBicubic(fluid_name="water", h_min=1e3, h_max=1.4e5, **eos)
    p0 = RESERVOIR_PRESSURE
    Tg0 = GAS_STAGNATION_TEMPERATURE
    Tl0 = LIQUID_STAGNATION_TEMPERATURE
    quality = GAS_MASS_FLOW_FRACTION
    slip = INLET_SLIP
    sg = gas.get_state(jxp.PT_INPUTS, p0, Tg0)
    sl = liquid.get_state(jxp.PT_INPUTS, p0, Tl0)
    hg, hl, rg, rl = float(sg["h"]), float(sl["h"]), float(sg["rho"]), float(sl["rho"])
    alpha = quality * rl / ((1 - quality) * rg * slip + quality * rl)

    # The port's Stadtke FVS + nonconservative work has scheme ID
    # model_compatibility. Generic flux="stadtke" and HLL are unsupported here.
    # MUSCL Van Albada matches this NICFD case; other available limiters include
    # muscl_van_leer/minmod/superbee. Changing them is a new numerical choice.
    scheme = Scheme(
        flux="model_compatibility",
        reconstruction="muscl_van_albada",
        time_integrator="euler-implicit",
        boundary_treatment="coupled",
        newton_backtracking=True,
        newton_iterations=20,
    )
    compatibility = CompatibilityConfig(
        nc_form="wb",
        delta_factor=1e-6,
        lowmach_diss=1.0,
        lowmach_diss_boundary=0.0,
        du_blend_lo=5.0,
        du_blend_hi=20.0,
    )
    # Nitrogen and water exchange momentum and heat, but NOT mass. The original
    # number-density area law is the paper's law, rather than a later polynomial.
    closures = ClosureConfig(Nb=1e11, Nd=1e11, mass_transfer=False, area_law="original")
    stabilization = StabilizationConfig(
        void_strength=0.05,
        speed_scale="frozen",
        void_velocity_cutoff=None,
        void_velocity_width=12.0,
    )
    mesh = mesh_for(NUMBER_OF_CELLS)
    records = []
    for name in CASES_TO_RUN:
        folder, pressure, cfl = REGIMES[name]
        reference = np.genfromtxt(
            REFERENCE_ROOT / folder / "reference.csv", delimiter=",", names=True
        )
        boundaries = BoundaryConfig(
            inlet="stagnation",
            outlet="characteristic",
            H_g0=hg,
            H_l0=hl,
            s0=float(sg["s"]),
            dominant_phase="gas",
            quality=quality,
            slip=slip,
            back_pressure=pressure,
            pressure_relaxation=1.0,
        )
        if INITIALIZATION == "reference":
            # Explicit warm start: interpolate published primitive fields only.
            # The numerical residual and all stopping criteria are evaluated
            # afresh. This does not demonstrate independent cold convergence.
            initial = np.column_stack(
                [
                    np.interp(mesh.points, reference["x"], reference[key])
                    for key in ("p", "u_g", "u_l", "alpha_g", "h_g", "h_l")
                ]
            )
        else:
            speed = 10.0 * mesh.area_faces[0] / np.asarray(elliot(mesh.points).A)
            initial = np.column_stack(
                (
                    np.full_like(speed, p0),
                    slip * speed,
                    speed,
                    np.full_like(speed, alpha),
                    hg - 0.5 * (slip * speed) ** 2,
                    hl - 0.5 * speed**2,
                )
            )
        model = TwoFluidModel(
            mesh,
            gas,
            liquid,
            initial_primitives=initial,
            boundaries=boundaries,
            closures=closures,
            compatibility=compatibility,
            stabilization=stabilization,
            virtual_mass=0.5,
        )
        settings = SteadySettings(cfl=cfl, max_steps=MAX_PSEUDO_STEPS)
        print(
            f"\n=== {name}: N2 + water, p0=20 bar, pb={pressure/1e5:g} bar; "
            f"initialization={INITIALIZATION} ===",
            flush=True,
        )
        table = ConvergenceTable(model.equation_names, every=20)
        workspace = SteadyWorkspace(model, scheme, settings.residual)
        # Keep this workspace when repeating the same fixed model in-process;
        # disk caching handles eligible executables across fresh Python runs.
        result = solve_single_grid(
            model, scheme, settings, callback=table, workspace=workspace
        )
        if table.last_step != result.history[-1]["step"]:
            table(result.history[-1], force=True)
        values = flow_fields(model, result.state)
        mach = values["Ma_mix"]
        if name == "subsonic":
            regime_ok = np.max(mach) < 1
        elif name == "shocked":
            regime_ok = np.max(mach) > 1 and mach[-1] < 1
        else:
            regime_ok = mach[-1] > 1
        summary = solution_summary(result)
        summary.update(
            initialization=INITIALIZATION,
            regime_passed=bool(regime_ok),
            max_mach=float(np.max(mach)),
            outlet_mach=float(mach[-1]),
        )
        print(strict_json(summary), flush=True)
        (OUTPUT_DIRECTORY / f"{name}.json").write_text(
            strict_json(
                dict(
                    summary=summary,
                    history=result.history,
                    scheme=asdict(scheme),
                    settings=asdict(settings),
                    boundaries=asdict(boundaries),
                    closures=asdict(closures),
                    compatibility=asdict(compatibility),
                    stabilization=asdict(stabilization),
                    eos=dict(gas="nitrogen HEOS", liquid="water HEOS", grid=[145, 200]),
                )
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            OUTPUT_DIRECTORY / f"{name}.npz",
            x=mesh.centers,
            full_state=result.state,
            **values,
        )
        records.append((name, result, values, reference, regime_ok, settings))
    plot_results(records, mesh, OUTPUT_DIRECTORY)
    if not all(s.success and ok for _, s, _, _, ok, _ in records):
        raise RuntimeError(
            "A requested case failed convergence or regime checks; actual results were saved."
        )


# Supporting solver, geometry and figure helpers.
def elliot(coordinate):
    """The paper's fixed contour, with lengths and radii explicitly in metres."""
    return get_nozzle_elliot(
        np.asarray(coordinate),
        roughness=0.0,
        convergent_length=0.09940,
        divergent_length=0.16652,
        radius_in=0.0254778,
        radius_throat=0.00655,
        radius_out=0.01413,
        throat_first_length=0.04223,
        radius_throat_in=0.007952,
        throat_second_length=0.011512,
        radius_throat_out=0.0068617,
    )


def mesh_for(cells):
    """Integrate cell volumes, splitting quadrature at every contour join."""
    faces = np.linspace(0.0, 0.31, cells + 1)
    centers = 0.5 * (faces[:-1] + faces[1:])
    z, weights = leggauss(24)
    positions, factors, indices = [], [], []
    for i, (left, right) in enumerate(zip(faces[:-1], faces[1:], strict=True)):
        splits = (
            [left]
            + [x for x in (0.09940, 0.14163, 0.153142) if left < x < right]
            + [right]
        )
        for a, b in zip(splits[:-1], splits[1:], strict=True):
            positions.extend(0.5 * (a + b) + 0.5 * (b - a) * z)
            factors.extend(0.5 * (b - a) * weights)
            indices.extend([i] * len(z))
    sampled, face_data = elliot(centers), elliot(faces)
    volumes = np.bincount(
        indices, weights=np.asarray(elliot(positions).A) * factors, minlength=cells
    )
    return Mesh(
        faces=faces,
        centers=centers,
        volumes=volumes,
        area_faces=np.asarray(face_data.A),
        area_centers=np.asarray(sampled.A),
        darea_centers=np.asarray(sampled.dAdx),
        area_derivative_integrals=np.diff(np.asarray(face_data.A)),
    )


def flow_fields(model, full):
    """Phase properties and the model's acoustic-mixture Mach definition."""
    U = primitives(jnp.asarray(full))[1:-1]
    gas = model.fluid_g.get_state(jxp.HmassP_INPUTS, U[:, 4], U[:, 0])
    liquid = model.fluid_l.get_state(jxp.HmassP_INPUTS, U[:, 5], U[:, 0])
    # This is the model-owned characteristic information, not an ideal-gas
    # sound-speed formula. The NICFD reproduction uses the same acoustic pair.
    acoustic = model.acoustic_diagnostics(jnp.asarray(full)[1:-1])
    return {
        key: np.asarray(value)
        for key, value in dict(
            p=U[:, 0],
            alpha_g=U[:, 3],
            u_g=U[:, 1],
            u_l=U[:, 2],
            T_g=gas["T"],
            T_l=liquid["T"],
            rho_g=gas["rho"],
            rho_l=liquid["rho"],
            Ma_mix=acoustic["mach"],
        ).items()
    }


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


def plot_results(records, mesh, output):
    """Render all regimes from plain arrays and the original solver histories."""
    plot_records = []
    summaries = {}
    for name, result, fields, reference, regime_ok, settings in records:
        summary = solution_summary(result)
        summary["regime_passed"] = bool(regime_ok)
        summaries[name] = summary
        exact = {key: reference[key] for key in ("x", *fields)}
        plot_records.append(
            dict(
                name=name,
                x=mesh.centers,
                fields=fields,
                reference=exact,
                history=result.history,
                settings=asdict(settings),
            )
        )
    plot_profiles(plot_records, output)
    plot_convergence(plot_records, output)
    (output / "summary.json").write_text(strict_json(summaries), encoding="utf-8")
    print(f"Saved phase fields and publication figures to {output.resolve()}")


if __name__ == "__main__":
    # One concise cache/compilation summary; details also go to compilation.json.
    with CompilationReport(OUTPUT_DIRECTORY / "compilation.json", verbose=VERBOSE_JAX):
        main()
