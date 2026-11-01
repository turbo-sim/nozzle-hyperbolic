"""Rebuild figures and the Markdown report from saved reference data only.

Run from the repository root with::

    .venv/Scripts/python.exe -B -m development.real_gas.co2_nozzle_reference.scripts.plot_results

This module imports neither JAXProp nor a nozzle solver. Plotting never reruns
the reference calculations or changes their saved numerical results.
"""

from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = (plt.colormaps["plasma"](0.2), plt.colormaps["plasma"](0.8))
STYLES = ("-", "--")
REGIMES = ("adapted", "shock")
LABELS = ("Adapted expansion", "Internal normal shock")


def read_profile(path):
    """Retain coincident shock traces and their side labels in stored order."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty reference profile: {path}")
    return {key: np.array([row[key] for row in rows], dtype=str if key == "side" else float)
            for key in rows[0]}


def save_figure(fig, folder, name):
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=190, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def decorate_spatial_axes(axes, case):
    for ax in np.asarray(axes).ravel():
        ax.axvline(0, color="0.55", linewidth=0.8, linestyle=":", zorder=0)
        ax.axvline(1e3 * case["shock"]["x"], color=COLORS[1], linewidth=0.8,
                   linestyle=":", alpha=0.75, zorder=0)
        ax.set_xlabel("Axial position x [mm]")
        ax.grid(alpha=0.2)


def plot_fields(case, profiles, folder):
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 10.5), layout="constrained")
    entries = (("p", 1e-6, "Pressure [MPa]"), ("T", 1, "Temperature [K]"),
               ("rho", 1, r"Density [kg m$^{-3}$]"), ("u", 1, r"Velocity [m s$^{-1}$]"),
               ("M_euler", 1, "Mach number"), ("Q", 1, "Raw table vapor mass quality"))
    for regime, label, color, style in zip(REGIMES, LABELS, COLORS, STYLES):
        data = profiles[regime]
        for ax, (key, scale, ylabel) in zip(axes.ravel(), entries):
            ax.plot(1e3 * data["x"], scale * data[key], color=color, linestyle=style,
                    linewidth=1.8, label=label)
            ax.set_ylabel(ylabel)
        axes[2, 0].plot(1e3 * data["x"], data["M_table"], color=color, linestyle=":",
                        linewidth=1.0, alpha=0.65, label=f"{label}: tabulated a")
    axes[2, 0].axhline(1, color="0.5", linewidth=0.7, zorder=0)
    axes[2, 1].axhline(0, color="0.5", linewidth=0.7, zorder=0)
    axes[2, 1].axhline(1, color="0.5", linewidth=0.7, zorder=0)
    axes[0, 0].legend(loc="best", fontsize=9)
    axes[2, 0].legend(loc="best", fontsize=8)
    decorate_spatial_axes(axes, case)
    exit_ratio = case["geometry"]["exit_ratio"]
    fig.suptitle(f"{case['label']} — exit area ratio {exit_ratio:g}\n"
                 "Continuous Euler reference with JAXProp bicubic 100 × 100 tables",
                 fontsize=14)
    fig.supxlabel("Dotted vertical lines: throat and solved shock. Quality is shown without clipping.",
                  fontsize=9)
    save_figure(fig, folder, f"{case['id']}_fields")


def plot_thermodynamics(case, profiles, folder):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), layout="constrained")
    for regime, label, color, style in zip(REGIMES, LABELS, COLORS, STYLES):
        data = profiles[regime]
        x = 1e3 * data["x"]
        axes[0, 0].plot(data["s"] / 1e3, data["T"], color=color, linestyle=style,
                        linewidth=1.8, label=label)
        axes[0, 1].plot(x, data["Z_table"], color=color, linestyle=style, linewidth=1.8,
                        label=label)
        entropy_anchor = np.full(data["x"].shape, case["reservoir"]["s"])
        if regime == "shock":
            downstream = np.isin(data["side"], ["shock_post", "downstream"])
            entropy_anchor[downstream] = case["shock"]["post"]["s"]
        axes[1, 0].plot(x, data["s"] - entropy_anchor, color=color, linestyle=style,
                        linewidth=1.8, label=label)
        axes[1, 1].plot(x, 100 * (data["a_table"] / data["a_euler"] - 1), color=color,
                        linestyle=style, linewidth=1.8, label=label)
    axes[0, 0].set(xlabel=r"Table entropy [kJ kg$^{-1}$ K$^{-1}$]", ylabel="Temperature [K]")
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0, 0].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.2)
    decorate_spatial_axes([axes[0, 1], axes[1, 0], axes[1, 1]], case)
    axes[0, 1].set_ylabel("Tabulated compressibility factor Z")
    axes[1, 0].set_ylabel(r"Smooth-path table entropy drift [J kg$^{-1}$ K$^{-1}$]")
    axes[1, 1].set_ylabel(r"Sound-speed difference 100 × ($a_{table}/a_{Euler}$ − 1) [%]")
    for ax in axes[1, :]:
        ax.axhline(0, color="0.5", linewidth=0.7, zorder=0)
    fig.suptitle(f"{case['label']} — exit area ratio {case['geometry']['exit_ratio']:g}\n"
                 "Thermodynamic interpolation diagnostics", fontsize=14)
    fig.supxlabel("Entropy drift is measured from the reservoir or the postshock anchor on each smooth branch.",
                  fontsize=9)
    save_figure(fig, folder, f"{case['id']}_thermodynamics")


def plot_geometry(summary, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6), sharex=True, layout="constrained")
    for i, case in enumerate(summary["cases"]):
        geometry = case["geometry"]
        x = np.linspace(geometry["x_in"], geometry["x_out"], 801)
        ratio = np.where(x < 0, 1 + (geometry["inlet_ratio"] - 1) * (x / geometry["x_in"])**2,
                         1 + (geometry["exit_ratio"] - 1) * (x / geometry["x_out"])**2)
        radius = geometry["throat_radius"] * np.sqrt(ratio)
        color, style = COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]
        label = f"{case['id']}: exit ratio {geometry['exit_ratio']:g}"
        axes[0].plot(x * 1e3, radius * 1e3, color=color, linestyle=style, linewidth=1.7,
                     label=label)
        axes[0].plot(x * 1e3, -radius * 1e3, color=color, linestyle=style, linewidth=1.7)
        axes[0].fill_between(x * 1e3, -radius * 1e3, radius * 1e3, color=color, alpha=0.025)
        axes[1].plot(x * 1e3, ratio, color=color, linestyle=style, linewidth=1.7, label=label)
    axes[0].set_ylabel("Equivalent wall radius [mm]")
    axes[0].text(0.02, 0.05, "Vertical scale enlarged", transform=axes[0].transAxes,
                 color="0.4", fontsize=9)
    axes[1].set(xlabel="Axial position x [mm]", ylabel=r"Area ratio $A/A_t$")
    axes[0].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.axvline(0, color="0.5", linestyle=":", linewidth=0.8)
        ax.grid(alpha=0.2)
    fig.suptitle("Synthetic nozzles with a common converger and different exit area ratios", fontsize=14)
    save_figure(fig, folder, "nozzle_geometry")


def display_number(value):
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if value is None:
        return "—"
    return str(value).replace("|", "\\|")


def acoustic_mismatch(case):
    return max(profile["table_sound_speed_relative_mismatch_max"]
               for profile in case["audit"]["profiles"].values())


def compact_audit_rows(audit):
    """Keep the report focused; the complete evidence remains in summary.json."""
    profiles = list(audit["profiles"].values())
    thresholds = audit["thresholds"]
    rh = audit["rankine_hugoniot"]
    refinement = audit.get("refinement", {})
    refinement_values = [value for key, value in refinement.items()
                         if key not in ("threshold", "settings", "passed")
                         and isinstance(value, (int, float)) and not isinstance(value, bool)]
    primitive = max(q["absolute_accumulated_defect_relative_to_h0"]
                    for q in audit["independent_smooth_primitive_quadrature"].values())
    entropy = max(q["max_absolute_drift_J_kgK"]
                  for q in audit["smooth_entropy_diagnostics"].values())
    quality = (min(q["quality_mass_raw_min"] for q in profiles),
               max(q["quality_mass_raw_max"] for q in profiles))
    rows = [
        ("All required checks", audit["passed"], "pass"),
        ("Maximum mass-flow relative error", max(q["mass_flow_relative_max"] for q in profiles),
         thresholds["mass_flow_relative"]),
        ("Maximum total-enthalpy relative error", max(q["total_enthalpy_relative_max"] for q in profiles),
         thresholds["total_enthalpy_relative"]),
        ("Maximum Rankine–Hugoniot relative error", max(rh[k] for k in
         ("mass_relative", "momentum_relative", "total_enthalpy_relative")),
         thresholds["rankine_hugoniot_relative"]),
        ("Throat Euler Mach absolute error", audit["throat_M_euler_absolute_error"],
         thresholds["throat_mach_absolute"]),
        ("Maximum backpressure relative error", max(audit["shock_backpressure_relative_error"],
         audit["sampled_shock_exit_backpressure_relative_error"]), thresholds["backpressure_relative"]),
        ("Maximum independent smooth-primitive defect / $\\lvert h_0\\rvert$", primitive,
         thresholds["smooth_primitive_relative_to_h0"]),
        ("Maximum relative change under ODE refinement", max(refinement_values) if refinement_values else "not run",
         refinement.get("threshold", "—")),
        ("Maximum smooth entropy drift [J/(kg K)]", entropy, "diagnostic"),
        ("Maximum sound-speed relative mismatch", max(q["table_sound_speed_relative_mismatch_max"]
                                                    for q in profiles), "diagnostic"),
        ("Raw table quality range", f"[{quality[0]:.6g}, {quality[1]:.6g}]", "diagnostic"),
        ("Intended phase setup", audit["setup"]["passed"], "pass"),
    ]
    return [f"| {label} | {display_number(value)} | {display_number(limit)} |"
            for label, value, limit in rows]


def write_report(summary, profiles_by_case, root):
    lines = [
        "# CO2 nozzle reference results", "",
        "These are continuous, semi-analytic references for steady, inviscid, adiabatic quasi-one-dimensional Euler flow. "
        "Every flow-state property is evaluated through JAXProp's equilibrium bicubic backend using 100 × 100 tables. "
        "Smooth branches use a numerical pressure-coordinate primitive and scalar roots; the internal shock satisfies "
        "the Rankine–Hugoniot conditions, and its location is iterated to match the specified backpressure. No FVM solution is included.", "",
        "The density interpolant defines the mechanical EOS. Smooth segments enforce $dh/dp=1/\\rho$; "
        "interpolated entropy and sound speed are retained as diagnostics. Accordingly, root accuracy and conservation "
        "accuracy do not establish the accuracy of the 100 × 100 property approximation. See [EQUATIONS.md](EQUATIONS.md) for the equations and interpretation.", "",
        "The geometries are synthetic smooth nozzles with a common Lettieri-inspired converger and different exit area ratios, "
        "rather than the measured Lettieri contour. The supercritical case uses a smaller exit area ratio to keep the full path supercritical. "
        "The equilibrium condensation case does not model nucleation delay or the experimental Wilson line.", "",
        "![Synthetic nozzle geometry](outputs/figures/nozzle_geometry.png)", "",
        "| Case | $A_e/A_t$ | Reservoir $p_0$ [MPa] | $T_0$ [K] | Adapted $p_b$ [MPa] | Shock $p_b$ [MPa] | Shock $x_s$ [mm] | $\\dot m$ [kg/s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        settings = case["settings"]
        lines.append(f"| {case['label']} | {case['geometry']['exit_ratio']:g} | "
                     f"{settings['p0'] / 1e6:.5f} | {settings['T0']:.3f} | "
                     f"{case['adapted_exit']['p'] / 1e6:.6f} | {case['backpressure'] / 1e6:.6f} | "
                     f"{case['shock']['x'] * 1e3:.6f} | {case['mdot']:.7f} |")
    lines += ["", "Each adapted backpressure equals the corresponding supersonic exit pressure. "
              "For the shock case, the pressure in the table is the prescribed outlet value used by the outer position solve.", ""]
    by_id = {case["id"]: case for case in summary["cases"]}
    if "lettieri_equilibrium" in by_id and "supercritical" in by_id:
        two_phase_mismatch = acoustic_mismatch(by_id["lettieri_equilibrium"])
        supercritical_mismatch = acoustic_mismatch(by_id["supercritical"])
        lines += [f"The largest local sound-speed consistency difference is {100 * two_phase_mismatch:.2f}% "
                  "in the two-phase case near saturation. For the supercritical case it is "
                  f"{supercritical_mismatch:.6g} relative ({100 * supercritical_mismatch:.6g}%). "
                  "Thus the interpolation diagnostics include a substantial local acoustic mismatch even though "
                  "the density-EOS reference satisfies its conservation equations tightly. This consistency check "
                  "does not measure the physical accuracy of the property table.", ""]
    for case in summary["cases"]:
        ident = case["id"]
        profiles = profiles_by_case[ident]
        pre, post = case["shock"]["pre"], case["shock"]["post"]
        adapted = profiles["adapted"]
        table = case["settings"]["table"]
        lines += [
            f"## {case['label']}", "",
            f"The shock connects Euler Mach numbers {pre['M_euler']:.5f} and {post['M_euler']:.5f}. "
            f"Pressure rises from {pre['p'] / 1e6:.6f} to {post['p'] / 1e6:.6f} MPa; "
            f"the table entropy rise is {post['s'] - pre['s']:.6g} J/(kg K). "
            f"The raw table quality at the adapted exit is {case['adapted_exit']['Q']:.6g}.", "",
            f"The 100 × 100 bicubic table spans $h={table['h_min'] / 1e3:g}$–"
            f"${table['h_max'] / 1e3:g}$ kJ/kg and $p={table['p_min'] / 1e6:g}$–"
            f"${table['p_max'] / 1e6:g}$ MPa, with uniform spacing in $h$ and $\\log p$.", "",
            f"Over the adapted profile, the raw table quality ranges from {np.min(adapted['Q']):.6g} "
            f"to {np.max(adapted['Q']):.6g}, and the tabulated compressibility factor ranges from "
            f"{np.min(adapted['Z_table']):.6g} to {np.max(adapted['Z_table']):.6g}. "
            "Quality values are physically meaningful only in equilibrium two-phase states; "
            "the table can also return 1 for a supercritical state. Values outside $[0,1]$ "
            "are displayed unchanged and have no physical mixture-quality interpretation.", "",
            f"![{case['label']}: expected nozzle fields](outputs/figures/{ident}_fields.png)", "",
            f"![{case['label']}: interpolation diagnostics](outputs/figures/{ident}_thermodynamics.png)", "",
            "The principal Mach curves use $a_\\mathrm{Euler}^2=[\\rho_p+\\rho_h/\\rho]^{-1}$, "
            "calculated by differentiating the bicubic density. Dotted Mach curves use the separately interpolated table sound speed. "
            "The entropy-drift panel resets its anchor at the shock so that it measures smooth-path interpolation drift separately from the entropy jump.", "",
            "The compact verification table takes maxima over both profiles and all applicable smooth segments. "
            "Rankine–Hugoniot error is the maximum of its mass, momentum, and energy residuals. "
            "The refinement row is the maximum dimensionless change among the recorded flow and position quantities; "
            "the entropy and acoustic rows have no zero-error acceptance criterion.", "",
            "| Audit metric | Value | Acceptance limit |", "|---|---:|---:|",
            *compact_audit_rows(case["audit"]), "",
            f"Phase requirement: {case['audit']['setup']['requirement']}. "
            "The full checks, definitions, normalization scales, and refinement settings are retained in "
            "[outputs/summary.json](outputs/summary.json).", "",
            f"Saved fields: [adapted CSV](outputs/{ident}_adapted.csv), [shock CSV](outputs/{ident}_shock.csv). "
            f"Export figures: [fields PDF](outputs/figures/{ident}_fields.pdf), "
            f"[thermodynamics PDF](outputs/figures/{ident}_thermodynamics.pdf).", "",
        ]
    lines += ["## Reproduction", "",
              "The source settings are in [cases.json](cases.json), and the numerical values and provenance are in "
              "[outputs/summary.json](outputs/summary.json). Both shock traces occupy the same axial position in each shock CSV. "
              "The `side` column identifies the upstream and downstream values; retain both when comparing a discretized solution.", "",
              "Regenerate this report and the figures from saved fields, without thermodynamic calls or flow solves:", "",
              "```powershell",
              ".\\.venv\\Scripts\\python.exe -B -m development.real_gas.co2_nozzle_reference.scripts.plot_results",
              "```", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(root=ROOT):
    root = Path(root)
    output = root / "outputs"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    profiles_by_case = {}
    plot_geometry(summary, folder)
    for case in summary["cases"]:
        profiles = {regime: read_profile(output / f"{case['id']}_{regime}.csv") for regime in REGIMES}
        profiles_by_case[case["id"]] = profiles
        plot_fields(case, profiles, folder)
        plot_thermodynamics(case, profiles, folder)
    write_report(summary, profiles_by_case, root)
    print(f"Rebuilt {2 * len(summary['cases']) + 1} PNG/PDF figures and {root / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT,
                        help="Study directory containing outputs/summary.json")
    args = parser.parse_args()
    build_report(args.study)


if __name__ == "__main__":
    main()
