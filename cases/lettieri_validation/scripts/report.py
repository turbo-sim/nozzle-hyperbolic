"""Build the small validation report from saved runs; never invokes a solver."""

import base64
import html
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ROOT, cfd, definitions, experiment


def markdown_table(header, rows):
    return ("| " + " | ".join(header) + " |\n|" + "|".join(["---"]*len(header)) + "|\n" +
            "\n".join("| " + " | ".join(row) + " |" for row in rows))


def html_table(header, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(v)}</th>" for v in header) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) +
                                             "</tr>" for row in rows) + "</tbody></table>")


def main():
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    records = []
    for path in sorted(output.glob("*/result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        check = path.with_name("verification.json")
        if check.exists():
            data["fresh_verification"] = json.loads(check.read_text(encoding="utf-8"))
        records.append((path.parent, data))
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 14), sharex=True, layout="constrained")
    for ax, case in zip(axes, definitions()["cases"]):
        x, p, origin = experiment(case)
        label = "Experiment (approximate tap x)" if "approximate" in origin else "Experiment (upstream digitized x)"
        ax.plot(x * 1e3, p / 1e5, "ks", ms=4, label=label)
        ax.axvline(definitions()["cases"][case]["onset_mm"], color="0.25", linestyle=":", linewidth=1,
                   label="Measured condensation onset")
        reference = cfd(case)
        if reference:
            xp, pp = reference["pressure"].T
            keep = (xp >= -0.032013) & (xp <= 0.066355)
            ax.plot(xp[keep]*1e3, pp[keep]/1e5, "--", color="0.4", label="Upstream reference CFD")
        runs = [(folder, r) for folder, r in records if r["case_id"] == case]
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, max(1, len(runs))))
        for color, (folder, run) in zip(colors, runs):
            file = folder / "solution.npz"
            if file.exists():
                state = np.load(file)
                suffix = "converged" if run["success"] else f"UNFINISHED: {run['reason']}"
                if not run.get("property_domain_audited"):
                    suffix = "preflight, property domain unaudited"
                mesh_label = "" if run["stretch"] == 1 else f", stretch={run['stretch']:g}"
                ax.plot(state["x_m"]*1e3, state["p"]/1e5, color=color,
                        linestyle="-" if run["stretch"] == 1 else ":",
                        label=f"HEM N={run['cells']}{mesh_label}, {suffix}")
        ax.set_ylabel("Pressure (bar)")
        ax.set_title(case, loc="left")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Axial position relative to throat (mm)")
    axes[-1].set_xlim(-32.013,66.355)
    fig.savefig(output / "pressure_profiles.png", dpi=160)
    fig.savefig(output / "pressure_profiles.pdf")
    plt.close(fig)

    rows = []
    for folder, run in records:
        comparison = run.get("comparison", {})
        metrics = run.get("metrics", {})
        norm = metrics.get("equation_norm", [])
        maximum = max(norm) if norm and all(v is not None for v in norm) else None
        rmse = comparison.get("pressure_rmse_bar")
        status = "converged" if run["success"] else run["reason"]
        if not run.get("property_domain_audited"):
            status += " (preflight; domain unaudited)"
        rows.append([run["case_id"], str(run["cells"]), str(run["stretch"]), status,
                     "—" if maximum is None else f"{maximum:.3e}",
                     "—" if rmse is None else f"{rmse:.5f}",
                     str(len([h for h in run.get("history",[]) if h.get("accepted") and h.get("phase") != "initial"]))])
    header = ["Case", "Cells", "Stretch", "Status", "Max equation L2", "Pressure RMSE (bar)", "Accepted updates"]
    table_md = markdown_table(header, rows)
    if not rows:
        table_md += "\nNo completed attempts are present."
    final_rows, refinement_rows, finest = [], [], []
    for case in definitions()["cases"]:
        candidates = [(folder, r) for folder, r in records if r["case_id"] == case and r.get("property_domain_audited") and r["stretch"] == 1]
        if not candidates:
            continue
        folder, run = max(candidates, key=lambda pair: (pair[1]["cells"], pair[0].name))
        finest.append(run)
        verification = folder / "verification.json"
        if verification.exists():
            checked = json.loads(verification.read_text(encoding="utf-8"))
            norms = checked["equation_l2"]
            final_rows.append([case, str(run["cells"]), "passed" if checked["verified"] else "FAILED",
                               *[f"{v:.2e}" for v in norms], f"{checked['boundary_max']:.2e}",
                               f"{max(checked['global_defect']):.2e}",
                               f"{checked['mass_flow']['numerical_inlet_kg_s']:.6f}",
                               f"{checked['mass_flow']['half_channel_per_depth_kg_s_m']:.4f}",
                               f"{checked['endpoint_mach'][-1]:.4f}"])
        coarse = [(f,r) for f,r in candidates if r["cells"] < run["cells"]]
        if coarse:
            small, old = max(coarse, key=lambda pair: (pair[1]["cells"], pair[0].name))
            fine_field, coarse_field = np.load(folder / "solution.npz"), np.load(small / "solution.npz")
            x, _, _ = experiment(case)
            pressure_change = np.interp(x, fine_field["points_m"], fine_field["state"][:,0]) - np.interp(x, coarse_field["points_m"], coarse_field["state"][:,0])
            refinement_rows.append([case, f"{old['cells']} to {run['cells']}", f"{np.max(np.abs(pressure_change))/1e5:.5f}",
                                    f"{old['comparison']['pressure_rmse_bar']:.5f}", f"{run['comparison']['pressure_rmse_bar']:.5f}"])
    budget_header = ["Case", "Cells", "Fresh audit", "Mass L2", "Momentum L2", "Energy L2", "BC max", "Global max", "Numerical flow (kg/s)", "Half-channel per depth (kg/(s m))", "Outlet Mach"]
    refinement_header = ["Case", "Cells", "Max pressure change at taps (bar)", "Coarse RMSE (bar)", "Fine RMSE (bar)"]
    budgets_md = markdown_table(budget_header, final_rows)
    refinement_md = markdown_table(refinement_header, refinement_rows)
    if finest:
        errors = [r.get("comparison", {}).get("pressure_rmse_bar") for r in finest]
        errors = [v for v in errors if v is not None]
        findings = f"{sum(bool(r['success']) for r in finest)} of {len(finest)} finest-grid cases converged."
        if errors:
            findings += f" Their pressure RMSE spans {min(errors):.3f}–{max(errors):.3f} bar."
        findings += " These errors measure agreement with the experiment; the residual tolerances measure the discrete root."
    else:
        findings = "No completed flow attempts are available."
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 12), sharex=True, layout="constrained")
    quality_diagnostics = {}
    for ax, case in zip(axes, definitions()["cases"]):
        candidates = [(folder, r) for folder,r in records if r["case_id"] == case and r.get("property_domain_audited") and r["stretch"] == 1]
        if candidates:
            folder, run = max(candidates, key=lambda pair: pair[1]["cells"])
            state = np.load(folder / "solution.npz")
            quality = state["quality"]
            valid = np.isfinite(quality) & (quality >= 0) & (quality <= 1)
            fraction = np.where(valid, 1-quality, np.nan)
            ax.plot(state["x_m"]*1e3, fraction, color=plt.cm.plasma(.8), label=f"Equilibrium HEM N={run['cells']}")
            quality_diagnostics[case] = dict(cells=run["cells"], quality_min=float(np.nanmin(quality)),
                                             quality_max=float(np.nanmax(quality)), masked_samples=int(np.sum(~valid)),
                                             mapping="liquid_mass_fraction=1-Q only where raw Q is finite and 0<=Q<=1; otherwise gap",
                                             onset_prediction_mm=None,
                                             onset_reason="No threshold-based prediction claimed from interpolated equilibrium quality")
        reference = cfd(case)
        if reference:
            xq, liquid = reference["liquid_fraction"].T
            ax.plot(xq*1e3, liquid, "--", color="0.4", label="Upstream reference CFD")
        onset = definitions()["cases"][case]["onset_mm"]
        ax.axvline(onset, color="0.25", linestyle=":", linewidth=1.3,
                   label=f"Measured condensation onset: {onset:g} mm")
        ax.set_title(case,loc="left")
        ax.set_ylabel("Liquid mass fraction (–)")
        ax.set_ylim(-.01,.35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Axial position relative to throat (mm)")
    axes[-1].set_xlim(-32.013,66.355)
    fig.savefig(output / "liquid_fraction_profiles.png", dpi=160)
    fig.savefig(output / "liquid_fraction_profiles.pdf")
    plt.close(fig)
    (output / "quality_diagnostics.json").write_text(json.dumps(quality_diagnostics,indent=2),encoding="utf-8")
    quality_summary = "; ".join(
        f"{case}: N={info['cells']}, {info['masked_samples']} masked samples, raw Q range {info['quality_min']:.6f} to {info['quality_max']:.6f}"
        for case,info in quality_diagnostics.items()) + "."
    fig, axes = plt.subplots(5, 1, figsize=(8.2, 12), sharex=True, layout="constrained")
    for ax, case in zip(axes, definitions()["cases"]):
        candidates = [r for _,r in records if r["case_id"] == case and r.get("property_domain_audited") and r["stretch"] == 1]
        if candidates:
            chosen = max(candidates, key=lambda r: r["cells"])
            history = [r for r in chosen.get("history",[]) if r.get("accepted") and "equation_l2" in r]
            if history:
                steps = [r["step"] for r in history]
                norms = np.asarray([r["equation_l2"] for r in history])
                for i,(label,color) in enumerate(zip(("Mass", "Momentum", "Energy"), plt.cm.plasma([.2,.5,.8]))):
                    ax.semilogy(steps, np.where(norms[:,i] > 0, norms[:,i], np.nan), color=color, label=label)
                ax.axhline(1e-9, color="0.4", linestyle="--", label="Absolute target")
            ax.set_title(f"{case}, N={chosen['cells']}",loc="left")
        ax.set_ylabel("Fixed-scale L2")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel("Accepted update (initial state is step 0)")
    fig.savefig(output / "residual_histories.png", dpi=160)
    fig.savefig(output / "residual_histories.pdf")
    plt.close(fig)
    text = """# Lettieri HEM validation results

Generated from saved outputs on 2 November 2026. All cases use the shared Euler/HEM
model, zero wall friction/heating, measured contour, and equilibrium JAXProp bicubic
properties. See [README.md](README.md) for sources, units and reproduction.

""" + findings + """

The data/geometry audit checks five 13-point pressure profiles, exact L1/L5
publication agreement, byte-exact upstream workbooks, uniform/nonuniform spline
volumes and pressure/geometry cancellation. It does not certify the flow solutions.

""" + table_md + """

Pressure errors for unsuccessful runs describe unfinished states and are not
validation conclusions. Convergence requires every transported equation, boundary
equation, global budget and the physical outlet audit to pass the unchanged fixed
targets. Exact final residuals, failure reasons and all attempted steps are retained
in each `outputs/<case>/result.json`. Preparation costs/table hashes are retained in
`outputs/properties.json` and in each result, separately from solve time.

## Fresh residual and physical-budget audit

The following audit re-evaluates the saved fields without advancing them, using
the original spatial equations and independently reducing volume-weighted L2
norms. The numerical flux telescoping identity, physical inlet/outlet regime and
admissibility are checked separately. These are dimensionless fixed-scale
residuals; thresholds are $10^{-9}$, with no relaxed acceptance rule.

""" + budgets_md + r"""

Mass flow in this table comes from the conservative numerical face flux.
The saved cell quantity $\rho u A$ varies at finite resolution and is a separate
physical diagnostic. The per-depth column means half-channel flow per unit
depth, not the mass flow through half of the 6.48 mm hardware depth.

## Mesh sensitivity

""" + refinement_md + r"""

These two resolutions are a sensitivity check, not proof of grid independence.
Pressure comparisons linearly interpolate saved primitive states to the
measurement positions. The reported errors combine discretization, geometry,
equilibrium-model and measurement differences; they are not uncertainty-weighted
scores. The nonuniform L1 run is an additional geometry/solver check.

![Five-case pressure profiles](outputs/pressure_profiles.png)

## Liquid fraction and measured condensation onset

![Equilibrium liquid fraction and measured onset](outputs/liquid_fraction_profiles.png)

The dotted lines show the measured condensation locations from Appendix Table 3,
not a computed onset. The only liquid-fraction reference curves available are
L1/L5 CFD; no experimental liquid-fraction samples are invented. The HEM curve
uses the saved raw JAXProp quality $Q$ and plots $1-Q$ only for finite
$0\le Q\le1$. Single-phase sentinels such as $Q=-1$, and interpolation overshoots
outside that interval, are left as gaps rather than clipped.

""" + quality_summary + r"""

This affects only this diagnostic plot, not the solver or EOS.
The exact per-run diagnostic counts are in `outputs/quality_diagnostics.json`.
No threshold-based numerical onset is claimed from this interpolated
equilibrium field. HEM phase appearance and delayed experimental condensation
are physically distinct; the measured onset remains an independent reference.

![Finest-grid per-equation residual histories](outputs/residual_histories.png)

The original L1/L5 Excel and reference CFD series are preserved. L2–L4 pressures
and reservoir inputs were recovered from Appendix Tables 2/3; their exact tap
coordinates, original workbooks and reference CFD files remain unavailable.
Nominal tap coordinates are explicitly approximate, with no invented uncertainty.
The 6.48 mm depth reproduces Andrea's convention and agrees with the primary
20 mm² throat area divided by the 3.0862 mm full throat height. Experimental
mass flow is not supplied. Full-channel and half-channel-per-depth
mass flows are reported separately in the machine-readable results.

The experiment measures nonequilibrium condensation. Equilibrium HEM is not
expected to reproduce delayed condensation identically. The report therefore
keeps numerical convergence separate from experimental agreement. No reference
CFD data were used to initialize the numerical solver.

The generated PNG pressure/residual plots were visually inspected. Structural
checks cover embedded images, internal navigation targets and the downloadable
JSON in the standalone HTML. Interactive browser verification could not be
completed because the browser access host was unavailable.
"""
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    table_html = html_table(header, rows)
    budgets_html = html_table(budget_header, final_rows)
    refinement_html = html_table(refinement_header, refinement_rows)
    image = base64.b64encode((output / "pressure_profiles.png").read_bytes()).decode()
    residual_image = base64.b64encode((output / "residual_histories.png").read_bytes()).decode()
    fraction_image = base64.b64encode((output / "liquid_fraction_profiles.png").read_bytes()).decode()
    download = base64.b64encode(json.dumps([r for _,r in records]).encode()).decode()
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Lettieri HEM validation</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#202124;background:#fff}}nav a{{margin-right:1rem}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #bbb;text-align:left;padding:.4rem}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}section{{margin:2rem 0}}.scroll{{overflow-x:auto}}</style>
<h1>Lettieri equilibrium HEM validation</h1><nav><a href="#results">Results</a><a href="#profiles">Pressure and convergence</a><a href="#fractions">Liquid fraction and onset</a><a href="#limits">Limits and provenance</a></nav>
<section id="results"><h2>Saved run status</h2><p>Shared Euler/HEM model, zero wall friction and heating, equilibrium bicubic JAXProp properties. Generated 2 November 2026. Unsuccessful attempts remain unfinished; their errors are not validation conclusions.</p><p>{html.escape(findings)}</p>{table_html}
<p><a download="lettieri_results.json" href="data:application/json;base64,{download}">Download settings, histories and results (JSON)</a></p><h2>Fresh residual and physical-budget audit</h2><div class="scroll">{budgets_html}</div><p>Each transported equation is checked separately. Numerical face mass flow is used for the budget; cell rho*u*A is a separate finite-grid diagnostic. The half-depth flow is per unit depth, in kg/(s m).</p><h2>Mesh sensitivity</h2>{refinement_html}<p>Two levels provide a sensitivity check, not proof of grid independence. Pressure errors use interpolation to measurement positions.</p></section>
<section id="profiles"><h2>Pressure profiles</h2><img alt="Pressure comparisons for all five Lettieri cases, with run status in each legend" src="data:image/png;base64,{image}"><h2>Finest-grid residual histories</h2><img alt="Separate mass, momentum and energy residuals against accepted updates" src="data:image/png;base64,{residual_image}"></section>
<section id="fractions"><h2>Liquid fraction and measured condensation onset</h2><img alt="Equilibrium liquid mass fraction for all five cases, reference CFD for L1 and L5, and measured condensation onset markers" src="data:image/png;base64,{fraction_image}"><p>The dotted lines show measured condensation locations, not computed onset. Only L1/L5 have reference CFD liquid-fraction curves; no experimental fraction samples are supplied. The saved raw JAXProp quality Q is mapped to 1−Q only when finite and between 0 and 1. Sentinels such as −1 and out-of-range interpolation values remain gaps, without clipping.</p><p>{html.escape(quality_summary)}</p><p>No threshold-based predicted onset is claimed from these interpolated equilibrium diagnostics.</p></section>
<section id="limits"><h2>Limits and provenance</h2><p>The source is Lettieri et al., <a href="https://doi.org/10.1115/1.4038082">DOI 10.1115/1.4038082</a>, 2018 issue, online 2017. Appendix Tables 2/3 supply first-repeat conditions and pressures for all five cases. Original L1/L5 Excel and CFD series came from Andrea revision bcfe6ae1ec8abd71528e57b218ac8291dc9461fa. L2–L4 exact tap coordinates and CFD series remain unavailable; nominal tap coordinates are explicitly approximate. No measurement uncertainty is invented.</p>
<p>The measured half-height spline uses x=-z from Appendix Table 4; its inlet half-height is corrected from the workbook's 6.38 mm to the publication's 6.35 mm. Depth is 6.48 mm as in Andrea, consistent with the primary 20 mm² throat area divided by the full throat height. Full-channel mass flow is 2 times depth times the half-channel per-depth flow; their units are kg/s and kg/(s m), respectively. Experimental mass flow is not supplied.</p><p>HEM imposes phase equilibrium; the experiment contains delayed condensation. Solver convergence, physical budgets and experimental agreement are separate checks. No CFD field was used as initialization.</p></section></html>"""
    (ROOT / "REPORT.html").write_text(document, encoding="utf-8")
    print(f"Wrote reports for {len(records)} saved attempts.")


if __name__ == "__main__":
    main()
