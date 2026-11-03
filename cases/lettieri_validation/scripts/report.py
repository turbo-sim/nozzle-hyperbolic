"""Offline Lettieri explorer from saved runs, fields and experimental data."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from development.flux_methods.shared.offline_report import prepare_inputs, render
from .data import ROOT, cfd, definitions, experiment, workbook_rows
from .comparison_report import load_comparison,combined_application,attach_comparison_evidence

ASSETS = Path(__file__).resolve().parent / "report_assets"

SECTIONS = dict(
    TITLE="Lettieri equilibrium HEM validation",
    EYEBROW="NOZZLE HYPERBOLIC / EXPERIMENTAL VALIDATION",
    LEAD="Five CO₂ nozzle operating points. Inspect saved equilibrium solutions, measured pressures and the gap between a converged numerical model and delayed experimental condensation.",
    OVERVIEW_TITLE="Numerical convergence and experimental agreement are different results.",
    SCOPE="Thirteen constant-reconstruction HLL attempts cover L1–L5, including a domain-unaudited preflight and a stretched L1 mesh. Default comparisons use audited 96-cell uniform solutions. No case was rerun for this presentation.",
    CONTROLS='''<label>Operating case<select id="case"></select></label><label>Mesh level<select id="cells"></select></label><label>Grid geometry<select id="grid-kind"><option value="all">All available grids</option><option value="uniform">Uniform</option><option value="stretched">Stretched only</option></select></label><label>Attempt scope<select id="attempt-scope"><option value="audited">Property-domain audited</option><option value="all">All attempts, including preflight</option></select></label>
<fieldset class="choice-set"><legend>Saved solutions in the selected scope</legend><div id="runs" class="choices"></div><div class="choice-tools"><button id="runs-all" type="button">All available</button><button id="runs-none" type="button">Clear</button><label><input id="show-cfd" type="checkbox" checked> Upstream reference CFD, where supplied</label><label><input id="show-onset" type="checkbox" checked> Measured condensation onset</label></div></fieldset>''',
    OVERVIEW='''<div id="case-cards" class="case-grid"></div><div class="prose-grid"><article class="panel prose"><h3>What the saved results show</h3><div id="findings"></div></article><article class="panel prose"><h3>Interpretation</h3><p>The five finest audited uniform solutions satisfy the saved original-equation checks. That establishes a numerically converged equilibrium calculation, not agreement with delayed condensation in the experiment.</p><p>Mesh refinement changes the pressure predictions, but the experimental discrepancy remains several bar. Discretization, geometry, property representation, equilibrium modeling and measurement differences all contribute. Two levels do not prove grid independence.</p><p>The visible onset line is a measured condensation location. Equilibrium phase appearance is physically different; no threshold-derived predicted onset or experimental liquid-fraction samples are invented.</p></article></div><article class="panel plot-panel"><div id="gallery-plot" class="plot gallery"></div><p class="caption">One panel per operating case: finest property-audited uniform HEM solution, measured pressure taps, and the supplied L1/L5 reference CFD. This campaign-wide gallery is independent of the explorer filters.</p></article><article class="panel"><h3>All retained attempts</h3><div class="table-wrap"><table><thead><tr><th>Case / attempt</th><th>Cells</th><th>Stretch</th><th>Recorded root</th><th>Property domain</th><th>Saved audit</th><th>RMSE [bar]</th></tr></thead><tbody id="all-rows"></tbody></table></div></article>''',
    EXPLORER='''<p id="reference-notice" class="notice warning"></p><div class="plot-grid"><article class="panel plot-panel"><div id="pressure-plot" class="plot"></div><p class="caption">Pressure at 13 measured taps. Exact digitized positions exist for L1/L5; L2–L4 use explicitly approximate nominal coordinates. No uncertainty bars are supplied.</p></article><article class="panel plot-panel"><div id="quality-plot" class="plot"></div><p class="caption">Liquid mass fraction = 1−Q only where saved raw quality is finite and 0≤Q≤1. Sentinels and interpolation overshoots remain gaps, never clipped. The vertical line is measured onset.</p></article><article class="panel plot-panel"><div id="residual-plot" class="plot"></div><p class="caption">Actual accepted trajectories with separate mass, momentum and energy fixed-scale L2 norms. Dashed threshold: the original 10⁻⁹ absolute target.</p></article><article class="panel plot-panel"><div id="cfl-plot" class="plot"></div></article><article class="panel plot-panel"><div id="massflow-plot" class="plot"></div><p class="caption">Saved cell ρuA is a finite-grid diagnostic. The separately plotted conserved numerical face-flow value is the physical-budget quantity. Experimental mass flow was not supplied.</p></article><article class="panel plot-panel"><div id="mesh-plot" class="plot"></div><p class="caption">All property-audited uniform meshes for this operating point, plus the available stretched L1 run. These are sensitivity observations, not a grid-independence certificate.</p></article></div><article class="panel"><h3>Selected saved numerical checks and experimental error</h3><div class="table-wrap"><table><thead><tr><th>Attempt</th><th>Mass / momentum / energy L2</th><th>BC / global max</th><th>Saved audit</th><th>RMSE [bar]</th><th>Masked Q</th><th>Numerical flow [kg/s]</th></tr></thead><tbody id="selected-rows"></tbody></table></div></article>''',
    DISCUSSION='''<div class="prose-grid"><article class="panel prose"><h3>Equilibrium HEM versus delayed condensation</h3><p>The shared Euler/HEM model assumes phase equilibrium and uses JAXProp CO₂ bicubic properties. The experiment contains delayed condensation. A converged HEM pressure profile therefore need not reproduce the measured nonequilibrium transition.</p><p>The observed discrepancy is a physical validation result. These saved results alone do not isolate how much comes from equilibrium modeling versus discretization, contour reconstruction or experimental data limitations.</p></article><article class="panel prose"><h3>Liquid fraction is a diagnostic, not an onset detector</h3><p>Saved raw quality Q is mapped to 1−Q only on the finite interval [0,1]. Single-phase sentinels such as −1 and overshoots outside that range become gaps. Mask counts and raw ranges remain in the selected details.</p><p>The only liquid-fraction reference curves available are the upstream L1/L5 CFD series. They are not experimental liquid-fraction measurements. No computed threshold onset is asserted from interpolated equilibrium quality.</p></article><article class="panel prose"><h3>Geometry, depth and conservation</h3><p>The measured half-height spline uses x = −z from Appendix Table 4. Its raw workbook inlet half-height is 6.38 mm; the publication-corrected value is 6.35 mm. Hardware depth is 6.48 mm, consistent with the nominal 20 mm² throat area and 3.0862 mm full throat height.</p><p>Conservative numerical face mass flow is used in the physical budget. Cell ρuA may vary at finite resolution. Full-channel kg/s and half-channel flow per unit depth in kg/(s m) are different quantities, retained separately in the saved audits.</p></article><article class="panel prose"><h3>Experimental provenance and limits</h3><p>Appendix Tables 2/3 supply first-repeat reservoir conditions and pressure values for all five cases. Original L1/L5 Excel and reference CFD bytes come from Andrea revision bcfe6ae1ec8abd71528e57b218ac8291dc9461fa.</p><p>L2–L4 exact tap coordinates, original workbooks and reference CFD remain unavailable. Their nominal tap coordinates are approximate; no measurement uncertainty or missing CFD curve is fabricated.</p><p>Primary source: Lettieri et al., <a href="https://doi.org/10.1115/1.4038082">DOI 10.1115/1.4038082</a> (2018 issue, online 2017).</p></article></div>''',
    PROTOCOL='''<div class="prose-grid"><article class="panel prose"><h3>Numerical model and properties</h3><p>The preserved original baseline uses shared NozzleModel and generic FVM, equilibrium JAXProp CO₂ bicubic properties, zero wall shear and heat transfer. The saved runs use HLL, constant reconstruction, coupled backward Euler with Newton backtracking, colored AD and banded linear algebra.</p><p>The selected Andrea table has 200×200 nodes over 50–850 kJ/kg and 0.1–300 bar. The exposed operating rectangle is 300–550 kJ/kg and 10–100 bar, with supporting interpolation data audited separately.</p><p>The renderer reads NPZ field arrays and original JSON records. It never imports the flow model, rebuilds the property table, evaluates an EOS or advances a solution.</p></article><article class="panel prose"><h3>Original checks and recorded epochs</h3><p>Stored verification records independently re-evaluated each saved field at their recorded source epoch. This presentation only displays those existing results. It does not claim a fresh check against today's edited source.</p><p>The recorded equation policy requires both relative 10⁻⁶ and absolute 10⁻⁹ convergence, with separate 10⁻⁹ boundary/global targets. All physical branch and admissibility checks remain distinct. Preflight and unfinished attempts stay available. Elapsed/preparation costs are recorded metadata, not a repeated timing ranking.</p></article></div>''')


def main():
    prepare_inputs(ROOT, [ROOT / "settings.json", *(ROOT / "data").glob("*"),
        *(ROOT / "outputs").glob("*/result.json"), *(ROOT / "outputs").glob("*/verification.json"),
        *(ROOT / "outputs").glob("*/solution.npz"), *(ROOT / "outputs").glob("*.json")])
    doc = definitions()
    cases = []
    for name, definition in doc["cases"].items():
        x, p, origin = experiment(name)
        upstream = cfd(name)
        cases.append(dict(id=name, definition=definition, experiment=dict(x_m=x, pressure_pa=p, origin=origin), upstream_cfd=upstream))
    records = []
    for file in sorted((ROOT / "outputs").glob("*/result.json")):
        row = json.loads(file.read_text(encoding="utf-8"))
        saved = file.with_name("solution.npz")
        fields = {}
        if saved.is_file():
            with np.load(saved) as z:
                fields = {key: z[key] for key in z.files}
        verification = file.with_name("verification.json")
        checked = json.loads(verification.read_text(encoding="utf-8")) if verification.exists() else None
        q = np.asarray(fields.get("quality", []), dtype=float)
        valid = np.isfinite(q) & (q >= 0) & (q <= 1)
        fraction = np.where(valid, 1 - q, np.nan)
        diagnostic = dict(masked_samples=int(np.count_nonzero(~valid)), samples=len(q),
                          raw_min=float(np.min(q[np.isfinite(q)])) if np.isfinite(q).any() else None,
                          raw_max=float(np.max(q[np.isfinite(q)])) if np.isfinite(q).any() else None)
        records.append(dict(row, id=file.parent.name, input_file=file.relative_to(ROOT).as_posix(),
                            fields=fields, displayed_liquid_fraction=fraction, quality_diagnostic=diagnostic,
                            saved_verification=checked))
    data = dict(study="lettieri", cases=cases, records=records,
                settings=json.loads((ROOT / "settings.json").read_text(encoding="utf-8")),
                publication=doc)
    comparison,comparison_files=load_comparison(ROOT,records)
    sections=dict(SECTIONS)
    if comparison is not None:
        geometry=data["settings"]["geometry"]
        rows=workbook_rows(ROOT/"data"/geometry["workbook"])[1:]
        control_x=np.asarray([row["A"] for row in rows])
        control_y=np.asarray([row["B"] for row in rows]);control_y[0]=geometry["inlet_half_height_m"]
        spline=CubicSpline(control_x,control_y)
        profile_x=np.linspace(control_x[0],control_x[-1],501)
        check=next(row for row in records if row["cells"]==96 and row.get("property_domain_audited") and row["stretch"]==1)
        np.testing.assert_allclose(2*geometry["width_m"]*spline(check["fields"]["x_m"]),check["fields"]["area_m2"],rtol=3e-14,atol=1e-18)
        data["geometry_profile"]=dict(x_m=profile_x,half_height_m=spline(profile_x),control_x_m=control_x,
            control_half_height_m=control_y,depth_m=geometry["width_m"],
            convention="Corrected 33-point measured half-height; the same not-a-knot cubic spline as the saved FVM geometry",
            saved_geometry_sample_audit=True)
        data["solver_comparison"]=comparison
        data["comparison_markup"]=(ASSETS/"comparison.html").read_text(encoding="utf-8")
        sections.update(TITLE="Lettieri HEM · solver verification and validation",
            EYEBROW="SPACE MARCHING / TIME MARCHING / EXPERIMENT",
            LEAD="Compare independently choked NozzleX space marches with steady FVM solutions, with and without Haaland friction. The original thirteen inviscid attempts and experimental comparisons remain available.",
            SCOPE="The current comparison uses HLL with MUSCL and the van Leer limiter on every grid, sequenced through 50, 100 and 200 cells for L1–L5 with and without Haaland friction. Thirty new FVM states are compared with independently calculated critical space marches. The original thirteen constant-reconstruction validation attempts and the earlier 48/96-cell comparison remain preserved. Numerical verification remains distinct from experimental agreement.")
        sections["PROTOCOL"]+='''<article class="panel prose"><h3>Independent space-marching comparison</h3><p>The Solver comparison page reads a separate saved campaign under outputs/solver_comparison/. Its critical inlet is determined independently, preserving the existing CO₂ table, tabulated sound-speed and Grüneisen conventions. No FVM profile is used as a shooting seed.</p><p>Both wall variants now use HLL, MUSCL with the van Leer limiter, and coupled implicit Euler. Each 50-cell case starts from the model's cold Mach ramp; its converged state is linearly prolonged to 100 and then 200 cells. Each grid resets CFL and residual normalization, and must pass the unchanged original-equation, boundary and global-budget targets. The Haaland case uses smooth walls, zero wall heat, the full rectangular wetted perimeter and matching hydraulic diameter.</p><p>Comparison inputs have a separate SHA-256 manifest in the evidence download, which also contains the earlier constant-reconstruction comparison archive. The original input manifest and saved baseline bytes are not replaced. Presentation rebuilds evaluate no EOS and advance no numerical solution.</p></article>'''
    application=combined_application(ROOT,ASSETS)
    render(ROOT,data,sections,application,files=[Path(__file__),ASSETS/"app.js",ASSETS/"comparison.js",
        ASSETS/"comparison.html",Path(__file__).with_name("comparison_report.py"),
        Path(__file__).with_name("audit_comparison.py"),Path(__file__).with_name("comparison.py"),
        Path(__file__).with_name("comparison_data.py"),ROOT/"comparison_settings.json",ROOT/"SOLVER_COMPARISON.md",
        Path(__file__).with_name("data.py"),ROOT/"README.md",ROOT/"REPORT.md"])
    attach_comparison_evidence(ROOT,comparison_files)


if __name__ == "__main__":
    main()
