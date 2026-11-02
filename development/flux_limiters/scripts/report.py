"""Modern limiter report from saved JSON only; no solver imports."""

import argparse
import json
from pathlib import Path

from development.flux_methods.shared.offline_report import prepare_inputs, render

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
METHODS = ("van_leer", "minmod", "superbee", "van_albada")

SECTIONS = dict(
    TITLE="MUSCL limiter comparison",
    EYEBROW="NOZZLE HYPERBOLIC / RECONSTRUCTION STUDY",
    LEAD="Four slope limiters, one controlled nozzle shock. Compare saved profiles, analytical error and nonlinear work without confusing verification with a timing benchmark.",
    OVERVIEW_TITLE="One physical case. Eight saved configurations.",
    SCOPE="The saved campaign covers the perfect-gas Euler nozzle shock at 35% of the divergent section, with four MUSCL limiters on 24 and 48 cells. No additional cases or meshes have been simulated for this report.",
    CONTROLS='''<label>Physical case<select id="case"><option value="euler_nozzle/shock_35">Euler nozzle · shock at 35%</option></select></label>
<label>Physical cells<select id="cells"></select></label><label>Profile range<select id="profile-range"><option value="domain">Full domain</option><option value="shock">Near analytical shock</option></select></label>
<label>Convergence measure<select id="residual-mode"><option value="ratio">Worst equation / allowed norm</option><option value="equations">Separate fixed-scale equation L2</option><option value="global">Global defect / tolerance</option><option value="boundary">Boundary residual / tolerance</option></select></label>
<fieldset class="choice-set"><legend>Limiters included in the comparison</legend><div id="methods" class="choices"></div><div class="choice-tools"><button id="methods-all" type="button">All limiters</button><button id="methods-none" type="button">Clear</button><span>Same cold state, corrected Stadtke flux, backward Euler and mesh.</span></div></fieldset>''',
    OVERVIEW='''<div id="case-cards" class="case-grid"></div><div class="prose-grid"><article class="panel prose"><h3>What the saved results show</h3><div id="findings"></div></article><article class="panel prose"><h3>How to interpret the comparison</h3><p>More compressive limiting does not automatically give the smallest integrated error or the easiest nonlinear solve. These are coupled Euler equations with implicit Newton updates, rather than a scalar transport experiment.</p><p>All comparisons use independent cold starts. No reference profile or other limiter solution is a hidden initial guess. The plotted pressure reference is the saved continuous analytical curve; errors use the separately saved conservative cell projection.</p><p>Two coarse meshes reveal sensitivity and solver work, not an asymptotic order of accuracy at a discontinuity. Compile-inclusive elapsed values are retained as metadata, but cannot support a performance ranking.</p></article></div><article class="panel plot-panel"><div id="overview-errors" class="plot"></div><p class="caption">Both saved meshes are retained. Solid markers indicate recorded convergence; unsuccessful runs, if present, use open markers and remain diagnostic.</p></article><article class="panel"><h3>Every saved configuration</h3><div class="table-wrap"><table><thead><tr><th>Limiter</th><th>Cells</th><th>Recorded root</th><th>Pressure L1 / p0</th><th>Max conserved L1 / scale</th><th>Accepted / rejected</th></tr></thead><tbody id="all-rows"></tbody></table></div></article>''',
    EXPLORER='''<div class="plot-grid"><article class="panel plot-panel"><div id="pressure-plot" class="plot"></div><p class="caption">Continuous analytical point curve and projected cell-pressure reference are distinct saved quantities.</p></article><article class="panel plot-panel"><div id="mach-plot" class="plot"></div><p class="caption">Only numerical Mach profiles were saved. No analytical Mach curve has been recomputed or invented.</p></article><article class="panel plot-panel"><div id="residual-plot" class="plot"></div><p class="caption">Actual accepted trajectories, including the initial state. Rejected attempts remain in the evidence download and separate work counts.</p></article><article class="panel plot-panel"><div id="cfl-plot" class="plot"></div></article><article class="panel plot-panel"><div id="work-plot" class="plot"></div></article><article class="panel plot-panel"><div id="shock-plot" class="plot"></div><p class="caption">Shock location uses the strongest downstream pressure gradient in the original saved assessment, not a newly substituted Mach-crossing metric.</p></article></div>''',
    DISCUSSION='''<div class="prose-grid"><article class="panel prose"><h3>Four different slope choices</h3><p><strong>Minmod</strong> chooses the smaller compatible signed slope and is relatively diffusive. <strong>Superbee</strong> favors larger compatible magnitudes and is relatively compressive. <strong>Van Leer</strong> combines compatible slopes harmonically; <strong>Van Albada</strong> uses the selected rational slope with an explicit extrema gate.</p><p>The implementation preserves the common slope sign, including negative gradients. The retained Andrea JAX Superbee reference used a signed maximum, which can choose the wrong candidate for negative slopes. The generic implementation selects magnitude and then restores sign.</p></article><article class="panel prose"><h3>What is and is not established</h3><p>These results compare this nozzle geometry, physical operating point, discretization and solver settings only. A limiter with lower error here is not universally superior. Piecewise-smooth slope formulas do not provide a general positivity theorem, smooth Newton behavior or guaranteed convergence for a coupled fluid system.</p><p>The two meshes are below the packaged case's recommended 400-cell finest grid. We preserve reported errors as diagnostics; this study does not promote a coarse-grid convergence result into a complete fine-grid accuracy certification.</p></article><article class="panel prose"><h3>Separate numerical work from runtime</h3><p>Accepted pseudo-time steps, rejected steps and Newton corrections are different work measures. The work plots show actual recorded counts, including rejected attempts. No elapsed-time chart or speedup is derived from compilation-inclusive, potentially contended measurements.</p></article><article class="panel prose"><h3>Separate references from initialization</h3><p>The analytical nozzle solution is used for independent pressure/conserved error assessment. Every limiter and mesh starts from the model-owned cold guess. Pressure projection and continuous pointwise profiles are retained separately.</p><p>Residuals, BCs and global budgets characterize the numerical equations. Agreement with the analytical branch is a separate question, and errors of failed states would remain unfinished-state diagnostics.</p></article></div>''',
    PROTOCOL='''<div class="prose-grid"><article class="panel prose"><h3>Controlled numerical choices</h3><p>Corrected Stadtke flux (both custom corrections), safeguarded backward Euler, no additional filter, the recorded fixed residual policy and the same CFL controller. Only reconstruction limiter and mesh size vary.</p><p>Source fingerprints, package versions, exact scheme settings, histories and the original eight result JSON files are in the data ZIP. This renderer imports no physical model or solver.</p></article><article class="panel prose"><h3>Recorded evidence and limitations</h3><p>Success flags are the recorded production-driver outcomes. No fresh saved-state operator evaluation is performed by this presentation. The independent numerical/physical checks already saved in each record remain inspectable.</p><p>The Andrea reference revision is retained in metadata. Display unit conversions and reading saved arrays do not change the numerical data. Input SHA-256 checks protect the nine campaign JSON files.</p></article></div>''')


def build_report(output=None):
    output = Path(output or STUDY / "outputs/comparison").resolve()
    prepare_inputs(STUDY, output.glob("*.json"))
    records = []
    for path in sorted(output.glob("*.json")):
        if path.name == "metadata.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "limiter" in record and "cells" in record:
            records.append(dict(record, id=path.stem, input_file=path.relative_to(STUDY).as_posix()))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    data = dict(study="limiters", case_id="euler_nozzle/shock_35", records=records,
                metadata=metadata, methods=list(METHODS), cells=sorted({r["cells"] for r in records}))
    render(STUDY, data, SECTIONS, ASSETS / "app.js", files=[Path(__file__), ASSETS / "app.js", STUDY / "README.md", STUDY / "RESULTS.md"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/comparison")
    build_report(parser.parse_args().input)


if __name__ == "__main__":
    main()
