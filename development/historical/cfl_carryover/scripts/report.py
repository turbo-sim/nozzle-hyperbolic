"""Interactive CFL carryover report; saved evidence only, no numerical reruns."""

import argparse
import json
from pathlib import Path
import statistics

from development.flux_methods.shared.offline_report import prepare_inputs,render

STUDY=Path(__file__).resolve().parents[1]
ASSETS=Path(__file__).resolve().parent/"report_assets"
POLICIES=("reset","carry")

SECTIONS=dict(
    TITLE="Carry CFL across refined grids",
    EYEBROW="NOZZLE HYPERBOLIC / GRID-SEQUENCE ACCELERATION",
    LEAD="Preserve a successful controller state instead of repeating the startup ramp. Explore the recorded 50 → 100 → 200 → 400-cell HLL benchmark: two shocks, two reconstructions, and three warmed samples per policy.",
    OVERVIEW_TITLE="A useful default, supported by this small comparison.",
    SCOPE="24 measured complete hierarchies and eight excluded warmups. All 128 saved mesh-stage audits pass. Mean reset/carry time ratios are 1.74–2.53 for these configurations; this is evidence for the selected cases, not a universal speedup guarantee.",
    CONTROLS='''<label>01 / Physical case<select id="case"><option value="euler_nozzle/shock_35">Euler nozzle · shock at 35%</option><option value="euler_nozzle/shock_70">Euler nozzle · shock at 70%</option></select></label>
<label>02 / Reconstruction<select id="reconstruction"><option value="constant">Constant</option><option value="muscl_van_leer" selected>MUSCL · Van Leer</option></select></label>
<label>03 / Mesh stage<select id="cells"></select></label>
<label>04 / Actual recorded run<select id="sample"><option value="median">Median-duration run, per policy</option><option value="1">Measured repetition 1</option><option value="2">Measured repetition 2</option><option value="3">Measured repetition 3</option></select></label>
<label>Profile range<select id="profile-range"><option value="domain">Full nozzle</option><option value="shock">Around the analytical shock</option></select></label>
<label>Convergence view<select id="residual-mode"><option value="ratio">Worst equation / allowed norm</option><option value="equations">Separate equation L2</option><option value="global">Global balance / tolerance</option><option value="boundary">Boundary equations / tolerance</option></select></label>
<fieldset class="choice-set"><legend>CFL policies included in the comparison</legend><div id="methods" class="choices"></div><div class="choice-tools"><button id="methods-all" type="button">Both policies</button><button id="methods-none" type="button">Clear</button><span>Same cold 50-cell state, flux, reconstruction, mesh sequence and stopping rules.</span></div></fieldset>''',
    OVERVIEW='''<article class="notice" role="note"><strong>Adopted after this study: grid sequencing carries accepted CFL by default.</strong><p>Use <code>carry_cfl=False</code> for per-grid reset. Cold starts and retries reset either way. These saved timings predate the production change; a separate saved API check later reproduced both final states and iteration counts exactly.</p></article><div id="case-cards" class="prose-grid"></div>
<div class="prose-grid"><article class="panel prose"><h3>What the saved evidence shows</h3><div id="findings"></div></article><article class="panel prose"><h3>Why the difference matters</h3><p>A converged coarse-grid state already provides a good fine-grid initial guess. Restarting CFL at 0.2 repeats a conservative startup ramp even though that state has successfully supported much larger steps.</p><p>Carrying CFL avoids that repeated ramp. The dimensional timestep is recomputed from the new mesh and wave speeds; the previous absolute timestep is not copied. A failed warm attempt remains visible before a cold retry.</p><p>Carryover can increase rejected steps and backtracking. The relevant result is complete time to a checked root, including rejected work, rather than accepted-step count alone.</p></article></div>
<div class="plot-grid"><article class="panel plot-panel"><div id="timing-plot" class="plot"></div><p class="caption">All three complete-hierarchy samples and their means for all four case/reconstruction configurations. Every coarse solve, transfer, monitor and rejection is included.</p></article><article class="panel plot-panel"><div id="savings-plot" class="plot"></div><p class="caption">Ratio of mean complete-hierarchy reset time to carryover time. Above one favors carryover. Saved original residual/budget/branch checks and same-root agreement gate every ratio.</p></article></div>
<article class="panel"><h3>All four controlled comparisons</h3><div class="table-wrap"><table><thead><tr><th>Shock / reconstruction</th><th>Reset mean [min–max] s</th><th>Carry mean [min–max] s</th><th>Reset / carry</th><th>Same-root checks</th><th></th></tr></thead><tbody id="all-rows"></tbody></table></div></article>''',
    EXPLORER='''<div class="plot-grid">
<article class="panel plot-panel"><div id="selected-timing" class="plot"></div><p class="caption">All three complete-hierarchy timings per selected policy. Choosing a coarse mesh changes stage diagnostics, not the timing boundary.</p></article>
<article class="panel plot-panel"><div id="stage-time" class="plot"></div><p class="caption">Mean recorded stage time across all three repetitions, including transfer, monitoring and retained retry work.</p></article>
<article class="panel plot-panel"><div id="pressure-plot" class="plot"></div><p class="caption">Saved physical-cell pressure at the selected mesh. Only the 400-cell analytical projection was saved and is shown there. Coarse centers follow the recorded uniform domain.</p></article>
<article class="panel plot-panel"><div id="velocity-plot" class="plot"></div><p class="caption">Saved axial velocity; no new EOS or analytical evaluation.</p></article>
<article class="panel plot-panel"><div id="residual-plot" class="plot"></div><p class="caption">Actual accepted trajectory, including the initial state. Each policy has its own recorded median-duration run; no residual curve is averaged across repeats.</p></article>
<article class="panel plot-panel"><div id="cfl-plot" class="plot"></div><p class="caption">Accepted CFL history with crosses for rejected timestep attempts. Initial CFL is the value actually applied on this mesh.</p></article>
<article class="panel plot-panel"><div id="work-plot" class="plot"></div><p class="caption">Saved stage work includes rejected attempts and any cold retry. Counts distinguish accepted updates, rejected timesteps, Newton corrections and rejected Newton trials.</p></article>
<article class="panel plot-panel"><div id="entry-cfl-plot" class="plot"></div><p class="caption">Actual initial and last-accepted CFL across all four meshes of the selected run. Reset repeats CFL 0.2; carry transfers the preceding accepted value.</p></article>
</div><article class="panel"><h3>Selected stage and hierarchy evidence</h3><div class="table-wrap"><table><thead><tr><th>Policy / run</th><th>Saved root / audit</th><th>Start / last CFL</th><th>Accepted / rejected</th><th>Newton corrections</th><th>Equation L2 max</th><th>BC / global max</th><th>Whole mean [min–max] s</th></tr></thead><tbody id="selected-rows"></tbody></table></div></article>''',
    DISCUSSION='''<div class="prose-grid"><article class="panel prose"><h3>01 / Reset every mesh</h3><p>Every grid starts at the configured CFL 0.2, even after interpolating a converged preceding state. Accepted updates multiply CFL by 1.2; rejected attempts multiply it by 0.5, with cap 10<sup>6</sup>.</p><div class="equation">CFL<sub>start,new grid</sub> = CFL<sub>configured</sub><br>Δτ = CFL · min<sub>i</sub>(Δx<sub>i</sub> / s<sub>i</sub>)</div><p>The repeated ramp improves caution during startup but can repeat unnecessary work after a successful transfer.</p></article>
<article class="panel prose"><h3>02 / Carry accepted CFL</h3><p>The first mesh starts cold at CFL 0.2. A successful transfer inherits the preceding mesh's last accepted CFL, capped at the common maximum. The new mesh and wave speeds determine its dimensional timestep.</p><div class="equation">CFL<sub>start,new grid</sub> = min(CFL<sub>last accepted</sub>, CFL<sub>max</sub>)</div><p>It does not carry the old absolute timestep or the next proposed CFL. Failed preparation, cold starts and cold retries reset. Missing controller history falls back to the configured value; an already-steady stage forwards its initial CFL.</p></article>
<article class="panel prose"><h3>03 / Same equations and stopping tests</h3><p>Both policies retain HLL, backward Euler, coupled algebraic traces, colored AD, the banded Newton solve and backtracking. Physical scales and per-equation absolute-or-relative convergence targets remain fixed; each mesh and retry gets its own relative baseline.</p><p>Original boundary/global budgets and the physical branch are checked independently. A successful inner update, a high CFL or fewer accepted steps alone is not proof of a converged solution.</p></article>
<article class="panel prose"><h3>04 / Evidence, scope and recommendation</h3><p>The four mean time ratios support carryover as a useful default for these cases, with reset retained as an opt-out. Three samples and one machine/session do not establish a general guarantee across stiff sources, different models or difficult transfers.</p><p>The production driver adopted the policy after measurement. A separate saved 35% MUSCL API check matched both experimental final states exactly and reproduced 673 carryover versus 1156 reset Newton corrections. That was a correctness check, not a new timing campaign.</p><p>Grid sequencing is coarse-to-fine initialization, not a defect-correction multigrid cycle. Each grid solves its own original equations.</p></article></div>''',
    PROTOCOL='''<div class="prose-grid"><article class="panel prose"><h3>Measured population</h3><p>Two perfect-gas Euler nozzle shocks at 35% and 70% of divergence, constant or MUSCL Van Leer reconstruction, HLL, and the fixed sequence 50 → 100 → 200 → 400 cells.</p><p>One excluded complete warmup per policy/configuration, followed by three measured complete hierarchies per option. Order alternates; runs are sequential. Each hierarchy resets its cold initial field, controller and history.</p></article><article class="panel prose"><h3>Timing boundary</h3><p>Models, operators and samplers are prebound and reused. Complete hierarchy timing includes coarse solves, grid transfers, monitoring, rejected steps and cold-retry work. Binding, final independent audits, output and plotting are excluded; device work is synchronized.</p><p>All timings, including excluded warmup costs, remain in the evidence. Stage durations are diagnostic components; no per-step wall-clock axis is invented.</p></article><article class="panel prose"><h3>Original checks and final-state agreement</h3><p>All 24 measured hierarchies and eight warmups converged; all 128 saved mesh-stage audits passed. The four comparison populations agree within 3.6 × 10<sup>−7</sup> in fixed variable scales.</p><p>Ratios require original equation/boundary/global/branch audits, agreement within 10<sup>−6</sup> and a shock within two cells of its reference, allowing only coordinate roundoff at that boundary. This presentation reads saved checks; it performs no new operator evaluation.</p></article><article class="panel prose"><h3>Measurement-time source identity</h3><p>The data were recorded before the production grid-sequence default changed. The source snapshot and per-file fingerprints identify the measurement epoch; later working-tree changes do not relabel these historical timings.</p><p>The evidence includes every original record, settings, full histories, actual final fields and the separate production API check. Input SHA-256 hashes are checked before and after rendering. Uniform coarse coordinates are reconstructed from the saved domain; no missing analytical projection or time history is invented.</p></article></div>
<article class="panel prose"><h3>Rebuild this HTML without simulations</h3><pre>poetry run python -B -m development.historical.cfl_carryover.scripts.report
poetry run python -B -m development.historical.cfl_carryover.scripts.audit_report</pre></article>''',
)


def build_report(output=None):
    out=Path(output or STUDY/"outputs/comparison").resolve()
    names=json.loads((out/"index.json").read_text())
    paths=[out/name for name in names]
    records=[dict(json.loads(p.read_text()),id=p.stem,input_file=p.relative_to(STUDY).as_posix()) for p in paths]
    metadata=json.loads((out/"metadata.json").read_text())
    saved=json.loads((out/"report_summary.json").read_text())
    production=json.loads((out/"production_default_check.json").read_text())
    prepare_inputs(STUDY,paths+[out/name for name in ("index.json","metadata.json","report_summary.json",
        "production_default_check.json","source_at_measurement.zip")]+[STUDY/"RESULTS.md"])
    groups=[]
    for summary in saved:
        group=dict(summary,timing={})
        for policy in POLICIES:
            rows=[r for r in records if r["case_id"]==summary["case_id"] and r["reconstruction"]==summary["reconstruction"] and r["phase"]=="measured" and r["policy"]==policy]
            assert len(rows)==3
            values=[r["elapsed_s"] for r in rows]
            group["timing"][policy]=dict(samples_s=values,mean_s=statistics.mean(values),min_s=min(values),max_s=max(values),
                sample_ids=[r["id"] for r in rows],representative_id=sorted(rows,key=lambda r:r["elapsed_s"])[1]["id"],
                work_mean={key:statistics.mean(r["total_work"][key] for r in rows) for key in rows[0]["total_work"]})
        if group["timing_eligible"]:
            assert group["reset_over_carry"]==group["timing"]["reset"]["mean_s"]/group["timing"]["carry"]["mean_s"]
        groups.append(group)
    data=dict(study="cfl_carryover",metadata=metadata,records=records,comparisons=groups,production_check=production,
        methods=list(POLICIES),cells=list(metadata["grids"]),cases=[dict(id=k,definition=v) for k,v in metadata["cases"].items()])
    sections=dict(SECTIONS)
    sections["OVERVIEW"]='<style>'+(ASSETS/"style.css").read_text()+'</style>'+sections["OVERVIEW"]
    render(STUDY,data,sections,ASSETS/"app.js",files=[Path(__file__),ASSETS/"app.js",ASSETS/"style.css",
        ASSETS/"audit_app.mjs",Path(__file__).with_name("audit_report.py"),STUDY/"README.md",STUDY/"RESULTS.md"])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--input",type=Path,default=STUDY/"outputs/comparison")
    build_report(parser.parse_args().input)


if __name__=="__main__": main()
