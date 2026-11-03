"""Render the shared offline study explorer from saved benchmark evidence only."""

import argparse
import json
from pathlib import Path
import statistics

from development.flux_methods.shared.offline_report import prepare_inputs, render

STUDY = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "report_assets"
METHODS = ("global", "local")
MODES = ("sequence", "single_400")

SECTIONS = dict(
    TITLE="Local pseudo-time stepping",
    EYEBROW="NOZZLE HYPERBOLIC / STEADY-SOLVER PERFORMANCE",
    LEAD="Overall outcome: limited and inconsistent acceleration. These results are not compelling enough to recommend local stepping as a production default; the experiment is retained for reference and reproducibility.",
    OVERVIEW_TITLE="Limited benefit. Keep local stepping experimental.",
    SCOPE="48 measured complete solves, 16 excluded warmups and 160 retained mesh stages. Both grid-sequenced and cold 400-cell solutions use HLL, backward Euler, the same CFL controller and unchanged convergence criteria. This presentation performs no numerical reruns.",
    CONTROLS='''<label>01 / Physical case<select id="case"><option value="euler_nozzle/shock_35">Euler nozzle · shock at 35%</option><option value="euler_nozzle/shock_70">Euler nozzle · shock at 70%</option></select></label>
<label>02 / Initialization<select id="strategy"><option value="sequence">Grid sequence · 50 → 100 → 200 → 400</option><option value="single_400">Cold single grid · 400 cells</option></select></label>
<label>03 / Reconstruction<select id="reconstruction"><option value="constant">Constant</option><option value="muscl_van_leer" selected>MUSCL · Van Leer</option></select></label>
<label>04 / Mesh stage<select id="cells"></select></label>
<label>05 / Actual recorded run<select id="sample"><option value="median">Median-duration run, per method</option><option value="1">Measured repetition 1</option><option value="2">Measured repetition 2</option><option value="3">Measured repetition 3</option></select></label>
<label>Profile range<select id="profile-range"><option value="domain">Full nozzle</option><option value="shock">Around the analytical shock</option></select></label>
<label>Convergence view<select id="residual-mode"><option value="ratio">Worst equation / allowed norm</option><option value="equations">Separate equation L2</option><option value="global">Global balance / tolerance</option><option value="boundary">Boundary equations / tolerance</option></select></label>
<fieldset class="choice-set"><legend>Pseudo-time methods in the comparison</legend><div id="methods" class="choices"></div><div class="choice-tools"><button id="methods-all" type="button">Both methods</button><button id="methods-none" type="button">Clear</button><span>Global and local use the same physical equations and coupled implicit solver.</span></div></fieldset>''',
    OVERVIEW='''<article class="notice warning" role="note"><strong>Assessment: not recommended for production adoption.</strong><p>The overall results were not sufficiently compelling. Constant reconstruction shows little consistent benefit; MUSCL gains are modest and the three-sample timing comparison has noticeable scatter. Retain this experiment as a documented result and continue using global stepping with CFL carryover for these cases. The measured gains and every timing sample remain visible below.</p></article><div id="case-cards" class="prose-grid"></div>
<div class="prose-grid"><article class="panel prose"><h3>What the recorded timings show</h3><div id="findings"></div></article><article class="panel prose"><h3>Read the small differences carefully</h3><p>The constant-reconstruction ratios are close to one. In the 35% cold case the three local samples span 0.756–1.188 s; the mean reverses a small apparent benefit. Every sample is shown and retained.</p><p>For the 35% constant grid sequence, local stepping uses <strong>581 Newton corrections versus 571 globally</strong>, despite a slightly lower mean duration. A few percent in this small session should not be promoted into a robust speedup.</p><p>Grid sequencing remains the larger improvement. Original-equation convergence, agreement with the same discrete root and analytical shock accuracy are separate checks.</p></article></div>
<div class="plot-grid"><article class="panel plot-panel"><div id="timing-plot" class="plot"></div><p class="caption">All three measured samples and their means, for every case/reconstruction at the selected initialization strategy. Warmups are excluded; no samples are discarded.</p></article><article class="panel plot-panel"><div id="strategy-plot" class="plot"></div><p class="caption">Mean cold 400-cell time divided by complete grid-sequence time. Above one favors sequencing. Every coarse solve and grid transfer is included, and cross-strategy state agreement gates the ratios.</p></article></div>
<article class="panel"><h3>All eight controlled comparisons</h3><div class="table-wrap"><table><thead><tr><th>Shock / reconstruction</th><th>Initialization</th><th>Global mean [s]</th><th>Local mean [s]</th><th>Global / local</th><th>Same-root checks</th><th></th></tr></thead><tbody id="all-rows"></tbody></table></div></article>''',
    EXPLORER='''<div class="plot-grid">
<article class="panel plot-panel"><div id="selected-timing" class="plot"></div><p class="caption">Three complete-solve timings per selected method. This chart remains a whole-hierarchy/cold-solve measurement when inspecting a coarse stage.</p></article>
<article class="panel plot-panel"><div id="stage-time" class="plot"></div><p class="caption">Mean recorded stage durations across the three repetitions. Includes monitoring, transfer and failed warm/cold-retry work; setup and serialization are excluded from complete solve timing.</p></article>
<article class="panel plot-panel"><div id="pressure-plot" class="plot"></div><p class="caption">Pressure from the saved physical-cell state at the selected mesh. The analytical conserved-cell projection was saved only on 400 cells and is shown only there. Coarse cell centers follow the recorded uniform domain.</p></article>
<article class="panel plot-panel"><div id="velocity-plot" class="plot"></div><p class="caption">Saved axial velocity, without any new EOS or analytical evaluation.</p></article>
<article class="panel plot-panel"><div id="residual-plot" class="plot"></div><p class="caption">An actual recorded trajectory, with the initial state at zero. Median selection is independent per method; no averaged or best-so-far residual curve is constructed.</p></article>
<article class="panel plot-panel"><div id="cfl-plot" class="plot"></div><p class="caption">Lines show accepted CFL. Crosses identify rejected timestep attempts at the corresponding accepted-update counter.</p></article>
<article class="panel plot-panel"><div id="work-plot" class="plot"></div><p class="caption">Selected stage counts include failed warm attempts and rejected work. Accepted updates, rejected timesteps, Newton corrections and rejected Newton trials are distinct measures.</p></article>
<article class="panel plot-panel"><div id="local-step-plot" class="plot"></div><p class="caption">Saved final cell-step multipliers at this stage. Global is one everywhere; local is the cell wave time divided by its minimum. These are final-state diagnostics, not a physical-time trajectory.</p></article>
</div><article class="panel"><h3>Selected stage and complete-solve evidence</h3><div class="table-wrap"><table><thead><tr><th>Method / run</th><th>Saved root / audit</th><th>Start / last CFL</th><th>Stage accepted / rejected</th><th>Stage Newton</th><th>Equation L2 max</th><th>BC / global max</th><th>Whole mean [min–max] s</th></tr></thead><tbody id="selected-rows"></tbody></table></div></article>''',
    DISCUSSION='''<div class="prose-grid">
<article class="panel prose"><h3>01 / Global pseudo-time continuation</h3><p>The production driver computes a common step from the most restrictive cell wave time, including model-owned endpoint bounds. It adapts one CFL after accepted or rejected nonlinear solves.</p><div class="equation">b<sub>i</sub> = Δx<sub>i</sub> / s<sub>i</sub><br>Δτ = CFL · min<sub>i</sub>(b<sub>i</sub>)</div><p>The grid-sequence default carries the last accepted CFL across successful meshes. The new mesh still determines the dimensional timestep; cold starts and retries reset CFL.</p></article>
<article class="panel prose"><h3>02 / Local weights, coupled Newton</h3><p>The development adapter freezes one wave time per physical cell at the old outer iterate. It passes a cellwise timestep array into the unchanged implicit kernel. The full coupled Newton solve and exact AD/banded linear algebra are retained.</p><div class="equation">Δτ<sub>i</sub> = CFL · b<sub>i</sub><br>V<sub>i</sub>[Q(W<sup>new</sup><sub>i</sub>) − Q(W<sup>old</sup><sub>i</sub>)] + Δτ<sub>i</sub>R<sub>i</sub>(W<sup>new</sup>) = 0</div><p>Local weights are fixed during Newton rather than differentiated with its trial state. Both methods retain the same cellwise residual scaling rule and fixed physical scales.</p></article>
<article class="panel prose"><h3>03 / Same root, different route</h3><p>Positive pseudo-time weights preserve the intended steady equation R(W) = 0. Physical boundary equations keep zero storage and their original definition. Final convergence is judged using the original spatial equations, independent boundary and global budgets, admissibility and the physical branch.</p><p>Same-root timing eligibility requires all saved audits to pass and measured final-state spread at most 10<sup>−6</sup> in fixed variable scales. Shock location is checked independently against the two-cell target, allowing only coordinate roundoff at its boundary.</p></article>
<article class="panel prose"><h3>04 / Where further acceleration may remain</h3><p>At large pseudo-time steps, the storage contribution weakens and the implicit solve approaches steady Newton. Local timesteps then have less influence. Strongly stretched meshes could give a different result from these uniform grids.</p><p>The measured MUSCL benefit is modest. Constant reconstruction gives little consistent timing improvement, while some sample ranges overlap substantially. Three repetitions establish a small session-specific comparison, not universal superiority.</p><p>Local iterates have no common physical time. This adapter is an experimental steady method, not a transient integration implementation or a production-default change.</p></article>
</div>''',
    PROTOCOL='''<div class="prose-grid"><article class="panel prose"><h3>Controlled numerical comparison</h3><p>Two packaged perfect-gas Euler nozzle shocks, at 35% and 70% of divergence. HLL, constant or MUSCL Van Leer, backward Euler with coupled traces, colored AD, a banded solve and Newton backtracking.</p><p>The same original convergence thresholds, CFL ramp, EOS, geometry and boundary laws apply to both methods. The local method's additional wave-time evaluation is included in measured work. No source file was edited by the experiment.</p></article><article class="panel prose"><h3>Timing boundary and resets</h3><p>One excluded complete warmup per option/configuration/strategy, then exactly three measured solves with alternating method order. Kernels, models and samplers are prebound and reused; numerical states, controllers and histories reset on every solve.</p><p>Timers include complete solves, interpolation, monitoring, rejected work and cold retries; device work is synchronized. Binding, final audits, output and plotting are excluded. Single-grid solves start cold, without a sequenced state.</p></article><article class="panel prose"><h3>Saved validation, unchanged evidence</h3><p>All 48 measured solves and 16 warmups converged; all 160 saved stage audits passed. Maximum cross-method/strategy state spread is below 4.8 × 10<sup>−7</sup> in fixed variable scales.</p><p>Uniform-step equivalence, boundary zero storage, colored/dense AD and directional finite differences were checked before the campaign. This presentation reads those saved checks; it does not rerun them.</p></article><article class="panel prose"><h3>Source identity and reproducibility</h3><p>Measurement-time source hashes, a source snapshot, exact settings, all warmups/repetitions, actual final states and full histories are preserved in the evidence ZIP. Later working-tree changes do not relabel historical timings.</p><p>The presentation manifest hashes every numerical input before and after rendering. Coarse pressure/velocity curves use saved primitive states; their uniform centers are reconstructed from the saved 400-cell domain. No missing coarse analytical reference, Mach profile or per-step wall time is invented.</p></article></div>
<article class="panel prose"><h3>Rebuild this HTML without simulations</h3><pre>poetry run python -B -m development.historical.local_timestep.scripts.report
poetry run python -B -m development.historical.local_timestep.scripts.audit_report</pre></article>''',
)


def comparisons(records):
    groups = []
    for mode in MODES:
        for case in ("euler_nozzle/shock_35", "euler_nozzle/shock_70"):
            for recon in ("constant", "muscl_van_leer"):
                selected = [r for r in records if (r["mode"], r["case_id"], r["reconstruction"]) == (mode, case, recon)]
                measured = [r for r in selected if r["phase"] == "measured"]
                timing = {}
                for method in METHODS:
                    rows = [r for r in measured if r["method"] == method]
                    assert len(rows) == 3
                    values = [r["elapsed_s"] for r in rows]
                    timing[method] = dict(samples_s=values, mean_s=statistics.mean(values),
                        min_s=min(values), max_s=max(values), sample_ids=[r["id"] for r in rows],
                        representative_id=sorted(rows, key=lambda r: r["elapsed_s"])[1]["id"],
                        work_mean={k:statistics.mean(r["total_work"][k] for r in rows) for k in rows[0]["total_work"]})
                groups.append(dict(mode=mode, case_id=case, reconstruction=recon, timing=timing))
    return groups


def build_report(output=None):
    output = Path(output or STUDY / "outputs/comparison").resolve()
    names = json.loads((output / "index.json").read_text())
    paths = [output / name for name in names]
    originals = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    records = [dict(r, id=p.stem, input_file=p.relative_to(STUDY).as_posix()) for r,p in zip(originals,paths)]
    metadata = json.loads((output / "metadata.json").read_text())
    summary = json.loads((output / "report_summary.json").read_text())
    cross = json.loads((output / "cross_strategy_check.json").read_text())
    frozen = paths + [output / name for name in ("index.json", "metadata.json", "report_summary.json",
        "cross_strategy_check.json", "source_verification.json", "source_at_measurement.zip")]
    frozen += [STUDY / "outputs/implementation_checks.json", STUDY / "RESULTS.md"]
    prepare_inputs(STUDY, frozen)
    groups = comparisons(records)
    for group, saved in zip(groups, summary):
        assert (group["mode"],group["case_id"],group["reconstruction"]) == (saved["mode"],saved["case_id"],saved["reconstruction"])
        group.update(saved)
        if saved["eligible"]:
            assert group["timing"]["global"]["mean_s"] / group["timing"]["local"]["mean_s"] == saved["global_over_local"]
    data = dict(study="local_timestep", records=records, metadata=metadata,
        comparisons=groups, cross_strategy=cross, methods=list(METHODS), modes=list(MODES),
        cells=list(metadata["grids"]), cases=[dict(id=key,definition=value) for key,value in metadata["cases"].items()])
    sections=dict(SECTIONS)
    sections["OVERVIEW"]='<style>'+ (ASSETS/"style.css").read_text(encoding="utf-8")+'</style>'+sections["OVERVIEW"]
    render(STUDY, data, sections, ASSETS / "app.js", files=[Path(__file__), ASSETS / "app.js", ASSETS / "style.css",
        Path(__file__).with_name("audit_report.py"), ASSETS / "audit_app.mjs", STUDY / "README.md", STUDY / "RESULTS.md"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "outputs/comparison")
    build_report(parser.parse_args().input)


if __name__ == "__main__":
    main()
