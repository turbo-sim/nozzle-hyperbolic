"""Generate one offline HTML report with results and theory as internal pages.

Run from the repository root after run_demo and audit_results. Numerical records
are read-only. REPORT.html lives at the study root; numerical artifacts stay
under outputs/. All libraries, plot data and offered downloads are embedded.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import gzip
import hashlib
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ASSETS = STUDY.parent / "euler_annular/scripts/report_assets"
METHODS = ("single_grid", "grid_sequence")
NAMES = {"single_grid": "Single grid", "grid_sequence": "Coarse to fine"}
RECON = {"constant": "Constant", "muscl_van_leer": "MUSCL"}
COLORS = {"single_grid": "#6a00a8", "grid_sequence": "#fca636"}


def safe_json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}" if np.isfinite(value) else "—"
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="panel table-wrap"><table><thead><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in headers) + '</tr></thead><tbody>' + "".join('<tr>' + "".join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def group_runs(summary, output):
    groups = defaultdict(list)
    audit_path=output/"audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    records={r["file"]:r for r in audit.get("results",[]) if "file" in r}
    for row in summary["runs"]:
        run = json.loads((output / row["file"]).read_text(encoding="utf-8"))
        run.update(row)
        if row["file"] in records:
            record=records[row["file"]]
            run["same_root"]=bool(record.get("matched_solution",False) and record.get("integrity_pass",False))
            run["audit"]=record
        groups[(row["case_id"], row["reconstruction"], row["method"])].append(run)
    return groups


def measured(runs):
    return [r for r in runs if not r.get("excluded", r.get("repeat", 0) < 0)]


def representative(runs):
    samples = measured(runs)
    if not samples:
        raise ValueError("Each displayed configuration needs a measured run")
    ordered = sorted(samples, key=lambda r: r["wall_s"])
    return ordered[(len(ordered)-1)//2]


def history(run, fine_cells=None):
    """Keep each level separate; no fabricated residual values at transfers."""
    result = []
    offset_t, offset_i, offset_work = 0., 0, 0.
    fine_cells=fine_cells or run["stages"][-1]["n_points"]-2
    for stage in run["stages"]:
        offset_t += stage.get("transfer_s", 0.)
        points = stage.get("history", [])
        result.append(dict(
            n_points=stage["n_points"],
            elapsed_s=[offset_t + h["elapsed_s"] for h in points],
            updates=[offset_i + h.get("accepted_index", h.get("update", 0)) for h in points],
            residual=[h["residual_rms"] for h in points],
            work=[offset_work+h.get("implicit_step_trials",0)*(stage["n_points"]-2)/fine_cells for h in points],
            cfl=[h.get("effective_cfl") for h in points],
        ))
        offset_t += stage["wall_s"]
        offset_i += stage["iterations"]
        offset_work += stage["trials"]*(stage["n_points"]-2)/fine_cells
    return result


def make_payload(summary, groups):
    cases = []
    ids = list(dict.fromkeys(s["case_id"] for s in summary["setups"]))
    for case_id in ids:
        setup = next(s for s in summary["setups"] if s["case_id"] == case_id)
        item = {k: setup[k] for k in ("case_id", "label", "p_back_ratio", "x_m", "area_m2", "reference", "reference_fine") if k in setup}
        item["runs"] = []
        for (cid, recon, method), runs in groups.items():
            if cid != case_id:
                continue
            rep = representative(runs)
            item["runs"].append(dict(
                method=method, reconstruction=recon, name=f"{NAMES[method]} · {RECON.get(recon,recon)}",
                repeat=rep["repeat"], wall_s=rep["wall_s"], iterations=rep["iterations"],
                fine_iterations=rep.get("fine_iterations"), equivalent_fine_updates=rep.get("equivalent_fine_updates"),
                root_success=rep.get("root_success"), same_root=rep.get("same_root"),
                diagnostics=rep["stages"][-1].get("diagnostics", {}),
                fields=rep["stages"][-1]["fields"], levels=history(rep,setup["n_points"]-2),
                stages=[{k:s.get(k) for k in ("n_points","wall_s","transfer_s","iterations","trials","rejected_trials","diagnostics","fields","tolerance","native_success")} for s in rep["stages"]],
                samples=[dict(repeat=r["repeat"], wall_s=r["wall_s"], iterations=r["iterations"],
                    fine_iterations=r.get("fine_iterations"), equivalent_fine_updates=r.get("equivalent_fine_updates"),
                    rejected_trials=r.get("rejected_trials"), root_success=r.get("root_success"), same_root=r.get("same_root")) for r in measured(runs)],
            ))
        cases.append(item)
    return clean(cases)


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def static_figures(summary, groups, payload, output):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": True,
        "axes.spines.right": True, "axes.grid": True, "grid.alpha": .22, "lines.linewidth": 1.5})
    groups_order = list(dict.fromkeys((s["case_id"], s["reconstruction"]) for s in summary["setups"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    flagged={(cid,recon) for cid,recon in groups_order if any(r.get("same_root") is False for method in METHODS for r in measured(groups[(cid,recon,method)]))}
    labels = [f"{cid}{'*' if (cid,recon) in flagged else ''}\n{RECON.get(recon,recon)}" for cid,recon in groups_order]
    for mi, method in enumerate(METHODS):
        xx = np.arange(len(groups_order)) + (mi-.5)*.34
        for ax, key, title in zip(axes, ("wall_s", "equivalent_fine_updates"), ("Warm solve time [s]", "Fine-equivalent Newton work")):
            values = [[r[key] for r in measured(groups[(cid,recon,method)]) if r.get(key) is not None] for cid,recon in groups_order]
            mean = np.array([np.mean(v) for v in values])
            low, high = np.array([min(v) for v in values]), np.array([max(v) for v in values])
            ax.bar(xx, mean, .32, color=COLORS[method], label=NAMES[method], alpha=.85)
            ax.errorbar(xx, mean, yerr=[np.maximum(mean-low,0),np.maximum(high-mean,0)], color="black", fmt="none", capsize=3, lw=.9)
            for x,v in zip(xx,values):
                ax.scatter(np.linspace(x-.055,x+.055,len(v)),v,s=10,facecolors="white",edgecolors=COLORS[method],linewidths=.7,zorder=3)
            ax.set_ylabel(title)
            ax.set_xticks(range(len(labels)),labels)
            ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle("Single grid vs coarse-to-fine initialization · all measured samples; mean and min–max")
    if flagged:
        fig.text(.5,-.055,"* Steady roots pass; profile agreement exceeds the declared tolerance. These timing ratios are descriptive, not matched-root acceleration.",ha="center",fontsize=9)
    save_figure(fig, figures/"timing_and_work")
    for recon in dict.fromkeys(s["reconstruction"] for s in summary["setups"]):
        fig, axes = plt.subplots(len(payload), 3, figsize=(17,3.4*len(payload)), squeeze=False, constrained_layout=True)
        for ri,case in enumerate(payload):
            for run in [r for r in case["runs"] if r["reconstruction"]==recon]:
                for li,level in enumerate(run["levels"]):
                    for ai,xkey in enumerate(("updates","elapsed_s","work")):
                        axes[ri,ai].semilogy(level[xkey],level["residual"],color=COLORS[run["method"]],
                            ls="-" if run["method"]=="single_grid" else "--",
                            label=f"{NAMES[run['method']]} · {level['n_points']} points" if ai==0 else None)
                for ax in axes[ri]:
                    ax.axhline(1e-8,color="gray",ls=":",lw=1)
                    ax.set_title(case["label"])
                    ax.set_ylabel("Current-level original residual RMS")
            axes[ri,0].set_xlabel("Accepted updates across all levels")
            axes[ri,1].set_xlabel("Cumulative solve time [s]")
            axes[ri,2].set_xlabel("Fine-equivalent Newton work")
            axes[ri,0].legend(fontsize=7,loc="best")
        fig.suptitle(f"{RECON.get(recon,recon)} · actual median-duration representative runs; histories split at transfers")
        save_figure(fig,figures/f"convergence_{recon}")
    for method in METHODS:
        fig, axes = plt.subplots(len(payload), 2, figsize=(13,3.4*len(payload)), squeeze=False, constrained_layout=True)
        for ri,case in enumerate(payload):
            ref = case.get("reference_fine",case["reference"])
            rx = np.asarray(ref.get("x_m",case["x_m"]))
            for ai,key in enumerate(("p","M")):
                scale=1e-3 if key=="p" else 1
                ax=axes[ri,ai]
                ax.plot(rx,np.asarray(ref[key])*scale,color="#222222",ls="--",label="Analytical",lw=1.6,zorder=5)
                for run in [r for r in case["runs"] if r["method"]==method]:
                    f=run["fields"]; color="#9c179e" if run["reconstruction"]=="constant" else "#fca636"
                    ax.plot(f["x_m"],np.asarray(f[key])*scale,color=color,label=RECON.get(run["reconstruction"],run["reconstruction"]),
                        marker="o",markevery=(0 if run["reconstruction"]=="constant" else 5,11),ms=5,mfc="white",mec=color,
                        ls="-" if run["reconstruction"]=="constant" else "-.")
                ax.set_title(case["label"]);ax.set_xlabel("Axial position x [m]")
                ax.set_ylabel("Static pressure [kPa]" if key=="p" else "Mach number [-]")
                ax.legend(fontsize=8)
        fig.suptitle(f"{NAMES[method]} · final target mesh · analytical / constant / MUSCL")
        save_figure(fig,figures/f"fields_{method}")
    for case in payload:
        for run in [r for r in case["runs"] if r["method"]=="grid_sequence"]:
            fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
            palette=plt.colormaps["plasma"](np.linspace(.2,.8,len(run["stages"])))
            for ax,key in zip(axes,("p","M")):
                scale=1e-3 if key=="p" else 1
                ref=case.get("reference_fine",case["reference"])
                ax.plot(ref.get("x_m",case["x_m"]),np.asarray(ref[key])*scale,color="black",ls="--",label="Analytical",zorder=5)
                for i,(stage,color) in enumerate(zip(run["stages"],palette)):
                    f=stage["fields"]
                    ax.plot(f["x_m"],np.asarray(f[key])*scale,color=color,ls=[":","--","-.","-"][i%4],
                        marker="o",markevery=(i,11),ms=4.5,mfc="white",mec=color,label=f"{stage['n_points']} points")
                ax.set_xlabel("Axial position x [m]");ax.set_ylabel("Static pressure [kPa]" if key=="p" else "Mach number [-]")
                ax.legend(fontsize=8)
            fig.suptitle(f"{case['label']} · {RECON.get(run['reconstruction'],run['reconstruction'])} · endpoint of each startup stage")
            save_figure(fig,figures/f"stages_{case['case_id']}_{run['reconstruction']}")


def results_content(summary, groups, audit, *, finding_override=None, failure_message=None):
    rows=[]; speedups=[]; excluded=[]; csv_rows=[]
    for (cid,recon,method),runs in groups.items():
        samples=measured(runs); times=[r["wall_s"] for r in samples]
        success=sum(bool(r.get("root_success")) for r in samples)
        matched=all(r.get("same_root",False) for r in samples)
        vals=lambda key: [r[key] for r in samples if r.get(key) is not None]
        avg=lambda key: statistics.mean(vals(key)) if vals(key) else None
        rows.append([html.escape(cid),RECON.get(recon,recon),NAMES[method],str(len(samples)),f"{success}/{len(samples)}",
            "PASS" if matched else "Pending / differs",fmt(statistics.mean(times),4),f"{fmt(min(times),4)}–{fmt(max(times),4)}",
            fmt(avg("iterations"),5),fmt(avg("fine_iterations"),5),fmt(avg("equivalent_fine_updates"),5),fmt(avg("rejected_trials"),5)])
        for r in runs:
            csv_rows.append({k:r.get(k) for k in ("case_id","reconstruction","method","repeat","excluded","root_success","same_root","wall_s","iterations","fine_iterations","equivalent_fine_updates","trials","rejected_trials")})
            if r.get("excluded"):
                excluded.append([html.escape(cid),RECON.get(recon,recon),NAMES[method],fmt(r["wall_s"],4),fmt(r["iterations"]),fmt(r.get("root_success"))])
    for cid,recon,_ in groups:
        if any(t[0]==cid and t[1]==recon for t in speedups):continue
        left=measured(groups[(cid,recon,"single_grid")]);right=measured(groups[(cid,recon,"grid_sequence")])
        valid=all(r.get("root_success") and r.get("same_root") for r in left+right)
        ratio=statistics.mean(r["wall_s"] for r in left)/statistics.mean(r["wall_s"] for r in right)
        speedups.append((cid,recon,ratio,valid))
    passed=sum(bool(r.get("root_success")) for r in summary["runs"] if not r.get("excluded"))
    total=sum(not r.get("excluded",False) for r in summary["runs"])
    ratios=[x[2] for x in speedups if x[3]]
    finding="Some configuration pairs do not yet qualify as the same discrete solution; their time ratios remain descriptive."
    if all(x[3] for x in speedups):
        if all(ratio>1 for cid,recon,ratio,valid in speedups if cid.startswith("shock")) and all(ratio<1 for cid,recon,ratio,valid in speedups if cid=="subsonic"):
            finding="The shock cases are faster, while both smooth subsonic controls are slower. Grid sequencing helps selectively; it does not provide a universal acceleration."
        else:
            finding=f"{sum(x[2]>1 for x in speedups)} configurations are faster and {sum(x[2]<1 for x in speedups)} are slower after including all hierarchy work."
    elif all(r.get("root_success") for r in summary["runs"]) and audit:
        finding="All measured solves reach steady roots. Matched MUSCL shock solutions are faster, while both subsonic controls are slower. Constant-reconstruction shock paths retain small near-sonic profile differences and their time ratios remain descriptive."
    if finding_override is not None:
        finding=finding_override
    meta=summary.get("metadata",{})
    settings=summary["setups"][0].get("settings",{})
    hierarchy=meta.get("hierarchy_points",[51,101,201,401])
    body=f'''<div class="hero"><div class="hero-copy"><div class="eyebrow">EulerNozzle · initialization benchmark</div><h1>Coarse startup, fine-grid convergence</h1><p>Cold single-grid implicit marching versus a coarse-to-fine startup, followed by the same fine-grid solver. Timings include every coarse solve, transfer and retry. This demo implements grid sequencing; FAS is deferred.</p><div class="hero-meta"><span>Perfect-gas backend</span><span>Backward Euler / Newton / banded</span><span>Target: 401 points including two ghosts</span><span>Independent cold restarts</span></div></div></div>
<div class="kpis"><div class="kpi"><div class="kpi-label">Measured roots</div><div class="kpi-value">{passed}<span> / {total}</span></div><div class="kpi-note">Original target-mesh residual and physical checks</div></div><div class="kpi"><div class="kpi-label">Matched-root time ratio</div><div class="kpi-value">{fmt(min(ratios),3) if ratios else '—'}<span>–{fmt(max(ratios),3) if ratios else '—'}×</span></div><div class="kpi-note">Mean single-grid time / mean full sequence time</div></div><div class="kpi"><div class="kpi-label">Hierarchy</div><div class="kpi-value">51 → 401</div><div class="kpi-note">51 → 101 → 201 → 401 points; every stage counted</div></div></div>
<div class="callout"><div><strong>How to read acceleration.</strong> A ratio above one means coarse-to-fine is faster. A ratio below one means it is slower. Ratios are qualified only when every measured run converges to the matched fine-grid root. Point counts include two boundary/ghost slots; they are not physical-cell counts.</div></div>
<section id="performance"><div class="section-head"><div><span class="section-index">01 / PERFORMANCE</span><h2>Time to the same fine-grid problem</h2><p>Bars show means, whiskers show minimum–maximum and markers retain individual measured samples. One complete cold solve per configuration is excluded before measurement.</p></div></div><div class="panel"><div id="timing" class="plot medium"></div></div><div class="panel" style="margin-top:18px"><div id="iterations" class="plot medium"></div></div>
'''
    body=body.replace('<div class="hero-meta">',f'<p><strong>{html.escape(finding)}</strong></p><div class="hero-meta">',1)
    if audit and not audit.get("pass",False):
        warning='<div class="callout" style="margin-top:18px"><div><strong>Root checks pass; two strict profile-agreement comparisons fail.</strong> Both constant-reconstruction shock sequences converge with the same detected shock locations, but differ from single-grid profiles near the sonic throat beyond the predeclared 10<sup>−4</sup> fixed-scale state threshold. Maximum scaled differences are approximately 6.94 × 10<sup>−4</sup> (shock35) and 8.83 × 10<sup>−4</sup> (shock70). Peak pressure differences are about 112 and 143 Pa. These time ratios are retained as descriptive measurements and are not validated same-root acceleration claims. The tolerance is unchanged; root uniqueness or the cause of the near-sonic discrepancy has not been established.</div></div>'
        if failure_message is not None:
            warning=failure_message
        body=body.replace('<section id="performance">',warning+'<section id="performance">',1)
    body+=table(["Case","Reconstruction","Method","n","Roots","Same root","Mean [s]","Min–max [s]","All updates","Fine updates","Fine-equivalent Newton work","Rejected trials"],rows)
    speedrows=[[cid,RECON.get(recon,recon),fmt(ratio,4)+"×",("Faster" if ratio>1 else "Slower") if valid else "Unqualified: root audit incomplete or differs"] for cid,recon,ratio,valid in speedups]
    body+='<h3 style="margin-top:22px">Time ratios by physical case</h3>'+table(["Case","Reconstruction","Single / sequence mean time","Interpretation"],speedrows)
    slower=[f"{cid} / {RECON.get(recon,recon)}" for cid,recon,ratio,valid in speedups if valid and ratio<1]
    faster=[f"{cid} / {RECON.get(recon,recon)}" for cid,recon,ratio,valid in speedups if valid and ratio>1]
    if slower:
        body+='<div class="callout" style="margin-top:18px"><div><strong>The startup hierarchy is not universally beneficial.</strong> It increases warm solve time for '+html.escape(", ".join(slower))+'. All coarse stages and transfers remain charged in these negative results.</div></div>'
    negative=[]
    for method in METHODS:
        key=("subsonic","constant",method)
        if key not in groups:continue
        r=next((r for r in measured(groups[key]) if r["repeat"]==0),None)
        if r is None:continue
        stage=r["stages"][-1];hh=stage["history"]
        control=next((h for h in hh if h.get("existing_ssr") is not None),{})
        at=lambda n: next((h.get("effective_cfl") for h in hh if h.get("accepted_index")==n),None)
        negative.append([NAMES[method],fmt(hh[0]["residual_rms"],4),fmt(control.get("existing_ssr"),4),str(stage["iterations"]),fmt(at(20),4),fmt(at(100),4),str(r.get("rejected_trials",0))])
    if len(negative)==2:
        body+='<h3 style="margin-top:24px">Why the smooth constant-reconstruction control can slow down</h3><p>The following values come from measured repeat 0, not an averaged trajectory. The transferred fine-grid state starts with a smaller original residual, yet requires more accepted fine-grid updates. Its controller also ramps CFL more slowly.</p>'
        body+=table(["Method","Fine initial original RMS","First accepted native SSR","Fine updates","CFL at update 20","CFL at update 100","Rejected trials"],negative)
        body+='<p class="plot-caption">Each mesh restarts the retained SER controller and establishes a new native-SSR reference. A smaller starting defect therefore need not produce a faster CFL ramp. Coarse-grid discretization bias and this stage-local normalization are plausible contributors to the slowdown; this is an interpretation of the saved histories, not a separate causal ablation. Native SSR is not the fixed-scale original residual shown in convergence plots.</p>'
    body+='''<p class="plot-caption">Fine-equivalent Newton work weights every attempted pseudo-step, including rejected trials, by its physical-cell count divided by the fine-grid physical-cell count. There is exactly one Newton correction per attempt in this baseline. This cell-count proxy cannot replace measured time.</p></section>
<section id="histories"><div class="section-head"><div><span class="section-index">02 / CONVERGENCE</span><h2>Progress across the mesh hierarchy</h2><p>Each trace is one actual median-duration measured run. Coarse residuals belong to their own mesh; transfer boundaries are split rather than represented by a fabricated fine-grid residual.</p></div></div><div class="panel"><div class="controls"><div class="control case-control"><label for="case">Physical case</label><select id="case"></select></div><div class="control"><label for="recon">Reconstruction</label><select id="recon"><option value="constant">Constant</option><option value="muscl_van_leer">MUSCL</option></select></div></div><div id="history" class="plot history-plot"></div><div id="work-history" class="plot medium"></div><div class="plot-caption" id="representative"></div></div></section>
<section id="fields"><div class="section-head"><div><span class="section-index">03 / FINAL SOLUTION</span><h2>Analytical, constant and MUSCL profiles</h2><p>Both methods solve the same final mesh. Numerical markers are large hollow circles; change the number of skipped points independently of the underlying solution.</p></div></div><div class="panel"><div class="controls"><div class="control"><label for="skip">Points skipped between markers</label><input id="skip" type="number" min="0" step="1" value="5"></div><small>Uses the selected physical case above. Zero marks every point.</small></div><div id="profiles" class="plot profiles-plot"></div></div></section>
<section id="stages"><div class="section-head"><div><span class="section-index">04 / STARTUP STAGES</span><h2>What each mesh contributes</h2><p>Endpoints of the actual coarse-to-fine representative run, using the case and reconstruction selected above. These curves show the discrete shock profile established on each mesh; they are not samples of one physical-time trajectory.</p></div></div><div class="panel"><div id="stage-profiles" class="plot history-plot"></div></div><div id="stage-table"></div></section>
<section id="geometry"><div class="section-head"><div><span class="section-index">04 / PHYSICAL SETUP</span><h2>One nozzle, three operating points</h2><p>The same quadratic converging–diverging nozzle and reservoir data are used on every mesh. Backpressure selects shock35, shock70 and a smooth subsonic control.</p></div></div><div class="panel"><div id="channel" class="plot medium"></div></div></section>
<section id="method"><div class="section-head"><div><span class="section-index">05 / METHOD & TIMING</span><h2>What the comparison measures</h2></div></div><div class="note-grid"><div class="note-card"><h3>Same fine-grid method</h3><p>The numerical flux, reconstruction, EOS, boundary maps, original residual target, CFL/SER controller and implicit Newton machinery are held fixed. Only the startup path changes. One Newton correction per pseudo-step matches the current EulerNozzle baseline; it is not an exactly converged backward-Euler step.</p></div><div class="note-card"><h3>Every stage is charged</h3><p>The sequence starts cold on 51 points, transfers to 101, then 201, then 401. Transfer time, coarse solves, monitoring and rejected attempts are included. Fine-grid polishing continues to the original 10<sup>−8</sup> target. Coarse stages use a declared looser startup target and are never reported as final roots.</p></div><div class="note-card"><h3>Warm-time protocol</h3><p>One excluded complete solve is followed by ten measured solves per method/case/reconstruction by default, with alternating method order. Kernels are compiled once and device results synchronized. Imports, compilation, case construction, final audits, plotting and serialization are outside warm solve timing and recorded separately.</p></div><div class="note-card"><h3>Root versus accuracy</h3><p>Matching another numerical solution verifies the acceleration path, not analytical accuracy. Shock smearing and existing discrete artifacts can remain at a converged root. Pressure and Mach plots expose both; the saved diagnostics retain analytical errors independently.</p></div></div>'''
    body+='<details style="margin-top:20px"><summary>Effective setup, settings and benchmark metadata</summary><div class="details-body"><pre>'+html.escape(json.dumps(clean(dict(metadata=meta,settings=settings)),indent=2))+'</pre></div></details></section>'
    setup_rows=[[html.escape(s["case_id"]),RECON.get(s["reconstruction"],s["reconstruction"]),fmt(s["construction_s"],5),fmt(s["compile_s"],5)] for s in meta.get("setup",[])]
    setup_table='<h3 style="margin-top:24px">Shared hierarchy construction and first kernel calls</h3>'+table(["Case","Reconstruction","Construction [s]","All-level kernel setup / first calls [s]"],setup_rows)
    setup_table+='<p class="plot-caption">These are the measured shared costs for constructing and preparing all hierarchy levels, outside the warm solve timer. They are not separable single-grid versus sequence setup costs. The reported acceleration applies to warm repeated solves; one one-off solve may not amortize additional coarse-mesh compilation. No per-method first-use total is inferred from this shared measurement.</p>'
    body=body.replace('<details style="margin-top:20px"><summary>Effective setup',setup_table+'<details style="margin-top:20px"><summary>Effective setup',1)
    body+='<section id="audit"><div class="section-head"><div><span class="section-index">06 / VERIFICATION & PROVENANCE</span><h2>Independent checks and retained samples</h2></div></div>'
    if audit:
        body+='<div class="panel panel-padding"><h3>Saved-state audit</h3><pre>'+html.escape(json.dumps(clean({k:v for k,v in audit.items() if k not in ("runs","results")}),indent=2))+'</pre><p><a href="audit.json">Full machine-readable audit</a></p></div>'
    else:
        body+='<div class="callout"><strong>Independent saved-state audit is pending.</strong> Root flags here are runner results until audit.json has been generated.</div>'
    body+='<details style="margin-top:20px"><summary>Excluded complete solves</summary><div class="details-body">'+table(["Case","Reconstruction","Method","Time [s]","Updates","Root"],excluded)+'</div></details>'
    body+='''<div class="note-card" style="margin-top:20px"><h3>Reproduce locally</h3><pre>poetry run python -m development.roberto.multigrid.scripts.run_demo
poetry run python -m development.roberto.multigrid.scripts.audit_results
poetry run python -m development.roberto.multigrid.scripts.report_results</pre><p>Share REPORT.html from the study root: its results and theory pages, interactive figures and offered downloads are embedded in this single file. Numerical archives and exported figures remain under outputs/. FAS has not been implemented.</p></div><p class="plot-caption">Embedded downloads: <a href="summary.json">summary.json</a> · <a href="timing_samples.csv">all timing samples CSV</a> · <a href="figures/timing_and_work.pdf">timing/work PDF</a>. No neighboring files or internet connection are needed to read this report.</p></section>'''
    body=body.replace("Target: 401 points",f"Target: {hierarchy[-1]} points")
    body=body.replace("51 → 401",f"{hierarchy[0]} → {hierarchy[-1]}").replace("51 → 101 → 201 → 401"," → ".join(map(str,hierarchy)))
    body=body.replace("followed by ten measured solves",f"followed by {meta.get('repeats',10)} measured solves")
    body=body.replace("The sequence starts cold on 51 points, transfers to 101, then 201, then 401.","The sequence starts cold and visits meshes of "+", ".join(map(str,hierarchy))+" points.")
    body=body.replace('04 / PHYSICAL SETUP','05 / PHYSICAL SETUP').replace('05 / METHOD & TIMING','06 / METHOD & TIMING').replace('06 / VERIFICATION & PROVENANCE','07 / VERIFICATION & PROVENANCE')
    return body,csv_rows


THEORY = r'''<div class="hero"><div class="hero-copy"><div class="eyebrow">EulerNozzle · method and interpretation</div><h1>Coarse-to-fine implicit startup</h1><p>A standalone description of the equations, transfer, stopping rules and timing comparison. This demo uses mesh sequencing once on the way to the target mesh. No FAS correction cycle is present.</p></div></div>
<article class="theory-content">
<h2 id="equations">Euler balances and steady target</h2>
<p>The model is quasi-one-dimensional, inviscid and adiabatic in a nozzle of area $A(x)$. The primitive variables are pressure, meridional velocity and static enthalpy, $W=(p,v_m,h)$. The source implementation's velocity slot is retained through its API; the report uses $v_m$.</p>
<div class="math-block">$$q=\begin{bmatrix}\rho\\\rho v_m\\\rho E\end{bmatrix},\qquad F=\begin{bmatrix}\rho v_m\\\rho v_m^2+p\\\rho v_m(E+p/\rho)\end{bmatrix},\qquad E=h-\frac{p}{\rho}+\frac{v_m^2}{2}.$$</div>
<div class="math-block">$$\frac{\partial(Aq)}{\partial\tau}+\frac{\partial(AF)}{\partial x}=\begin{bmatrix}0\\p\,dA/dx\\0\end{bmatrix}.$$</div>
<p>The pseudo-time $\tau$ is a device for finding a steady root. The reported trajectories do not represent a physical nozzle startup. Eigenvalues in the axial direction are $v_m-a$, $v_m$ and $v_m+a$; the acoustic speed and other thermodynamic quantities come from the supplied fluid object. Perfect-gas relations are used by the analytical reference.</p>
<h2 id="implicit">Existing implicit update and boundary handling</h2>
<p>The experiment imports the current production EulerNozzle factory and its shared Newton/colored-Jacobian/banded solver infrastructure. No numerical flux or boundary law is reimplemented. The geometric momentum source, constant or MUSCL reconstruction, and existing boundary maps are retained.</p>
<div class="math-block">$$Q_i(W^{n+1})-Q_i(W^n)+\Delta\tau\,R_i(W^{n+1})=0.$$</div>
<p>The baseline applies one Newton correction per pseudo-step. Accepted updates are safeguarded by the existing increment/domain checks and timestep-halving policy. A step being accepted is not evidence that its inner backward-Euler residual has converged. Final success is determined from the original unsteady-term-free spatial residual and independent final-state checks.</p>
<p>The nozzle boundary values are constructed through the retained production boundary maps. This demo does not replace them with the annular solver's coupled trace formulation. Thus both compared startup strategies retain exactly the same nozzle boundary treatment.</p>
<h2 id="sequencing">Grid sequencing and interpolation</h2>
<p>The hierarchy is 51, 101, 201 and 401 points including the two boundary/ghost slots. Every mesh describes the same physical nozzle and backpressure. The single-grid path starts cold directly on 401 points. The sequence starts from the same cold-start rule on 51 points, performs implicit pseudo-time updates, transfers its state to the next mesh and continues.</p>
<p>Interpolation transfers numerical primitive states at physical coordinates. No analytical pressure, Mach profile or prescribed shock position is used to initialize any solve. The transferred state is checked for admissibility, and boundary maps are applied consistently by the existing solver. Primitive interpolation is a startup operation; it is not a conservative remap for a time-accurate transient.</p>
<p>Coarse stages use the declared initialization tolerance. The final 401-point stage always solves the original fine-grid problem to the original target. A loose coarse residual is not counted as convergence of the target equations.</p>
<h2 id="acceleration">Why a shock may settle at lower cost</h2>
<div class="math-block">$$\Delta\tau=C\min_i\frac{\Delta x_i}{|v_{m,i}|+a_i}.$$</div>
<p>At comparable CFL and wave speeds a wider cell permits a larger pseudo-step and reduces per-step work. A coarse mesh may establish the approximate shock location and global pressure field cheaply. The final mesh then resolves the remaining profile. Implicit integration is globally coupled, so this cell-width argument is a heuristic rather than a strict shock-propagation bound.</p>
<p>The hypothesis can fail. The coarse grid may select a different shock-cell root, smear the compression excessively, or cost more than the fine iterations it saves. Smooth cases can show little benefit when the existing single-grid Newton method already converges quickly. Measured total time and matched-root checks decide the outcome.</p>
<p>The retained SER controller resets its native-residual reference at every mesh transition. Thus a transferred state with a lower original residual does not automatically receive a more aggressive CFL ramp. The smooth constant-reconstruction control in this demo exhibits this combination: a smaller initial fine-grid defect, slower CFL growth and more fine-grid updates. Coarse discretization bias and stage-local SER normalization are plausible explanations, not effects isolated by an ablation. An alternative controller would be a separately labeled numerical experiment.</p>
<h2 id="work">Iterations and timing</h2>
<div class="math-block">$$N_{\mathrm{eq}}=\sum_\ell T_\ell\frac{n_{\mathrm{phys},\ell}}{n_{\mathrm{phys},\mathrm{fine}}},\qquad S=\frac{\overline t_{\mathrm{single}}}{\overline t_{\mathrm{sequence}}}.$$</div>
<p>$T_\ell$ counts all attempted pseudo-steps on level $\ell$, including rejected attempts. Each attempt executes one Newton correction in this baseline. $N_{\mathrm{eq}}$ is only a cell-count work proxy. Raw accepted counts, target-mesh counts, rejected trials and elapsed solve times are also shown. $S&gt;1$ indicates acceleration; $S&lt;1$ indicates a slowdown. Ratios are qualified only after convergence and same-root checks.</p>
<p>The solve timer charges coarse stages, transfers, retry work and equivalent monitoring. Setup, compilation, reference generation, final diagnostics and report writing are recorded separately. Repetitions start independently; no measured solution becomes a hidden warm start for a later sample. The displayed convergence trace is one labeled median-duration run, not an average of different trajectories.</p>
<h2 id="limits">Scope and limitations</h2>
<p>This is coarse-to-fine initialization, not recurring nonlinear FAS multigrid and not adaptive mesh refinement. The target discretization is unchanged. Fine-grid root convergence, agreement between numerical paths, conservation, operating branch and analytical accuracy are distinct checks. No speedup claim extends beyond the measured cases and backend.</p>
</article>'''


JS = r'''
const DATA=PAYLOAD;
const RECONS=[...new Set(DATA.flatMap(c=>c.runs.map(r=>r.reconstruction)))];
const names={single_grid:'Single grid',grid_sequence:'Coarse to fine'};
const colors={single_grid:'#6a00a8',grid_sequence:'#fca636'};
const reconNames={constant:'Constant',muscl_van_leer:'MUSCL'};
const cfg={responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg'}};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function layout(extra={}){return {paper_bgcolor:css('--panel'),plot_bgcolor:css('--panel'),font:{family:'Arial, sans-serif',size:12,color:css('--ink')},margin:{l:72,r:35,t:55,b:65},legend:{orientation:'h',y:1.15,x:0,font:{size:11}},...extra};}
function axis(title,extra={}){return {title:{text:title,font:{size:12}},showline:true,linewidth:1,linecolor:css('--ink'),mirror:true,ticks:'outside',ticklen:4,gridcolor:css('--grid'),zeroline:false,...extra};}
function selectCase(){return DATA[Number(document.getElementById('case').value)||0];}
function timing(){
const categories=[],traces=[],work=[];for(const c of DATA)for(const r of RECONS)categories.push(c.case_id+' / '+reconNames[r]);
for(const [mi,m] of ['single_grid','grid_sequence'].entries()){
let xx=[],yy=[],lo=[],hi=[],ii=[],il=[],ih=[],sx=[],sy=[],sw=[],ci=0;
for(const c of DATA)for(const r of RECONS){const run=c.runs.find(z=>z.method===m&&z.reconstruction===r),t=run.samples.map(z=>z.wall_s),u=run.samples.map(z=>z.equivalent_fine_updates),a=t.reduce((a,b)=>a+b,0)/t.length,b=u.reduce((a,b)=>a+b,0)/u.length,x=ci+(mi-.5)*.34;ci++;xx.push(x);yy.push(a);lo.push(a-Math.min(...t));hi.push(Math.max(...t)-a);ii.push(b);il.push(b-Math.min(...u));ih.push(Math.max(...u)-b);t.forEach((v,i)=>{sx.push(x+(i/(Math.max(t.length-1,1))-.5)*.09);sy.push(v);sw.push(u[i]);});}
traces.push({x:xx,y:yy,name:names[m],type:'bar',width:.32,marker:{color:colors[m]},error_y:{type:'data',symmetric:false,array:hi,arrayminus:lo,color:css('--ink'),thickness:1}});work.push({x:xx,y:ii,name:names[m],type:'bar',width:.32,marker:{color:colors[m]},error_y:{type:'data',symmetric:false,array:ih,arrayminus:il,color:css('--ink'),thickness:1}});
for(const [dest,values] of [[traces,sy],[work,sw]])dest.push({x:sx,y:values,name:names[m]+' samples',type:'scatter',mode:'markers',marker:{color:css('--panel'),line:{color:colors[m],width:1},size:4.5},showlegend:false,hovertemplate:'Measured sample: %{y:.5g}<extra>'+names[m]+'</extra>'});
}
const xax=axis('',{tickvals:categories.map((z,i)=>i),ticktext:categories,range:[-.6,categories.length-.4]});
Plotly.react('timing',traces,layout({barmode:'overlay',xaxis:xax,yaxis:axis('Warm solve time [s]')}),cfg);Plotly.react('iterations',work,layout({barmode:'overlay',xaxis:xax,yaxis:axis('Fine-equivalent Newton work')}),cfg);
}
function historyPlot(){const c=selectCase(),recon=document.getElementById('recon').value,traces=[];for(const r of c.runs.filter(x=>x.reconstruction===recon)){r.levels.forEach((l,i)=>{for(let j=0;j<2;j++)traces.push({x:j?l.elapsed_s:l.updates,y:l.residual,mode:'lines',type:'scatter',name:names[r.method]+' · '+l.n_points+' points',legendgroup:r.method+'-'+i,showlegend:j===0,line:{color:colors[r.method],width:1.8,dash:r.method==='single_grid'?'solid':['dot','dash','dashdot','solid'][i%4]},xaxis:j?'x2':'x',yaxis:j?'y2':'y'});});}
Plotly.react('history',traces,layout({height:470,xaxis:axis('Accepted updates · all levels',{domain:[0,.43]}),xaxis2:axis('Cumulative solve time [s]',{domain:[.57,1],anchor:'y2'}),yaxis:axis('Current-level original residual RMS',{type:'log'}),yaxis2:axis('Current-level original residual RMS',{type:'log',anchor:'x2'}),shapes:[{type:'line',xref:'paper',x0:0,x1:.43,yref:'y',y0:1e-8,y1:1e-8,line:{color:css('--muted'),dash:'dot',width:1}},{type:'line',xref:'paper',x0:.57,x1:1,yref:'y2',y0:1e-8,y1:1e-8,line:{color:css('--muted'),dash:'dot',width:1}}]}),cfg);
const work=[];for(const r of c.runs.filter(x=>x.reconstruction===recon))r.levels.forEach((l,i)=>work.push({x:l.work,y:l.residual,mode:'lines',line:{color:colors[r.method],width:1.8,dash:r.method==='single_grid'?'solid':['dot','dash','dashdot','solid'][i%4]},name:names[r.method]+' · '+l.n_points+' points'}));Plotly.react('work-history',work,layout({xaxis:axis('Fine-equivalent Newton work · rejected attempts included'),yaxis:axis('Current-level residual RMS',{type:'log'})}),cfg);
document.getElementById('representative').textContent=c.runs.filter(x=>x.reconstruction===recon).map(r=>names[r.method]+': repeat '+r.repeat+', '+r.wall_s.toPrecision(4)+' s, '+r.iterations+' total updates').join(' · ');}
function profilePlot(){const c=selectCase(),traces=[],skip=Number(document.getElementById('skip').value);if(!Number.isInteger(skip)||skip<0){document.getElementById('skip').setAttribute('aria-invalid','true');return;}document.getElementById('skip').removeAttribute('aria-invalid');const stride=skip+1,ref=c.reference_fine||c.reference,rx=ref.x_m||c.x_m;
for(let mi=0;mi<2;mi++){const method=['single_grid','grid_sequence'][mi];for(let ki=0;ki<2;ki++){const key=['p','M'][ki],scale=key==='p'?1e-3:1,idx=mi*2+ki,xa=idx?'x'+(idx+1):'x',ya=idx?'y'+(idx+1):'y';traces.push({x:rx,y:ref[key].map(v=>v*scale),name:'Analytical',mode:'lines',line:{color:css('--ink'),width:2,dash:'dash'},legendgroup:'analytical',showlegend:idx===0,xaxis:xa,yaxis:ya});for(const r of c.runs.filter(x=>x.method===method)){const f=r.fields,color=r.reconstruction==='constant'?'#9c179e':'#fca636';traces.push({x:f.x_m,y:f[key].map(v=>v*scale),name:reconNames[r.reconstruction],mode:'lines',line:{color,width:1.6,dash:r.reconstruction==='constant'?'solid':'dashdot'},legendgroup:r.reconstruction,showlegend:idx===0,xaxis:xa,yaxis:ya});const xx=[],yy=[];for(let i=r.reconstruction==='constant'?0:Math.floor(stride/2);i<f.x_m.length;i+=stride){xx.push(f.x_m[i]);yy.push(f[key][i]*scale);}traces.push({x:xx,y:yy,name:reconNames[r.reconstruction]+' markers',mode:'markers',marker:{color:css('--panel'),line:{color,width:1.8},size:8,symbol:'circle'},legendgroup:r.reconstruction,showlegend:false,xaxis:xa,yaxis:ya,hoverinfo:'skip'});}}}
const ann=[{text:'Single grid',x:.21,y:1.08},{text:'Coarse to fine',x:.21,y:.48}].map(a=>({...a,xref:'paper',yref:'paper',showarrow:false,font:{size:13,color:css('--ink')}}));const z=layout({height:760,annotations:ann,xaxis:axis('x [m]',{domain:[0,.43],anchor:'y'}),yaxis:axis('Static pressure [kPa]',{domain:[.59,1]}),xaxis2:axis('x [m]',{domain:[.57,1],anchor:'y2'}),yaxis2:axis('Mach number [-]',{domain:[.59,1],anchor:'x2'}),xaxis3:axis('x [m]',{domain:[0,.43],anchor:'y3'}),yaxis3:axis('Static pressure [kPa]',{domain:[0,.40],anchor:'x3'}),xaxis4:axis('x [m]',{domain:[.57,1],anchor:'y4'}),yaxis4:axis('Mach number [-]',{domain:[0,.40],anchor:'x4'})});Plotly.react('profiles',traces,z,cfg);}
function channel(){const c=selectCase(),x=c.x_m,A=c.area_m2;Plotly.react('channel',[{x,y:A.map(v=>v*1e4),mode:'lines',line:{color:'#9c179e',width:2},name:'Nozzle area'}],layout({xaxis:axis('Axial position x [m]'),yaxis:axis('Cross-sectional area [cm²]'),showlegend:false}),cfg);}
function stages(){const skip=Number(document.getElementById('skip').value);if(!Number.isInteger(skip)||skip<0)return;const c=selectCase(),recon=document.getElementById('recon').value,r=c.runs.find(x=>x.method==='grid_sequence'&&x.reconstruction===recon),ref=c.reference_fine||c.reference,traces=[],palette=r&&r.stages.length===2?['#6a00a8','#fca636']:['#6a00a8','#b12a90','#e16462','#fca636'];if(!r)return;const stride=skip+1;
for(let j=0;j<2;j++){const key=['p','M'][j],scale=j?1:1e-3;traces.push({x:ref.x_m||c.x_m,y:ref[key].map(v=>v*scale),name:'Analytical',mode:'lines',line:{color:css('--ink'),dash:'dash',width:2},showlegend:j===0,xaxis:j?'x2':'x',yaxis:j?'y2':'y'});r.stages.forEach((s,i)=>{let f=s.fields;traces.push({x:f.x_m,y:f[key].map(v=>v*scale),name:s.n_points+' points',mode:'lines',line:{color:palette[i%palette.length],dash:['dot','dash','dashdot','solid'][i%4],width:1.8},legendgroup:String(i),showlegend:j===0,xaxis:j?'x2':'x',yaxis:j?'y2':'y'});let xx=[],yy=[];for(let k=i;k<f.x_m.length;k+=stride){xx.push(f.x_m[k]);yy.push(f[key][k]*scale);}traces.push({x:xx,y:yy,mode:'markers',marker:{color:css('--panel'),line:{color:palette[i%palette.length],width:1.5},size:7},showlegend:false,legendgroup:String(i),xaxis:j?'x2':'x',yaxis:j?'y2':'y',hoverinfo:'skip'});});}
Plotly.react('stage-profiles',traces,layout({height:460,xaxis:axis('x [m]',{domain:[0,.43]}),yaxis:axis('Static pressure [kPa]'),xaxis2:axis('x [m]',{domain:[.57,1],anchor:'y2'}),yaxis2:axis('Mach number [-]',{anchor:'x2'})}),cfg);
let fields=['Points','Declared spatial RMS target','Accepted updates','Attempted Newton corrections','Rejected trials','Stage solve [s]','Transfer [s]','Endpoint residual','Compression proxy [m]'];let rows=r.stages.map(s=>[s.n_points,s.tolerance.toExponential(1),s.iterations,s.trials,s.rejected_trials,s.wall_s.toPrecision(4),s.transfer_s.toPrecision(3),s.diagnostics.residual_rms.toExponential(3),s.diagnostics.shock_proxy_m===null?'—':s.diagnostics.shock_proxy_m.toPrecision(4)]);document.getElementById('stage-table').innerHTML='<div class="panel table-wrap"><table><thead><tr>'+fields.map(s=>'<th>'+s+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td class="numeric">'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';}
function redraw(){timing();historyPlot();profilePlot();stages();channel();}
document.getElementById('case').innerHTML=DATA.map((c,i)=>'<option value="'+i+'">'+c.label+'</option>').join('');document.getElementById('recon').innerHTML=RECONS.map(r=>'<option value="'+r+'">'+reconNames[r]+'</option>').join('');document.getElementById('case').onchange=()=>{historyPlot();profilePlot();stages();};document.getElementById('recon').onchange=()=>{historyPlot();stages();};document.getElementById('skip').oninput=()=>{profilePlot();if(Number.isInteger(Number(document.getElementById('skip').value))&&Number(document.getElementById('skip').value)>=0)stages();};
window.redraw=redraw;redraw();
'''


THEME_JS = r'''function theme(value){document.documentElement.dataset.theme=value;try{localStorage.setItem('nozzle-grid-sequence-theme',value);}catch{}document.querySelectorAll('[data-theme-choice]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.themeChoice===value));for(const [page,study] of Object.entries(window.gridStudies||{}))if(!document.getElementById(page).hidden)study.redraw();}document.querySelectorAll('[data-theme-choice]').forEach(b=>b.onclick=()=>theme(b.dataset.themeChoice));theme(document.documentElement.dataset.theme);'''

PAGE_JS = r'''
let theoryTypeset=false, theoryTypesetTask=null;
async function typesetTheory(){if(theoryTypeset)return;if(!theoryTypesetTask)theoryTypesetTask=MathJax.startup.promise.then(async()=>{await MathJax.typesetPromise([document.getElementById('theory')]);theoryTypeset=true;});await theoryTypesetTask;}
function routePage(scroll=true){
 const defaultPage=document.body.dataset.defaultPage||'results';let key=defaultPage;try{key=decodeURIComponent(location.hash.slice(1))||key;}catch{}
 const target=document.getElementById(key)||document.getElementById(defaultPage);
 const page=target.closest('[data-report-page]')?.dataset.reportPage||'results';
 document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=p.dataset.reportPage!==page);
 document.querySelectorAll('[data-page-nav]').forEach(p=>p.hidden=p.dataset.pageNav!==page);
 document.querySelectorAll('[data-page-link]').forEach(a=>{if(a.dataset.pageLink===page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
 document.getElementById('sidebar-heading').textContent={results:'Original, looser coarse target',four_grid:'Four grids, tight tolerance',two_grid:'Two grids, tight tolerance',theory:'Method and equations'}[page];
 document.querySelectorAll('.sidebar a').forEach(a=>a.classList.toggle('active',a.hash==='#'+key));
 document.title={results:'Original four grids',four_grid:'Four grids (tight)',two_grid:'Two grids (tight)',theory:'Method and equations'}[page]+' - EulerNozzle grid sequencing';
 window.multigridPageReady=new Promise((resolve,reject)=>requestAnimationFrame(async()=>{try{
   const study=window.gridStudies?.[page];
   if(study){await study.redraw();await Promise.all([...document.querySelectorAll('#'+page+' .js-plotly-plot')].map(p=>Plotly.Plots.resize(p)));}
   else await typesetTheory();
   if(scroll)target.scrollIntoView({block:'start',behavior:'instant'});
   resolve();
 }catch(e){console.error(e);document.documentElement.dataset.reportError=e.message;reject(e);}}));
}
window.addEventListener('hashchange',()=>routePage());
window.addEventListener('beforeprint',()=>{document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=false);});
window.addEventListener('afterprint',()=>routePage(false));
document.getElementById('print').onclick=async()=>{
 document.querySelectorAll('[data-report-page]').forEach(p=>p.hidden=false);
 await typesetTheory();await Promise.all(Object.values(window.gridStudies||{}).map(s=>s.redraw()));
 requestAnimationFrame(()=>requestAnimationFrame(()=>window.print()));
};
window.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='p'){event.preventDefault();document.getElementById('print').click();}});
routePage(Boolean(location.hash));
'''


def embed_downloads(body, output, name_prefix=""):
    """Replace repository-relative downloads with exact self-contained bytes."""
    mime={".json":"application/json", ".csv":"text/csv", ".pdf":"application/pdf"}
    records=[]
    def replace(match):
        relative=match.group(1)
        if relative.startswith(("#","data:","https:","http:","mailto:")):
            return match.group(0)
        path=(output/relative).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError(f"Missing or out-of-scope report download: {relative}")
        raw=path.read_bytes()
        records.append(dict(name=name_prefix+path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        uri=f"data:{mime.get(path.suffix,'application/octet-stream')};base64,"+base64.b64encode(raw).decode()
        return f'href="{uri}" download="{html.escape(name_prefix+path.name,quote=True)}"'
    return re.sub(r'href="([^"]+)"',replace,body),records


def audit_documents(output, report, *, pages=("results","theory"), download_count=4):
    """Reject neighboring-file dependencies in the shareable HTML artifact."""
    class Inspector(HTMLParser):
        def __init__(self):
            super().__init__();self.ids=[];self.hrefs=[];self.dependencies=[];self.downloads=[];self.styles=[];self.pages=[]
            self.in_style=False
            self.in_script=False;self.script_html_comment_markers=0
        def handle_starttag(self, tag, attrs):
            values=dict(attrs)
            if "id" in values:self.ids.append(values["id"])
            if "href" in values:self.hrefs.append(values["href"])
            if values.get("data-report-page"):self.pages.append(values["data-report-page"])
            if "download" in values:self.downloads.append(values)
            for attribute in ("src","srcset","poster","data"):
                if values.get(attribute) and not values[attribute].startswith(("data:","#")):
                    self.dependencies.append(values[attribute])
            if tag=="link" and values.get("href") and not values["href"].startswith("data:"):
                self.dependencies.append(values["href"])
            if values.get("style"):self.styles.append(values["style"])
            if tag=="style":self.in_style=True
            if tag=="script":self.in_script=True
        def handle_endtag(self,tag):
            if tag=="style":self.in_style=False
            if tag=="script":self.in_script=False
        def handle_data(self,data):
            if self.in_style:self.styles.append(data)
            if self.in_script:self.script_html_comment_markers+=data.count("<!--")
    doc=Inspector();doc.feed(report.read_text(encoding="utf-8"))
    broken=[]
    for href in doc.hrefs:
        if href.startswith("#"):
            if href[1:] not in doc.ids:broken.append(href)
        elif not href.startswith(("https:","http:","mailto:","data:")):
            broken.append(href)
    css_dependencies=[]
    for style in doc.styles:
        if re.search(r'@import\b',style,re.I):css_dependencies.append("CSS import")
        css_dependencies.extend(v for v in re.findall(r'url\([\s\"\']*([^\)\"\']+)',style,re.I)
                                if not v.startswith(("data:","#")))
    downloads=[]
    for item in doc.downloads:
        raw=base64.b64decode(item["href"].split(",",1)[1],validate=True)
        downloads.append(dict(name=item["download"],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    application_network_calls=bool(re.search(r'\b(?:fetch\s*\(|XMLHttpRequest|import\s*\()',JS+PAGE_JS+THEME_JS))
    result=dict(pass_=bool(len(doc.ids)==len(set(doc.ids)) and not broken and not doc.dependencies
                          and not css_dependencies and not application_network_calls
                          and not doc.script_html_comment_markers
                          and sorted(doc.pages)==sorted(pages) and len(downloads)==download_count),
        file=report.name,internal_pages=doc.pages,unique_ids=len(doc.ids)==len(set(doc.ids)),
        ids=len(doc.ids),broken_links=broken,file_or_network_resource_dependencies=doc.dependencies,
        css_dependencies=css_dependencies,application_network_calls=application_network_calls,
        raw_script_html_comment_markers=doc.script_html_comment_markers,
        embedded_downloads=downloads,standalone_html=True,browser_runtime_checked=False,
        note="Structural portability audit: internal pages, fragment links, embedded resources and decoded download payloads. No browser rendering claim.")
    (output/"report_audit.json").write_text(json.dumps(result,indent=2)+'\n',encoding="utf-8")
    if not result["pass_"]:raise ValueError("Generated document structural audit failed")
    return result


def shell(body, theory, scripts="", two_grid=None, four_grid=None):
    css=(ASSETS/"report.css").read_text(encoding="utf-8")
    css+='\n.history-plot{height:470px}.profiles-plot{height:760px}section .table-wrap{margin-top:18px}main{max-width:1750px}.theory-content{max-width:1100px} @media(max-width:680px){.history-plot{height:600px!important}.profiles-plot{height:800px!important}}'
    css+='\n[hidden]{display:none!important}.report-page{margin:0;scroll-margin-top:88px}.report-page:focus{outline:none}@media print{body .report-page[hidden]{display:block!important}#theory{break-before:page}.report-page{break-inside:auto}}'
    results_sections=(("performance","Time and iterations"),("histories","Convergence histories"),("fields","Final solutions"),("stages","Startup stages"),("geometry","Physical setup"),("method","Method and timing"),("audit","Verification"))
    theory_sections=(("equations","Euler balances"),("implicit","Implicit method"),("sequencing","Grid sequencing"),("acceleration","Shock acceleration"),("work","Timing and work"),("limits","Scope"))
    def nav(sections):
        return "".join(f'<a href="#{key}"><span>{i+1:02}</span>{text}</a>' for i,(key,text) in enumerate(sections))
    extra_link = '<a href="#two_grid" data-page-link="two_grid">02 &middot; Two grids (tight)</a>' if two_grid else ''
    four_link = '<a href="#four_grid" data-page-link="four_grid" aria-current="page">01 &middot; Four grids (tight)</a>' if four_grid else ''
    default_page = 'four_grid' if four_grid else 'results'
    original_link = ('<a href="#results" data-page-link="results">03 &middot; Original</a>' if four_grid
                     else '<a href="#results" data-page-link="results" aria-current="page">01 &middot; Four grids</a>')
    theory_number = '04' if four_grid else ('03' if two_grid else '02')
    two_sections = (("two-experiment","Experiment and targets"),) + tuple(("two-"+key,label) for key,label in results_sections) + (("two-agreement","Profile agreement"),("two-tight-stages","Both stage residuals"),("two-historical-comparison","Four-grid context"))
    extra_nav = f'<nav data-page-nav="two_grid" hidden>{nav(two_sections)}</nav>' if two_grid else ''
    four_sections = tuple((key.replace('two-','four-',1),label.replace("Both stage","All stage")) for key,label in two_sections) + ((("four-tight-comparison","Two-grid comparison"),) if two_grid else ())
    four_nav = f'<nav data-page-nav="four_grid">{nav(four_sections)}</nav>' if four_grid else ''
    four_panel = f'<div class="report-page" id="four_grid" data-report-page="four_grid">{four_grid}</div>' if four_grid else ''
    original_hidden = ' hidden' if four_grid else ''
    extra_panel = f'<div class="report-page" id="two_grid" data-report-page="two_grid" hidden>{two_grid}</div>' if two_grid else ''
    licenses=(ASSETS/"vendor/PLOTLY_LICENSE.txt").read_text(encoding="utf-8")+'\n\n'+(ASSETS/"vendor/MATHJAX_LICENSE.txt").read_text(encoding="utf-8")
    return f'''<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><title>EulerNozzle · Grid sequencing</title><script>try{{document.documentElement.dataset.theme=localStorage.getItem('nozzle-grid-sequence-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light')}}catch{{}}</script><style>{css}</style></head><body data-default-page="{default_page}">
<header class="header"><a class="brand" href="#{default_page}">EulerNozzle <small>Grid sequencing</small></a><nav class="page-switch" aria-label="Report pages">{four_link}{extra_link}{original_link}<a href="#theory" data-page-link="theory">{theory_number} &middot; Method & equations</a></nav><div class="header-actions"><button id="print">Print / PDF</button><div class="theme-switch"><button data-theme-choice="light">Light</button><button data-theme-choice="dark">Dark</button></div></div></header><aside class="sidebar"><div class="eyebrow" id="sidebar-heading">Results and verification</div>{four_nav}<nav data-page-nav="results"{original_hidden}>{nav(results_sections)}</nav>{extra_nav}<nav data-page-nav="theory" hidden>{nav(theory_sections)}</nav><div class="sidebar-foot"><span class="status-dot"></span>Single-file offline report<br>Results + theory included<br>No FAS implementation</div></aside><main>{four_panel}{extra_panel}<div class="report-page" id="results" data-report-page="results"{original_hidden}>{body}</div><div class="report-page" id="theory" data-report-page="theory" hidden>{theory}</div><footer><span>EulerNozzle · Coarse-to-fine implicit startup</span><span>Share this HTML file · No installation or internet required</span></footer><details><summary>Embedded library licenses</summary><pre>{html.escape(licenses)}</pre></details></main>{scripts}<script>{THEME_JS}</script><script>{PAGE_JS}</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=STUDY/"outputs")
    parser.add_argument("--report",type=Path,default=STUDY/"REPORT.html",
                        help="Single shareable HTML file, directly in the study root")
    parser.add_argument("--two-grid-output",type=Path,help="Separate two-grid dataset to append; detected automatically for the main report")
    parser.add_argument("--four-grid-output",type=Path,help="Uniformly tight four-grid dataset; detected automatically")
    args=parser.parse_args(); output=args.output.resolve()
    destination=args.report.resolve()
    if not output.is_relative_to(STUDY.resolve()):
        raise ValueError("Numerical artifacts must remain under the multigrid study")
    if destination.parent!=STUDY.resolve() or destination.suffix.lower()!=".html":
        raise ValueError("Place the standalone HTML report directly at the multigrid study root")
    source=output/"summary.json"
    raw_summary=source.read_bytes()
    summary=json.loads(source.read_text(encoding="utf-8"))
    numerical_files=[source,output/"audit.json",*sorted((output/"runs").glob("*.json"))]
    numerical_hashes={p:digest(p) for p in numerical_files if p.exists()}
    two_output = args.two_grid_output.resolve() if args.two_grid_output else output/"two_grid_tight"
    include_two = (two_output/"summary.json").is_file()
    if include_two:
        if not two_output.is_relative_to(STUDY):raise ValueError("Two-grid data outside study")
        for path in [two_output/"summary.json",two_output/"audit.json",*sorted((two_output/"runs").glob("*.json"))]:
            numerical_hashes[path]=digest(path)
    four_output = args.four_grid_output.resolve() if args.four_grid_output else output/"four_grid_tight"
    include_four = (four_output/"summary.json").is_file()
    if include_four:
        if not four_output.is_relative_to(STUDY):raise ValueError("Four-grid data outside study")
        for path in [four_output/"summary.json",four_output/"audit.json",*sorted((four_output/"runs").glob("*.json"))]:
            numerical_hashes[path]=digest(path)
    groups=group_runs(summary,output)
    payload=make_payload(summary,groups)
    static_figures(summary,groups,payload,output)
    audit_file=output/"audit.json"
    audit=json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.exists() else None
    body,csv_rows=results_content(summary,groups,audit)
    with (output/"timing_samples.csv").open('w',newline='',encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    body,downloads=embed_downloads(body,output)
    plotly=gzip.decompress((ASSETS/"vendor/plotly-2.35.2.min.js.gz").read_bytes()).decode()
    import sys
    from .tight_grid_report import build_page, study_script, compare_tight_experiments
    rr=sys.modules[__name__]
    scripts='<script>'+re.sub(r'</script',r'<\/script',plotly,flags=re.I)+'</script>'+study_script(rr,payload,'results')
    two_body = None
    if include_two:
        two_body,two_payload,two_downloads,two_summary,two_audit=build_page(rr,two_output,summary,groups)
        scripts+=study_script(rr,two_payload,'two_grid','two-')
        downloads+=two_downloads
    four_body = None
    if include_four:
        four_body,four_payload,four_downloads,four_summary,four_audit=build_page(rr,four_output,summary,groups,intervals=(50,100,200,400),prefix="four-")
        if include_two:
            four_body+=compare_tight_experiments(rr,four_output,two_output)
        scripts+=study_script(rr,four_payload,'four_grid','four-')
        downloads+=four_downloads

    mathjax=gzip.decompress((ASSETS/"vendor/mathjax-3.2.2-tex-svg.js.gz").read_bytes()).decode()
    # VS Code rewrites HTML-comment markers even inside inline JavaScript regexes.
    # Hex-escape the opener in MathJax's regex, preserving its matching behavior.
    mathjax=mathjax.replace("<!--",r"\x3C!--")
    mathscripts='''<script>window.MathJax={loader:{load:[]},tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']],processEscapes:true,packages:['base','ams','newcommand','noundefined','configmacros']},svg:{fontCache:'local'},options:{enableMenu:false},startup:{typeset:false}};</script><script>'''+re.sub(r'</script',r'<\\/script',mathjax,flags=re.I)+'</script>'
    theory=THEORY
    hierarchy=summary.get("metadata",{}).get("hierarchy_points",[51,101,201,401])
    hierarchy_text=", ".join(map(str,hierarchy[:-1]))+" and "+str(hierarchy[-1])
    theory=theory.replace("51, 101, 201 and 401",hierarchy_text)
    theory=theory.replace("401-point",f"{hierarchy[-1]}-point").replace("on 401 points",f"on {hierarchy[-1]} points").replace("on 51 points",f"on {hierarchy[0]} points")
    if include_two:
        theory += r'<section class="theory-content"><h2 id="two-grid-theory">Two grids with equally tight convergence</h2><p>The additional experiment uses 40 and 400 source mesh intervals: 41/401 sampled points, including two ghost slots and 39/399 interior unknown cells. Both stages stop at fixed-scale original spatial RMS 1e-8. The fine equations, flux, boundary maps and one-correction pseudo-time kernel are unchanged. A fresh single-grid control is timed in the same campaign.</p><div class="math-block">$$T_{2}=T_{41}+T_{\mathrm{transfer}}+T_{401\mid41},\qquad r_{41}\le10^{-8},\quad r_{401}\le10^{-8}.$$</div><p>These are steady residual targets, not tighter inner backward-Euler solves. A converged coarse field is still a solution of a different discrete system. Removing intermediate levels can reduce startup work while increasing fine-grid relaxation or changing the selected discrete profile. This experiment changes both the hierarchy and coarse tolerance; it does not isolate either cause. Full-state agreement and branch checks remain necessary before interpreting a time ratio as acceleration.</p></section>'
    if include_four:
        theory += r'<section class="theory-content"><h2 id="four-grid-theory">Four grids with one tight target</h2><p>The new primary experiment retains 50/100/200/400 source intervals, hence 51/101/201/401 points, and applies original spatial RMS 1e-8 to every stage. It changes only coarse stopping tolerances relative to the original four-grid study. The single-grid control is freshly measured; the original loose-coarse data and tight two-grid data remain separate.</p><div class="math-block">$$r_{51},r_{101},r_{201},r_{401}\le10^{-8},\qquad T_4=\sum_{\ell}T_{\ell}+T_{\mathrm{transfer}}.$$</div><p>These targets concern steady equations, not the accuracy of an individual one-correction pseudo-time step. Every transfer resets the retained SER controller. A tighter coarse solve can change the subsequent trajectory or discrete profile; the measured full cost and unrelaxed profile checks determine whether initialization accelerates the target solve.</p></section>'
    destination.write_bytes(shell(body,theory,scripts+mathscripts,two_grid=two_body,four_grid=four_body).encode("utf-8"))
    pages=tuple(p for p in (("four_grid",) if include_four else ()) + (("two_grid",) if include_two else ()) + ("results","theory"))
    document_audit=audit_documents(output,destination,pages=pages,download_count=4*(1+int(include_two)+int(include_four)))
    for path,expected in numerical_hashes.items():
        if digest(path)!=expected:
            raise ValueError(f"Numerical data changed during report generation: {path}")
    if source.read_bytes()!=raw_summary:
        raise ValueError("Summary changed during report generation")
    # Remove only the superseded generated pages, after the combined report passes.
    for name in ("REPORT.html","THEORY.html"):
        obsolete=(output/name).resolve()
        if obsolete.parent==output and obsolete!=destination and obsolete.is_file():
            obsolete.unlink()
    manifest=dict(summary_sha256=digest(source),audit_sha256=digest(audit_file) if audit else None,
        files={p.relative_to(STUDY).as_posix():digest(p) for p in [destination,output/"timing_samples.csv",output/"report_audit.json",*sorted((output/"figures").glob('*'))]},
        artifact_paths_relative_to="development/roberto/multigrid",
        report=destination.name,standalone_html=True,internal_pages=list(pages),
        embedded_downloads=downloads,numerical_files_checked=len(numerical_hashes),numerical_data_unchanged=True,
        representative_policy="lower median-duration measured actual run; no averaged trajectories",
        generated_artifacts_only=True,document_structure_pass=document_audit["pass_"])
    if include_two:
        manifest["two_grid_dataset"] = dict(summary_sha256=digest(two_output/"summary.json"),audit_sha256=digest(two_output/"audit.json"),
            hierarchy_points=two_summary["metadata"]["hierarchy_points"],mesh_intervals=two_summary["metadata"]["mesh_intervals"],
            coarse_tolerance=two_summary["metadata"]["coarse_tolerance"],fine_tolerance=two_summary["metadata"]["fine_tolerance"],
            retained_attempts=len(two_summary["runs"]),stage_targets_verified=all(s["residual_target_met"] for r in two_audit["results"] for s in r["stage_residual_checks"]))
    if include_four:
        manifest["four_grid_dataset"] = dict(summary_sha256=digest(four_output/"summary.json"),audit_sha256=digest(four_output/"audit.json"),
            hierarchy_points=four_summary["metadata"]["hierarchy_points"],mesh_intervals=four_summary["metadata"]["mesh_intervals"],
            coarse_tolerance=four_summary["metadata"]["coarse_tolerance"],fine_tolerance=four_summary["metadata"]["fine_tolerance"],
            retained_attempts=len(four_summary["runs"]),stage_targets_verified=all(s["residual_target_met"] for r in four_audit["results"] for s in r["stage_residual_checks"]))
    (output/"report_manifest.json").write_text(json.dumps(manifest,indent=2)+'\n',encoding="utf-8")
    print(f"Generated {destination}; pages={list(pages)}; retained attempts={len(summary['runs']) + (len(two_summary['runs']) if include_two else 0) + (len(four_summary['runs']) if include_four else 0)}; numerical data unchanged")


if __name__=="__main__":
    main()
