"""Saved-data calculations shared by the production report and its audit."""

import base64
import csv
from hashlib import sha256
from html import escape
from io import StringIO
import json
from pathlib import Path

import numpy as np

NAMES={"single_stadtke":"Stadtke · single","multigrid_stadtke":"Stadtke · sequence",
    "multigrid_stadtke_modified":"Modified Stadtke · sequence","multigrid_hll":"HLL · sequence","multigrid_rusanov":"Rusanov · sequence"}

def number(value):return "—" if value is None else f"{value:.4g}"

def embedded(name,data,mime):
    digest=sha256(data).hexdigest()
    return f'<a download="{name}" data-sha256="{digest}" href="data:{mime};base64,{base64.b64encode(data).decode()}">{escape(name)}</a>'

def csv_data(records):
    stream=StringIO();writer=csv.writer(stream)
    writer.writerow(["case","reconstruction","method","n","mean_s","min_s","max_s","fine_mean_s","success","all_stages_success","analytical_passed","branch_passed","fallbacks","residual_us","jacobian_us","linear_us","direction_us","fixed20_ms"])
    for r in records:
        k=r.get("kernel_timing",{});a=r.get("assessment",{});t=r["timing"]
        writer.writerow([r["case_id"],r["reconstruction"],r["method"]["id"],len(r["samples"]),t["mean_s"],t["min_s"],t["max_s"],r["fine_timing"]["mean_s"],
            r.get("success"),r.get("all_stages_success"),a.get("analytical_passed"),a.get("branch_passed"),r.get("fallback_count"),
            *[k.get(n,{}).get("mean_s",0)*f if n in k else None for n,f in (("residual",1e6),("jacobian_blocks",1e6),("linear_solve",1e6),("direction",1e6),("fixed20",1e3))]])
    return stream.getvalue().encode()

def same_flux_comparison(records):
    result=[]
    index={(r["case_id"],r["reconstruction"],r["method"]["id"]):r for r in records}
    for (case,recon,method),single in index.items():
        if method!="single_stadtke":continue
        sequence=index.get((case,recon,"multigrid_stadtke"))
        if sequence is None or not single["samples"] or not sequence["samples"]:continue
        a,b=single["assessment"],sequence["assessment"]
        deviations={name:float(np.max(np.abs(np.asarray(a["state"]["fields"][name])-np.asarray(b["state"]["fields"][name]))))/norm["scale"]
            for name,norm in a["errors"].items() if name in a["state"]["fields"] and name in b["state"]["fields"]}
        acceptable=all(r.get("repeat_consistency",{}).get("all_fine_success") and r["assessment"]["branch_passed"] and r["assessment"]["analytical_passed"] for r in (single,sequence))
        ratio=sequence["timing"]["mean_s"]/single["timing"]["mean_s"]
        correction=None
        if "fixed20" in single.get("kernel_timing",{}) and "fixed20" in sequence.get("kernel_timing",{}):
            correction=sequence["kernel_timing"]["fixed20"]["mean_s"]/single["kernel_timing"]["fixed20"]["mean_s"]
        deviation=max(deviations.values())
        result.append(dict(case=case,reconstruction=recon,single_mean_s=single["timing"]["mean_s"],sequence_mean_s=sequence["timing"]["mean_s"],
            sequence_over_single_total=ratio,sequence_over_single_correction=correction,original_checks_passed=bool(acceptable),
            maximum_field_scaled_state_deviation=deviation,field_scaled_deviations=deviations,
            comparison_field_agreement_tolerance=1e-6,comparison_field_agreement_passed=deviation<=1e-6,
            timing_interpretation="Same-equation timing comparison; original checks pass and fine fields agree within 1e-6 of fixed scales" if acceptable and deviation<=1e-6 else "Descriptive timing only: original accuracy or fine-field agreement check failed"))
    return result

def measured_findings(records,comparisons,folder):
    paragraphs=[]
    ratios=[c["sequence_over_single_correction"] for c in comparisons if c["sequence_over_single_correction"] is not None]
    if ratios:
        paragraphs.append(f"With vanilla Stadtke, sequencing changes measured 400-cell correction cost by {(min(ratios)-1)*100:+.1f}% to {(max(ratios)-1)*100:+.1f}% across these cases. Complete-solve totals also reflect different convergence paths and coarse-grid overhead.")
    validation=folder/"api_removal_validation.json"
    if validation.is_file():
        v=json.loads(validation.read_text(encoding="utf-8"))
        equal=sum(c["bitwise_equal"] for c in v["state_checks"])
        paragraphs.append(f"The public-input removal passed {v['production_tests']['passed']} focused tests. {equal}/{len(v['state_checks'])} vanilla single-grid final states are bitwise identical to the prior campaign with reuse enabled.")
    identical=sum(r.get("repeat_consistency",{}).get("max_variable_scaled_state_difference")==0 for r in records)
    paragraphs.append(f"Repeated final states are numerically identical in {identical}/{len(records)} configurations. Original analytical targets pass in {sum(bool(r.get('assessment',{}).get('analytical_passed')) for r in records)}/{len(records)} configurations; misses remain flagged below.")
    lookup={(r["case_id"],r["reconstruction"],r["method"]["id"]):r for r in records}
    for method in ("multigrid_stadtke_modified","multigrid_hll","multigrid_rusanov"):
        relative=[]
        for (case,recon,key),r in lookup.items():
            if key!=method or "fixed20" not in r.get("kernel_timing",{}):continue
            base=lookup.get((case,recon,"multigrid_stadtke"),{})
            if "fixed20" in base.get("kernel_timing",{}):relative.append(r["kernel_timing"]["fixed20"]["mean_s"]/base["kernel_timing"]["fixed20"]["mean_s"])
        if relative:paragraphs.append(f"{NAMES[method]}: measured correction cost is {min(relative):.3g}–{max(relative):.3g} times sequenced vanilla Stadtke across {len(relative)} case/reconstruction pairs. These kernel diagnostics use each method's own final state, including cases missing analytical targets.")
    for c in comparisons:
        if c["comparison_field_agreement_passed"]:continue
        a=lookup[c["case"],c["reconstruction"],"single_stadtke"]["assessment"]["state"]["fields"]
        b=lookup[c["case"],c["reconstruction"],"multigrid_stadtke"]["assessment"]["state"]["fields"]
        velocity=float(np.max(np.abs(np.asarray(a["velocity"])-np.asarray(b["velocity"]))))
        paragraphs.append(f"{c['case']} with {c['reconstruction']}: single-grid and sequenced fine states differ by up to {c['maximum_field_scaled_state_deviation']:.4g} of a fixed field scale (maximum velocity difference {velocity:.3g} m/s). Their {c['single_mean_s']:.3g} s versus {c['sequence_mean_s']:.3g} s totals are descriptive convergence-path timings, not a verified time-to-identical-state speedup. Original checks passed for both: {c['original_checks_passed']}.")
    agreeing=[c["maximum_field_scaled_state_deviation"] for c in comparisons if c["comparison_field_agreement_passed"]]
    if agreeing:paragraphs.append(f"{len(agreeing)}/{len(comparisons)} same-flux pairs agree within the separate comparison threshold of 1e-6 of fixed field scales; their maximum deviation is {max(agreeing):.4g}. This check does not change solver or analytical acceptance.")
    return paragraphs
