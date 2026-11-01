"""Study-specific saved-data findings; no numerical solver imports."""
from hashlib import sha256
import json
VARIANTS={"stadtke":("stadtke_original","stadtke_jump","stadtke_endpoint","stadtke_both")}

def epoch_recovery(study, records, originals):
    name = "supplemental/source_isolation_completion.json"
    result = dict(verified=False, statement="Flow and fresh-audit results refer to their recorded source/case/client epoch, not current-checkout byte identity.")
    if name not in originals:
        return result
    completion = json.loads(originals[name])
    bindings = {"source_isolation_manifest.json": "source_isolation_manifest_sha256",
                "source_guard_interruption.json": "guard_interruption_sha256",
                "source_isolation_worker.json": "actual_worker_proof_sha256"}
    okay = completion.get("passed") is True and all(completion.get("checks", {}).values())
    for filename, key in bindings.items():
        blob = originals.get("supplemental/" + filename)
        okay = okay and blob is not None and sha256(blob).hexdigest() == completion.get(key)
    okay = okay and all(r["source_sha256"] == completion.get("source_sha256")
                       and r["implementation_sha256"] == completion.get("implementation_sha256") for r in records)
    if study == "osher":
        okay = okay and sha256(originals["summary.json"]).hexdigest() == completion.get("canonical_summary_sha256")
        okay = okay and sha256(originals["fresh_audit.json"]).hexdigest() == completion.get("canonical_fresh_audit_sha256")
    if not okay:
        raise ValueError("Recorded-epoch completion proof differs from its evidence or campaign hashes")
    worker = json.loads(originals["supplemental/source_isolation_worker.json"])
    new = completion["new_rows_admitted"]
    count = len(new) if isinstance(new, (list, dict)) else int(new)
    result.update(verified=True, evidence=name, evidence_sha256=sha256(originals[name]).hexdigest(),
        statement=f"The Osher source guard stopped after {completion['original_immutable_rows']} admitted rows. The remaining {count} rows and final fresh checks were completed with the same frozen epoch in isolation; the original rows were unchanged. The actual worker verified {worker['verified_loaded_module_count']} loaded package origins and {worker['deployed_file_count']} deployed file hashes before/after. The external current-checkout Roe edit was preserved; this is not a current full-suite or Roe-branch validation.")
    return result


def matrix_findings(study, records, audits, originals, valid_audit):
    name = f"supplemental/{study}_matrix_postprocess.json"
    result = dict(verified=False, statements=[], scope="Current production matrix only; historical shifted evidence is not included")
    if name not in originals:
        return result
    review = json.loads(originals[name])
    if not review.get("coverage_complete") or not review.get("saved_data_review_passed"):
        return result
    actual = {row["source_file"]: row["file_sha256"] for row in records}
    native = review.get("records", [])
    audit_sha = sha256(originals.get("fresh_audit.json", b"")).hexdigest()
    bound = len(native) == len(actual) and all(actual.get(r["file"]) == r["file_sha256"] for r in native)
    bound = bound and review.get("fresh_audit_at_postprocessing", {}).get("sha256") == audit_sha
    bound = bound and all(valid_audit(r, audits) for r in records)
    bound = bound and review.get("source_epoch_unchanged") is True and review.get("controls_match_except_flags") is True
    bound = bound and review.get("all_native_succeeded") is True and all(r["success"] and r["alignment"] == 0 for r in records)
    if not bound:
        raise ValueError("Matrix postprocessing is not bound to the current rows and fresh audit")
    aggregate = review["aggregate"]
    counts = {v: (sum(r["local_pressure_peak_count"] for r in aggregate if r["variant"] == v),
                  sum(r["count"] for r in aggregate if r["variant"] == v)) for v in VARIANTS[study]}
    comparison = next(r for r in review["comparisons_to_original"] if r["variant"] == "stadtke_both")
    original_linear = [r for r in native if r["variant"] == "stadtke_original" and r["reconstruction"] == "muscl_van_leer"]
    peak = max(original_linear, key=lambda r: r["pressure_local_neighbor_peak_pct"])
    statements = [
        f"All {len(records)} saved solves and their recorded-epoch fresh original-equation/branch/identity checks pass in this unshifted production matrix.",
        f"Endpoint-only has {counts['stadtke_endpoint'][0]}/{counts['stadtke_endpoint'][1]} immediate-neighbor pressure peaks; both-on has {counts['stadtke_both'][0]}/{counts['stadtke_both'][1]}. This is a tested local-shape result, not a universal no-peak theorem.",
        f"Both-on has larger conservative-cell-projection pressure L1 error than the original in {comparison['pressure_cell_l1_worse']}/{comparison['matched_configurations']} matched configurations, and larger maximum total-enthalpy error in {comparison['h0_linf_worse']}/{comparison['matched_configurations']}. There is no blanket analytical-accuracy improvement.",
        f"The original MUSCL results contain one local pressure peak: {peak['pressure_local_neighbor_peak_pct']:.6g}% in {peak['case']} at N{peak['cells']}. The large shifted MUSCL example belongs to archived historical evidence, not this unshifted matrix.",
    ]
    details = []
    tradeoff = next((r for r in review.get("accuracy_tradeoffs", []) if r["variant"] == "stadtke_both"), None)
    if tradeoff:
        pressure = tradeoff["metrics"]["pressure_cell_l1"]
        enthalpy = tradeoff["metrics"]["h0_relative_linf_pct"]
        statements[2] += f" Median corrected/original error ratios are {pressure['ratio_all']['median']:.4g} for pressure and {enthalpy['ratio_all']['median']:.4g} for total enthalpy."
        details.append(f"Pressure conservative-cell L1 ratios span {pressure['ratio_all']['min']:.4g}–{pressure['ratio_all']['max']:.4g}; the median absolute change is {pressure['absolute_delta_all']['median']:.4g} Pa. Maximum relative total-enthalpy error ratios span {enthalpy['ratio_all']['min']:.4g}–{enthalpy['ratio_all']['max']:.4g}; the median change is {enthalpy['absolute_delta_all']['median']:.4g} percentage points.")
        p_example, h_example = pressure["largest_ratio_example"], enthalpy["largest_ratio_example"]
        details.append(f"Large ratios need an absolute scale: {p_example['case']}, N{p_example['cells']}, {p_example['reconstruction']}, pressure cell L1 changes from {p_example['original_error']:.6g} to {p_example['corrected_error']:.6g} Pa; maximum relative total-enthalpy error changes from {h_example['original_error']:.6g}% to {h_example['corrected_error']:.6g}%. These are deterministic matched errors, not statistical timing evidence.")
    result.update(verified=True, statements=statements, accuracy_details=details,
                  evidence=name, evidence_sha256=sha256(originals[name]).hexdigest())
    return result


def make_notes(records,audits,originals,profile,helpers):
    findings=(matrix_findings(profile["study"],records,audits,originals,helpers["valid_audit"]) if helpers["audit_complete"]
              else {"verified":False,"statements":[]})
    return {"findings":findings,"epoch_recovery":epoch_recovery(profile["study"],records,originals)}
