"""Audit saved numerical-only steady evidence without advancing a solver."""

from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path

from development.historical.benchmarking.scripts.audit import expected_configurations, _residual_acceptance
from development.historical.benchmarking.scripts.common import identity, source_snapshot, harness_snapshot, save


def artifact_claim_issues(row,directory):
    """Compare index claims with their independently hashed complete records."""
    directory=Path(directory).resolve();issues=[];loaded={}
    for name in ("raw_result","scenario"):
        filename=row.get(name+"_file");digest=row.get(name+"_sha256")
        if not filename:
            if name=="raw_result" and not row.get("success"):continue
            issues.append(name+" is missing");continue
        path=(directory/filename).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            issues.append(name+" path is unsafe/missing");continue
        raw=path.read_bytes()
        if sha256(raw).hexdigest()!=digest:
            issues.append(name+" SHA256 differs");continue
        try:loaded[name]=json.loads(raw)
        except (ValueError,TypeError):issues.append(name+" is invalid JSON")
    scenario=loaded.get("scenario");raw=loaded.get("raw_result")
    if scenario is not None:
        for key,value in row.items():
            if key in ("file","scenario_file","scenario_sha256"):continue
            if scenario.get(key)!=value:issues.append("Index/scenario differ: "+key)
    if raw is not None and scenario is not None:
        for key in ("success","reason","work","history","roe_diagnostics"):
            if scenario.get(key)!=raw.get(key):issues.append("Scenario/raw differ: "+key)
        for key,value in raw.get("metrics",{}).items():
            if scenario.get("metrics",{}).get(key)!=value:issues.append("Scenario/raw metrics differ: "+key)
        if scenario.get("state",{}).get("full")!=raw.get("state"):
            issues.append("Scenario/raw full state differs")
    return issues


def audit_verification(campaign, directory, *, compare_live=True):
    directory=Path(directory).resolve();issues=[];records=[]
    protocol=campaign.get("protocol",{});provenance=campaign.get("provenance",{})
    expected=expected_configurations(protocol);rows=campaign.get("results",[])
    cases={item["id"]:item for item in campaign.get("cases",[])}
    counts=Counter(row.get("id") for row in rows)
    coverage_complete=bool(campaign.get("completed")) and set(counts)==set(expected) and all(n==1 for n in counts.values())
    def issue(check,detail,record=None):
        issues.append(dict(check=check,detail=detail,record=record))
    if campaign.get("verification_only") is not True or protocol.get("verification_only") is not True:
        issue("verification_only","Campaign must explicitly exclude performance timing")
    if identity(protocol)!=campaign.get("protocol_sha256"):
        issue("protocol_identity","Saved resolved protocol does not match its hash")
    if not campaign.get("completed"):
        issue("coverage","Campaign is incomplete")
    if set(counts)!=set(expected) or any(n!=1 for n in counts.values()):
        issue("coverage",dict(missing=sorted(set(expected)-set(counts)),unexpected=sorted(set(counts)-set(expected)),duplicates=[key for key,n in counts.items() if n!=1]))
    for name in ("source","harness"):
        snapshot=provenance.get(name,{})
        if identity(snapshot.get("files",{}))!=snapshot.get("sha256"):
            issue(name+"_identity","Invalid saved file-map hash")
        if campaign.get(name+"_unchanged") is not True:
            issue(name+"_unchanged","Not verified unchanged during the numerical campaign")
        if compare_live and (source_snapshot() if name=="source" else harness_snapshot())["sha256"]!=snapshot.get("sha256"):
            issue("live_"+name,"Saved revision differs from the current checkout")
    for row in rows:
        key=row.get("id");requested=expected.get(key,{})
        for name in ("case_id","cells","reconstruction","cap","backend","flux"):
            if row.get(name)!=requested.get(name):issue("configuration",name+" differs from protocol",key)
        if row.get("timing_valid") is not False or row.get("samples") or row.get("timing",{}).get("samples_s") or row.get("timing",{}).get("mean_s") is not None:
            issue("no_timing","Numerical-only records must not contain eligible timing samples",key)
        if row.get("source_sha256")!=provenance.get("source",{}).get("sha256") or row.get("harness_sha256")!=provenance.get("harness",{}).get("sha256"):
            issue("record_source_identity","Record revision differs from campaign",key)
        if row.get("case_sha256")!=cases.get(row.get("case_id"),{}).get("source_sha256"):
            issue("record_case_identity","Record case differs from campaign resource",key)
        for detail in artifact_claim_issues(row,directory):issue("artifact_claim",detail,key)
        metrics=row.get("metrics",{});monitor=_residual_acceptance(metrics,protocol["residual"])
        if row.get("success") and not monitor["passed"]:
            issue("convergence_flag","Success differs from independent saved spatial/BC/global assessment",key)
        metadata=row.get("metadata",{})
        if metadata and metadata.get("flux")!=row.get("flux"):
            issue("bound_flux","Bound flux differs from label",key)
        if row.get("flux")=="roe" and row.get("success"):
            for name,value in protocol.get("roe_options",{}).items():
                if (metadata.get("roe") or {}).get(name)!=value:issue("roe_policy",name+" differs",key)
            if (row.get("roe_diagnostics") or {}).get("invalid_faces")!=0:
                issue("roe_faces","Successful Roe result has invalid/missing final face audit",key)
        accuracy_assessed=row.get("accuracy_assessed") is True
        if row.get("accuracy_checks") and row.get("analytical_passed") is not all(row["accuracy_checks"].values()):
            issue("accuracy_flag","Analytical flag differs from saved independent component checks",key)
        if row.get("branch_checks") and row.get("branch_passed") is not all(row["branch_checks"].values()):
            issue("branch_flag","Branch flag differs from saved independent component checks",key)
        accepted=bool(row.get("success") and monitor["passed"] and row.get("branch_passed") is True
                      and (not accuracy_assessed or row.get("analytical_passed") is True))
        records.append(dict(id=key,status=row.get("status"),original_residual_passed=monitor["passed"],
                            nonlinear_success=row.get("success") is True,branch_passed=row.get("branch_passed") is True,
                            accuracy_assessed=accuracy_assessed,analytical_passed=row.get("analytical_passed") is True,
                            numerical_and_analytical_accepted=accepted))
    return dict(schema_version=1,scope="Untimed numerical-only coverage, saved original spatial/BC/global norms, branch/accuracy flags, identities and artifact bytes; no fresh PDE solve",
                verification_only=True,passed=not issues,complete_requested_coverage=coverage_complete,
                numerical_and_analytical_acceptance=coverage_complete and not issues and bool(records) and all(r["numerical_and_analytical_accepted"] for r in records),
                configurations_expected=len(expected),configurations_retained=len(rows),timings_collected=0,
                counts=dict(statuses=dict(Counter(row.get("status") for row in rows)),
                            nonlinear_success=sum(r["nonlinear_success"] for r in records),
                            original_residual_passed=sum(r["original_residual_passed"] for r in records),
                            branch_passed=sum(r["branch_passed"] for r in records),
                            accuracy_assessed=sum(r["accuracy_assessed"] for r in records),
                            analytical_passed_at_required_resolution=sum(r["accuracy_assessed"] and r["analytical_passed"] for r in records)),issues=issues,records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    result=audit_verification(json.loads(args.input.read_text(encoding="utf-8")),args.input.parent)
    save(args.output or args.input.with_name("numerical_audit.json"),result)
    print("Numerical evidence integrity:",result["passed"],"acceptance:",result["numerical_and_analytical_acceptance"],result["counts"])
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
