"""Read-only final joins of maintained production-study evidence."""
from __future__ import annotations
import json
from pathlib import Path
from hashlib import sha256
from .common import ROOT,FAMILY,save,digest,identity,source_hashes,case_hashes,client_hashes,utc


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    checks={}
    evidence=FAMILY/"shared/evidence"
    epoch=read(FAMILY/"stadtke/outputs/verification/settings.json")
    source,physical,client=epoch["source_sha256"],epoch["case_sha256"],epoch["implementation_sha256"]
    current_source,current_physical,current_client=source_hashes(),case_hashes(),client_hashes()
    current_drift={name:dict(recorded=source.get(name),current=current_source.get(name))
                   for name in source.keys()|current_source.keys() if source.get(name)!=current_source.get(name)}
    tests=read(evidence/"production_tests_summary.json")
    checks["production_suite"]=tests["exit_code"]==0 and tests["failed"]==tests["errors"]==0
    checks["production_test_artifacts_intact"]=all(digest(evidence/name)==sha for name,sha in tests["artifacts_sha256"].items())
    checks["tested_python_source_matches_recorded_epoch"]={name:sha for name,sha in tests["files_sha256"].items()
        if name.startswith("src/") and name.endswith(".py")}==source
    checks["tested_selected_case_resources_match_recorded_epoch"]=all(
        tests["files_sha256"]["src/nozzle_hyperbolic/models/"+case.split("/")[0]+"/cases/"+case.split("/")[1]+".yaml"]==sha
        for case,sha in physical.items())
    tested_current_drift={name:dict(tested=sha,current=digest(ROOT/name) if (ROOT/name).is_file() else None)
                          for name,sha in tests["files_sha256"].items() if not (ROOT/name).is_file() or digest(ROOT/name)!=sha}
    focused=read(evidence/"current_focus_tests_summary.json")
    checks["current_focused_supplement_passed_stably"]=focused["exit_code"]==0 and focused["failed"]==focused["errors"]==0 and focused["input_files_unchanged"]
    checks["current_focus_artifacts_intact"]=all(digest(evidence/name)==sha for name,sha in focused["artifacts_sha256"].items())
    api=read(evidence/"production_api_audit.json")
    checks["api_audit_recorded_source_and_cases"]=api["passed"] and api["source_sha256"]==source and api["case_sha256"]==physical
    roe=read(FAMILY/"roe/outputs/final_report_audit/public_name_migration.json")
    checks["roe_names_updated_without_rewriting_evidence"]=bool(roe["passed"] and roe["original_archive_entries_preserved"]
         and roe["original_file_bytes_preserved"])
    if current_drift:
        recovery=read(evidence/"source_isolation_completion.json")
        checks["source_drift_recovered_with_frozen_epoch"]=recovery["passed"] and recovery["source_sha256"]==source
    studies={}
    for study in ("stadtke","osher"):
        directory=FAMILY/study/"outputs/verification"
        settings,summary,audit=[read(directory/name) for name in ("settings.json","summary.json","fresh_audit.json")]
        signature=identity(dict(config=settings["profile"],source=settings["source_sha256"],client=settings["implementation_sha256"],
                                physical=settings["case_sha256"],alignment=settings["alignment"]))
        keys=lambda r:(r["case"],r["cells"],r["reconstruction"],r["variant"])
        requested={keys(r) for r in summary["requested"]}
        population=summary["records"]
        fresh={r["file"]:r for r in audit["records"]}
        report=FAMILY/study/"REPORT.html"
        structure,logic=[read(directory/name) for name in ("report_audit.json","report_logic_audit.json")]
        local={
            "exact_requested_population":len(population)==len(requested)==128 and {keys(r) for r in population}==requested,
            "campaign_complete":summary["completed"] and summary["final"],
            "campaign_identity":settings["campaign_identity"]==summary["campaign_identity"]==signature,
            "same_recorded_source_client_cases":settings["source_sha256"]==source and settings["implementation_sha256"]==client and settings["case_sha256"]==physical,
            "recorded_source_snapshots_intact":all(digest(directory/"implementation_snapshot"/name)==sha for name,sha in source.items()),
            "source_and_cases_unchanged":summary["source_unchanged"] and summary["cases_unchanged"],
            "native_outcomes_agree_with_fresh_checks":all(r["file"] in fresh and fresh[r["file"]].get("saved_success_agrees") is True for r in population),
            "all_saved_row_hashes_match":all(digest(directory/r["file"])==r["sha256"] for r in population),
            "full_fresh_audit":len(fresh)==128 and audit["complete"] and audit["audit_valid"],
            "native_successes_freshly_confirmed":all(not r["success"] or fresh.get(r["file"],{}).get("all_checks_passed") is True for r in population),
            "all_fresh_hash_joins":all(r["file"] in fresh and fresh[r["file"]]["file_sha256"]==r["sha256"] for r in population),
            "all_snapshots_match":all(audit[k] is True for k in ("snapshot_source_match","snapshot_client_match","snapshot_case_match")),
            "report_structure":structure["passed"] and structure["record_count"]==128 and structure["report_sha256"]==digest(report),
            "report_ui_logic":logic["passed"] and logic["record_count"]==128 and logic["report_sha256"]==digest(report),
        }
        checks.update({study+"."+name:bool(value) for name,value in local.items()})
        studies[study]=dict(native_count=len(population),fresh_count=len(fresh),native_successes=sum(r["success"] for r in population),
                            summary_sha256=digest(directory/"summary.json"),fresh_audit_sha256=digest(directory/"fresh_audit.json"),
                            report_sha256=digest(report),browser_rendering_verified=False,checks=local)
    result=dict(utc=utc(),passed=all(checks.values()),checks=checks,studies=studies,
                source_sha256=source,case_sha256=physical,numerical_client_sha256=client,
                current_checkout_matches_recorded_source=not current_drift,current_checkout_source_drift=current_drift,
                tested_inputs_current_drift=tested_current_drift,
                current_cases_match_recorded_epoch=current_physical==physical,current_numerical_clients_match_recorded_epoch=current_client==client,
                production_tests=dict(passed=tests["passed"],expected_failures=tests["xfailed"]),
                current_focused_tests=dict(passed=focused["passed"],full_current_suite_validated=False,
                      inputs_still_match=all((ROOT/name).is_file() and digest(ROOT/name)==sha for name,sha in focused["files_sha256"].items())),
                limitations=["Browser/static rendering unavailable; standalone structure and actual inline UI logic checked.",
                             "Root convergence and eliminated measured neighbor peaks do not establish universal monotonicity or all-field accuracy improvement.",
                             "Verification elapsed diagnostics are not matched repeated performance timings."],
                interpretation="No flow advanced and no file removed. Checks validate the recorded study/test source epoch, not later concurrent checkout edits. Completed source-backed studies are the maintained execution paths; the retired prototype archive is no longer required.")
    save(evidence/"delivery_audit.json",result)
    print(json.dumps(dict(passed=result["passed"],failed=[k for k,v in checks.items() if not v]),indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
