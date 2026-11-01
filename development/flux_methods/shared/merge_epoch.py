"""Admit byte-verified isolated completion without changing working source."""
import json
from pathlib import Path
import shutil
from .common import ROOT,FAMILY,save,digest,source_hashes,utc


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def copy_exact(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    assert digest(source)==digest(target)


def main():
    evidence=FAMILY/"shared/evidence"
    manifest_path=evidence/"source_isolation_manifest.json"
    manifest=read(manifest_path)
    isolated=Path(manifest["isolated_root"])
    destination=FAMILY/"osher/outputs/verification"
    origin=isolated/"development/flux_methods/osher/outputs/verification"
    summary,audit,settings=[read(origin/name) for name in ("summary.json","fresh_audit.json","settings.json")]
    worker_path=isolated/"verified_workers/final_audit.json"
    worker=read(worker_path)
    expected=manifest["deployed_files"]
    checks=dict(
        all_flat_src_client_resource_marker_bytes_unchanged=all(digest(isolated/name)==sha for name,sha in expected.items()),
        full_native_population=len(summary["records"])==summary["requested_count"]==128 and summary["completed"],
        full_fresh_population=len(audit["records"])==128 and audit["complete"] and audit["audit_valid"],
        all_native_outcomes_freshly_confirmed=all(r["saved_success_agrees"] for r in audit["records"]),
        actual_numerical_worker_verified=worker["passed"] and worker["before_after_deployed_hashes_identical"],
        worker_manifest_sha_bound=worker["source_isolation_manifest_sha256"]==digest(manifest_path),
        original_settings_byte_identical=digest(origin/"settings.json")==digest(destination/"settings.json"),
        same_recorded_epoch=settings["source_sha256"]==manifest["source_sha256"] and settings["implementation_sha256"]==manifest["implementation_sha256"] and settings["case_sha256"]==manifest["case_sha256"],
        frozen_source_and_cases_unchanged=summary["source_unchanged"] and summary["cases_unchanged"],
    )
    records={r["file"]:r for r in summary["records"]}
    checks["original_115_records_unchanged"]=all(file in records and records[file]["sha256"]==sha
        and digest(destination/file)==digest(origin/file)==sha for file,sha in manifest["original_row_identities"].items())
    checks["all_isolated_row_sha_match"]=all(digest(origin/r["file"])==r["sha256"] for r in records.values())
    fresh={r["file"]:r for r in audit["records"]}
    checks["all_fresh_row_sha_join"]=all(file in fresh and fresh[file]["file_sha256"]==r["sha256"] for file,r in records.items())
    if not all(checks.values()):
        raise RuntimeError("Isolated completion not admitted: "+repr([k for k,v in checks.items() if not v]))
    preserved=destination/"interrupted_source_epoch"
    preserved.mkdir(exist_ok=True)
    for name in ("summary.json","fresh_audit.json"):
        target=preserved/("before_isolation_"+name)
        if not target.exists():
            copy_exact(destination/name,target)
    admitted=[]
    for file,row in records.items():
        target=destination/file
        if target.exists():
            assert digest(target)==row["sha256"]
        else:
            copy_exact(origin/file,target)
            admitted.append(dict(file=file,sha256=row["sha256"]))
    for name in ("summary.json","fresh_audit.json"):
        copy_exact(origin/name,destination/name)
    for name in ("isolated_completion.log","isolated_full_audit.log","isolated_final_audit.log"):
        path=origin.parent/name
        if path.exists():
            copy_exact(path,destination.parent/name)
    copy_exact(worker_path,evidence/"source_isolation_worker.json")
    closed=[]
    for checkpoint in (destination/"checkpoints").glob("*.json"):
        saved=preserved/checkpoint.name
        if saved.exists() and digest(saved)==digest(checkpoint):
            closed.append(dict(file=checkpoint.relative_to(ROOT).as_posix(),preserved=saved.relative_to(ROOT).as_posix(),sha256=digest(saved)))
            checkpoint.unlink()  # One exact completed/interrupted checkpoint; its identical preserved copy remains.
    current=source_hashes()
    drift={name:dict(recorded=manifest["source_sha256"].get(name),current=current.get(name))
           for name in manifest["source_sha256"].keys()|current.keys() if manifest["source_sha256"].get(name)!=current.get(name)}
    result=dict(category="byte_verified_recorded_epoch_completion",passed=True,utc=utc(),checks=checks,
         source_sha256=manifest["source_sha256"],implementation_sha256=manifest["implementation_sha256"],case_sha256=manifest["case_sha256"],
         isolated_root=str(isolated),source_isolation_manifest_sha256=digest(manifest_path),
         guard_interruption_sha256=digest(evidence/"source_guard_interruption.json"),
         actual_worker_proof_sha256=digest(evidence/"source_isolation_worker.json"),
         original_immutable_rows=manifest["original_rows_copied"],new_rows_admitted=admitted,
         native_successes=sum(r["success"] for r in summary["records"]),
         original_records_rewritten=False,canonical_summary_sha256=digest(destination/"summary.json"),
         canonical_fresh_audit_sha256=digest(destination/"fresh_audit.json"),
         closed_interrupted_checkpoints=closed,current_checkout_drift=drift,current_working_source_modified=False,
         interpretation="All128 rows and fresh checks use the same recorded source/client/case bytes. The concurrent Roe edit is preserved; source_unchanged flags refer to the recorded frozen workspace, not the later editable checkout.")
    save(evidence/"source_isolation_completion.json",result)
    print("Verified isolated completion admitted:",len(admitted),"new rows; original rows unchanged")


if __name__=="__main__":
    main()
