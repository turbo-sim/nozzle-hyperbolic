"""Deploy recorded source bytes to an isolated, ordinary flat-package workspace.

No working source, environment, import path or numerical client is modified.
Python's normal current-directory package discovery selects the deployed
package; an identical src tree is retained for the frozen client fingerprints.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from .common import ROOT,FAMILY,save,digest,identity,source_hashes,utc


def main():
    original=FAMILY/"osher/outputs/verification"
    settings=json.loads((original/"settings.json").read_text(encoding="utf-8"))
    summary=json.loads((original/"summary.json").read_text(encoding="utf-8"))
    epoch=identity(settings["source_sha256"])[:12]
    # Keep the deployment root short enough for Windows' nested snapshot paths.
    isolated=ROOT.parents[1]/("frozen_flux_"+epoch)
    if not isolated.resolve().is_relative_to(ROOT.parents[1].resolve()):
        raise ValueError("Isolated destination must remain in the authorized projects root")
    if isolated.exists():
        raise ValueError("Use a new destination or the already prepared workspace; never overwrite it")
    isolated.mkdir(parents=True)
    snapshot=original/"implementation_snapshot"
    deployed={}
    for name,sha in settings["source_sha256"].items():
        source=snapshot/name
        if digest(source)!=sha:
            raise RuntimeError("Recorded source snapshot mismatch: "+name)
        for relative in (name,name.removeprefix("src/")):
            target=isolated/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            if digest(target)!=sha:
                raise RuntimeError("Deployed source mismatch: "+relative)
            deployed[relative]=sha
    for case,sha in settings["case_sha256"].items():
        model,name=case.split("/")
        source=snapshot/"cases"/(case+".yaml")
        if digest(source)!=sha:
            raise RuntimeError("Recorded physical resource mismatch: "+case)
        for prefix in ("src/nozzle_hyperbolic","nozzle_hyperbolic"):
            target=isolated/prefix/"models"/model/"cases"/(name+".yaml")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            deployed[target.relative_to(isolated).as_posix()]=sha
    for name,sha in settings["implementation_sha256"].items():
        source=snapshot/"client"/name
        if digest(source)!=sha:
            raise RuntimeError("Recorded numerical client mismatch: "+name)
        target=isolated/"development/flux_methods/shared"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        deployed[target.relative_to(isolated).as_posix()]=sha
    markers=("development/__init__.py","development/flux_methods/__init__.py",
             "development/flux_methods/shared/__init__.py","development/flux_methods/osher/__init__.py")
    for name in markers:
        target=isolated/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        deployed[name]=digest(target)
    profile=isolated/"development/flux_methods/osher/verification_profile.yaml"
    shutil.copyfile(original/"profile.yaml",profile)
    destination=isolated/"development/flux_methods/osher/outputs/verification"
    shutil.copytree(original,destination)
    for row in summary["records"]:
        if digest(destination/row["file"])!=row["sha256"]:
            raise RuntimeError("Copied immutable row mismatch: "+row["file"])
    current=source_hashes()
    drift={name:dict(recorded=sha,current=current.get(name)) for name,sha in settings["source_sha256"].items() if current.get(name)!=sha}
    manifest=dict(category="recorded_source_epoch_isolation",prepared_utc=utc(),isolated_root=isolated,
         original_campaign=original,campaign_identity=settings["campaign_identity"],source_sha256=settings["source_sha256"],
         case_sha256=settings["case_sha256"],implementation_sha256=settings["implementation_sha256"],
         deployed_files=deployed,all_deployed_hashes_match=True,original_rows_copied=len(summary["records"]),
         original_row_identities={r["file"]:r["sha256"] for r in summary["records"]},
         current_checkout_drift=drift,working_source_modified=False,environment_modified=False,
         import_path_modified=False,selection="Normal Python -m invocation from the isolated flat-package working directory",
         guard_interruption="The all-source byte guard refused admission of a row after a concurrent Roe edit. The previous checkpoint/log remain preserved; this is not a native convergence failure.")
    save(FAMILY/"shared/evidence/source_isolation_manifest.json",manifest)
    print(isolated)


if __name__=="__main__":
    main()
