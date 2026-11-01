"""Rebuild source Operators on saved states; no flow solves or tolerance changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import traceback
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from nozzle_hyperbolic.solvers.fvm import Operator, Scheme
from .common import ROOT,save,digest,source_hashes,case_hashes,case_resources,client_hashes,study_path,utc
from .assessment import shifted_problem,independent_check
from .branch_metrics import local_shock_metrics
from .migration import OriginArchive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    campaign=(args.input or study_path(args.study)/"outputs"/"verification").resolve()
    summary=json.loads((campaign/"summary.json").read_text(encoding="utf-8"))
    derived=summary.get("record_type")=="api_migration_view"
    origin=OriginArchive(campaign) if derived else None
    destination=campaign/"fresh_audit.json"
    source,client=source_hashes(),client_hashes()
    previous={}
    if args.resume and destination.exists():
        old=json.loads(destination.read_text(encoding="utf-8"))
        if old.get("evaluation_source_sha256",old.get("source_sha256"))==source and old.get("audit_implementation_sha256")==client:
            previous={r["file"]:r for r in old["records"]}
    snapshot = campaign/("evaluation_snapshot" if derived else "implementation_snapshot")
    if derived and not snapshot.exists():
        for name,sha in source.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if digest(target)!=sha:raise RuntimeError("Evaluation source snapshot differs")
        for name,sha in client.items():
            target=snapshot/"client"/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(__file__).parent/name,target)
            if digest(target)!=sha:raise RuntimeError("Evaluation client snapshot differs")
        for case,resource in case_resources().items():
            target=snapshot/"cases"/(case+".yaml");target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(resource.read_bytes())
    snapshot_source_match = all((snapshot/name).is_file() and digest(snapshot/name)==sha for name,sha in source.items())
    snapshot_client_match = all((snapshot/"client"/name).is_file() and digest(snapshot/"client"/name)==sha for name,sha in client.items())
    physical=case_hashes()
    snapshot_case_match=all((snapshot/"cases"/(case+".yaml")).is_file() and digest(snapshot/"cases"/(case+".yaml"))==sha for case,sha in physical.items())
    records=[]
    epoch_stable=True

    def write_audit():
        identity_keys=("evaluation_source_hashes_match","evaluation_clients_hashes_match","case_hashes_match","initial_state_matches_model_cold")
        if derived:
            identity_keys+=("recorded_source_matches_origin","numeric_payload_matches_origin","configuration_matches_origin","operator_equivalent_at_saved_state")
        else:
            identity_keys+=("source_hashes_match","numerical_clients_match")
        invalid=[r["file"] for r in records if r.get("error")
                 or any(r.get(k) is not True for k in identity_keys)
                 or r.get("saved_success_agrees") is not True]
        audit_valid=bool(epoch_stable and snapshot_source_match and snapshot_client_match and snapshot_case_match and not invalid)
        save(destination,dict(schema_version=2,study=args.study,updated_utc=utc(),records=records,
             source_sha256=source,audit_implementation_sha256=client,
             evaluation_source_sha256=source,evaluation_clients_sha256=client,
             record_type="api_migration_evaluation" if derived else "native_saved_state_evaluation",
             new_native_solves=0,required_identity_flags=identity_keys,
             evaluation_epoch_stable=epoch_stable,
             snapshot_source_match=snapshot_source_match,snapshot_client_match=snapshot_client_match,
             snapshot_case_match=snapshot_case_match,case_sha256=physical,
             complete=len(records)==len(summary["records"]),audit_valid=audit_valid,
             invalid_audit_files=invalid,
             description="Fresh source Operators; independent NumPy fixed-scale stopping reductions, branch and path admissibility. Audit validity distinguishes faithful checks of a failed solve from an audit/provenance failure. No solves."))
        return audit_valid

    for item in summary["records"]:
        path=campaign/item["file"]
        sha=digest(path)
        if sha!=item["sha256"]:
            raise ValueError("Saved row hash mismatch: "+item["file"])
        if item["file"] in previous and previous[item["file"]]["file_sha256"]==sha:
            records.append(previous[item["file"]]);continue
        row=json.loads(path.read_text(encoding="utf-8"))
        result=dict(file=item["file"],file_sha256=sha,case=row["case"],cells=row["cells"],
                    variant=row["variant"],reconstruction=row["reconstruction"],
                    saved_success=row["success"],saved_reason=row["reason"],
                    evaluation_source_hashes_match=snapshot_source_match,
                    evaluation_clients_hashes_match=snapshot_client_match)
        if not derived:
            result["source_hashes_match"]=row["source_sha256"]==source
            result["numerical_clients_match"]=row["implementation_sha256"]==client
        else:
            result.update(origin.verify(row))
        try:
            problem=shifted_problem(row["case"],row["cells"],row.get("alignment",0.))
            result["case_hashes_match"]=row["case_sha256"]==problem.definition.source_sha256
            result["initial_state_matches_model_cold"]=bool(np.array_equal(row["initial_state"],problem.initial_state))
            operator=Operator(problem.model,Scheme(**row["scheme"]))
            state=jnp.asarray(row["state"],dtype=jnp.float64)
            result.update(independent_check(operator,state,row["initial_state"],row["settings"]["residual"]))
            result["local_shock_metrics"]=local_shock_metrics(problem,np.asarray(state))
            result["fresh_vs_saved"]={}
            equivalence=[]
            for name,value in (("raw_spatial_residual",operator.spatial_residual(state)),
                               ("face_flux",operator.face_flux(state)),
                               ("boundary_residual",operator.boundary_residual(state,0.))):
                if name in row:
                    difference=abs(np.asarray(value)-np.asarray(row[name]))
                    result["fresh_vs_saved"][name+"_absolute_max"]=float(np.max(difference))
                    scale=1.0 if name=="boundary_residual" else np.asarray(problem.model.global_flux_scales)
                    scaled=float(np.max(difference/scale))
                    result["fresh_vs_saved"][name+"_scaled_max"]=scaled
                    equivalence.append(np.isfinite(scaled) and scaled<=1e-12)
            if derived:
                result["operator_equivalent_at_saved_state"]=bool(equivalence and all(equivalence))
            result["saved_success_agrees"]=row["success"]==result["all_checks_passed"]
        except Exception as error:
            result["error"]=dict(message=repr(error),traceback=traceback.format_exc())
        records.append(result)
        write_audit()
        print(f'AUDIT {item["file"]}: {result.get("all_checks_passed",False)}',flush=True)
        jax.clear_caches()
    print(destination,flush=True)
    if origin is not None:
        origin.close()
    epoch_stable=source_hashes()==source and client_hashes()==client and case_hashes()==physical
    if not epoch_stable:
        write_audit()
        raise RuntimeError("Evaluation source/client/case epoch changed; checks are not a stable current evaluation")
    if not write_audit():
        raise SystemExit("Audit/provenance failure; inspect invalid_audit_files and snapshot flags")


if __name__=="__main__":
    main()
