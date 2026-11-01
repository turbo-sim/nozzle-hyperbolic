"""Run the selected steady matrix once per cold configuration, without timing.

This client reuses the maintained worker, assessment and physical cases. It
does not require an idle machine and never produces performance samples.
"""

import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[_name]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import yaml

from development.roberto.benchmarking.scripts.run import Worker, assess, flux_profile, config_key, resolved_fluxes
from development.roberto.benchmarking.scripts.common import ROOT, clean, save, identity, provenance, source_snapshot, harness_snapshot
from development.roberto.benchmarking.scripts.audit import _residual_acceptance
from nozzle_hyperbolic.verification import available_cases, load_case, build_case
from .audit_steady import audit_verification, artifact_claim_issues

STUDY=Path(__file__).resolve().parents[1]


def client_snapshot():
    files={name:sha256((STUDY/"scripts"/name).read_bytes()).hexdigest() for name in ("verify_steady.py","audit_steady.py")}
    return dict(files=files,sha256=identity(files))


def assert_roe_revision(audit,source):
    checked=[]
    for filename,digest in audit["source"]["fingerprints_sha256"].items():
        normalized="/"+filename.replace("\\","/").lstrip("/");marker="/src/nozzle_hyperbolic/"
        if marker not in normalized:continue
        relative="src/nozzle_hyperbolic/"+normalized.split(marker,1)[1]
        if source["files"].get(relative)!=digest:raise RuntimeError("Roe audit does not match source: "+relative)
        checked.append(relative)
    if not checked or audit.get("source",{}).get("files_changed_during_audit"):
        raise RuntimeError("Roe audit source identity unavailable/changed")
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=STUDY/"verification_profile.yaml")
    parser.add_argument("--output",type=Path,default=STUDY/"outputs/roe_steady_verification")
    parser.add_argument("--roe-audit",type=Path,default=STUDY/"outputs/roe_audit.json")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cases",nargs="+");parser.add_argument("--grids",nargs="+",type=int)
    parser.add_argument("--reconstructions",nargs="+");args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8"));fluxes=resolved_fluxes(config)
    cases=args.cases or (list(available_cases()) if config["cases"]=="all" else config["cases"])
    grids=sorted(set(args.grids or config["grids"]));reconstructions=args.reconstructions or config["reconstructions"]
    if not cases or any(case not in available_cases() for case in cases):raise ValueError("Unknown physical case")
    output=args.output.resolve()
    if not output.is_relative_to((STUDY/"outputs").resolve()):raise ValueError("Verification outputs must stay under development/flux_methods/roe/outputs")
    output.mkdir(parents=True,exist_ok=True);summary_path=output/"summary.json"
    protocol=dict(config,cases=cases,grids=grids,reconstructions=reconstructions,repeats=1,refinement_repeats=1,output=str(output),
                  verification_only=True,verification_solves_per_configuration=1,excluded_warmup_solves=0,
                  timing="NOT MEASURED: one complete cold compiled solve; elapsed retained only as a load-dependent diagnostic",
                  initialization="Independent model-owned cold initial state; no analytical guess, multigrid or grid sequencing",
                  precision="float64",complete_solve_timeout_s=config["complete_solve_timeout_s"])
    revision=provenance();client=client_snapshot();audit_raw=args.roe_audit.read_bytes();audit=json.loads(audit_raw)
    audit_match=assert_roe_revision(audit,revision["source"])
    print("Verified Roe audit source fingerprints:",len(audit_match),"production files",flush=True)
    protocol_sha=identity(protocol)
    if summary_path.exists():
        if not args.resume:raise FileExistsError("Use --resume or a new verification output directory")
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary["protocol_sha256"]!=protocol_sha or summary["provenance"]["source"]["sha256"]!=revision["source"]["sha256"]
            or summary["provenance"]["harness"]["sha256"]!=revision["harness"]["sha256"]
            or summary["verification_client"]["sha256"]!=client["sha256"]
            or summary["provenance"]["machine_sha256"]!=revision["machine_sha256"]
            or summary["roe_audit_sha256"]!=sha256(audit_raw).hexdigest()):
            raise RuntimeError("Resume requires identical protocol/source/harness/client/Roe-audit identities")
        for row in summary["results"]:
            failures=artifact_claim_issues(row,output)
            if failures:raise RuntimeError("Cannot trust retained configuration "+row["id"]+": "+"; ".join(failures))
    else:
        definitions=[load_case(case) for case in cases]
        summary=dict(schema_version=2,campaign_id=output.name,title="Untimed four-flux steady verification",
                     description="Numerical-only matrix; performance timings were not collected",created_at=revision["created_at"],
                     completed=False,verification_only=True,protocol=protocol,protocol_sha256=protocol_sha,
                     provenance=revision,verification_client=client,roe_audit_sha256=sha256(audit_raw).hexdigest(),roe_audit_source_files_verified=audit_match,
                     timing_environment=dict(status="not_required",reason="Verification-only; concurrent workload permitted; elapsed diagnostics must not be interpreted as timings"),
                     cases=[dict(id=d.case_id,label=d.label,model=d.model_name,source=d.source_name,source_sha256=d.source_sha256,
                                 description=d.data.get("description"),definition=d.to_dict(),known_limit=d.data.get("known_limit")) for d in definitions],results=[])
    completed={row["id"] for row in summary["results"]};total=len(cases)*len(grids)*len(reconstructions)*len(config["newton_caps"])*len(fluxes)
    worker=None;started=perf_counter();save(summary_path,summary)
    try:
      for case_id in cases:
       for cells in grids:
        physical=None;setup_error=None
        try:physical=build_case(case_id,cells)
        except Exception as error:setup_error=repr(error)
        definition=load_case(case_id)
        for reconstruction in reconstructions:
         for cap in config["newton_caps"]:
          for flux in fluxes:
            key=config_key(case_id,reconstruction,cells,cap,"current",flux)
            if key in completed:continue
            if source_snapshot()["sha256"]!=revision["source"]["sha256"] or harness_snapshot()["sha256"]!=revision["harness"]["sha256"]:
                raise RuntimeError("Source/harness changed during verification; checkpoint retained; use a separate campaign")
            profile=flux_profile(config,flux)
            record=dict(id=key,case_id=case_id,model=definition.model_name,label=definition.label,backend="current",flux=flux,
                        reconstruction=reconstruction,cells=cells,cap=cap,verification_only=True,
                        settings=dict(profile,cap=cap,reconstruction=reconstruction),numerical_profile_sha256=identity(dict(profile,cap=cap,reconstruction=reconstruction)),
                        case_sha256=definition.source_sha256,source_sha256=revision["source"]["sha256"],harness_sha256=revision["harness"]["sha256"],
                        machine_sha256=revision["machine_sha256"],timing_valid=False,
                        timing_environment=summary["timing_environment"],timing=dict(samples_s=[],mean_s=None,min_s=None,max_s=None,std_s=None,expected_samples=0),samples=[],metadata={})
            print(f'[{len(completed)+1}/{total}] {case_id} n={cells} {reconstruction} {flux}: cold verification',flush=True)
            result=None
            try:
                if setup_error:raise RuntimeError("physical_case_setup_failed: "+setup_error)
                if worker is None:worker=Worker("current",output/"logs",config["complete_solve_timeout_s"])
                bound=worker.ask(dict(command="bind",case_id=case_id,cells=cells,reconstruction=reconstruction,cap=cap,profile=profile,payload=physical.legacy_payload))
                record["metadata"]=bound["metadata"]
                result=worker.ask(dict(command="solve"))
                raw_file="raw/"+key+".json";save(output/raw_file,result)
                record.update(raw_result_file=raw_file,raw_result_sha256=sha256((output/raw_file).read_bytes()).hexdigest(),
                              success=bool(result["success"]),reason=result["reason"],metrics=result["metrics"],work=result["work"],history=result["history"],
                              roe_diagnostics=result.get("roe_diagnostics"),diagnostic_elapsed_s=result["elapsed_s"],diagnostic_elapsed_includes_compilation=True)
                try:record.update(assess(physical,result,"current",cells))
                except Exception as error:
                    record.update(assessment_error=repr(error),branch_passed=False,analytical_passed=False,
                                  accuracy_assessed=cells>=definition.data["accuracy"]["recommended_finest_cells"],
                                  state=dict(x=result["x"],fields=result["fields"],full=result["state"]),errors={},conservative_errors={},reference={})
                monitor=_residual_acceptance(record["metrics"],config["residual"])
                record["independent_convergence"]=monitor
                record["status"]="passed" if record["success"] and monitor["passed"] else "failed"
                if record["status"]=="passed" and record.get("accuracy_assessed") and not record.get("analytical_passed"):record["status"]="accuracy_failed"
                if record["success"] and not record.get("branch_passed"):record["status"]="branch_failed"
                if record.get("assessment_error"):record["status"]="assessment_failed"
                print(f'  {record["status"]}: {record["reason"]}; steps={record["work"]["accepted_steps"]}, Newton={record["work"]["newton_corrections"]}; timing excluded',flush=True)
            except Exception as error:
                record.update(success=False,status="timeout" if isinstance(error,TimeoutError) else "failed",reason=str(error))
                print("  "+record["status"]+": "+str(error)[-450:],flush=True)
                if worker is not None:worker.close();worker=None
            filename="scenarios/"+key+".json";save(output/filename,record)
            index={name:value for name,value in record.items() if name not in ("state","reference","history","errors","conservative_errors")}
            index.update(file=filename,scenario_file=filename,scenario_sha256=sha256((output/filename).read_bytes()).hexdigest())
            summary["results"].append(clean(index));completed.add(key)
            summary.update(completed=False,configurations_expected=total,counts=dict(Counter(row["status"] for row in summary["results"])),
                           source_unchanged=True,harness_unchanged=True,diagnostic_campaign_elapsed_s=perf_counter()-started)
            save(summary_path,summary)
    finally:
        if worker is not None:worker.close()
        summary.update(source_unchanged=source_snapshot()["sha256"]==revision["source"]["sha256"],harness_unchanged=harness_snapshot()["sha256"]==revision["harness"]["sha256"])
        save(summary_path,summary)
    summary.update(completed=len(completed)==total,diagnostic_campaign_elapsed_s=perf_counter()-started)
    save(summary_path,summary)
    result=audit_verification(summary,output);save(output/"numerical_audit.json",result)
    print("Verification complete:",len(completed),"/",total,summary["counts"],"evidence audit:",result["passed"],"timings collected: 0",flush=True)


if __name__=="__main__":main()
