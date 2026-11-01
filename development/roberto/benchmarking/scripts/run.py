"""Execute versioned steady verification cases and repeatable backend timings."""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Thread
from time import perf_counter
from zoneinfo import ZoneInfo

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml

from nozzle_hyperbolic.verification import available_cases, load_case, build_case
from .common import STUDY, ROOT, clean, save, identity, provenance, source_snapshot, harness_snapshot
from .environment import inspect_environment, wait_until_idle


class Worker:
    def __init__(self, backend, folder, timeout):
        self.backend=backend;self.timeout=timeout
        folder.mkdir(parents=True,exist_ok=True)
        self.log=(folder/(backend+".log")).open("a",encoding="utf-8")
        self.process=subprocess.Popen([sys.executable,"-B","-m","development.roberto.benchmarking.scripts.worker","--backend",backend],
             cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
             env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1"))
        ready=self._read()
        if not ready.get("ready"):
            self.close();raise RuntimeError(f"Worker startup failed: {ready}")

    def _read(self):
        queue=Queue()
        Thread(target=lambda:queue.put(self.process.stdout.readline()),daemon=True).start()
        try:line=queue.get(timeout=self.timeout)
        except Empty:
            self.process.kill();self.process.wait(timeout=15)
            raise TimeoutError(f"{self.backend} complete solve exceeded {self.timeout}s; see worker log")
        if not line:raise RuntimeError(f"{self.backend} worker exited ({self.process.poll()}); see worker log")
        return json.loads(line)

    def ask(self, command):
        self.process.stdin.write(json.dumps(clean(command),allow_nan=False)+"\n");self.process.stdin.flush()
        result=self._read()
        if not result.get("ok"):
            error=result.get("traceback",result)
            if "UnsupportedLegacyCase" in str(error):raise NotImplementedError(error)
            raise RuntimeError(error)
        return result.get("result")

    def close(self):
        if getattr(self,"process",None) is not None and self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"shutdown"}\n');self.process.stdin.flush();self.process.wait(timeout=10)
            except Exception:
                self.process.kill();self.process.wait(timeout=10)
        if getattr(self,"log",None):self.log.close()


def error_norms(numeric, exact, weights, scale):
    numeric,exact=np.asarray(numeric,float),np.asarray(exact,float)
    difference=numeric-exact;weights=np.asarray(weights,float);weights=weights/weights.sum()
    l1=float(np.sum(weights*np.abs(difference)));l2=float(np.sqrt(np.sum(weights*difference*difference)))
    linf=float(np.max(np.abs(difference)));denominator=float(np.sum(weights*np.abs(exact)))
    active=denominator>max(float(scale),1e-300)*1e-14
    return dict(l1=l1,l2=l2,linf=linf,normalized_l1=l1/scale,normalized_l2=l2/scale,normalized_linf=linf/scale,
                relative_l1=l1/denominator if active else None,relative_active=active,scale=scale)


def assess(problem, result, backend, cells):
    x=np.asarray(result["x"]);volumes=np.asarray(result["volumes"]);fields={k:np.asarray(v) for k,v in result["fields"].items()}
    fields["velocity"]=fields["v_m"] if problem.definition.model_name=="euler_annular" else fields["u"]
    if problem.definition.model_name=="euler_annular":
        inlet_pressure=float(problem.reference_fields([float(problem.mesh.faces[0])])["p"][0])
        reservoir_pressure=float(problem.definition.data["inlet"]["p0"])
        fields["Cp"]=(fields["p"]-inlet_pressure)/(reservoir_pressure-inlet_pressure)
    # Match representation: current FV cells use exact conserved-volume
    # projection; the native nodal nozzle uses independent point values.
    projected=backend=="current" or problem.definition.model_name=="euler_annular"
    exact=problem.reference_cell_fields() if projected else problem.reference_fields(x)
    exact_q=problem.cell_reference()["q"] if projected else problem.reference_fields(x)["q"]
    norms={}
    for name,scale in problem.field_scales.items():
        if name in fields and name in exact and np.shape(fields[name])==np.shape(exact[name]):
            weights=np.asarray(problem.mesh.widths) if name=="Cp" else volumes
            norms[name]=error_norms(fields[name],exact[name],weights,float(scale))
    q=np.asarray(result["q"]);conservative={}
    for index,label in enumerate(problem.model.equation_names):
        conservative[label]=error_norms(q[:,index],exact_q[:,index],volumes,float(problem.model.conservative_scales[index]))
    max_conservative=max((item["relative_l1"] if item["relative_active"] else item["normalized_l1"] for item in conservative.values()),default=float("inf"))
    max_conservative_normalized=max((item["normalized_l1"] for item in conservative.values()),default=float("inf"))
    targets=problem.definition.data["accuracy"];checks={}
    if "normalized_field_l1" in targets:
        selected=[norms[name]["normalized_l1"] for name in ("p","rho","velocity","M","T") if name in norms]
        checks["normalized_field_l1"]=max(selected,default=float("inf"))<=targets["normalized_field_l1"]
    if "conservative_normalized_l1" in targets:
        checks["conservative_normalized_l1"]=max_conservative_normalized<targets["conservative_normalized_l1"]
    if "Cp_length_mean_absolute" in targets:
        checks["Cp_length_mean_absolute"]=norms.get("Cp",{}).get("l1",float("inf"))<=targets["Cp_length_mean_absolute"]
    shock={"expected":problem.shock_position,"computed":None,"error_m":None,"error_cells":None,"method":None}
    if problem.shock_position is not None:
        mach=np.asarray(fields.get("M_m",fields.get("M")))
        crossings=np.where((mach[:-1]>1)&(mach[1:]<=1))[0]
        if len(crossings):
            index=int(crossings[0]);position=float(.5*(x[index]+x[index+1]))
            shock["method"]="measured interior Mach crossing"
        elif abs(problem.shock_position-float(problem.mesh.faces[-1]))<=2*np.max(np.diff(problem.mesh.faces)) and mach[-1]>1:
            position=float(problem.mesh.faces[-1])
            shock["method"]="inferred outgoing boundary limit; no interior crossing"
        else:position=None
        if position is not None:
            error=abs(position-problem.shock_position);spacing=float(np.max(np.diff(problem.mesh.faces)))
            shock.update(computed=position,error_m=error,error_cells=error/spacing)
        checks["shock_position"]=shock["error_cells"] is not None and shock["error_cells"]<=targets.get("shock_position_cells",2.)
    finest=cells>=targets["recommended_finest_cells"]
    analytical=bool(all(checks.values()))
    mach=np.asarray(fields.get("M_m",fields.get("M")))
    boundaries=problem.definition.data["boundaries"]
    # Endpoint regimes belong to the physical traces (an exit shock can leave
    # the last physical cell supersonic). Use the model's trace/EOS audit.
    branch_checks=dict(forward_meridional=bool(np.all(fields["velocity"]>0)))
    if problem.shock_position is not None:
        branch_checks["shock_present"]=shock["computed"] is not None
    elif (problem.definition.model_name=="euler_annular" and boundaries["inlet"]=="subsonic") or problem.definition.data["reference"].get("condition") in ("throat_mach","critical"):
        branch_checks["subsonic_branch"]=bool(np.max(mach)<=1+1e-8)
    branch_checks["model_solution_audit"]=bool(problem.model.solution_audit(np.asarray(result["state"])))
    dense_x=np.linspace(float(problem.mesh.faces[0]),float(problem.mesh.faces[-1]),1001)
    if problem.shock_position is not None:
        epsilon=float(problem.mesh.faces[-1]-problem.mesh.faces[0])*1e-10
        dense_x=np.unique(np.r_[dense_x,max(dense_x[0],problem.shock_position-epsilon),min(dense_x[-1],problem.shock_position+epsilon)])
    dense=problem.reference_fields(dense_x)
    reference_fields={k:v for k,v in dense.items() if np.shape(v)==dense_x.shape and np.issubdtype(np.asarray(v).dtype,np.number)}
    return dict(errors=norms,conservative_errors=conservative,metrics=dict(result["metrics"],max_conservative_relative_l1=max_conservative,max_conservative_normalized_l1=max_conservative_normalized),
                analytical_passed=analytical,accuracy_assessed=finest,accuracy_checks=checks,accuracy_targets=targets,
                branch_passed=bool(all(branch_checks.values())),branch_checks=branch_checks,
                shock=shock,reference_representation="conservative finite-volume projection" if projected else "native coordinate point sampling",
                reference=dict(x=dense_x,fields=reference_fields),
                state=dict(x=x,fields=fields,full=result["state"],conserved_density=q),
                known_limit=problem.definition.data.get("known_limit"))


def stats(samples):
    values=np.asarray(samples,float)
    if not len(values):return dict(samples_s=[],mean_s=None,min_s=None,max_s=None,std_s=None)
    return dict(samples_s=values.tolist(),mean_s=float(values.mean()),min_s=float(values.min()),max_s=float(values.max()),std_s=float(values.std(ddof=1)) if len(values)>1 else 0.)


def config_key(case_id,reconstruction,cells,cap,backend,flux=None):
    """Retain schema-1 identities; schema 2 explicitly identifies every flux."""
    key=f'{case_id.replace("/","__")}__{reconstruction}__n{cells}__cap{cap}__{backend}'
    return key if flux is None else key+'__flux_'+flux


def flux_profile(config,flux):
    profile={key:deepcopy(config[key]) for key in ("scheme","steady","residual")}
    profile["scheme"]["flux"]=flux
    if flux=="roe" and "roe_options" in config:
        profile["roe_options"]=deepcopy(config["roe_options"])
    return profile


def resolved_fluxes(config):
    """Validate sweep identities before launching or resuming any workers."""
    if "fluxes" not in config:
        return [config["scheme"]["flux"]]
    if config.get("schema_version")!=2:
        raise ValueError("A flux sweep requires schema_version: 2")
    fluxes=config["fluxes"]
    if not fluxes or len(set(fluxes))!=len(fluxes) or any(f not in ("roe","rusanov","hll","characteristic") for f in fluxes):
        raise ValueError("Select distinct supported fluxes")
    if "flux" in config["scheme"]:
        raise ValueError("Sweep profiles select fluxes at top level; remove scheme.flux")
    if config["backends"]!=["current"]:
        raise ValueError("Flux sweeps use the current backend; keep native comparisons in separate schema-1 campaigns")
    return fluxes


def rotating_order(methods,index):
    if not methods:return []
    offset=index%len(methods)
    return methods[offset:]+methods[:offset]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=STUDY/"settings.yaml")
    parser.add_argument("--output",type=Path);parser.add_argument("--cases",nargs="+")
    parser.add_argument("--models",nargs="+",choices=("euler_nozzle","euler_annular"))
    parser.add_argument("--grids",nargs="+",type=int);parser.add_argument("--reconstructions",nargs="+")
    parser.add_argument("--fluxes",nargs="+",choices=("roe","rusanov","hll","characteristic"))
    parser.add_argument("--backends",nargs="+",choices=("current","legacy"));parser.add_argument("--caps",nargs="+",type=int)
    parser.add_argument("--repeats",type=int);parser.add_argument("--refinement-repeats",type=int)
    parser.add_argument("--timeout",type=float);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--wait-for-idle",action="store_true",help="Wait for 45 seconds without competing Python research/test jobs")
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for arg,key in ((args.grids,"grids"),(args.reconstructions,"reconstructions"),(args.backends,"backends"),(args.caps,"newton_caps"),(args.repeats,"repeats"),(args.refinement_repeats,"refinement_repeats"),(args.timeout,"complete_solve_timeout_s")):
        if arg is not None:config[key]=arg
    if args.fluxes:
        config.update(schema_version=2,fluxes=args.fluxes)
        config["scheme"].pop("flux",None)
    fluxes=resolved_fluxes(config);sweep="fluxes" in config
    cases=list(available_cases()) if config["cases"]=="all" else list(config["cases"])
    if args.cases:cases=args.cases
    if args.models:cases=[c for c in cases if c.split('/')[0] in args.models]
    if not cases or any(c not in available_cases() for c in cases):raise ValueError("Select known physical case IDs")
    grids=sorted(set(config["grids"]))
    if not grids or min(grids)<2 or config["repeats"]<1 or config["refinement_repeats"]<1:raise ValueError("Positive repeats and cell counts >=2 required")
    output=args.output or Path(config["output"])
    if not output.is_absolute():output=STUDY/output
    output=output.resolve()
    if not output.is_relative_to(STUDY.resolve()):raise ValueError("Campaign outputs must stay under benchmarking/")
    output.mkdir(parents=True,exist_ok=True);summary_path=output/'summary.json'
    protocol=dict(config,grids=grids,cases=cases,timing="Production public driver vs strict native-kernel controller; reusable bindings; JIT/setup/audits excluded; monitoring and rejected work included",
                  order="One excluded complete cold solve per configuration, then sequential rotating backend/flux order; configuration-block rotation offsets warmup and measured order",
                  initialization="Cold model-owned state; identical physical-cell values supplied by index; no analytical initialization or grid sequencing",
                  source_defaults="CFL 0.2, growth1.2, max1e6; Newton cap10",precision="float64")
    environment=wait_until_idle() if args.wait_for_idle else inspect_environment()
    if environment["status"]!="clean":
        raise RuntimeError("Timing environment is not clean; use --wait-for-idle or inspect: "+str(environment))
    run_provenance=provenance();protocol_sha=identity(protocol)
    definitions=[load_case(c) for c in cases]
    if args.resume and summary_path.exists():
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["protocol_sha256"]!=protocol_sha or summary["provenance"]["source"]["sha256"]!=run_provenance["source"]["sha256"] or summary["provenance"].get("harness",{}).get("sha256")!=run_provenance["harness"]["sha256"] or summary["provenance"]["legacy_manifest_sha256"]!=run_provenance["legacy_manifest_sha256"]:
            raise RuntimeError("Cannot resume across case/protocol/source/harness changes; use a new output directory")
        retained=[]
        for record in summary["results"]:
            previous=record.get("timing_environment",{})
            if isinstance(previous,dict) and previous.get("status")=="clean":retained.append(record)
            else:
                stamp=datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d_%H%M%S")
                filename='interrupted/'+record['id']+'__'+stamp+'.json'
                save(output/filename,record)
                summary.setdefault("environment_retries",[]).append(dict(id=record["id"],file=filename))
        summary["results"]=retained
    else:
        if summary_path.exists():raise FileExistsError("Existing campaign: use --resume or a new output directory")
        summary=dict(schema_version=config.get("schema_version",1),campaign_id=output.name,created_at=run_provenance["created_at"],completed=False,
                     protocol=protocol,protocol_sha256=protocol_sha,provenance=run_provenance,
                     cases=[dict(id=d.case_id,label=d.label,model=d.model_name,description=d.data.get("description"),
                                 source=d.source_name,source_sha256=d.source_sha256,definition=d.to_dict(),known_limit=d.data.get("known_limit")) for d in definitions],results=[])
    summary["timing_environment"]=environment
    completed={row["id"] for row in summary["results"]}
    started=perf_counter();save(summary_path,summary)
    block_index=0
    total=len(fluxes)*sum(len(config["reconstructions"])*len(config["newton_caps"])*(len(config["backends"]) if n==max(grids) or config.get("legacy_refinement") else int("current" in config["backends"])) for _ in cases for n in grids)
    for case_id in cases:
      for cells in grids:
        print(f'Prepare {case_id}, {cells} cells',flush=True)
        try:
            physical=build_case(case_id,cells);payload=physical.legacy_payload
        except Exception as error:
            definition=load_case(case_id)
            for reconstruction in config["reconstructions"]:
              for cap in config["newton_caps"]:
                selected=config["backends"] if cells==max(grids) or config.get("legacy_refinement") else [b for b in config["backends"] if b=="current"]
                for backend,flux in [(b,f) for b in selected for f in fluxes]:
                    key=config_key(case_id,reconstruction,cells,cap,backend,flux if sweep else None)
                    if key in completed:continue
                    record=dict(id=key,case_id=case_id,model=definition.model_name,label=definition.label,backend=backend,
                                flux=flux,reconstruction=reconstruction,cells=cells,cap=cap,status="failed",success=False,timing_valid=False,
                                reason="physical_case_setup_failed: "+repr(error),case_sha256=definition.source_sha256,timing=stats([]),timing_environment=environment)
                    save(output/'scenarios'/(key+'.json'),record);summary["results"].append(record);completed.add(key)
            save(summary_path,summary);print('  Physical setup failed:',repr(error),flush=True);continue
        # Compute projections outside timing, but retain a structured error if
        # an analytical branch/projection is unavailable on this mesh.
        try:physical.cell_reference()
        except Exception as error:print('  Reference projection unavailable:',repr(error),flush=True)
        for reconstruction in config["reconstructions"]:
          for cap in config["newton_caps"]:
            backends=config["backends"] if cells==max(grids) or config.get("legacy_refinement") else [b for b in config["backends"] if b=="current"]
            pending=[(b,f) for b in backends for f in fluxes if config_key(case_id,reconstruction,cells,cap,b,f if sweep else None) not in completed]
            pending=rotating_order(pending,block_index);block_index+=1
            if not pending:continue
            block_environment=inspect_environment()
            if block_environment["status"]=="contended" and args.wait_for_idle:
                block_environment=wait_until_idle()
            if block_environment["status"]!="clean":
                summary["timing_environment"]=block_environment;save(summary_path,summary)
                raise RuntimeError("Concurrent workload detected before timing; use --wait-for-idle --resume")
            print(f'[{len(completed)+1}/{total}] {case_id} n={cells} {reconstruction} cap={cap}: {pending}',flush=True)
            workers={};data={};failures={};metadata={}
            repetitions=config["repeats"] if cells==max(grids) else config["refinement_repeats"]
            try:
              for method in pending:
                backend,flux=method;profile=flux_profile(config,flux)
                try:
                  worker=Worker(backend,output/'logs'/config_key(case_id,reconstruction,cells,cap,backend,flux if sweep else None),config['complete_solve_timeout_s']);workers[method]=worker
                  bound=worker.ask(dict(command="bind",case_id=case_id,cells=cells,reconstruction=reconstruction,cap=cap,profile=profile,payload=payload));metadata[method]=bound["metadata"]
                  warm=worker.ask(dict(command="solve"));data[method]=dict(warmup=warm,samples=[])
                  print(f'  {backend}/{flux} warmup: {warm["success"]} {warm["reason"]}, {warm["elapsed_s"]:.3f}s, {warm["work"]["accepted_steps"]} steps',flush=True)
                except Exception as error:
                  failures[method]=dict(status="unsupported" if isinstance(error,NotImplementedError) else "timeout" if isinstance(error,TimeoutError) else "failed",reason=str(error))
                  print(f'  {backend}/{flux}: {failures[method]["status"]}: {str(error)[-350:]}',flush=True)
              for repeat in range(repetitions):
                order=rotating_order(pending,repeat)
                for method in order:
                  backend,flux=method
                  if method in failures or not data[method]["warmup"]["success"]:continue
                  try:
                    result=workers[method].ask(dict(command="solve"));data[method]["samples"].append(result)
                    print(f'  {backend}/{flux} repeat {repeat+1}/{repetitions}: {result["elapsed_s"]:.3f}s, {result["success"]}',flush=True)
                    if not result["success"]:failures[method]=dict(status="failed",reason="Timed solve failed: "+result["reason"])
                  except Exception as error:
                    failures[method]=dict(status="timeout" if isinstance(error,TimeoutError) else "failed",reason=str(error));print(f'  {backend}/{flux}: {failures[method]["status"]}',flush=True)
            finally:
              for worker in workers.values():worker.close()
            after_environment=inspect_environment()
            environment_clean=block_environment["status"]=="clean" and after_environment["status"]=="clean"
            block_environment=dict(status="clean" if environment_clean else after_environment["status"],before=block_environment,after=after_environment)
            for method in pending:
                backend,flux=method;profile=flux_profile(config,flux)
                key=config_key(case_id,reconstruction,cells,cap,backend,flux if sweep else None)
                record=dict(id=key,case_id=case_id,model=physical.definition.model_name,label=physical.definition.label,backend=backend,
                            flux=flux,reconstruction=reconstruction,cells=cells,cap=cap,settings=dict(profile,cap=cap,reconstruction=reconstruction),
                            numerical_profile_sha256=identity(dict(profile,cap=cap,reconstruction=reconstruction)),
                            case_sha256=physical.definition.source_sha256,machine_sha256=run_provenance["machine_sha256"],
                            source_sha256=run_provenance["source"]["sha256"],metadata=metadata.get(method,{}),provenance=run_provenance,timing_environment=block_environment)
                pack=data.get(method);samples=pack["samples"] if pack else []
                if pack:
                    selected=sorted(samples,key=lambda r:r["elapsed_s"])[len(samples)//2] if samples else pack["warmup"]
                    try:
                        assessment=assess(physical,selected,backend,cells)
                    except Exception as error:
                        assessment=dict(errors={},conservative_errors={},metrics=selected["metrics"],analytical_passed=False,
                          accuracy_assessed=cells>=physical.definition.data["accuracy"]["recommended_finest_cells"],accuracy_checks={},
                          accuracy_targets=physical.definition.data["accuracy"],branch_passed=False,branch_checks={},
                          state=dict(x=selected["x"],fields=selected["fields"],full=selected["state"]),reference={},
                          assessment_error=repr(error),known_limit=physical.definition.data.get("known_limit"))
                    passed=bool(selected["success"] and len(samples)==repetitions and all(s["success"] for s in samples))
                    record.update(assessment,success=passed,reason=selected["reason"],work=selected["work"],history=selected["history"],roe_diagnostics=selected.get("roe_diagnostics"),
                                  warmup=dict(elapsed_s=pack["warmup"]["elapsed_s"],success=pack["warmup"]["success"],reason=pack["warmup"]["reason"],work=pack["warmup"]["work"]),
                                  timing=dict(stats([s["elapsed_s"] for s in samples]),warmup_s=pack["warmup"]["elapsed_s"],expected_samples=repetitions),
                                  failure_diagnostics=[dict(repeat=i+1,**s) for i,s in enumerate(samples) if not s["success"]],
                                  samples=[dict(elapsed_s=s["elapsed_s"],success=s["success"],reason=s["reason"],work=s["work"],metrics=s["metrics"],state_sha256=sha256(np.asarray(s["state"],float).tobytes()).hexdigest()) for s in samples])
                    record["status"]="passed" if passed else "failed"
                    if passed and record["accuracy_assessed"] and not record["analytical_passed"]:
                        record["status"]="accuracy_failed"
                    if passed and not record["branch_passed"]:
                        record["status"]="branch_failed"
                    if record.get("assessment_error"):
                        record["status"]="assessment_failed";record["reason"]="assessment_failed: "+record["assessment_error"]
                    record["timing_valid"]=passed and record["branch_passed"] and (not record["accuracy_assessed"] or record["analytical_passed"])
                else:record.update(success=False,status="failed",timing=stats([]),timing_valid=False,reason="No completed solve")
                if method in failures:record.update(failures[method],success=False,timing_valid=False)
                if not environment_clean:record["timing_valid"]=False
                save(output/'scenarios'/(key+'.json'),record)
                # The index includes complete representative data so it remains
                # a standalone input to the offline report and future baselines.
                summary["results"].append(clean(record));completed.add(key)
                summary.update(elapsed_s=perf_counter()-started,completed=False)
                save(summary_path,summary)
            if not environment_clean:
                summary["timing_environment"]=block_environment
                summary["interrupted_reason"]="Competing workload detected after sample block; block retained but excluded. Resume when idle."
                save(summary_path,summary)
                raise RuntimeError(summary["interrupted_reason"])
    summary.update(completed=True,elapsed_s=perf_counter()-started,source_unchanged=source_snapshot()["sha256"]==run_provenance["source"]["sha256"],
                   harness_unchanged=harness_snapshot()["sha256"]==run_provenance["harness"]["sha256"])
    if not summary["source_unchanged"] or not summary["harness_unchanged"]:
        for record in summary["results"]:record["timing_valid"]=False
    summary["counts"]={status:sum(r["status"]==status for r in summary["results"]) for status in sorted({r["status"] for r in summary["results"]})}
    save(summary_path,summary)
    print('Campaign complete:',summary["counts"],'source unchanged:',summary["source_unchanged"],flush=True)

if __name__=="__main__":
    main()
