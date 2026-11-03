"""Sequential, checkpointed production solver and fixed-work campaign."""

import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
os.environ["MPLBACKEND"]="Agg"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import yaml
from .common import ROOT,STUDY,save,identity,provenance,source_snapshot,runtime_snapshot
from .runtime_support import WorkerProtocol as SharedWorker,assess,stats,rotating_order
from .environment import inspect_environment,wait_until_idle
from nozzle_hyperbolic.verification import build_case,load_case


class Worker(SharedWorker):
    def __init__(self,folder,timeout):
        self.backend="production";self.timeout=timeout
        folder.mkdir(parents=True,exist_ok=True)
        self.log=(folder/"worker.log").open("a",encoding="utf-8")
        self.process=subprocess.Popen([sys.executable,"-B","-m","development.time_benchmarking.scripts.worker"],
            cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            env=dict(os.environ,PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1"))
        ready=self._read()
        if not ready.get("ready"):self.close();raise RuntimeError(str(ready))


def freeze(output,summary):
    names=set(summary["provenance"]["source"]["files"])|set(summary["provenance"]["harness"]["files"])
    names.update(d["path"] for d in summary["cases"])
    names.update(name for name in ("pyproject.toml","poetry.lock") if (ROOT/name).is_file())
    target=output/"source_and_cases.zip"
    with ZipFile(target,"w",ZIP_DEFLATED) as archive:
        for name in sorted(names):archive.write(ROOT/name,name)
        archive.writestr("campaign_protocol.json",json.dumps(summary["protocol"],indent=2))
    summary["source_archive"]=dict(file=target.name,sha256=sha256(target.read_bytes()).hexdigest())


def compact_stage(stage):
    keys=("cells","success","reason","metrics","work","elapsed_s","stage_total_s","initialization",
          "cfl_policy","initial_cfl","last_accepted_cfl","residual_normalization")
    value={key:stage[key] for key in keys}
    value["failed_warm_attempts"]=[{key:attempt[key] for key in ("reason","elapsed_s","metrics","initialization","initial_cfl")} for attempt in stage["attempts"]]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--cases",nargs="+")
    parser.add_argument("--reconstructions",nargs="+")
    parser.add_argument("--methods",nargs="+")
    parser.add_argument("--repeats",type=int)
    parser.add_argument("--kernel-repeats",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--wait-for-idle",action="store_true")
    args=parser.parse_args()
    config=yaml.safe_load((STUDY/"settings.yaml").read_text(encoding="utf-8"))
    if args.cases:config["cases"]=args.cases
    if args.reconstructions:config["reconstructions"]=args.reconstructions
    if args.methods:
        config["methods"]=[m for m in config["methods"] if m["id"] in args.methods]
        if len(config["methods"])!=len(args.methods):raise ValueError("Unknown method")
    if args.repeats is not None:config["repeats"]={name:args.repeats for name in config["repeats"]}
    if args.kernel_repeats is not None:config["kernel_repeats"]=args.kernel_repeats
    if min(config["repeats"].values())<1 or config["kernel_repeats"]<0:raise ValueError("Invalid repeats")
    output=(args.output or STUDY/config["output"]).resolve()
    if not output.is_relative_to(STUDY):raise ValueError("Outputs must stay inside this study")
    output.mkdir(parents=True,exist_ok=True)
    environment=wait_until_idle() if args.wait_for_idle else inspect_environment()
    if environment["status"]!="clean":raise RuntimeError(str(environment))
    prov=provenance()
    protocol=dict(config,precision="float64",timing="Reusable production bindings; compilation/setup/reference/serialization excluded; full hierarchy, interpolation, rejected work and cold retries included",
        multigrid="Production nested-grid initialization, not FAS/defect correction",initialization="Cold physical guess on first grid; numerical componentwise linear prolongation only within each hierarchy",
        monitors="Original equation, boundary, global and model audits enabled; no optional scalar quantity monitors in any method",
        order="One excluded complete hierarchy per method, rotating method order for measured repetitions; fresh numerical/controller state every solve",
        kernels="Converged (or last failed) fine state; dt=0.2*wave_dt; three excluded calls, 20 checked Newton corrections, zero early-convergence tolerances")
    summary_path=output/"summary.json"
    if summary_path.exists():
        if not args.resume:raise FileExistsError("Use --resume or another output")
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["protocol_sha256"]!=identity(protocol) or any(summary["provenance"][key]["sha256"]!=prov[key]["sha256"] for key in ("source","harness")) or summary["provenance"]["machine_sha256"]!=prov["machine_sha256"]:
            raise ValueError("Cannot resume with different source/runtime/protocol/machine")
    else:
        cases=[]
        for name in config["cases"]:
            definition=load_case(name)
            # Definition identities and exact files are archived beside source.
            files=[p for p in (ROOT/"src/nozzle_hyperbolic").rglob("*.yaml") if sha256(p.read_bytes()).hexdigest()==definition.source_sha256]
            if len(files)!=1:raise ValueError("Cannot locate unique case file: "+name)
            cases.append(dict(id=name,path=files[0].relative_to(ROOT).as_posix(),sha256=definition.source_sha256))
        summary=dict(schema_version=1,campaign_id=output.name,created_at=prov["created_at"],completed=False,
            protocol=protocol,protocol_sha256=identity(protocol),provenance=prov,cases=cases,results=[],blocks=[],environment_start=environment)
        freeze(output,summary);save(summary_path,summary)
    completed={b["id"] for b in summary["blocks"] if b["status"]=="complete"}
    for case_index,case in enumerate(config["cases"]):
        for recon_index,recon in enumerate(config["reconstructions"]):
            block_id=case.replace("/","__")+"__"+recon
            if block_id in completed:continue
            # An interrupted block is rerun into a fresh attempt folder; prior
            # artifacts remain intact and are never mixed into new statistics.
            prior=sum(b["id"]==block_id for b in summary["blocks"])
            folder=output/"raw"/(block_id+f"__attempt{prior+1}")
            folder.mkdir(parents=True,exist_ok=True)
            before=inspect_environment()
            if before["status"]!="clean":raise RuntimeError("Competing job: "+str(before))
            block=dict(id=block_id,status="running",folder=folder.relative_to(output).as_posix(),environment_before=before)
            summary["blocks"].append(block);save(summary_path,summary)
            problem=build_case(case,400)
            methods=config["methods"];ids=[m["id"] for m in methods]
            entries={key:dict(warmups=[],samples=[],failures=[]) for key in ids}
            worker=None;warmed=set();generation=0
            def ask(kind,method):
                nonlocal worker,generation,warmed
                if worker is None:
                    generation+=1;worker=Worker(folder,config["complete_solve_timeout_s"]);warmed=set()
                    binding=worker.ask(dict(command="bind",config=config,case_id=case,reconstruction=recon))
                    save(folder/f"binding_{generation}.json",binding)
                if kind!="solve_warmup" and method not in warmed:
                    warm=worker.ask(dict(command="solve",method=method))
                    file=folder/f"{method}__warmup_g{generation}.json";save(file,warm)
                    entries[method]["warmups"].append(dict(file=file.relative_to(output).as_posix(),elapsed_s=warm["elapsed_s"],success=warm["success"]))
                    warmed.add(method)
                value=worker.ask(dict(command="solve" if kind=="solve_warmup" else kind,method=method))
                if kind=="solve_warmup":warmed.add(method)
                return value
            def failure(method,phase,error):
                nonlocal worker,warmed
                entries[method]["failures"].append(dict(phase=phase,error=str(error)))
                print(f"FAILED {block_id} {method} {phase}: {str(error)[-600:]}",flush=True)
                if worker is not None:worker.close();worker=None;warmed=set()
            try:
                for key in rotating_order(ids,case_index+recon_index):
                    print(f"WARM {block_id} {key}",flush=True)
                    try:
                        value=ask("solve_warmup",key)
                        file=folder/f"{key}__warmup_g{generation}.json";save(file,value)
                        entries[key]["warmups"].append(dict(file=file.relative_to(output).as_posix(),elapsed_s=value["elapsed_s"],success=value["success"]))
                    except Exception as error:failure(key,"warmup",error)
                count=config["repeats"][case.split("/")[0]]
                for repeat in range(count):
                    for key in rotating_order(ids,repeat+case_index+recon_index):
                        if entries[key]["failures"]:continue
                        try:
                            value=ask("solve",key)
                            file=folder/f"{key}__repeat{repeat+1}.json";save(file,value)
                            entries[key]["samples"].append(dict(file=file.relative_to(output).as_posix(),sha256=sha256(file.read_bytes()).hexdigest(),
                                elapsed_s=value["elapsed_s"],fine_total_s=value["fine_total_s"],coarse_total_s=value["coarse_total_s"],success=value["success"],
                                all_stages_success=value["all_stages_success"],fallback_count=value["fallback_count"],total_work=value["total_work"],stages=[compact_stage(s) for s in value["stages"]]))
                            print(f"SAMPLE {block_id} {key} {repeat+1}/{count} {value['elapsed_s']:.3f}s {value['reason']} fallbacks={value['fallback_count']}",flush=True)
                        except Exception as error:failure(key,"sample",error)
                for key in ids:
                    if not entries[key]["samples"]:continue
                    try:entries[key]["kernel_metadata"]=ask("prepare_kernels",key);entries[key]["kernel_samples"]=[]
                    except Exception as error:failure(key,"prepare_kernels",error)
                for repeat in range(config["kernel_repeats"]):
                    for key in rotating_order(ids,repeat):
                        if "kernel_samples" not in entries[key] or entries[key]["failures"]:continue
                        try:entries[key]["kernel_samples"].append(ask("sample_kernels",key))
                        except Exception as error:failure(key,"sample_kernels",error)
                after=inspect_environment()
                unchanged=source_snapshot()["sha256"]==prov["source"]["sha256"] and runtime_snapshot()["sha256"]==prov["harness"]["sha256"]
                for method in methods:
                    key=method["id"];entry=entries[key]
                    record=dict(id=block_id+"__"+key,case_id=case,reconstruction=recon,method=method,cells=400,**entry,
                        timing=stats([s["elapsed_s"] for s in entry["samples"]]),fine_timing=stats([s["fine_total_s"] for s in entry["samples"]]),
                        timing_environment=dict(before=before,after=after,status="clean" if before["status"]==after["status"]=="clean" else "unverified"),source_unchanged=unchanged)
                    if entry["samples"]:
                        ordered=sorted(entry["samples"],key=lambda s:s["elapsed_s"]);selected=ordered[len(ordered)//2]
                        result=json.loads((output/selected["file"]).read_text(encoding="utf-8"))
                        record.update(representative_file=selected["file"],success=result["success"],reason=result["reason"],
                            all_stages_success=result["all_stages_success"],fallback_count=result["fallback_count"],
                            assessment=assess(problem,result["fine"],"current",400),fine_history=result["fine"]["history"],
                            representative_stages=[compact_stage(s) for s in result["stages"]])
                        reference_state=np.asarray(result["fine"]["state"])
                        scales=np.asarray(problem.model.variable_scales)
                        deltas=[]
                        for sample in entry["samples"]:
                            repeated=json.loads((output/sample["file"]).read_text(encoding="utf-8"))
                            deltas.append(float(np.max(np.abs(np.asarray(repeated["fine"]["state"])-reference_state)/scales)))
                        record["repeat_consistency"]=dict(max_variable_scaled_state_difference=max(deltas),
                            all_fine_success=all(s["success"] for s in entry["samples"]),
                            all_hierarchies_success=all(s["all_stages_success"] for s in entry["samples"]),
                            intended_repeats=count,actual_repeats=len(entry["samples"]))
                    if entry.get("kernel_samples"):
                        record["kernel_timing"]={name:stats([s[name] for s in entry["kernel_samples"]]) for name in entry["kernel_samples"][0]}
                    summary["results"]=[r for r in summary["results"] if r["id"]!=record["id"]]+[record]
                block.update(status="complete",environment_after=after,source_unchanged=unchanged)
                save(folder/"checkpoint.json",entries);save(summary_path,summary)
                print(f"BLOCK COMPLETE {block_id}",flush=True)
                if after["status"]!="clean" or not unchanged:raise RuntimeError("Environment/source changed; retained block marked explicitly")
            finally:
                if worker is not None:worker.close()
    summary["completed"]=True;summary["environment_end"]=inspect_environment();save(summary_path,summary)
    print(f"CAMPAIGN COMPLETE {summary_path}",flush=True)


if __name__=="__main__":main()
