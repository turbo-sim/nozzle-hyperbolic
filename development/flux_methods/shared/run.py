"""Execute a declarative, source-backed flux study; retain every outcome."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from importlib.metadata import version
import shutil
from time import perf_counter
import traceback

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import yaml

from nozzle_hyperbolic.solvers.fvm import Scheme, SteadySettings, Operator, solve_single_grid
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from .common import CASES, ROOT, clean, save, digest, identity, source_hashes, case_hashes, case_resources, client_hashes, git_context, study_path, utc
from .assessment import shifted_problem, assess, independent_check

def configuration_scheme(config, reconstruction, variant):
    """Resolve one named configuration; labels never select numerical methods."""
    options = dict(config["scheme"])
    options.update(config["configurations"][variant])
    options["reconstruction"] = reconstruction
    return Scheme(**options)


def serialized_scheme(scheme):
    options = asdict(scheme)
    if scheme.flux == "osher":
        # Options owned by another flux family do not describe this calculation.
        options.pop("custom_jump_correction", None)
        options.pop("custom_endpoint_envelope", None)
    return options


class StudyBudget(Exception):
    """Declared study budget at a completed driver callback, not convergence."""


def work(history):
    return dict(accepted_steps=sum(r.get("phase")=="accepted" for r in history),
                rejected_attempts=sum(r.get("phase")=="rejected" for r in history),
                newton_corrections=sum(r.get("inner_iterations",0) for r in history),
                rejected_newton_trials=sum(r.get("rejected_newton_trials",0) for r in history))


def run_one(study, config, case, cells, reconstruction, variant, campaign, source, client, physical, alignment=0.):
    scheme = configuration_scheme(config,reconstruction,variant)
    settings = SteadySettings(**config["steady"],residual=ResidualPolicy(**config["residual"]))
    row_id = f'{case.replace("/","__")}__n{cells}__{reconstruction}__{variant}__a{alignment:g}'
    destination = campaign/"rows"/(row_id+".json")
    problem = shifted_problem(case,cells,alignment)
    if problem.definition.source_sha256 != physical[case]:
        raise RuntimeError("Physical case definition changed before solving: "+case)
    initial = problem.initial_state.copy()
    history = []
    snapshot = initial.copy()
    start = perf_counter()
    checkpoint_time = start
    latest_metrics = {}
    checkpoint = campaign/"checkpoints"/(row_id+".json")

    def capture(point, face_flux):
        nonlocal snapshot
        snapshot = np.asarray(point).copy()
        return {}

    def progress(row):
        nonlocal checkpoint_time, latest_metrics
        history.append(clean(row))
        if row["accepted"]:
            latest_metrics = clean(row)
        elapsed = perf_counter()-start
        if perf_counter()-checkpoint_time >= 30 or row.get("phase")=="initial":
            save(checkpoint,dict(id=row_id,updated_utc=utc(),state=snapshot,history=history,
                 source_sha256=source,implementation_sha256=client,elapsed_with_compile_s=elapsed))
            checkpoint_time = perf_counter()
        native_done = all(row.get(k,False) for k in ("finite","admissible","all_equations_passed","boundary_passed","global_passed"))
        if elapsed > config["study_wall_budget_s"] and not native_done:
            raise StudyBudget("Declared study wall budget exceeded; last accepted state retained")

    operator = None
    exception = None
    try:
        solution = solve_single_grid(problem.model,scheme,settings,initial_state=initial,
                                     quantity_evaluator=capture,callback=progress)
        state, success, reason, metrics = solution.state,solution.success,solution.reason,solution.metrics
        operator = solution.operator
    except Exception as error:
        state, success = snapshot,False
        reason = "study_wall_budget" if isinstance(error,StudyBudget) else "exception"
        metrics = latest_metrics
        exception = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    elapsed = perf_counter()-start
    row = dict(id=row_id,study=study,case=case,cells=cells,reconstruction=reconstruction,
               variant=variant,method=scheme.flux,flux=scheme.flux,
               schema_version=2,record_type="native_solve",
               quadrature_points=scheme.quadrature_points,integration=scheme.integration,
               scheme=serialized_scheme(scheme),settings=asdict(settings),initialization="cold",alignment=alignment,
               source_sha256=source,implementation_sha256=client,
               case_sha256=problem.definition.source_sha256,case_definition=asdict(problem.definition),
               initial_state=initial,state=state,success=success,reason=reason,metrics=metrics,
               history=history,work=work(history),elapsed_with_compile_s=elapsed,
               elapsed_definition="Includes compile, driver monitoring, common accepted-state checkpoint capture; excludes final assessment. Not a timing benchmark.",
               completed_utc=utc(),exception=exception)
    try:
        if operator is None:
            operator = Operator(problem.model,scheme)
        row.update(assess(problem,np.asarray(state),operator))
        row["independent_check"] = independent_check(operator,state,initial,asdict(settings.residual))
    except Exception as error:
        row["assessment_error"] = dict(message=repr(error),traceback=traceback.format_exc())
    if source_hashes()!=source or client_hashes()!=client or case_hashes()!=physical:
        raise RuntimeError("Source or numerical study client changed while solving; record not admitted to frozen campaign")
    save(destination,row)
    if checkpoint.exists():
        checkpoint.unlink()
    print(f'{row_id}: {success} {reason}; {row["work"]}; {elapsed:.2f}s diagnostic',flush=True)
    return dict(id=row_id,file=destination.relative_to(campaign).as_posix(),sha256=digest(destination),
                case=case,cells=cells,reconstruction=reconstruction,variant=variant,success=success,reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cells",type=int,nargs="+")
    parser.add_argument("--cases",choices=CASES,nargs="+")
    parser.add_argument("--reconstructions",choices=("constant","muscl_van_leer"),nargs="+")
    parser.add_argument("--alignment",type=float,default=0.)
    args = parser.parse_args()
    study = study_path(args.study)
    config_path = args.config or study/"verification_profile.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants=config.get("variants",[])
    if (config["cases"] != list(CASES) or not variants or len(variants)!=len(set(variants))
        or set(variants)!=set(config.get("configurations",{}))):
        raise ValueError("Declare the four physical cases and one unique parameter mapping per configuration")
    for variant in variants:
        for reconstruction in config["reconstructions"]:
            configuration_scheme(config,reconstruction,variant)
    campaign = (args.output or study/"outputs"/"verification").resolve()
    if not campaign.is_relative_to(study.resolve()):
        raise ValueError("Study outputs must remain inside the chosen study directory")
    source,client,physical = source_hashes(),client_hashes(),case_hashes()
    requested = [(case,cells,rec,v) for case in config["cases"] for cells in config["grids"]
                 for rec in config["reconstructions"] for v in config["variants"]]
    signature = identity(dict(config=config,source=source,client=client,physical=physical,alignment=args.alignment))
    settings_path = campaign/"settings.json"
    summary_path = campaign/"summary.json"
    records = []
    if settings_path.exists():
        old = json.loads(settings_path.read_text(encoding="utf-8"))
        if not args.resume or old["campaign_identity"]!=signature:
            raise ValueError("Existing campaign: --resume requires identical profile, source, clients and alignment")
        if summary_path.exists():
            records = json.loads(summary_path.read_text(encoding="utf-8"))["records"]
            for record in records:
                if digest(campaign/record["file"]) != record["sha256"]:
                    raise ValueError("Saved row hash mismatch: "+record["file"])
    elif campaign.exists() and any(campaign.iterdir()):
        raise ValueError("Nonempty output folder has no valid settings manifest")
    else:
        for name,sha in source.items():
            target=campaign/"implementation_snapshot"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if digest(target)!=sha:
                raise RuntimeError("Source snapshot mismatch: "+name)
        for name,sha in client.items():
            target=campaign/"implementation_snapshot"/"client"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(__file__).parent/name,target)
            if digest(target)!=sha:
                raise RuntimeError("Client snapshot mismatch: "+name)
        for case,resource in case_resources().items():
            target=campaign/"implementation_snapshot"/"cases"/(case+".yaml")
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(resource.read_bytes())
            if digest(target)!=physical[case]:
                raise RuntimeError("Physical-case snapshot mismatch: "+case)
        shutil.copyfile(config_path,campaign/"profile.yaml")
        save(settings_path,dict(schema_version=2,record_type="native_campaign",study=args.study,started_utc=utc(),profile=config,profile_path=config_path,
             profile_sha256=digest(config_path),campaign_identity=signature,source_sha256=source,
             implementation_sha256=client,git=git_context(),alignment=args.alignment,
             case_sha256=physical,
             python=__import__("sys").version,jax_version=jax.__version__,dtype="float64",
             packages={name:version(name) for name in ("numpy","jax","jaxlib","jaxprop","nurbspy")},
             backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
             monitoring="Same extra face-flux call at initial/accepted state for checkpoint capture; no physical monitor stopping rule"))

    def summarize():
        complete = len(records)==len(requested)
        save(summary_path,dict(schema_version=2,record_type="native_campaign",study=args.study,campaign_identity=signature,records=records,
             requested=[dict(case=c,cells=n,reconstruction=r,variant=v) for c,n,r,v in requested],
             requested_count=len(requested),completed_count=len(records),completed=complete,final=complete,
             source_before=source,source_after=source_hashes(),source_unchanged=source==source_hashes(),
             case_before=physical,case_after=case_hashes(),cases_unchanged=physical==case_hashes(),
             implementation_sha256=client,updated_utc=utc(),alignment=args.alignment))

    summarize()
    done = {(r["case"],r["cells"],r["reconstruction"],r["variant"]) for r in records}
    for case,cells,rec,variant in requested:
        if ((case,cells,rec,variant) in done or (args.cells and cells not in args.cells)
            or (args.cases and case not in args.cases) or (args.reconstructions and rec not in args.reconstructions)):
            continue
        if source_hashes()!=source or client_hashes()!=client or case_hashes()!=physical:
            raise RuntimeError("Frozen numerical identity changed between solves")
        records.append(run_one(args.study,config,case,cells,rec,variant,campaign,source,client,physical,args.alignment))
        summarize()
        jax.clear_caches()
    print(summary_path,flush=True)


if __name__ == "__main__":
    main()
