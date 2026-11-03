"""Global/local pseudo-time: three samples with grid sequencing and on 400 cells."""

from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[name]="1"

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
import zipfile

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

from nozzle_hyperbolic.solvers.fvm import Operator,Scheme,SteadySettings,drivers
from nozzle_hyperbolic.verification import build_case,load_case
from .adapter import local_operator
from .common import STUDY,ROOT,GRIDS,METHODS,MODES,save,source_hashes,work,audit


def solve(physical,scheme,settings,operators,samplers,method,mode):
    captured=[]
    def on_stage(model,solution,metadata):
        captured.append((model,solution,metadata[-1]))
    grids=GRIDS if mode=="sequence" else (400,)
    jax.effects_barrier()
    started=perf_counter()
    with patch.object(drivers,"Operator",lambda model,selected:operators[method][model.mesh.n_cells]), \
         patch.object(drivers,"sampler",lambda op,policy:samplers[op.mesh.n_cells]):
        final,metadata=drivers.solve_grid_sequence(lambda n:physical[n].model,grids,scheme,settings,
            carry_cfl=True,on_stage=on_stage,quantity_evaluator_factory=lambda model:None)
    jax.block_until_ready(final.state)
    elapsed=perf_counter()-started
    stages=[]
    for model,solution,stage in captured:
        histories=[solution.history]+[attempt["history"] for attempt in stage["attempts"]]
        counts={key:sum(work(h)[key] for h in histories) for key in work(solution.history)}
        local=operators["local"][model.mesh.n_cells]
        times=np.asarray(local.local_wave_times(jnp.asarray(solution.state)))
        multipliers=times/min(times) if method=="local" else np.ones_like(times)
        stages.append(dict(cells=model.mesh.n_cells,success=solution.success,reason=solution.reason,
            initial_cfl=stage["initial_cfl"],last_accepted_cfl=stage["last_accepted_cfl"],
            stage_elapsed_s=stage["elapsed_s"],solve_elapsed_s=stage["solve_elapsed_s"],
            initialization=stage["initialization"],attempts=stage["attempts"],work=counts,
            history=solution.history,state=solution.state,metrics=solution.metrics,
            final_cell_step_multipliers=multipliers,final_local_wave_time_ratio=float(max(times)/min(times)),
            audit=audit(solution,physical[model.mesh.n_cells],settings.residual)))
    problem=physical[400]
    fields=problem.numerical_fields(final.state)
    reference=problem.cell_reference()
    weights=problem.mesh.volumes/sum(problem.mesh.volumes)
    error=float(np.sum(weights*np.abs(fields["p"]-reference["fields"]["p"]))/problem.model.p0)
    crossings=np.where((fields["M"][:-1]>1)&(fields["M"][1:]<=1))[0]
    shock=float(np.mean(problem.mesh.centers[crossings[0]:crossings[0]+2])) if len(crossings) else None
    return dict(method=method,mode=mode,elapsed_s=elapsed,success=bool(all(s["success"] for s in stages)),
        audits_passed=bool(all(s["audit"]["passed"] for s in stages)),reason=final.reason,stages=stages,
        total_work={key:sum(s["work"][key] for s in stages) for key in stages[0]["work"]},
        fallback_count=sum(bool(s["attempts"]) for s in stages),
        coordinate=problem.mesh.centers,pressure=fields["p"],mach=fields["M"],
        reference_pressure=reference["fields"]["p"],reference_shock_m=problem.shock_position,
        shock_m=shock,pressure_l1_over_p0=error,variable_scales=problem.model.variable_scales)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,default=STUDY/"outputs/comparison")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    settings=SteadySettings()
    source=source_hashes()
    client={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((STUDY/"scripts").glob("*.py"))}
    metadata=dict(grids=GRIDS,methods=METHODS,modes=MODES,measured_repetitions=3,
        excluded_warmups_per_option=1,case_ids=["euler_nozzle/shock_35","euler_nozzle/shock_70"],
        reconstructions=["constant","muscl_van_leer"],settings=asdict(settings),
        source_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_hashes=source,study_hashes=client,platform=platform.platform(),processor=platform.processor(),
        python=sys.version,packages={name:importlib.metadata.version(name) for name in ("jax","jaxlib","jaxprop","numpy","scipy")},
        backend=jax.default_backend(),devices=[str(v) for v in jax.devices()],x64=bool(jax.config.jax_enable_x64),
        threads={name:os.environ[name] for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        timing_boundary="prebound reusable kernels; entire solve/hierarchy including interpolation/monitoring/rejected work/cold retries; synchronized; excludes audits/output",
        cases={key:load_case(key).to_dict() for key in ("euler_nozzle/shock_35","euler_nozzle/shock_70")},
        method="old-state wave times, cellwise dt frozen within Newton; original coupled algebraic traces; no physical-time interpretation")
    check_path=STUDY/"outputs/implementation_checks.json"
    if check_path.exists(): metadata["implementation_checks"]=json.loads(check_path.read_text())
    save(args.output/"metadata.json",metadata)
    with zipfile.ZipFile(args.output/"source_at_measurement.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in {**source,**client}: archive.writestr(path,(ROOT/path).read_bytes())
    index=[]
    for case_id in metadata["case_ids"]:
        for recon in metadata["reconstructions"]:
            started=perf_counter()
            physical={n:build_case(case_id,cells=n) for n in GRIDS}
            scheme=Scheme(flux="hll",reconstruction=recon,newton_backtracking=True)
            global_ops={n:Operator(p.model,scheme) for n,p in physical.items()}
            operators={"global":global_ops,"local":{n:local_operator(op) for n,op in global_ops.items()}}
            samplers={n:drivers.sampler(op,settings.residual) for n,op in global_ops.items()}
            binding_s=perf_counter()-started
            for mode in MODES:
                for phase,repeats in (("warmup",1),("measured",3)):
                    for repeat in range(repeats):
                        order=METHODS if repeat%2==0 else tuple(reversed(METHODS))
                        for method in order:
                            print(f"{case_id} {recon} {mode} {phase} {repeat+1} {method}: starting",flush=True)
                            start=perf_counter()
                            try:
                                row=solve(physical,scheme,settings,operators,samplers,method,mode)
                            except Exception as exc:
                                row=dict(method=method,mode=mode,success=False,audits_passed=False,
                                    reason="exception",error=repr(exc),traceback=traceback.format_exc(),elapsed_s=perf_counter()-start)
                            row.update(case_id=case_id,reconstruction=recon,phase=phase,repeat=repeat+1,
                                       scheme=asdict(scheme),binding_s=binding_s)
                            name=f"{case_id.split('/')[-1]}_{recon}_{mode}_{phase}_{repeat+1}_{method}.json"
                            save(args.output/name,row)
                            index.append(name)
                            save(args.output/"index.json",index)
                            print(f"  {row['elapsed_s']:.6f}s success={row['success']} audit={row['audits_passed']} "
                                  f"reason={row['reason']} work={row.get('total_work')}",flush=True)
            del physical,operators,global_ops,samplers
            jax.clear_caches()
    unchanged=source_hashes()==source
    save(args.output/"source_verification.json",dict(src_unchanged=unchanged))
    assert unchanged,"src changed during benchmark"
    print("Complete: 48 measured solves and 16 excluded warmups, across both strategies.",flush=True)


if __name__=="__main__": main()
