"""Reusable production bindings for one physical case and reconstruction.

All methods use the public grid-sequence driver. A single-grid baseline is the
same driver with (400,), hence performs no interpolation/coarse solve. Models,
operators and samplers are prebound outside timers; numerical/controller state
resets on each call. Failed warm attempts and coarse stages are retained.
"""

import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="1"
os.environ["MPLBACKEND"]="Agg"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"

from contextlib import redirect_stdout
import json
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64",True)
import jax.numpy as jnp
import numpy as np

from .common import clean,identity
from nozzle_hyperbolic.verification import build_case
from nozzle_hyperbolic.solvers.fvm import Operator,Scheme,SteadySettings,drivers
from nozzle_hyperbolic.solvers.residuals import ResidualPolicy
from nozzle_hyperbolic.numerics.jacobians import make_block_operators
from nozzle_hyperbolic.solvers.time_marching.implicit import make_newton_solver


def work(history):
    return dict(accepted_steps=sum(row.get("phase")=="accepted" for row in history),
        rejected_steps=sum(row.get("phase")=="rejected" for row in history),
        newton_corrections=sum(row.get("inner_iterations",0) for row in history),
        rejected_newton_trials=sum(row.get("rejected_newton_trials",0) for row in history))


def main():
    cache={}

    def bind(command):
        config=command["config"]
        policy=ResidualPolicy(**config["residual"])
        settings=SteadySettings(residual=policy,**config["steady"])
        grids=sorted({n for method in config["methods"] for n in method["grids"]})
        physical={n:build_case(command["case_id"],n) for n in grids}
        operators={};samplers={};schemes={};methods={m["id"]:m for m in config["methods"]}
        for method in methods.values():
            scheme=Scheme(reconstruction=command["reconstruction"],**config["scheme"],
                **{key:method[key] for key in ("flux","custom_jump_correction","custom_endpoint_envelope")})
            schemes[method["id"]]=scheme
            for n in method["grids"]:
                key=(scheme,n)
                if key not in operators:
                    op=Operator(physical[n].model,scheme)
                    operators[key]=op;samplers[id(op)]=drivers.sampler(op,policy)
        cache.update(config=config,physical=physical,operators=operators,samplers=samplers,
            schemes=schemes,settings=settings,methods=methods,last={})
        return dict(case_id=command["case_id"],grids=grids,bindings=len(operators),
            implementation="src production Operator and solve_grid_sequence; unconditional local derivative reuse",
            package_file=__import__("nozzle_hyperbolic").__file__)

    def pack(model,solution,metadata):
        physical=cache["physical"][model.mesh.n_cells]
        state=np.asarray(solution.state)
        fields={name:np.asarray(value) for name,value in physical.numerical_fields(state).items()
                if isinstance(value,(jax.Array,np.ndarray,list,float,int))}
        q=np.asarray(model.cell_conserved(jnp.asarray(state[1:-1])))
        metrics=dict(solution.metrics)
        initial=next((row for row in solution.history if row.get("phase")=="initial"),{})
        metrics["initial_norm"]=initial.get("equation_norm")
        return dict(cells=model.mesh.n_cells,state=state,fields=fields,q=q,
            x=np.asarray(model.mesh.centers),volumes=np.asarray(model.mesh.volumes),
            success=solution.success,reason=solution.reason,metrics=metrics,history=solution.history,
            work=work(solution.history),elapsed_s=solution.elapsed_s,
            stage_total_s=solution.elapsed_s+sum(a["elapsed_s"] for a in metadata["attempts"]),
            initialization=metadata["initialization"],attempts=metadata["attempts"],
            **{key:metadata[key] for key in ("cfl_policy","initial_cfl","last_accepted_cfl","residual_normalization")})

    def solve(method_id):
        method=cache["methods"][method_id];scheme=cache["schemes"][method_id]
        stages=[]
        def stage(model,solution,metadata):
            # Capture references only. Field extraction and serialization are
            # deliberately deferred until the hierarchy timer has stopped.
            stages.append((model,solution,metadata[-1]))
        started=perf_counter()
        with patch.object(drivers,"Operator",lambda model,selected:cache["operators"][(selected,model.mesh.n_cells)]),\
             patch.object(drivers,"sampler",lambda op,policy:cache["samplers"][id(op)]):
            final,metadata=drivers.solve_grid_sequence(
                lambda n:cache["physical"][n].model,tuple(method["grids"]),scheme,cache["settings"],
                on_stage=stage,quantity_evaluator_factory=lambda model:None,
                **cache["config"]["grid_sequence"])
        jax.block_until_ready(final.state)
        elapsed=perf_counter()-started
        packed=[pack(model,solution,meta) for model,solution,meta in stages]
        fine=packed[-1]
        cache["last"][method_id]=fine["state"]
        return dict(elapsed_s=elapsed,success=bool(final.success),reason=final.reason,
            all_stages_success=all(s["success"] for s in packed),fine=fine,stages=packed,
            coarse_total_s=sum(s["stage_total_s"] for s in packed[:-1]),fine_total_s=fine["stage_total_s"],
            fallback_count=sum(bool(s["attempts"]) for s in packed),
            total_work={key:sum(s["work"][key]+sum(work(a["history"])[key] for a in s["attempts"]) for s in packed)
                        for key in fine["work"]})

    def prepare(method_id):
        method=cache["methods"][method_id];scheme=cache["schemes"][method_id]
        op=cache["operators"][(scheme,400)]
        full=jnp.asarray(cache["last"][method_id]);dt=jnp.float64(.2*float(op.timestep(full,1.).wave_dt))
        args=(full.ravel(),full.ravel(),dt,jnp.float64(0.))
        blocks=make_block_operators(op.residual,op.shape[0],op.shape[1],op.radius,
            jacobian_mode=scheme.jacobian_mode,linear_solver=scheme.linear_solver)
        rhs=op.residual(*args);band=jax.jit(blocks["jacobian_blocks"])(*args)
        fixed=jax.jit(make_newton_solver(op.residual,blocks["direction"],
            lambda point:jnp.broadcast_to(jnp.asarray(op.model.variable_scales),op.shape).ravel(),
            max_iterations=20,tolerance=0.,residual_tolerance=0.,backtracking=False,
            admissible=lambda flat:op.is_admissible(flat.reshape(op.shape))))
        kernels=dict(residual=(op.residual,args),jacobian_blocks=(jax.jit(blocks["jacobian_blocks"]),args),
            linear_solve=(jax.jit(blocks["solve_blocks"]),(band,-rhs)),
            direction=(op.direction,args),fixed20=(fixed,args))
        for function,values in kernels.values():
            for _ in range(3):jax.block_until_ready(function(*values))
        state,info=fixed(*args)
        if int(info.iterations)!=20 or not np.isfinite(np.asarray(state)).all():
            raise ValueError("Invalid fixed-work component diagnostic")
        cache.setdefault("kernels",{})[method_id]=kernels
        return dict(cells=400,state_sha256=identity(np.asarray(full)),dt=float(dt),colors=op.shape[1]*(2*op.radius+1),
            fixed_corrections=20,excluded_calls=3,inner_admissibility=True,local_derivative_reuse="always enabled")

    def sample(method_id):
        result={}
        for label,(function,args) in cache["kernels"][method_id].items():
            started=perf_counter();value=function(*args);jax.block_until_ready(value)
            result[label]=perf_counter()-started
        return result

    print('{"ready":true,"backend":"production"}',flush=True)
    for line in sys.stdin:
        try:
            command=json.loads(line)
            if command["command"]=="shutdown":print('{"ok":true}',flush=True);break
            with redirect_stdout(sys.stderr):
                kind=command["command"]
                if kind=="bind":result=bind(command)
                elif kind=="solve":result=solve(command["method"])
                elif kind=="prepare_kernels":result=prepare(command["method"])
                elif kind=="sample_kernels":result=sample(command["method"])
                else:raise ValueError("Unknown production worker command")
            print(json.dumps(clean(dict(ok=True,result=result)),allow_nan=False),flush=True)
        except Exception as error:
            print(json.dumps(dict(ok=False,error=repr(error),traceback=traceback.format_exc())),flush=True)


if __name__=="__main__":main()
